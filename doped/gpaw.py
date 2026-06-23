"""
Code to generate and parse GPAW defect calculations.
"""

import os
from typing import Any

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.interpolate import RegularGridInterpolator

from doped.analysis import defect_from_structures
from doped.core import Defect, DefectEntry
from doped.utils.parsing import _get_defect_supercell


class GPAWDefectRelaxSet:
    """
    Class for generating input files (Python scripts) for GPAW defect
    relaxation.
    """

    def __init__(
        self,
        defect_entry: DefectEntry | Structure,
        charge_state: int | None = None,
        gpaw_settings: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Args:
            defect_entry (DefectEntry, Structure):
                doped/pymatgen DefectEntry or Structure object.
            charge_state (int):
                Charge state of the defect. Overrides DefectEntry.charge_state.
            gpaw_settings (dict):
                Dictionary of GPAW settings. Defaults used if not specified:
                - "mode": {"name": "pw", "ecut": 400}
                - "xc": "PBE"
                - "kpts": {"size": (1, 1, 1), "gamma": True}
                - "txt": "gpaw_output.txt"
                - "spinpol": True
                - "fmax": 0.05
                - "optimizer": "BFGS"
            **kwargs:
                Additional keyword arguments.
        """
        self.defect_entry = defect_entry
        self.charge_state = charge_state
        if self.charge_state is None:
            self.charge_state = kwargs.get("charge")  # Catch it if passed as kwarg
        if self.charge_state is None and isinstance(self.defect_entry, DefectEntry):
            self.charge_state = self.defect_entry.charge_state

        self.gpaw_settings = gpaw_settings or {}
        self.kwargs = kwargs
        self.poscar_comment = kwargs.get("poscar_comment")

        if isinstance(self.defect_entry, Structure):
            self.defect_supercell = self.defect_entry
        elif isinstance(self.defect_entry, DefectEntry):
            self.defect_supercell = _get_defect_supercell(self.defect_entry)

    def write_input(
        self,
        output_path: str,
        filename: str = "relax.py",
        make_dir_if_not_present: bool = True,
    ):
        """
        Writes the input files (structure and script) to a directory.
        """
        if make_dir_if_not_present:
            os.makedirs(output_path, exist_ok=True)

        # Write structure to a file
        structure_filename = "structure.cif"

        from pymatgen.io.cif import CifWriter

        # Do not use symprec arg inside CifWriter. It reduces supercells to primitives.
        writer = CifWriter(self.defect_supercell)
        writer.write_file(os.path.join(output_path, structure_filename))

        # Generate Python script
        script_content = self._generate_script(structure_filename)

        with open(os.path.join(output_path, filename), "w") as f:
            f.write(script_content)

    def _generate_script(self, structure_filename: str) -> str:
        """
        Generates the content of the GPAW script.
        """
        settings = self.gpaw_settings.copy()

        # Extract known parameters
        mode_params = settings.pop("mode", {"name": "pw", "ecut": 400})
        xc = settings.pop("xc", "PBE")
        kpts = settings.pop("kpts", {"size": (1, 1, 1), "gamma": True})
        txt = settings.pop("txt", "gpaw_output.txt")
        convergence = settings.pop("convergence", {})
        optimizer = settings.pop("optimizer", "BFGS")

        # Determine charge
        charge = self.charge_state or 0

        # Determine spinpol (default True for defects if not specified)
        spinpol = settings.pop("spinpol", True)

        # Relaxation params
        fmax = settings.pop("fmax", 0.05)

        # Prepare mode string
        if isinstance(mode_params, dict):
            name = mode_params.pop("name", "pw")
            args = ", ".join([f"{k}={v!r}" for k, v in mode_params.items()])
            mode_str = f"{name.upper()}({args})"
        else:
            mode_str = repr(mode_params)

        # Prepare other settings
        other_kwargs = ""
        if settings:
            other_kwargs = ",\n    " + ",\n    ".join([f"{k}={v!r}" for k, v in settings.items()])

        return f"""
from ase.io import read
from gpaw import GPAW, PW, LCAO, FD
from ase.optimize import {optimizer}
import json

# Read structure
atoms = read('{structure_filename}')

# Setup calculator
calc = GPAW(
    mode={mode_str},
    xc='{xc}',
    kpts={kpts},
    txt='{txt}',
    convergence={convergence},
    charge={charge},
    spinpol={spinpol}{other_kwargs}
)

atoms.calc = calc

print("Starting calculation...")
energy = atoms.get_potential_energy()
print(f"Initial Energy: {{energy}} eV")

# Relaxation
dyn = {optimizer}(atoms, trajectory='relax.traj')
dyn.run(fmax={fmax})

# Save the final state
calc.write('relaxed.gpw')

print(f"Final Energy: {{atoms.get_potential_energy()}} eV")
"""


def _get_site_potentials_from_calc(calc, beta_bohr: float = 1.5) -> np.ndarray:
    """
    Helper to extract site potentials from a GPAW calculator using Gaussian
    spherical averaging in reciprocal space.
    """
    atoms = calc.get_atoms()
    v_ext = calc.get_electrostatic_potential()  # 3D grid in eV natively
    nx, ny, nz = v_ext.shape

    # Setup reciprocal lattice and broadening
    ang_to_bohr = 1.8897259886
    reci_cell = atoms.cell.reciprocal() * 2 * np.pi

    dgx = np.linalg.norm(reci_cell[0]) / ang_to_bohr
    dgy = np.linalg.norm(reci_cell[1]) / ang_to_bohr
    dgz = np.linalg.norm(reci_cell[2]) / ang_to_bohr

    gx = np.roll(np.arange(-nx // 2, nx // 2, 1, dtype=int), int(nx // 2)) * dgx
    gy = np.roll(np.arange(-ny // 2, ny // 2, 1, dtype=int), int(ny // 2)) * dgy
    gz = np.roll(np.arange(-nz // 2, nz // 2, 1, dtype=int), int(nz // 2)) * dgz

    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing="ij")
    g2 = Gx**2 + Gy**2 + Gz**2

    # Gaussian averaging via FFT
    gaussian = np.exp(-0.5 * (beta_bohr**2) * g2)

    v_G = np.fft.fftn(v_ext)
    v_G *= gaussian
    smoothed_potential = np.real(np.fft.ifftn(v_G))

    # Robust Parsing Logic
    xpoints = np.linspace(0.0, 1.0, nx, endpoint=False)
    ypoints = np.linspace(0.0, 1.0, ny, endpoint=False)
    zpoints = np.linspace(0.0, 1.0, nz, endpoint=False)

    # pad the grid with periodic images so (cubic) interpolation works at cell boundaries:
    xpoints_padded = np.concatenate([xpoints[-1:] - 1.0, xpoints, xpoints[:1] + 1.0])
    ypoints_padded = np.concatenate([ypoints[-1:] - 1.0, ypoints, ypoints[:1] + 1.0])
    zpoints_padded = np.concatenate([zpoints[-1:] - 1.0, zpoints, zpoints[:1] + 1.0])

    padded = np.concatenate(
        [smoothed_potential[-1:, :, :], smoothed_potential, smoothed_potential[:1, :, :]], axis=0
    )
    padded = np.concatenate([padded[:, -1:, :], padded, padded[:, :1, :]], axis=1)
    padded = np.concatenate([padded[:, :, -1:], padded, padded[:, :, :1]], axis=2)

    interpolator = RegularGridInterpolator(
        (xpoints_padded, ypoints_padded, zpoints_padded),
        padded,
        method="cubic",
        bounds_error=True,
    )

    atomic_site_potentials = np.zeros(len(atoms))
    for i, frac in enumerate(atoms.get_scaled_positions()):
        # Need to use fractional coordinates modulo 1.0 to interpolate the potentials
        atomic_site_potentials[i] = float(interpolator(frac % 1.0)[0])

    return atomic_site_potentials


def _get_planar_averaged_potential_from_calc(calc) -> dict[str, np.ndarray]:
    """
    Helper to extract planar-averaged potentials from a GPAW calculator.
    """
    v_ext = calc.get_electrostatic_potential()
    planar_averages = {}
    for i in range(3):
        axes = [0, 1, 2]
        axes.remove(i)
        planar_averages[str(i)] = v_ext.mean(axis=tuple(axes))

    return planar_averages


def get_gpaw_site_potentials(gpw_file: str) -> np.ndarray:
    """
    Extracts atomic site potentials from a GPAW .gpw file.
    """
    from gpaw import GPAW

    calc = GPAW(gpw_file)
    site_potentials = _get_site_potentials_from_calc(calc)

    if hasattr(calc, "close"):
        calc.close()

    if hasattr(calc, "atoms") and calc.atoms:
        calc.atoms.calc = None

    return site_potentials


def get_gpaw_planar_averaged_potential(gpw_file: str) -> dict[str, np.ndarray]:
    """
    Extracts planar-averaged potential from a GPAW .gpw file.
    """
    from gpaw import GPAW

    calc = GPAW(gpw_file)
    planar_averages = _get_planar_averaged_potential_from_calc(calc)

    if hasattr(calc, "close"):
        calc.close()

    return planar_averages


class GPAWParser:
    """
    Parser for GPAW calculations to interface with doped.

    Note:
        The Kumagai (eFNV) finite-size charge correction is applied by default
        during parsing, as it is generally preferred. However, the standard
        Freysoldt (FNV) correction is also fully supported. If preferred, users
        can manually apply it to the parsed defects using:
        `defect_entry.get_freysoldt_correction()`
    """

    def __init__(self, gpw_file: str):
        """
        Args:
            gpw_file (str): Path to GPAW .gpw file.
        """
        from gpaw import GPAW

        self.gpw_file = gpw_file
        self.calc = GPAW(gpw_file)
        self.atoms = self.calc.get_atoms()
        self.structure = AseAtomsAdaptor.get_structure(self.atoms)
        self.energy = self.calc.get_potential_energy()

        # Pull charge directly from calculation parameters
        try:
            self.charge = self.calc.parameters.get("charge", None)
        except Exception:
            self.charge = None

    def get_computed_structure_entry(self) -> ComputedStructureEntry:
        """
        Returns a ComputedStructureEntry for the calculation.
        """
        return ComputedStructureEntry(self.structure, self.energy)

    def get_computed_entry(self) -> ComputedEntry:
        """
        Returns a ComputedEntry for the calculation.
        """
        return ComputedEntry(self.structure.composition, self.energy)

    def get_site_potentials(self) -> np.ndarray:
        """
        Returns atomic site potentials.
        """
        return _get_site_potentials_from_calc(self.calc)

    def get_locpot_dict(self) -> dict[str, np.ndarray]:
        """
        Returns planar-averaged potential dictionary.
        """
        return _get_planar_averaged_potential_from_calc(self.calc)

    def get_eigenvalue_properties(self) -> tuple:
        """
        Returns (band_gap, cbm, vbm, efermi).
        """
        # Basic implementation
        efermi = self.calc.get_fermi_level()
        # GPAW can give eigenvalues for each k-point and spin
        # This is a simplification to get VBM/CBM
        energies = []
        for s in range(self.calc.get_number_of_spins()):
            for k in range(len(self.calc.get_ibz_k_points())):
                energies.extend(self.calc.get_eigenvalues(kpt=k, spin=s))

        energies = sorted(energies)
        # Identify VBM and CBM based on efermi
        vbm = max([e for e in energies if e <= efermi]) if any(e <= efermi for e in energies) else efermi
        cbm = min([e for e in energies if e > efermi]) if any(e > efermi for e in energies) else efermi
        band_gap = cbm - vbm

        return band_gap, cbm, vbm, efermi

    def close(self):
        """
        Closes the underlying GPAW calculator.
        """
        if hasattr(self.calc, "close"):
            self.calc.close()

        # Break reference cycle
        if self.atoms:
            self.atoms.calc = None
        self.calc = None
        self.atoms = None


def get_gpaw_defect_entry(
    defect_path: str,
    bulk_path: str,
    dielectric: float | np.ndarray | None = None,
    charge_state: int = 0,
    bulk_parser: GPAWParser | None = None,
) -> DefectEntry:
    """
    Convenience function to create a DefectEntry from GPAW directories.

    Assumes 'relaxed.gpw' exists in both directories.
    """
    defect_parser = GPAWParser(os.path.join(defect_path, "relaxed.gpw"))

    close_bulk = False
    if bulk_parser is None:
        bulk_parser = GPAWParser(os.path.join(bulk_path, "relaxed.gpw"))
        close_bulk = True

    # Identify defect
    defect = defect_from_structures(defect_parser.structure, bulk_parser.structure)
    assert isinstance(defect, Defect)  # typing

    # Band edge data
    band_gap, cbm, vbm, efermi = bulk_parser.get_eigenvalue_properties()

    defect_entry = DefectEntry(
        defect=defect,
        charge_state=charge_state,
        sc_entry=defect_parser.get_computed_structure_entry(),
        bulk_entry=bulk_parser.get_computed_structure_entry(),
        sc_defect_frac_coords=defect.site.frac_coords,
        defect_supercell=defect_parser.structure,
        bulk_supercell=bulk_parser.structure,
        defect_supercell_site=defect.site,
        calculation_metadata={
            "bulk_path": bulk_path,
            "defect_path": defect_path,
            "dielectric": dielectric,
            "bulk_site_potentials": bulk_parser.get_site_potentials(),
            "defect_site_potentials": defect_parser.get_site_potentials(),
            "bulk_locpot_dict": bulk_parser.get_locpot_dict(),
            "defect_locpot_dict": defect_parser.get_locpot_dict(),
            "vbm": vbm,
            "band_gap": band_gap,
            "cbm": cbm,
            "efermi": efermi,
        },
    )

    defect_parser.close()
    if close_bulk:
        bulk_parser.close()

    return defect_entry


class GPAWDefectsParser:
    """
    Class for rapidly parsing multiple GPAW defect supercell calculations.
    """

    def __init__(
        self,
        output_path: str = ".",
        bulk_path: str | None = None,
        dielectric: float | np.ndarray | None = None,
        subfolder: str | None = None,
    ):
        """
        Args:
            output_path (str): Path to directory containing defect folders.
            bulk_path (str): Path to bulk reference folder.
            dielectric (float or matrix): Dielectric constant for corrections.
            subfolder (str): Optional subfolder within each defect folder.
        """
        self.output_path = output_path
        self.dielectric = dielectric
        self.subfolder = subfolder

        if bulk_path is None:
            # Try to find bulk folder
            folders = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]
            bulk_folders = [f for f in folders if "bulk" in f.lower()]
            if not bulk_folders:
                raise ValueError("Could not find bulk folder. Please specify bulk_path.")
            self.bulk_path = os.path.join(output_path, bulk_folders[0])
        else:
            self.bulk_path = bulk_path

    def parse_all(self) -> dict[str, DefectEntry]:
        """
        Parses all defect folders in output_path.
        """
        defect_dict = {}
        folders = [
            f for f in os.listdir(self.output_path) if os.path.isdir(os.path.join(self.output_path, f))
        ]

        # Exclude bulk folder
        defect_folders = [
            f
            for f in folders
            if os.path.abspath(os.path.join(self.output_path, f)) != os.path.abspath(self.bulk_path)
        ]

        # Instantiate bulk parser once
        bulk_parser = GPAWParser(os.path.join(self.bulk_path, "relaxed.gpw"))

        for folder in defect_folders:
            defect_dir = os.path.join(self.output_path, folder)
            if self.subfolder:
                defect_dir = os.path.join(defect_dir, self.subfolder)

            if not os.path.exists(os.path.join(defect_dir, "relaxed.gpw")):
                continue

            print(f"Parsing {folder}...")
            try:
                # Get charge from the calculation file directly
                defect_parser = GPAWParser(os.path.join(defect_dir, "relaxed.gpw"))
                charge_state = defect_parser.charge
                defect_parser.close()

                # Fallback to folder name only if calculation lacked a charge parameter
                if charge_state is None:
                    charge_state = 0
                    if "_" in folder:
                        suffix = folder.rsplit("_", 1)[-1]
                        if suffix.startswith(("+", "-")):
                            charge_state = int(suffix)

                defect_entry = get_gpaw_defect_entry(
                    defect_dir,
                    self.bulk_path,
                    dielectric=self.dielectric,
                    charge_state=charge_state,
                    bulk_parser=bulk_parser,
                )

                # Apply Kumagai correction if possible
                if self.dielectric is not None and charge_state != 0:
                    try:
                        defect_entry.get_kumagai_correction()
                    except Exception as e:
                        print(f"Warning: Kumagai correction failed for {folder}: {e}")

                defect_dict[defect_entry.name] = defect_entry
            except Exception as e:
                print(f"Failed to parse {folder}: {e}")

        bulk_parser.close()

        return defect_dict
