from ase.io import read
from gpaw import GPAW, PW

# Read structure
atoms = read("structure.cif")

# Setup calculator
calc = GPAW(
    mode=PW(ecut=250),
    xc="PBE",
    kpts={"size": (1, 1, 1), "gamma": True},
    txt="gpaw_output.txt",
    convergence={},
    charge=1,
    spinpol=True,
)

atoms.calc = calc

print("Starting calculation...")
energy = atoms.get_potential_energy()
print(f"Initial Energy: {energy} eV")

# Relaxation

# Save the final state
calc.write("relaxed.gpw.gz")

print(f"Final Energy: {atoms.get_potential_energy()} eV")

# Force static energy calculation (unrelaxed) and save
energy = atoms.get_potential_energy()
calc.write("relaxed.gpw.gz")
