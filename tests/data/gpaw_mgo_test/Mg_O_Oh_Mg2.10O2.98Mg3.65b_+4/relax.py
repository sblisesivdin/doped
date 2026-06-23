from ase.io import read
from ase.optimize import BFGS
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
    charge=4,
    spinpol=True,
)

atoms.calc = calc

print("Starting calculation...")
energy = atoms.get_potential_energy()
print(f"Initial Energy: {energy} eV")

# Relaxation
dyn = BFGS(atoms, trajectory="relax.traj")
dyn.run(fmax=0.05)

# Save the final state
calc.write("relaxed.gpw.gz")

print(f"Final Energy: {atoms.get_potential_energy()} eV")
