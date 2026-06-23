import os
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from doped.generation import DefectsGenerator
from doped.gpaw import GPAWDefectRelaxSet

def main():
    print("Generating MgO bulk structure natively...")
    # Native structure generation
    lattice = Lattice.cubic(4.21)
    bulk_structure = Structure.from_spacegroup("Fm-3m", lattice, ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    print("Generating MgO defects (forcing Mg_O antisites)...")
    # Force the generation of the Mg_O antisite using the extrinsic flag
    defect_gen = DefectsGenerator(bulk_structure, extrinsic={"O": "Mg"})

    # GPAW Parameters
    gpaw_settings = {
        "mode": {"name": "pw", "ecut": 250},
        "kpts": {"size": (1, 1, 1), "gamma": True},
        "xc": "PBE"
    }

    print("Writing GPAW input files...")
    
    # Setup Bulk using the finalized API parameters
    bulk_set = GPAWDefectRelaxSet(
        defect_gen.bulk_supercell, 
        charge_state=0, 
        gpaw_settings=gpaw_settings
    )
    bulk_set.write_input("bulk")

    # Setup Defects
    for defect_name, defect_entry in defect_gen.defect_entries.items():
        #Example of Filtering defects: Allow both Magnesium Vacancies AND Magnesium Antisites.
        #if "v_Mg" not in defect_name and "Mg_O" not in defect_name:
        #    continue

        print(f"Setting up {defect_name}...")
        
        # Pass the entry directly, and use the gpaw_settings dictionary
        defect_set = GPAWDefectRelaxSet(
            defect_entry, 
            charge_state=defect_entry.charge_state, 
            gpaw_settings=gpaw_settings
        )
        defect_set.write_input(defect_name)
        
        # Create an example of an unrelaxed structure for the +1 state
        if "+1" in defect_name:
            unrelaxed_name = defect_name + "_unrelaxed"
            print(f"Setting up {unrelaxed_name}...")
            
            defect_set.write_input(unrelaxed_name)
            
            # Open the generated relax.py and modify it for a static single-point run
            relax_file = os.path.join(unrelaxed_name, "relax.py")
            with open(relax_file, "r") as f:
                lines = f.readlines()
                
            with open(relax_file, "w") as f:
                for line in lines:
                    # Strip out the ASE optimizer logic robustly
                    if "ase.optimize" in line or "BFGS" in line or "dyn" in line:
                        continue
                    f.write(line)
                
                # Force a static energy calculation and save the .gpw file
                f.write("\n# Force static energy calculation (unrelaxed) and save\n")
                f.write("energy = atoms.get_potential_energy()\n")
                f.write("calc.write('relaxed.gpw')\n")

    print("Workflow setup complete! You can now run the calculations.")

if __name__ == "__main__":
    main()
