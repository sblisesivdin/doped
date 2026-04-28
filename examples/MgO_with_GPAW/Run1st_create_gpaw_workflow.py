import os
from pymatgen.core import Structure
from doped.generation import DefectsGenerator
from doped.gpaw import GPAWDefectRelaxSet

def main():
    # Point to the relaxed bulk POSCAR
    poscar_path = "../MgO/Bulk_relax/POSCAR" 
    
    print(f"Loading MgO bulk structure from {poscar_path}...")
    bulk_structure = Structure.from_file(poscar_path)

    print("Generating MgO defects...")
    # Generate the defects using doped
    defect_gen = DefectsGenerator(bulk_structure)

    # GPAW Parameters
    gpaw_kwargs = {
        "mode": "PW",
        "ecut": 250,
        "kpts": (1, 1, 1),
        "xc": "PBE",
        "symmetry": "off"
    }

    print("Writing GPAW input files...")
    # Setup Bulk
    # Note: For bulk, we need to use a supercell so it matches the defect size!
    # The DefectsGenerator automatically determines the optimal supercell matrix.
    bulk_supercell = bulk_structure.copy()
    bulk_supercell.make_supercell(defect_gen.supercell_matrix)
    
    bulk_set = GPAWDefectRelaxSet(bulk_supercell, **gpaw_kwargs)
    bulk_set.write_input("bulk")

    # Setup Defects
    for defect_name, defect_entry in defect_gen.items():
        # To save time and space, let's just generate the Magnesium Vacancy
        # specifically the -2 charge state which is typically used for tests.
        if "v_Mg" not in defect_name:
            continue

        print(f"Setting up {defect_name}...")
        
        # defect_entry.defect.get_supercell_structure() gets the defective supercell
        defect_struct = defect_entry.defect.get_supercell_structure(
            sc_mat=defect_gen.supercell_matrix
        )
        
        # defect_entry.charge_state gives the specific charge for this entry
        defect_set = GPAWDefectRelaxSet(
            defect_struct, 
            charge=defect_entry.charge_state, 
            **gpaw_kwargs
        )
        defect_set.write_input(defect_name)

    print("Workflow setup complete! You can now run the calculations.")

if __name__ == "__main__":
    main()
