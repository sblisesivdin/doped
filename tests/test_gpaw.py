import unittest
import os
import shutil
import numpy as np
from pymatgen.core.structure import Structure
from doped.gpaw import GPAWDefectRelaxSet, GPAWDefectsParser

class GPAWTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.output_dir = "gpaw_test_outputs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Create a simple structure for testing input generation
        self.structure = Structure.from_file(os.path.join(self.data_dir, "Cu_prim_POSCAR"))

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_gpaw_defect_relax_set(self):
        # Test with Structure
        relax_set = GPAWDefectRelaxSet(self.structure, charge_state=1)
        relax_set.write_input(self.output_dir)
        
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "relax.py")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "structure.cif")))
        
        with open(os.path.join(self.output_dir, "relax.py"), "r") as f:
            content = f.read()
            self.assertIn("charge=1", content)
            self.assertIn("mode=PW(ecut=400)", content) # Default

    def test_gpaw_defect_relax_set_custom(self):
        # Test with custom settings
        gpaw_settings = {
            "mode": {"name": "pw", "ecut": 400},
            "xc": "PBE",
            "kpts": {"size": (2, 2, 2), "gamma": True},
        }
        relax_set = GPAWDefectRelaxSet(self.structure, charge_state=-1, gpaw_settings=gpaw_settings, poscar_comment="Test Comment")
        relax_set.write_input(self.output_dir)
        
        self.assertEqual(relax_set.poscar_comment, "Test Comment")
        
        with open(os.path.join(self.output_dir, "relax.py"), "r") as f:
            content = f.read()
            self.assertIn("charge=-1", content)
            self.assertIn("mode=PW(ecut=400)", content)
            self.assertIn("'size': (2, 2, 2)", content)
            self.assertIn("from gpaw import GPAW, PW, LCAO, FD", content)

    def test_gpaw_defect_relax_set_lcao(self):
        # Test with LCAO mode
        gpaw_settings = {
            "mode": {"name": "lcao", "basis": "dzp"},
        }
        relax_set = GPAWDefectRelaxSet(self.structure, charge_state=0, gpaw_settings=gpaw_settings)
        relax_set.write_input(self.output_dir)
        
        with open(os.path.join(self.output_dir, "relax.py"), "r") as f:
            content = f.read()
            self.assertIn("mode=LCAO(basis='dzp')", content)
            self.assertIn("from gpaw import GPAW, PW, LCAO, FD", content)

    def test_gpaw_kumagai_correction_mgo(self):
        """
        Test that the GPAW parser correctly extracts electrostatic potentials 
        and calculates the eFNV (Kumagai) correction for multiple charge states
        using real static .gpw files.
        """
        # Path to the static test data directories
        gpaw_mgo_dir = os.path.join(self.data_dir, "gpaw_mgo_test")
        gpaw_bulk_dir = os.path.join(gpaw_mgo_dir, "bulk")
        
        self.assertTrue(os.path.exists(gpaw_bulk_dir), "Bulk test directory missing!")
        
        # Initialize the parser
        dp_gpaw = GPAWDefectsParser(
            output_path=gpaw_mgo_dir, 
            bulk_path=gpaw_bulk_dir, 
            dielectric=10.0 
        )
        
        # Parse all relaxed defects in the folder
        defect_dict = dp_gpaw.parse_all()
        self.assertGreaterEqual(len(defect_dict), 4, "Not all defects were parsed!")
        
        # Expected Kumagai corrections mapped by charge state for relaxed defects
        expected_corrections = {
            1: 0.303790,
            -1: 0.091121,
            -2: 0.575566,
            0: 0.0  # Neutral defects have no Kumagai correction
        }
        
        for defect_name, entry in defect_dict.items():
            charge = entry.charge_state
            if charge in expected_corrections:
                expected_energy = expected_corrections[charge]
                if expected_energy == 0.0:
                    self.assertNotIn('kumagai_charge_correction', entry.corrections)
                else:
                    self.assertIn('kumagai_charge_correction', entry.corrections)
                    calculated_energy = float(entry.corrections['kumagai_charge_correction'])
                    np.testing.assert_allclose(
                        calculated_energy, expected_energy, atol=1e-4, 
                        err_msg=f"Failed for charge state {charge}!"
                    )

        # --- Explicitly Test the Unrelaxed +1 State ---
        # We parse this manually to avoid dictionary key collisions with the relaxed state
        unrelaxed_dir = os.path.join(gpaw_mgo_dir, "v_Mg_+1_unrelaxed")
        if os.path.exists(unrelaxed_dir):
            from doped.gpaw import get_gpaw_defect_entry
            from doped.gpaw import GPAWParser
            
            bulk_parser = GPAWParser(os.path.join(gpaw_bulk_dir, "relaxed.gpw"))
            unrelaxed_entry = get_gpaw_defect_entry(
                defect_path=unrelaxed_dir, 
                bulk_path=gpaw_bulk_dir, 
                dielectric=10.0,
                charge_state=1,
                bulk_parser=bulk_parser
            )
            unrelaxed_entry.get_kumagai_correction()
            
            self.assertIn('kumagai_charge_correction', unrelaxed_entry.corrections)
            calculated_unrelaxed = float(unrelaxed_entry.corrections['kumagai_charge_correction'])
            
            # Since atoms barely move in this coarse test, it should identically match the relaxed value
            np.testing.assert_allclose(
                calculated_unrelaxed, 0.303790, atol=1e-4,
                err_msg="Failed for unrelaxed +1 state!"
            )
            
    def test_gpaw_freysoldt_correction_mgo(self):
        """
        Test that the GPAW parser supports the Freysoldt (FNV) correction
        via manual invocation after parsing, using the MgO test data for all charge states.
        """
        gpaw_mgo_dir = os.path.join(self.data_dir, "gpaw_mgo_test")
        gpaw_bulk_dir = os.path.join(gpaw_mgo_dir, "bulk")
        
        self.assertTrue(os.path.exists(gpaw_bulk_dir), "MgO bulk test directory missing!")
        
        # Initialize the parser
        dp_gpaw = GPAWDefectsParser(
            output_path=gpaw_mgo_dir, 
            bulk_path=gpaw_bulk_dir, 
            dielectric=10.0 
        )
        
        # Parse the relaxed defects
        defect_dict = dp_gpaw.parse_all()
        self.assertGreaterEqual(len(defect_dict), 4, "Not all relaxed MgO defects were parsed!")
        
        print("\n--- Calculated Freysoldt (FNV) Corrections ---")
        
        # 1. Loop through all relaxed parsed defects
        for defect_name, defect_entry in defect_dict.items():
            if defect_entry.charge_state == 0:
                continue  # Neutral defects do not get charge corrections
                
            # Manually apply the Freysoldt (FNV) correction
            freysoldt_correction = defect_entry.get_freysoldt_correction()
            
            # Verify that the FNV correction was successfully calculated and stored
            self.assertIsNotNone(freysoldt_correction, f"FNV failed for {defect_name}")
            self.assertIn('freysoldt_charge_correction', defect_entry.corrections)
            
            # Ensure it calculates a valid float value
            calculated_energy = float(defect_entry.corrections['freysoldt_charge_correction'])
            self.assertIsInstance(calculated_energy, float)
            print(f"{defect_name} (Charge {defect_entry.charge_state}): {calculated_energy:.4f} eV")

        # 2. Explicitly Test the Unrelaxed +1 State
        unrelaxed_dir = os.path.join(gpaw_mgo_dir, "v_Mg_+1_unrelaxed")
        if os.path.exists(unrelaxed_dir):
            from doped.gpaw import get_gpaw_defect_entry, GPAWParser
            
            bulk_parser = GPAWParser(os.path.join(gpaw_bulk_dir, "relaxed.gpw"))
            unrelaxed_entry = get_gpaw_defect_entry(
                defect_path=unrelaxed_dir, 
                bulk_path=gpaw_bulk_dir, 
                dielectric=10.0,
                charge_state=1,
                bulk_parser=bulk_parser
            )
            
            unrelaxed_entry.get_freysoldt_correction()
            
            self.assertIn('freysoldt_charge_correction', unrelaxed_entry.corrections)
            calculated_unrelaxed = float(unrelaxed_entry.corrections['freysoldt_charge_correction'])
            self.assertIsInstance(calculated_unrelaxed, float)
            print(f"v_Mg_+1_unrelaxed (Charge 1): {calculated_unrelaxed:.4f} eV")
    
    def test_gpaw_graphene_2d_handling(self):
        """
        Test that the GPAW parser handles highly anisotropic 2D supercells (Graphene)
        without crashing during the Kumagai correction / defect region radius calculation.
        Tests multiple defects spanning vacancies, interstitials, and substitutions to ensure robustness.
        """
        # Path to the static test data directories
        gpaw_graphene_dir = os.path.join(self.data_dir, "gpaw_graphene_test")
        gpaw_bulk_dir = os.path.join(gpaw_graphene_dir, "bulk")
        
        self.assertTrue(os.path.exists(gpaw_bulk_dir), "Graphene bulk test directory missing!")
        
        # Initialize the parser
        dp_gpaw = GPAWDefectsParser(
            output_path=gpaw_graphene_dir, 
            bulk_path=gpaw_bulk_dir, 
            dielectric=10.0 # Dummy dielectric
        )
        
        # Parse the graphene defects
        defect_dict = dp_gpaw.parse_all()
        self.assertGreaterEqual(len(defect_dict), 5, "Not enough Graphene defects were parsed!")
        
        # Expected Kumagai corrections mapped by defect name (values from local run)
        # Note: The +4 charge state correction is ~41 eV due to the q^2 scaling 
        # of the charge correction in a small 2D supercell.
        expected_corrections = {
            "v_C_D3h_C1.42_+1": 2.5146,
            "C_i_C3v_C2.00_+4": 41.0205,
            "N_i_C3v_C2.00_-3": 4.8420,
            "v_C_D3h_C1.42_-1": 1.5464,
            "N_C_D3h_C1.42_-2": 1.5279
        }
        
        for defect_name, expected_energy in expected_corrections.items():
            self.assertIn(defect_name, defect_dict, f"{defect_name} missing from parsed defects!")
            entry = defect_dict[defect_name]
            
            # Verify the Kumagai correction was calculated (even if physically inaccurate for 2D)
            self.assertIn('kumagai_charge_correction', entry.corrections)
            calculated_energy = float(entry.corrections['kumagai_charge_correction'])
            
            np.testing.assert_allclose(
                calculated_energy, expected_energy, atol=1e-3, 
                err_msg=f"Graphene 2D Kumagai calculation failed for {defect_name}!"
            )   

if __name__ == "__main__":
    unittest.main()
