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

if __name__ == "__main__":
    unittest.main()
