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
        and calculates the eFNV (Kumagai) correction using real static .gpw files.
        """
        # Path to the static test data directories
        gpaw_mgo_dir = os.path.join(self.data_dir, "gpaw_mgo_test")
        gpaw_bulk_dir = os.path.join(gpaw_mgo_dir, "bulk")
        
        # Ensure the test directories exist
        self.assertTrue(os.path.exists(gpaw_bulk_dir), "Bulk test directory missing!")
        
        # Initialize the parser exactly as it's used in the automation script
        dp_gpaw = GPAWDefectsParser(
            output_path=gpaw_mgo_dir, 
            bulk_path=gpaw_bulk_dir, 
            dielectric=10.0 
        )
        
        # Parse the defects
        defect_dict = dp_gpaw.parse_all()
        self.assertGreater(len(defect_dict), 0, "No defects were parsed!")
        
        # Extract the defect entry and its corrections dict
        # Since we only saved one defect folder (v_Mg_-2), it will be the only item
        defect_entry = list(defect_dict.values())[0]
        corrections = defect_entry.corrections
        
        # Verify the Kumagai correction exists
        self.assertIn('kumagai_charge_correction', corrections, "Kumagai correction missing from parsed output!")
        
        # Verify the value matches the exact output of our local coarse calculation
        expected_energy = 0.588754 
        calculated_energy = float(corrections['kumagai_charge_correction'])
        
        # Assert they match tightly
        np.testing.assert_allclose(
            calculated_energy, 
            expected_energy, 
            atol=1e-5, 
            err_msg=f"GPAW Kumagai calculation changed! Expected ~{expected_energy}, got {calculated_energy}"
        )

if __name__ == "__main__":
    unittest.main()
