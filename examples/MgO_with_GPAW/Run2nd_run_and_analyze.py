import os
import sys
import subprocess
from doped.gpaw import GPAWDefectsParser
from doped.analysis import DefectThermodynamics

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
bulk_dir = os.path.join(base_dir, "bulk")
CORES = 8 # Set number of cores for MPI

def run_calculation(folder):
    """Runs the relax.py script in the given folder."""
    print(f"Processing {folder}...")
    relax_script = os.path.join(folder, "relax.py")
    
    if os.path.exists(os.path.join(folder, "relaxed.gpw.gz")):
        print(f"  relaxed.gpw.gz exists, skipping calculation.")
        return

    if os.path.exists(os.path.join(folder, "gpaw_output.txt")):
        print(f"  gpaw_output.txt exists but relaxed.gpw.gz does not. Skipping likely failed calculation.")
        return

    if not os.path.exists(relax_script):
        print(f"  relax.py not found in {folder}, skipping.")
        return

    try:
        if CORES > 1:
            cmd = ["mpirun", "-np", str(CORES), "gpaw", "python", "relax.py"]
        else:
            # Use sys.executable to ensure it runs in the exact same Python environment!
            cmd = [sys.executable, "relax.py"]

        # Capture output instead of using DEVNULL so we can see actual errors
        subprocess.run(cmd, cwd=folder, check=True, capture_output=True, text=True)
        print(f"  Calculation completed.")
        
    except subprocess.CalledProcessError as e:
        # Smart check for the MPI garbage collection errors we discussed previously
        if os.path.exists(os.path.join(folder, "relaxed.gpw.gz")):
            print("  Done!")
        else:
            print(f"  Calculation failed. Error log:\n{e.stderr}")
    except FileNotFoundError:
        print(f"  Execution failed. Check if mpirun/gpaw are in your PATH.")

# 1. Run Bulk Calculation
print("--- Step 1: Running Bulk Calculation ---")
run_calculation(bulk_dir)

# 2. Run Defect Calculations
print("\n--- Step 2: Running Defect Calculations ---")
defect_folders = sorted([
    os.path.join(base_dir, d) for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d)) and d != "bulk" and os.path.exists(os.path.join(base_dir, d, "relax.py"))
])

for folder in defect_folders:
    run_calculation(folder)

# 3. Parse Results with Doped
print("\n--- Step 3: Parsing Results with Doped ---")
try:
    parser = GPAWDefectsParser(
        output_path=base_dir,
        bulk_path=bulk_dir,
        dielectric=10 
    )
    defect_dict = parser.parse_all()
    
    print(f"Parsed {len(defect_dict)} defects.")
    
    if defect_dict:
        # 4. Thermodynamic Analysis
        print("\n--- Step 4: Thermodynamic Analysis ---")
        
        for defect_name, defect_entry in defect_dict.items():
             # FIX: Use sc_entry.energy instead of sc_entry_energy
             print(f"Defect: {defect_name}")
             print(f"  Charge: {defect_entry.charge_state}")
             print(f"  Supercell Energy: {defect_entry.sc_entry.energy:.4f} eV")
             
             if defect_entry.charge_state != 0:
                 # Ensure the correction is calculated explicitly
                 defect_entry.get_kumagai_correction()
                 clean_corrections = {k: round(float(v), 4) for k, v in defect_entry.corrections.items()}
                 print(f"  Corrections: {clean_corrections}")
             else:
                 print("  Corrections: None (Neutral defect)")
                 
             # Simple formation energy calculation ignoring chempots
             e_form_raw = defect_entry.formation_energy(fermi_level=0)
             print(f"  Formation Energy (at VBM, no chempots): {e_form_raw:.4f} eV\n")
             
        # Initialize thermodynamics object for advanced downstream use
        thermo = DefectThermodynamics(list(defect_dict.values()))
        print("Successfully initialized DefectThermodynamics object!")
             
except Exception as e:
    print(f"Analysis failed: {e}")
    import traceback
    traceback.print_exc()

print("\nFull Automation Test Complete.")

import os
os._exit(0)
