"""
Run All EDA Scripts
"""
import subprocess
import sys
from pathlib import Path

scripts = [
    "01_basic_statistics.py",
    "02_target_distribution.py",
    "03_text_analysis.py",
    "04_identity_analysis.py"
]

eda_dir = Path(__file__).parent

print("="*60)
print("RUNNING ALL EDA SCRIPTS")
print("="*60)

for i, script in enumerate(scripts, 1):
    script_path = eda_dir / script
    print(f"\n[{i}/{len(scripts)}] Running {script}...")
    print("-"*60)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(eda_dir)
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"ERROR in {script}:")
            print(result.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"FAILED to run {script}: {e}")
        sys.exit(1)

print("\n" + "="*60)
print("ALL EDA SCRIPTS COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\nResults saved to: {eda_dir}")
