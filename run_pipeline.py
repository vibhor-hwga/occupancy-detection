"""Run all three steps for one config, stopping at the first failure.

Usage:  python run_pipeline.py configs/baseline.yaml
"""

import subprocess
import sys

if len(sys.argv) != 2:
    sys.exit("Usage: python run_pipeline.py configs/<name>.yaml")

config = sys.argv[1]
for step in ["normalize", "train", "evaluate"]:
    print(f"\n=== {step} ===")
    result = subprocess.run([sys.executable, f"src/{step}.py", "--config", config])
    if result.returncode != 0:
        sys.exit(f"Step '{step}' failed; stopping.")
print("\nDone. Run  mlflow ui  and open http://127.0.0.1:5000 to compare runs.")
