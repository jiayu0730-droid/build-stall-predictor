"""Run the whole pipeline end to end:
generate practice data -> clean -> features -> train/evaluate -> demo scoring.
With real data: skip step 1, drop the real export at data/builds_raw.csv,
and run steps 2-5 unchanged.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

steps = [
    ("1/5 generate synthetic practice data", ["data/generate_synthetic_data.py"]),
    ("2/5 clean", ["src/clean.py"]),
    ("3/5 features", ["src/features.py"]),
    ("4/5 train + evaluate", ["src/train.py"]),
]

for label, script in steps:
    print(f"\n=== {label} ===")
    subprocess.run([PY, *script], cwd=ROOT, check=True)

print("\n=== 5/5 deployment demo: score two example builds ===")
demo_high_risk = [PY, "src/predict.py", "--has-bom", "0", "--has-cad", "0",
                  "--quote-spread-pct", "55", "--quantity", "500",
                  "--requested-lead-days", "14", "--design-revisions", "4"]
demo_low_risk = [PY, "src/predict.py", "--quote-spread-pct", "12",
                 "--quantity", "50", "--requested-lead-days", "45"]
print("High-risk example (no CAD, no BOM, wide quote spread, rush order):")
subprocess.run(demo_high_risk, cwd=ROOT, check=True)
print("Low-risk example (complete package, tight quotes, relaxed lead time):")
subprocess.run(demo_low_risk, cwd=ROOT, check=True)
print("\nPipeline complete.")
