# Rolling Origin Evaluation - Pilot Guide

## Smoke Test Results
- Smoke test was run on the 2024 fold using a 2-epoch override:
  `python scripts/rolling_origin_eval.py --fold 2024 --epochs 2`
- The script successfully generated temporary config files with the correct expanding window dates (e.g. Test: 2024, Val: 2023, Train: < 2022-12-31).
- Outputs were safely directed to `outputs/rolling_origin/fold_2024/`.
- Tested the aggregator companion script: `python scripts/aggregate_rolling_origin.py`. It correctly aggregates fold results into `outputs/rolling_origin/rolling_origin_summary.csv`.

## Kaggle Notebook Edits for the Full Sweep

Since you want to run the full rolling-origin evaluation on your GPU, simply add the following cell to your Kaggle notebook (e.g., as Step 12). 

```python
# 12. Full Rolling Origin Evaluation (2020-2024)
print("\n================ ROLLING ORIGIN SWEEP ================")
# Run the expanding window evaluation for all folds using the standard epochs (no --epochs override)
!python scripts/rolling_origin_eval.py

# Aggregate the rolling origin CSVs into a single summary table
!python scripts/aggregate_rolling_origin.py

# Repackage outputs including the new rolling origin artifacts
import shutil
from pathlib import Path
if Path("outputs").exists():
    zpath = shutil.make_archive("artifacts_outputs", "zip", root_dir=".", base_dir="outputs")
    print(f"Updated {zpath}")
print("Rolling Origin sweep complete. Download artifacts_outputs.zip from the Output tab.")
```

*Note: In Step 5 of your notebook, the `SEEDS` array now dynamically reads from the `seeds` key in `configs/default_config.yaml`. (It defaults to `[42, 7, 123, 2024, 17]` if missing). `scripts/aggregate_seeds.py` has been updated to dynamically report on N-seeds rather than a hardcoded 3.*
