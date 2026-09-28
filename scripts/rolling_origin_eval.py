import argparse
import yaml
import shutil
import subprocess
from pathlib import Path
import sys

# 2015-2016 added to satisfy the approved design's validation requirement:
# "Validate on 2014-2016 (severe El Nino 2015-16) and 2023-2024 (recent El Nino)".
# Fold 2015 = the strongest El Nino on record (ONI +1.55 annual, +1.74 JJAS).
FOLDS = [2015, 2016, 2020, 2021, 2022, 2023, 2024]
EL_NINO_FOLDS = {2015: "severe El Nino", 2023: "recent El Nino onset"}

def run_fold(fold_year, epochs, config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    config['data']['test_years'] = [fold_year]
    config['data']['val_years'] = [fold_year - 1]
    config['data']['train_end'] = f"{fold_year - 2}-12-31"
    
    if epochs is not None:
        config['training']['finetune']['epochs'] = epochs
        
    out_dir = Path(f"outputs/rolling_origin/fold_{fold_year}")
    # Wipe the fold dir first: a re-run must not leave stale artifacts (e.g. a
    # previous local run's tfevents or metric CSVs) mixed with the new ones.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    temp_config = Path(f"configs/temp_fold_{fold_year}.yaml")
    with open(temp_config, "w") as f:
        yaml.dump(config, f, sort_keys=False)
        
    tag = EL_NINO_FOLDS.get(fold_year, "")
    print(f"================ FOLD {fold_year} {('['+tag+']') if tag else ''} ================")
    print(f"Train end: {config['data']['train_end']}, Val: {fold_year-1}, Test: {fold_year}")
    
    cmd = [sys.executable, "main.py", "--config", str(temp_config), "--output-dir", str(out_dir)]
    subprocess.run(cmd, check=True)
    
    if temp_config.exists():
        temp_config.unlink()

def main():
    parser = argparse.ArgumentParser(description="Rolling origin evaluation")
    parser.add_argument("--fold", type=int, choices=FOLDS, help="Specific fold year to run")
    parser.add_argument("--epochs", type=int, help="Epochs override for smoke testing")
    parser.add_argument("--config", type=str, default="configs/rolling_origin.yaml")
    args = parser.parse_args()
    
    folds = [args.fold] if args.fold else FOLDS
    
    for f in folds:
        run_fold(f, args.epochs, args.config)

if __name__ == "__main__":
    main()
