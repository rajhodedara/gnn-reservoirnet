import argparse
import yaml
import subprocess
from pathlib import Path
import sys

FOLDS = [2020, 2021, 2022, 2023, 2024]

def run_fold(fold_year, epochs, config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    config['data']['test_years'] = [fold_year]
    config['data']['val_years'] = [fold_year - 1]
    config['data']['train_end'] = f"{fold_year - 2}-12-31"
    
    if epochs is not None:
        config['training']['finetune']['epochs'] = epochs
        
    out_dir = Path(f"outputs/rolling_origin/fold_{fold_year}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    temp_config = Path(f"configs/temp_fold_{fold_year}.yaml")
    with open(temp_config, "w") as f:
        yaml.dump(config, f, sort_keys=False)
        
    print(f"================ FOLD {fold_year} ================")
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
