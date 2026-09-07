import sys
from pathlib import Path
import pandas as pd

def main():
    rolling_dir = Path("outputs/rolling_origin")
    if not rolling_dir.exists():
        print(f"Directory {rolling_dir} not found.")
        return 1

    fold_dirs = sorted([d for d in rolling_dir.glob("fold_*") if d.is_dir()])
    if not fold_dirs:
        print("No fold directories found.")
        return 1

    results = []
    
    for d in fold_dirs:
        fold_year = int(d.name.split("_")[1])
        week_csv = d / "evaluation_metrics_by_week_test.csv"
        res_csv = d / "evaluation_metrics_per_reservoir_test.csv"
        
        row = {"fold": fold_year}
        
        if week_csv.exists():
            week_df = pd.read_csv(week_csv)
            # Assuming columns: 'Week', 'NSE', ...
            for _, w_row in week_df.iterrows():
                row[f"week_{int(w_row['Week'])}"] = w_row["NSE"]
                
        if res_csv.exists():
            res_df = pd.read_csv(res_csv)
            # Pooled mean NSE across all reservoirs in this fold
            row["pooled_mean_NSE"] = res_df["NSE"].mean()
            
        if len(row) > 1:
            results.append(row)
            
    if not results:
        print("No valid CSVs found.")
        return 1
        
    df = pd.DataFrame(results)
    df.set_index("fold", inplace=True)
    df.sort_index(inplace=True)
    
    out_file = rolling_dir / "rolling_origin_summary.csv"
    df.to_csv(out_file)
    
    print("=" * 80)
    print("ROLLING ORIGIN EVALUATION SUMMARY (Mean NSE)")
    print("=" * 80)
    print(df.to_string(float_format="{:.3f}".format))
    print("=" * 80)
    print(f"Saved: {out_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
