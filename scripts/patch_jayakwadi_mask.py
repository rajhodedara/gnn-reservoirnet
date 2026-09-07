"""Jayakwadi fake-zero mask: measured-data hygiene, no synthesis.

Rule: where Inflow == 0 but same-day Storage rose > 1.0 TMC, the zero is
falsified by the reservoir's own mass balance -> set to empty (NaN).
The existing documented pipeline policy (time-interpolation of missing days,
see data_formatter.py and manifest_v2.json 'days_interpolated') then fills
the gap. We remove false assertions; we invent nothing.

Also writes a manifest with the rule, evidence, and per-year counts.
"""
import json

import numpy as np
import pandas as pd

BASE = r"C:\Users\odeda\Desktop\Projects\PBL"
CSV = BASE + r"\data\raw\wris_v2\jayakwadi.csv"
MANIFEST = BASE + r"\data\raw\wris_v2\jayakwadi_mask_manifest.json"
DS_THRESHOLD = 1.0  # TMC/day storage rise that contradicts Inflow == 0

df = pd.read_csv(CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
inflow_col = [c for c in df.columns if "Inflow" in c][0]
storage_col = [c for c in df.columns if "Storage" in c][0]
df["ds"] = df[storage_col].astype(float).diff()

zero = df[inflow_col].astype(float) == 0
contradicted = zero & (df["ds"] > DS_THRESHOLD)
n_before = int(zero.sum())

per_year = (
    df[contradicted].groupby(df["Date"].dt.year).size().rename("masked").to_frame()
)
per_year["zeros_total"] = df[zero].groupby(df["Date"].dt.year).size()
per_year["mask_pct"] = (per_year["masked"] / per_year["zeros_total"] * 100).round(1)

df.loc[contradicted, inflow_col] = np.nan  # empty cell in CSV

# write back with the original schema; masked values become empty strings
out = df.drop(columns=["ds"]).copy()
out[inflow_col] = out[inflow_col].map(lambda v: "" if pd.isna(v) else v)
out[storage_col] = out[storage_col].map(lambda v: "" if pd.isna(v) else v)
out.to_csv(CSV, index=False)

# verify: reload and confirm NaNs land where intended
chk = pd.read_csv(CSV)
n_nan = int(chk[inflow_col].isna().sum())
assert n_nan == int(contradicted.sum()), f"mask count mismatch {n_nan} vs {contradicted.sum()}"

manifest = {
    "operation": "fake-zero mask (no synthesis)",
    "reservoir": "jayakwadi",
    "rule": f"Inflow == 0 AND same-day Storage rise > {DS_THRESHOLD} TMC -> set to missing (NaN)",
    "evidence": {
        "zero_inflow_days": n_before,
        "masked_days": int(contradicted.sum()),
        "mask_pct_of_zeros": round(contradicted.sum() / max(n_before, 1) * 100, 1),
        "storage_rise_on_masked_days_min_tmc": DS_THRESHOLD,
        "rationale": "storage cannot rise without inflow; releases/evaporation only lower it; direct rainfall on reservoir < 0.15 TMC/day",
        "weekly_monsoon_corr_gauge_vs_storage_delta_before": 0.127,
    },
    "per_year": per_year.to_dict(orient="index"),
    "policy": "masked days are treated as missing measurements and filled by the existing time-interpolation policy (data_formatter), consistent with manifest_v2.json days_interpolated convention",
}
json.dump(manifest, open(MANIFEST, "w", encoding="utf-8"), indent=1)
print(f"masked {contradicted.sum()} of {n_before} zero days ({manifest['evidence']['mask_pct_of_zeros']}%)")
print(per_year.to_string())
print("manifest written:", MANIFEST)
