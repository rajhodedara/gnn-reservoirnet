#!/usr/bin/env python
"""
Build data/raw/wris_v2/ — artifact-cleaned rebuild of the WRIS reservoir dataset.

The storage columns in data/raw/wris/ carry pipeline artifacts (unit-mixed
segments, impossible day-over-day jumps, exact-0.0 dead-storage floors, linear
interpolation bridges). This script rebuilds storage from the best available
sources, corrects units per segment, masks physically impossible values, and
re-interpolates. Inflow columns are copied verbatim — they are real gauge data.

Sources (true sparse download is lost; best available lineage):
  - almatti, krishnaraja_sagara, tungabhadra: data/raw/legacy_cwc_cache/<slug>.csv
    (pre-overwrite daily 3-col files, storage in BCM scale)
  - the other 7: Storage (TMC/MCM) column of the current data/raw/wris/<slug>.csv

Usage:
    python scripts/build_wris_v2.py [--wris-dir DIR] [--out-dir DIR]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
EXPECTED_ROWS = 5479
REQUIRED_COLUMNS = [
    "Date",
    "Reservoir_Name",
    "Inflow (cusecs/cumecs)",
    "Storage (TMC/MCM)",
]

MCM_PER_TMC = 28.3168466
TMC_PER_BCM = 35.3146667

# Gross capacities in MCM (from configs/reservoirs.yaml)
CAPACITY_MCM = {
    "almatti": 3440,
    "tungabhadra": 3760,
    "krishnaraja_sagara": 1400,
    "mettur": 2646,
    "nagarjuna_sagar": 11560,
    "srisailam": 8560,
    "jayakwadi": 2909,
    "ujjani": 3140,
    "sardar_sarovar": 9500,
    "ukai": 7414,
}

SLUGS = [
    "almatti",
    "jayakwadi",
    "krishnaraja_sagara",
    "mettur",
    "nagarjuna_sagar",
    "sardar_sarovar",
    "srisailam",
    "tungabhadra",
    "ujjani",
    "ukai",
]

# Storage sources: "legacy3" = pre-overwrite daily 3-col file (BCM scale),
# "current" = Storage (TMC/MCM) column of the current wris file.
LEGACY3_SLUGS = ["almatti", "krishnaraja_sagara", "tungabhadra"]

ARTIFACT_BOUND_FRAC = 0.10  # |dS| > 10% of capacity in one day = impossible
HARD_PLAUSIBLE_HI = 1.15  # values above 115% of capacity are impossible
SEGMENT_BOUND_FRAC = 0.10  # segment splitter threshold (same physics)
MIN_COVERAGE_SCORE = 0.60  # min fraction of segment values inside bounds


def capacity_tmc(slug: str) -> float:
    return CAPACITY_MCM[slug] / MCM_PER_TMC


def load_source_storage(slug: str, wris_dir: Path, legacy_dir: Path) -> tuple[pd.Series, str, str]:
    """Returns (raw storage series indexed by date, source path label, native scale hint)."""
    if slug in LEGACY3_SLUGS:
        path = legacy_dir / f"{slug}.csv"
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")
        s = pd.to_numeric(df["storage"], errors="coerce")
        s.index = df["Date"]
        s = s.sort_index()
        return s, str(path), "bcm_scale_assumed"
    path = wris_dir / f"{slug}.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")
    s = pd.to_numeric(df["Storage (TMC/MCM)"], errors="coerce")
    s.index = df["Date"]
    s = s.sort_index()
    return s, str(path), "tmc_assumed"


def find_boundaries(raw: pd.Series, cap_tmc: float) -> list[int]:
    """Indices where |day-over-day change| exceeds the physical bound (raw scale)."""
    d = raw.diff().abs()
    bound = cap_tmc * SEGMENT_BOUND_FRAC
    return list(np.where(d.values > bound)[0])


def split_segments(n: int, boundaries: list[int]) -> list[tuple[int, int]]:
    """Contiguous [start, end) index ranges between boundaries."""
    cuts = [0] + [b + 1 for b in boundaries] + [n]
    segments = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b > a:
            segments.append((a, b))
    return segments


def segment_scale_score(values: np.ndarray, cap_tmc: float, scale: str) -> float:
    """Fraction of values inside [0, 1.15 x cap] under the candidate scale."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return 0.0
    if scale == "identity":
        w = v
    elif scale == "mcm_to_tmc":
        w = v / MCM_PER_TMC
    elif scale == "bcm_to_tmc":
        w = v * TMC_PER_BCM
    else:
        raise ValueError(scale)
    return float(((w >= 0) & (w <= cap_tmc * HARD_PLAUSIBLE_HI)).mean())


def coverage_score(segment: pd.Series, cap_tmc: float, scale: str) -> float:
    """Median over full years of (annual max / capacity) — reservoirs refill.

    A series that never reaches 25% of capacity for whole years is a wrong-scale
    reading (e.g. BCM numbers read as TMC), even though values are in bounds.
    """
    v = segment.astype(float)
    if scale == "identity":
        w = v
    elif scale == "mcm_to_tmc":
        w = v / MCM_PER_TMC
    elif scale == "bcm_to_tmc":
        w = v * TMC_PER_BCM
    else:
        raise ValueError(scale)
    s = w.dropna()
    if s.empty:
        return 0.0
    annual_max = s.groupby(s.index.year).max()
    full_years = [y for y in annual_max.index if int((s.index.year == y).sum()) >= 300]
    if not full_years:
        return float(annual_max.max() / cap_tmc) if annual_max.size else 0.0
    return float(np.median([annual_max[y] / cap_tmc for y in full_years]))


def choose_scale(segment: pd.Series, cap_tmc: float, prior: str) -> tuple[str, float, float]:
    """Choose the unit scale for a segment.

    The source file declares a scale (tmc for current files, bcm for legacy3).
    Drought years where storage legitimately sits far below capacity are REAL,
    so coverage is recorded but never gates the decision. The prior scale wins
    whenever it is bound-plausible; another scale is accepted only when it
    rescues a segment that clearly violates bounds under the prior (>= 90%
    plausible under the alternative).
    """
    others = [s for s in ("identity", "mcm_to_tmc", "bcm_to_tmc") if s != prior]
    prior_hard = segment_scale_score(segment.values, cap_tmc, prior)
    prior_cov = coverage_score(segment, cap_tmc, prior)
    if prior_hard >= MIN_COVERAGE_SCORE:
        return prior, prior_hard, prior_cov
    for scale in others:
        hard = segment_scale_score(segment.values, cap_tmc, scale)
        cov = coverage_score(segment, cap_tmc, scale)
        if hard >= 0.90:
            return scale, hard, cov
    return "none", prior_hard, prior_cov


def apply_scale(values: np.ndarray, scale: str) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if scale == "identity":
        return v
    if scale == "mcm_to_tmc":
        return v / MCM_PER_TMC
    if scale == "bcm_to_tmc":
        return v * TMC_PER_BCM
    raise ValueError(scale)


def clean_storage(raw: pd.Series, cap_tmc: float, prior: str = "identity") -> tuple[pd.Series, dict]:
    """Full cleaning pipeline for one reservoir. Returns (cleaned series, stats)."""
    stats: dict = {}
    sigma_before = float(raw.diff().std())

    # 1. Segment-wise scale correction
    raw_v = raw.values.astype(float)
    boundaries = find_boundaries(raw, cap_tmc)
    segments = split_segments(len(raw), boundaries)
    corrected = np.full(len(raw), np.nan)
    seg_log = []
    scale_counts = {"identity": 0, "mcm_to_tmc": 0, "bcm_to_tmc": 0, "none": 0}
    for a, b in segments:
        seg = raw.iloc[a:b]
        scale, hard, cov = choose_scale(seg, cap_tmc, prior)
        scale_counts[scale] += 1
        seg_log.append(
            {
                "i": [int(a), int(b)],
                "dates": [str(raw.index[a].date()), str(raw.index[min(b, len(raw)) - 1].date())],
                "scale": scale,
                "hard_score": round(hard, 3),
                "coverage": round(cov, 3),
            }
        )
        if scale != "none":
            corrected[a:b] = apply_scale(seg.values, scale)
    scaled = pd.Series(corrected, index=raw.index)

    # 2. Drop values above hard physical bound
    over = (scaled < 0) | (scaled > cap_tmc * HARD_PLAUSIBLE_HI)
    over_count = int(over.sum())
    scaled[over] = np.nan

    # 3. Exact-zero floors are missing data, not hydrology (counted pre-mask)
    zero_mask = scaled == 0.0
    zero_days = int(zero_mask.sum())
    scaled[zero_mask] = np.nan

    # 4. Iterative artifact masking: remove the later day of each pair whose
    # day-over-day change exceeds the physical bound, re-interpolate, repeat
    # until stable (interpolation can expose new violations at hole edges).
    bound = cap_tmc * ARTIFACT_BOUND_FRAC
    artifact_masked = 0
    rounds = 0
    for rounds in range(1, 31):
        d = scaled.diff().abs()
        viol = d > bound
        n = int(viol.sum())
        if n == 0:
            rounds -= 1
            break
        scaled[viol] = np.nan
        artifact_masked += n
        scaled = scaled.interpolate(method="time")
    if scaled.isna().any():
        scaled = scaled.ffill().bfill()
    if scaled.isna().all():
        raise ValueError("clean_storage produced an all-NaN series")

    # 5. Final interpolation of any remaining gaps and clip
    cleaned = scaled.interpolate(method="time").ffill().bfill()
    interpolated = int(zero_days + over_count + artifact_masked)

    # 6. Clip and round
    cleaned = cleaned.clip(lower=0.0, upper=cap_tmc).round(3)

    sigma_after = float(cleaned.diff().std())

    stats = {
        "capacity_tmc": round(cap_tmc, 1),
        "boundaries": len(boundaries),
        "segments": len(segments),
        "scale_counts": scale_counts,
        "segment_log": seg_log[:50],
        "over_bound_dropped": over_count,
        "artifact_days_masked": artifact_masked,
        "artifact_rounds": rounds,
        "zero_days_converted": zero_days,
        "days_interpolated": interpolated,
        "sigma_ds_before_tmc_per_day": round(sigma_before, 2),
        "sigma_ds_after_tmc_per_day": round(sigma_after, 2),
    }
    return cleaned, stats


def build(wris_dir: Path, legacy_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": "Artifact-cleaned rebuild of WRIS reservoir dataset (storage corrected, inflow preserved verbatim).",
        "transformations": [
            "1. Storage loaded from best available source (legacy 3-col BCM-scale files for almatti/krishnaraja_sagara/tungabhadra; current TMC column otherwise).",
            f"2. Segment-wise unit scale correction with source-declared prior scale (legacy 3-col files: BCM x35.3146667; current files: TMC identity); alternatives accepted only when the prior violates physical bounds (candidates: TMC identity, MCM/28.3168466, BCM x35.3146667); segments split at |dS| > {int(SEGMENT_BOUND_FRAC*100)}% capacity.",
            f"3. Values outside [0, 115% capacity] dropped.",
            f"4. Day-over-day |dS| > {int(ARTIFACT_BOUND_FRAC*100)}% capacity masked iteratively (later day of each pair, re-interpolated, up to 6 rounds) until no violating pair remains.",
            "5. Exact-0.0 storage treated as missing (dead storage makes real 0 impossible).",
            "6. Time interpolation of masked/missing days; edge ffill/bfill; clip [0, capacity]; round 3.",
            "7. Inflow and Reservoir_Name copied verbatim from data/raw/wris/.",
        ],
        "reservoirs": {},
    }
    report_rows = []
    for slug in SLUGS:
        cap = capacity_tmc(slug)
        cur = pd.read_csv(wris_dir / f"{slug}.csv")
        cur["Date"] = pd.to_datetime(cur["Date"])
        if cur["Inflow (cusecs/cumecs)"].isna().any():
            raise ValueError(
                f"{slug}: inflow column contains NaN in data/raw/wris — refusing to "
                "propagate broken data into wris_v2"
            )
        cur = cur[(cur["Date"] >= START_DATE) & (cur["Date"] <= END_DATE)].reset_index(drop=True)
        if len(cur) != EXPECTED_ROWS:
            raise ValueError(f"{slug}: current file has {len(cur)} rows in window, expected {EXPECTED_ROWS}")

        raw, source_label, scale_hint = load_source_storage(slug, wris_dir, legacy_dir)
        prior = "bcm_to_tmc" if slug in LEGACY3_SLUGS else "identity"
        cleaned, stats = clean_storage(raw, cap, prior=prior)

        # Align cleaned storage to the window
        cleaned_w = cleaned.reindex(pd.date_range(START_DATE, END_DATE, freq="D"))
        if cleaned_w.isna().any():
            raise ValueError(f"{slug}: cleaned storage still has NaN after interpolation")

        original_w = cur["Storage (TMC/MCM)"].astype(float)
        corr = float(np.corrcoef(cleaned_w.values, original_w.values)[0, 1])
        stats["corr_cleaned_vs_original"] = round(corr, 3)
        stats["source"] = source_label
        stats["source_scale_hint"] = scale_hint

        out = pd.DataFrame(
            {
                "Date": cur["Date"].dt.strftime("%Y-%m-%d"),
                "Reservoir_Name": cur["Reservoir_Name"],
                "Inflow (cusecs/cumecs)": cur["Inflow (cusecs/cumecs)"],
                "Storage (TMC/MCM)": cleaned_w.values,
            }
        )
        assert list(out.columns) == REQUIRED_COLUMNS
        assert len(out) == EXPECTED_ROWS
        out.to_csv(out_dir / f"{slug}.csv", index=False)

        manifest["reservoirs"][slug] = stats
        report_rows.append(
            (
                slug,
                stats["sigma_ds_before_tmc_per_day"],
                stats["sigma_ds_after_tmc_per_day"],
                stats["artifact_days_masked"],
                stats["zero_days_converted"],
                stats["days_interpolated"],
                stats["scale_counts"],
                round(corr, 3),
                float(cleaned_w.min()),
                float(cleaned_w.max()),
            )
        )

    manifest_path = out_dir / "manifest_v2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 116)
    print(
        f"{'slug':<20}{'sig_bef':>8}{'sig_aft':>8}{'masked':>8}{'zeros':>7}{'interp':>8}"
        f"{'scales':>34}{'corr':>7}{'min':>8}{'max':>8}"
    )
    print("-" * 116)
    for r in report_rows:
        sc = ",".join(f"{k[:3]}:{v}" for k, v in r[6].items() if v)
        print(
            f"{r[0]:<20}{r[1]:>8}{r[2]:>8}{r[3]:>8}{r[4]:>7}{r[5]:>8}{sc:>34}{r[7]:>7}{r[8]:>8}{r[9]:>8}"
        )
    print("=" * 116)
    print(f"Manifest: {manifest_path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wris-dir", default=str(PROJECT_ROOT / "data" / "raw" / "wris"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "raw" / "wris_v2"))
    args = parser.parse_args()
    wris_dir = Path(args.wris_dir)
    out_dir = Path(args.out_dir)
    legacy_dir = PROJECT_ROOT / "data" / "raw" / "legacy_cwc_cache"
    build(wris_dir, legacy_dir, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
