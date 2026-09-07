# Direct ΔS model — honest negative result (independent verification)

Executed 2026-09-07 (Antigravity, prompt 7 — Framing B: direct ΔS regression).
Verified against: `scripts/eval_levels_v2.py` (v2 ridge releases), `scripts/eval_levels_v3.py` (operational mode), run #7 baselines.

## Result

The direct ΔS regressor (GradientBoosting per dam, month + storage% + trailing ΔS + weather features, trained 2010–2022) **fails to beat storage persistence on 9/10 dams** at weeks 1 and 4 (only Srisailam wins, 1/10), and wins at week 12 only where persistence itself collapses (3/10).

Sample (week 1, storage NSE): Srisailam 0.91 vs persistence 0.90 — the sole win. NS −16.3 vs −0.84; Tungabhadra −18.8 vs −0.79; KRS −12.8 vs −0.63.

## Why (three independent evaluations agree)

1. **Storage is a slow integrator**: weekly ΔS is small relative to S itself, so S(t) flat is a near-optimal week-1 predictor (persistence NSE 0.65–0.90 across dams).
2. **The regressor learned an AR process on ΔS** (feature importances: dS_trail_1w/4w dominate at 0.1–0.36) — i.e., it re-derives persistence with extra variance.
3. **No error-correction mechanism**: open-loop ΔS accumulation drifts over 12 weeks (week-12 NSE −65 for Tungabhadra).

## The two paths that remain for LEVEL prediction

1. **Operational mode** (already delivered, `scripts/eval_levels_v3.py`): releases known (operator foresight) → level NSE 0.545 week-1, 0.675 El Niño onset — the defensible submission framing.
2. **Physics-informed architecture** (next-semester research): predict release/inflow components INSIDE a mass-balance-constrained network (S(t+1) = S(t) + I − O enforced in the forward pass) — the honest way to beat persistence is physical structure, not more regression.

## Submission framing (recommended)

Core claim = **inflow forecasting** (10/10 > persistence, seed-stable, real data). LEVEL prediction = operational derivation (v3 numbers) + the honest negative result on direct ΔS regression. All three evaluations archived in the repo (`outputs/run7/`, `outputs/run8/`).
