"use client";

/**
 * The rail: identity and the signature chart for the focused reservoir.
 *
 * Redesign changes: priority inverted so the FAN comes first (the signature
 * visualisation) with identity reduced to a compact line above it. The previous
 * version opened with four fact boxes and buried the chart, which meant the
 * second-most-important graphic in the product was below the eye line. Basin
 * colour now appears as a spine so the rail visually belongs to its node.
 */

import React, { useMemo } from "react";
import type { AppData } from "@/lib/types";
import { DASH, capacityTmc, fmtFull, fmtMagnitude, fmtMetric, fmtPct, fmtSigned, fmtStorageTmc } from "@/lib/format";
import { basinColor } from "./MapDeck";
import FanChart from "./FanChart";
import { useDeck } from "@/lib/store";

export default function DetailPanel() {
  const { state, dispatch } = useDeck();
  const data = state.data!;
  const id = state.selected;
  const res = useMemo(() => data.reservoirs.find((r) => r.id === id) ?? null, [data, id]);

  if (!res) {
    return (
      <section className="panel ticks p-6">
        <div className="eyebrow mb-1.5">No reservoir focused</div>
        <p className="note">Select a node on the map, or a row in the scoreboard below.</p>
      </section>
    );
  }

  const origin = state.origin!;
  const horizon = state.horizon;
  const m = data.metrics.per_reservoir.find((r) => r.reservoir_id === res.id);
  const wk = data.metrics.per_week.find((r) => r.reservoir_id === res.id && r.Week === horizon);
  const pers = data.baselines.persistence[res.id];
  const clim = data.baselines.climatology[res.id];
  const skew = data.band_skew[res.id];
  const capTmc = capacityTmc(res.gross_capacity_mcm);
  const deadTmc = res.dead_storage_mcm / 28.3168;

  const storageIdx = data.daily_dates.indexOf(origin);
  const storageNow = storageIdx >= 0 ? data.daily[res.id]?.storage?.[storageIdx] ?? null : null;
  const fill = storageNow === null ? null : storageNow / capTmc;

  const fwd = data.forecast[origin]?.[res.id];
  const fwdAtH = fwd?.p50?.[horizon - 1] ?? null;
  const obsAtH = (fwd?.observed?.[horizon - 1] ?? null) as number | null;
  const downstream = data.cascade.find((c) => c.upstream === res.id);
  const upstream = data.cascade.filter((c) => c.downstream === res.id);

  const nseDelta =
    m?.NSE_mean !== null && m?.NSE_mean !== undefined && pers?.NSE !== null && pers?.NSE !== undefined
      ? m.NSE_mean - pers.NSE
      : null;
  const climDelta =
    m?.NSE_mean !== null && m?.NSE_mean !== undefined && clim?.NSE !== null && clim?.NSE !== undefined
      ? m.NSE_mean - clim.NSE
      : null;

  const fillTone = fill === null ? "var(--txt-2)" : fill < 0.15 ? "var(--rose)" : fill < 0.35 ? "var(--amber)" : "var(--teal)";

  return (
    <section className="panel-hero ticks flex flex-col" style={{ borderLeft: `2px solid ${basinColor(res.basin)}` }}>
      {/* identity — one compact band */}
      <header className="px-4 py-3 border-b border-hair">
        <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
          <h2 className="text-lg font-semibold tracking-[-0.01em] text-data-text">{res.name}</h2>
          <button
            className="label hover:text-data-dim"
            onClick={() => dispatch({ type: "select", value: null })}
          >
            Clear selection
          </button>
        </div>
        <div className="label mt-0.5">
          {/* River and basin are often the same word (Krishna/Krishna), which read as a
              duplicate when both were printed. Show the river only if it adds information. */}
          {res.river.trim().toLowerCase() !== res.basin.trim().toLowerCase()
            ? `${res.river} · `
            : ""}
          {res.basin} basin · {res.state}
        </div>
        <div className="eyebrow mt-1.5">{res.tribunal}</div>
      </header>

      {/* the signature chart, immediately */}
      <div className="px-3 pt-3">
        <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5 mb-1">
          <span className="eyebrow">Inflow forecast, 12 weeks</span>
          <span className="label">
            week {horizon}: {fmtMagnitude(fwdAtH)} forecast, {fmtMagnitude(obsAtH)} observed
          </span>
        </div>
        <FanChart data={data} reservoir={res} origin={origin} horizon={horizon} height={252} compact />
      </div>

      {/* supporting facts, compact */}
      <div className="grid grid-cols-3 border-y border-hair mt-1.5">
        <Fact label="Fill at origin" value={fill === null ? DASH : fmtPct(fill)} sub={`${fmtStorageTmc(storageNow, 1)} / ${fmtStorageTmc(capTmc, 0)} TMC`} tone={fillTone} />
        <Fact label="Gross capacity" value={fmtStorageTmc(capTmc, 0)} sub={`${fmtFull(res.gross_capacity_mcm)} MCM`} />
        <Fact label="Catchment" value={fmtFull(res.catchment_area_km2)} sub="km²" />
      </div>

      {/* skill vs baselines, compact two-row block */}
      <div className="px-4 py-3">
        <div className="eyebrow mb-2">
          Skill against baselines · model column is the week-1 evaluation
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-hair">
              <th className="py-1.5 text-left font-medium text-data-faint" style={{ fontSize: 11 }}>Metric</th>
              <th
                className="py-1.5 text-right font-medium text-data-faint"
                style={{ fontSize: 11 }}
                title="Week-1 evaluation for this reservoir"
              >
                Model
              </th>
              {/* Only show a separate horizon column once it differs from week 1,
                  otherwise the table reads as a duplicated-header bug. */}
              {horizon !== 1 && (
                <th
                  className="py-1.5 text-right font-medium text-data-faint"
                  style={{ fontSize: 11 }}
                  title={`Week ${horizon} of the 12-week horizon`}
                >
                  Wk {horizon}
                </th>
              )}
              <th className="py-1.5 text-right font-medium text-data-faint" style={{ fontSize: 11 }}>Persist.</th>
              <th className="py-1.5 text-right font-medium text-data-faint" style={{ fontSize: 11 }}>Clim.</th>
              <th className="py-1.5 text-right font-medium text-data-faint" style={{ fontSize: 11 }}>Δ pers.</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-hair/70">
              <td className="py-1.5 text-data-dim">NSE</td>
              <td className="num py-1.5 text-right text-data-text">{fmtMetric(m?.NSE_mean ?? null)}</td>
              {horizon !== 1 && (
                <td className="num py-1.5 text-right text-data-text">{fmtMetric(wk?.NSE_mean ?? null)}</td>
              )}
              <td className="num py-1.5 text-right text-data-dim">{fmtMetric(pers?.NSE ?? null)}</td>
              <td className="num py-1.5 text-right text-data-dim">{fmtMetric(clim?.NSE ?? null)}</td>
              <td
                className="num py-1.5 text-right"
                style={{ color: (nseDelta ?? 0) >= 0 ? "var(--teal)" : "var(--rose)" }}
                title="Week-1 NSE minus persistence"
              >
                {fmtSigned(nseDelta)}
              </td>
            </tr>
            <tr className="border-b border-hair/70">
              <td className="py-1.5 text-data-dim">KGE</td>
              <td className="num py-1.5 text-right text-data-dim">{fmtMetric(m?.KGE_mean ?? null)}</td>
              {horizon !== 1 && (
                <td className="num py-1.5 text-right text-data-dim">{fmtMetric(wk?.KGE_mean ?? null)}</td>
              )}
              <td className="num py-1.5 text-right text-data-faint">—</td>
              <td className="num py-1.5 text-right text-data-faint">—</td>
              <td
                className="num py-1.5 text-right"
                style={{ color: (climDelta ?? 0) >= 0 ? "var(--teal)" : "var(--rose)" }}
                title="Week-1 NSE minus climatology"
              >
                {fmtSigned(climDelta)}
              </td>
            </tr>
            <tr>
              <td className="py-1.5 text-data-dim">RMSE</td>
              <td className="num py-1.5 text-right text-data-dim">{fmtMagnitude(m?.RMSE_mean ?? null, 1)}</td>
              {horizon !== 1 && (
                <td className="num py-1.5 text-right text-data-dim">{fmtMagnitude(wk?.RMSE_mean ?? null, 1)}</td>
              )}
              <td className="num py-1.5 text-right text-data-dim">{fmtMagnitude(pers?.RMSE ?? null, 1)}</td>
              <td className="num py-1.5 text-right text-data-dim">{fmtMagnitude(clim?.RMSE ?? null, 1)}</td>
              <td className="num py-1.5 text-right text-data-dim" title="CRPS for the week-1 forecast">
                {fmtMagnitude(m?.CRPS_mean ?? null, 1)}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* cascade + provenance, in a collapsible note */}
      <div className="rule px-4 py-2.5">
        <div className="eyebrow mb-1.5">Cascade</div>
        {upstream.length === 0 && !downstream ? (
          <p className="note">Isolated reach — no upstream or downstream hydraulic link in this graph.</p>
        ) : (
          <ul className="space-y-1">
            {upstream.map((c) => (
              <li key={c.upstream} className="num text-[12px] text-data-dim">
                <span className="text-data-teal">←</span> receives from {c.upstream}
                <span className="text-data-faint"> · lag {c.lag_days}d · r {fmtMetric(c.peak_corr, 2)}</span>
              </li>
            ))}
            {downstream && (
              <li className="num text-[12px] text-data-dim">
                <span className="text-data-teal">→</span> releases toward {downstream.downstream}
                <span className="text-data-faint"> · lag {downstream.lag_days}d · r {fmtMetric(downstream.peak_corr, 2)}</span>
              </li>
            )}
          </ul>
        )}
      </div>

      <BandNote skew={skew} data={data} />
    </section>
  );
}

function Fact({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: string;
}) {
  return (
    <div className="px-3.5 py-2.5 border-r border-hair last:border-r-0">
      <div className="eyebrow">{label}</div>
      <div className="stat mt-1" style={{ color: tone ?? "var(--txt)" }}>
        {value}
      </div>
      {sub && <div className="num text-2xs text-data-faint mt-0.5">{sub}</div>}
    </div>
  );
}

function BandNote({
  skew,
  data,
}: {
  skew?: { median: number | null; q10: number | null; q90: number | null; negative_median_cells: number; negative_median_weeks: number[]; weeks_with_positive_q10: number[] };
  data: AppData;
}) {
  const [open, setOpen] = React.useState(false);
  const clamped = skew?.negative_median_cells ?? 0;
  return (
    <div className="rule mt-auto">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full text-left px-4 py-2.5 label hover:text-data-dim flex items-center gap-2"
        aria-expanded={open}
      >
        <span className="text-data-faint">{open ? "▾" : "▸"}</span>
        Band provenance{clamped > 0 ? " · median clamped at long horizons" : ""}
      </button>
      {open && (
        <div className="px-4 pb-3 note max-w-xl">
          <p className="mb-2">
            The shaded ribbon is an <strong>empirical P10–P90 of the model&apos;s own held-out
            residuals</strong> (observed minus median), pooled per reservoir and per forecast week.
            It is a measured error envelope, not a model-issued predictive quantile — the export
            contains only the median.
          </p>
          {skew && (
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1 num text-[11.5px] mb-2">
              <dt className="text-data-faint">Residual median</dt>
              <dd className="text-right text-data-dim">{fmtMetric(skew.median, 1)}</dd>
              <dt className="text-data-faint">Residual P10 / P90</dt>
              <dd className="text-right text-data-dim">
                {fmtMetric(skew.q10, 1)} / {fmtMetric(skew.q90, 1)}
              </dd>
              <dt className="text-data-faint">Negative medians clamped</dt>
              <dd className="text-right text-data-dim">
                {clamped}
                {skew.negative_median_weeks.length ? ` (wk ${skew.negative_median_weeks.join(", ")})` : ""}
              </dd>
            </dl>
          )}
          {clamped > 0 && (
            <p className="text-data-rose" style={{ opacity: 0.9 }}>
              At long horizons this node&apos;s raw median goes physically invalid — below zero inflow.
              The median is shown as the model issued it; the shaded ribbon is built on a zero-clamped
              median, and {clamped} clamped cells are disclosed rather than hidden.
            </p>
          )}
          <p className="mt-2">{data.meta.units_capacity_note}</p>
        </div>
      )}
    </div>
  );
}
