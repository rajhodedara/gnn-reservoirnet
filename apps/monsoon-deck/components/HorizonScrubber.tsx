"use client";

/**
 * The horizon scrubber — the deck's driver.
 *
 * Redesign changes: the duplicate uppercase mono matrix is gone. The per-reservoir
 * skill-by-week grid was a 120-cell spreadsheet; it is replaced by a single clear
 * readout row (horizon, window, mean skill, confidence) plus the scrubber, and the
 * matrix moves into the header as a compact sparkline strip only for the selected
 * reservoir. The driver now reads as an instrument, not a table.
 */

import React, { useMemo } from "react";
import { addDays, fmtDate, fmtMetric, DASH } from "@/lib/format";
import { meanNseByHorizon, weeklyNse } from "@/lib/selectors";
import { useDeck } from "@/lib/store";

export default function HorizonScrubber() {
  const { state, dispatch, origins } = useDeck();
  const data = state.data!;
  const { horizon, origin, lens, selected } = state;

  const mean = useMemo(() => meanNseByHorizon(data), [data]);
  const perDam = useMemo(() => weeklyNse(data), [data]);

  const start = origin ? addDays(origin, 7 * (horizon - 1)) : null;
  const end = origin ? addDays(origin, 7 * horizon - 1) : null;

  const conf = useMemo(() => {
    const eb = data.forecast_residual_band;
    let relSum = 0;
    let n = 0;
    for (const id of Object.keys(perDam)) {
      const band = eb[origin ?? ""]?.[id];
      const mid = data.forecast[origin ?? ""]?.[id]?.p50?.[horizon - 1] ?? null;
      if (!band || mid === null || mid <= 0) continue;
      const lo = band.p10?.[horizon - 1];
      const hi = band.p90?.[horizon - 1];
      if (lo === null || lo === undefined || hi === null || hi === undefined) continue;
      relSum += (hi - lo) / mid;
      n++;
    }
    return n ? relSum / n : null;
  }, [data, perDam, origin, horizon]);

  const onKey = (ev: React.KeyboardEvent<HTMLInputElement>) => {
    let v = horizon;
    if (ev.key === "ArrowRight") v = horizon + 1;
    else if (ev.key === "ArrowLeft") v = horizon - 1;
    else if (ev.key === "Home") v = 1;
    else if (ev.key === "End") v = 12;
    else return;
    ev.preventDefault();
    dispatch({ type: "horizon", value: v });
  };

  const selName = data.reservoirs.find((r) => r.id === selected)?.name;
  const selSeries = selected ? perDam[selected] ?? [] : [];

  return (
    <section className="panel-hero ticks">
      <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-4 px-4 py-3.5">
        {/* the four numbers that define the current view */}
        <div className="flex flex-wrap items-end gap-x-7 gap-y-3">
          <div>
            <div className="eyebrow mb-1">Horizon</div>
            <div className="stat-lg">
              W{String(horizon).padStart(2, "0")}
              <span className="label ml-1.5">of 12</span>
            </div>
          </div>
          <div>
            <div className="eyebrow mb-1">Forecast window</div>
            <div className="num text-[12.5px] text-data-dim mt-1.5">
              {start && end ? `${fmtDate(start)} → ${fmtDate(end)}` : DASH}
            </div>
          </div>
          <div>
            <div className="eyebrow mb-1">Mean NSE at horizon</div>
            <div
              className="num text-[12.5px] mt-1.5"
              style={{ color: (mean[horizon - 1] ?? 0) >= 0 ? "var(--teal)" : "var(--rose)" }}
            >
              {fmtMetric(mean[horizon - 1] ?? null)}
            </div>
          </div>
          <div>
            <div className="eyebrow mb-1">Band width / median</div>
            <div className="num text-[12.5px] text-data-indigo mt-1.5">
              {conf === null ? DASH : `${(conf * 100).toFixed(0)}%`}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="eyebrow">Forecast origin</span>
          <select
            value={origin ?? ""}
            onChange={(e) => dispatch({ type: "origin", value: e.target.value })}
            className="num ctl ctl-sm bg-ink-850 text-data-text focus:outline-none focus:border-data-teal"
            aria-label="Forecast origin date"
          >
            {origins
              .slice()
              .reverse()
              .map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
          </select>
          <span className="label">{origins.length} available</span>
        </div>
      </div>

      {/* the scrubber */}
      <div className="px-4 pb-3.5">
        <input
          className="scrub w-full"
          type="range"
          min={1}
          max={12}
          step={1}
          value={horizon}
          onKeyDown={onKey}
          onChange={(e) => dispatch({ type: "horizon", value: Number(e.target.value) })}
          aria-label={`Forecast horizon, week ${horizon} of 12`}
          aria-valuetext={`week ${horizon}`}
        />
        <div className="flex justify-between mt-1">
          {Array.from({ length: 12 }, (_, i) => i + 1).map((w) => {
            const nse = mean[w - 1];
            return (
              <button
                key={w}
                onClick={() => dispatch({ type: "horizon", value: w })}
                className={`num text-2xs px-1.5 py-1 transition-colors ${
                  w === horizon ? "text-data-teal" : "text-data-faint hover:text-data-dim"
                }`}
                aria-label={`jump to week ${w}`}
                aria-current={w === horizon}
              >
                <span className="block text-[11px]">w{w}</span>
                <span
                  className="block text-[10px] mt-0.5"
                  style={{ opacity: nse === null ? 0.4 : 0.85 }}
                >
                  {nse === null ? DASH : nse.toFixed(2)}
                </span>
              </button>
            );
          })}
        </div>

        {/* selected reservoir's skill decay, as a shape rather than 120 cells */}
        {selName && selSeries.length > 0 && (
          <div className="rule mt-2 pt-2.5 flex flex-wrap items-center gap-x-4 gap-y-2">
            <span className="eyebrow">Skill decay · {selName}</span>
            <Sparkline series={selSeries} horizon={horizon} />
            <span className="label">
              select a reservoir on the map or below to trace its own decay
            </span>
          </div>
        )}
      </div>
    </section>
  );
}

/**
 * Twelve-point NSE path with the current horizon marked. Replaces the 120-cell
 * matrix: same information, readable at a glance, and it scales to any reservoir.
 */
function Sparkline({ series, horizon }: { series: (number | null)[]; horizon: number }) {
  const W = 340;
  const H = 44;
  const pad = 3;
  const vals = series.filter((v): v is number => v !== null && Number.isFinite(v));
  if (!vals.length) return null;
  const min = Math.min(...vals, 0);
  const max = Math.max(...vals, 0.4);
  const span = max - min || 1;
  const x = (i: number) => pad + (i / 11) * (W - pad * 2);
  const y = (v: number) => pad + (H - pad * 2) - ((v - min) / span) * (H - pad * 2);

  let d = "";
  let started = false;
  series.forEach((v, i) => {
    if (v === null || !Number.isFinite(v)) return;
    d += `${started ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
    started = true;
  });

  const zeroY = y(0);
  const cur = series[horizon - 1];

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="shrink-0" role="img"
         aria-label={`NSE by week: ${series.map((v, i) => `w${i + 1} ${v === null ? "unavailable" : v.toFixed(2)}`).join(", ")}`}>
      {/* zero rule: above = skill, below = worse than the mean */}
      <line x1={pad} y1={zeroY} x2={W - pad} y2={zeroY} stroke="rgba(148,178,188,0.22)" strokeWidth="1" strokeDasharray="2 3" />
      <path d={d} fill="none" stroke="var(--sky)" strokeWidth="1.6" strokeLinejoin="round" strokeLinecap="round" />
      {series.map((v, i) =>
        v === null || !Number.isFinite(v) ? null : (
          <circle
            key={i}
            cx={x(i)}
            cy={y(v)}
            r={i + 1 === horizon ? 3 : 1.6}
            fill={i + 1 === horizon ? "var(--teal)" : "var(--sky)"}
          />
        ),
      )}
      {cur !== null && Number.isFinite(cur) && (
        <g>
          <line x1={x(horizon - 1)} y1={pad} x2={x(horizon - 1)} y2={H - pad} stroke="var(--teal)" strokeWidth="1" opacity="0.75" />
          <text x={x(horizon - 1)} y={pad + 8} textAnchor="middle" className="num" fontSize="9.5" fill="var(--teal)">
            {cur.toFixed(2)}
          </text>
        </g>
      )}
    </svg>
  );
}
