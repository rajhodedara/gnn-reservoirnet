"use client";

/**
 * Level & fill — storage trajectory and the physics model's honest result.
 *
 * Redesign changes: the shell supplies the panel and heading; the three headline
 * numbers became stat blocks at the new scale; the chart runs taller with legible
 * axis text; the level table uses sentence-case headers; and the negative-result
 * disclosure plus the closed-loop explanation moved into a collapsible note so
 * they inform without dominating. The `ds_nse` column is clearly labelled so the
 * reader knows what the ΔS number is.
 *
 * The honest headline is NOT softened: the physics model beats storage persistence
 * on only a minority of dams, below the project's own 6/10 acceptance bar, and the
 * UI states the bar it missed.
 */

import React, { useMemo, useState } from "react";
import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import type { AppData, Reservoir } from "@/lib/types";
import {
  DASH,
  TMC_IN_MCM,
  addDays,
  capacityTmc,
  deadFraction,
  fmtFull,
  fmtMetric,
  fmtPct,
  fmtStorageTmc,
} from "@/lib/format";
import { levelsAt } from "@/lib/selectors";
import { useDeck } from "@/lib/store";

const M = { top: 26, right: 62, bottom: 30, left: 54 };

export default function LevelView() {
  const { state, dispatch } = useDeck();
  const data = state.data!;
  const [which, setWhich] = useState<"levels" | "physics_levels">("physics_levels");
  const [open, setOpen] = useState(false);

  const rows = useMemo(() => levelsAt(data, which, state.horizon), [data, which, state.horizon]);
  const allPhysicsW1 = useMemo(() => levelsAt(data, "physics_levels", 1), [data]);
  const allOpW1 = useMemo(() => levelsAt(data, "levels", 1), [data]);

  const physicsW1Wins = allPhysicsW1.filter((r) => r.beatsPersistence).length;
  const opW1Wins = allOpW1.filter((r) => r.beatsPersistence).length;
  const opMean = meanOf(allOpW1.map((r) => r.level_nse));

  const selected = data.reservoirs.find((r) => r.id === state.selected) ?? data.reservoirs[0];

  return (
    <div>
      {/* the honest headline, before any chart */}
      <div className="grid grid-cols-3 border-b border-hair">
        <Cell
          label="Physics model, week 1"
          value={`${physicsW1Wins}/${allPhysicsW1.length}`}
          sub="beats storage persistence — below the 6/10 acceptance bar"
          tone={physicsW1Wins >= 6 ? "var(--teal)" : "var(--rose)"}
        />
        <Cell
          label="Operational mode, week 1"
          value={`${opW1Wins}/${allOpW1.length}`}
          sub="known releases assumed"
          tone="var(--teal)"
        />
        <Cell
          label="Mean level NSE, week 1"
          value={fmtMetric(opMean)}
          sub="operational mode, held-out year"
          tone="var(--teal)"
        />
      </div>

      <div className="px-4 py-3 border-b border-hair flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <span className="eyebrow">
          Mode ·{" "}
          {which === "physics_levels"
            ? "physics-constrained mass balance"
            : "operational (known releases)"}
        </span>
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setWhich("physics_levels")}
            className="text-[11.5px] px-2 py-1 border transition-colors"
            style={{
              color: which === "physics_levels" ? "var(--txt)" : "var(--txt-3)",
              borderColor: which === "physics_levels" ? "var(--hair-2)" : "var(--hair)",
            }}
            aria-pressed={which === "physics_levels"}
          >
            Physics
          </button>
          <button
            onClick={() => setWhich("levels")}
            className="text-[11.5px] px-2 py-1 border transition-colors"
            style={{
              color: which === "levels" ? "var(--txt)" : "var(--txt-3)",
              borderColor: which === "levels" ? "var(--hair-2)" : "var(--hair)",
            }}
            aria-pressed={which === "levels"}
          >
            Operational
          </button>
        </div>
      </div>

      <div className="px-4 pt-3.5">
        <StorageChart data={data} reservoir={selected} origin={state.origin!} horizon={state.horizon} />
        <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-1.5">
          <LegendItem color="var(--sky)" label="Storage (TMC)" />
          <LegendItem color="var(--txt-3)" label="Gross capacity" />
          <LegendItem color="var(--rose)" label="Dead storage" dashed />
          <LegendItem color="rgba(99,176,218,0.16)" label="Observed window" block />
        </div>
      </div>

      <div className="rule overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-hair">
              <th className="px-4 py-2.5 text-left font-medium text-data-faint" style={{ fontSize: 11.5 }}>
                Reservoir
              </th>
              <th className="px-4 py-2.5 text-right font-medium text-data-faint" style={{ fontSize: 11.5 }}>
                Level NSE
              </th>
              <th className="px-4 py-2.5 text-right font-medium text-data-faint" style={{ fontSize: 11.5 }}>
                Persistence
              </th>
              <th className="px-4 py-2.5 text-right font-medium text-data-faint" style={{ fontSize: 11.5 }}>
                Climatology
              </th>
              {which === "levels" ? (
                <th className="px-4 py-2.5 text-right font-medium text-data-faint" style={{ fontSize: 11.5 }}
                    title="Direct ΔS regressor, for comparison">
                  ΔS model
                </th>
              ) : (
                <>
                  <th className="px-4 py-2.5 text-right font-medium text-data-faint" style={{ fontSize: 11.5 }}>
                    Samples
                  </th>
                  <th className="px-4 py-2.5 text-right font-medium text-data-faint" style={{ fontSize: 11.5 }}>
                    Gross TMC
                  </th>
                </>
              )}
              <th className="px-4 py-2.5 text-right font-medium text-data-faint" style={{ fontSize: 11.5 }}>
                Verdict
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const isSel = state.selected === r.dam;
              return (
                <tr
                  key={r.dam}
                  onClick={() => dispatch({ type: "select", value: r.dam })}
                  className={`border-b border-hair/70 cursor-pointer transition-colors ${
                    isSel ? "bg-data-teal/[0.08]" : "hover:bg-white/[0.022]"
                  }`}
                >
                  <td className={`px-4 py-2 whitespace-nowrap ${isSel ? "text-data-teal" : "text-data-text"}`}>
                    {r.name}
                  </td>
                  <td
                    className="num px-4 py-2 text-right"
                    style={{ color: r.beatsPersistence ? "var(--teal)" : "var(--rose)" }}
                  >
                    {fmtMetric(r.level_nse)}
                  </td>
                  <td className="num px-4 py-2 text-right text-data-dim">{fmtMetric(r.pers_nse)}</td>
                  <td className="num px-4 py-2 text-right text-data-dim">{fmtMetric(r.clim_nse)}</td>
                  {which === "levels" ? (
                    <td className="num px-4 py-2 text-right text-data-dim">{fmtMetric(r.ds_nse ?? null)}</td>
                  ) : (
                    <>
                      <td className="num px-4 py-2 text-right text-data-dim">
                        {r.n_samples === null || r.n_samples === undefined ? DASH : fmtFull(r.n_samples)}
                      </td>
                      <td className="num px-4 py-2 text-right text-data-dim">
                        {r.gross_tmc === null || r.gross_tmc === undefined
                          ? DASH
                          : fmtStorageTmc(r.gross_tmc, 0)}
                      </td>
                    </>
                  )}
                  <td className="px-4 py-2 text-right whitespace-nowrap">
                    {r.beatsPersistence === null ? (
                      <span className="num text-2xs text-data-faint">{DASH}</span>
                    ) : (
                      <span
                        className="ctl ctl-xs num inline-flex gap-1"
                        style={{
                          color: r.beatsPersistence ? "var(--teal)" : "var(--rose)",
                          borderColor: r.beatsPersistence
                            ? "rgba(62,201,182,0.45)"
                            : "rgba(207,101,99,0.45)",
                          background: r.beatsPersistence
                            ? "rgba(62,201,182,0.08)"
                            : "rgba(207,101,99,0.08)",
                        }}
                      >
                        <span aria-hidden>{r.beatsPersistence ? "▲" : "▼"}</span>
                        {r.beatsPersistence ? "WIN" : "LOSS"}
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="rule">
        <button
          onClick={() => setOpen((v) => !v)}
          className="w-full text-left px-4 py-2.5 label hover:text-data-dim flex items-center gap-2"
          aria-expanded={open}
        >
          <span className="text-data-faint">{open ? "▾" : "▸"}</span>
          Why level skill is reported as a negative result
        </button>
        {open && (
          <div className="px-4 pb-3 note max-w-4xl">
            <p className="mb-2">
              The physics-constrained model rolls a differentiable mass balance inside the forward
              pass and assumes no known releases. It beats persistence on a minority of dams, so no
              persistence-beating self-contained level skill is claimed. Its measured strength is
              robustness — it avoids the catastrophic drift a learned ΔS regressor suffers.
            </p>
            <p className="mb-0">
              Recovering level skill by routing GNN inflow through mass balance was tested closed-loop
              and loses to persistence on every dam at every horizon, because the level error equals
              the inflow error while the weekly ΔS signal is only 1–5% of capacity. Storage is
              persistence-dominated week to week, which is why level skill must come from predicting
              ΔS directly.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function meanOf(vals: (number | null)[]): number | null {
  const f = vals.filter((v): v is number => v !== null && Number.isFinite(v));
  return f.length ? f.reduce((a, b) => a + b, 0) / f.length : null;
}

function Cell({ label, value, sub, tone }: { label: string; value: string; sub: string; tone: string }) {
  return (
    <div className="px-4 py-3 border-r border-hair last:border-r-0">
      <div className="eyebrow">{label}</div>
      <div className="stat-lg mt-1" style={{ color: tone }}>
        {value}
      </div>
      <div className="label mt-0.5">{sub}</div>
    </div>
  );
}

function LegendItem({
  color,
  label,
  dashed,
  block,
}: {
  color: string;
  label: string;
  dashed?: boolean;
  block?: boolean;
}) {
  return (
    <span className="inline-flex items-center gap-2">
      {block ? (
        <span className="inline-block w-3.5 h-2.5" style={{ background: color }} />
      ) : (
        <svg width="16" height="6" aria-hidden>
          <line x1="0" y1="3" x2="16" y2="3" stroke={color} strokeWidth="2" strokeDasharray={dashed ? "4 3" : undefined} />
        </svg>
      )}
      <span className="label">{label}</span>
    </span>
  );
}

/**
 * Storage trajectory around the origin with capacity and dead-storage rules.
 */
function StorageChart({
  data,
  reservoir,
  origin,
  horizon,
}: {
  data: AppData;
  reservoir: Reservoir;
  origin: string;
  horizon: number;
}) {
  const W = 1180;
  const H = 226;
  const iw = W - M.left - M.right;
  const ih = H - M.top - M.bottom;

  const cap = capacityTmc(reservoir.gross_capacity_mcm);
  const dead = deadFraction(reservoir.dead_storage_mcm, reservoir.gross_capacity_mcm) * cap;

  const series = useMemo(() => {
    const dates = data.daily_dates;
    const idx = dates.indexOf(origin);
    if (idx < 0) return [];
    const from = Math.max(0, idx - 180);
    const to = Math.min(dates.length - 1, idx + 7 * 12);
    const out: Array<{ i: number; d: string; v: number | null; rel: number }> = [];
    for (let i = from; i <= to; i++) {
      out.push({ i, d: dates[i], v: data.daily[reservoir.id]?.storage?.[i] ?? null, rel: i - idx });
    }
    return out;
  }, [data, reservoir.id, origin]);

  const x = useMemo(
    () => scaleLinear().domain([0, Math.max(1, series.length - 1)]).range([M.left, M.left + iw]),
    [series.length, iw],
  );
  const y = useMemo(() => {
    const vals = series.map((s) => s.v).filter((v): v is number => v !== null);
    const hi = Math.max(cap * 1.05, ...(vals.length ? vals : [0]));
    const lo = Math.min(0, ...(vals.length ? vals : [0]));
    return scaleLinear().domain([lo, hi]).range([M.top + ih, M.top]);
  }, [series, cap, ih]);

  const obsPath = useMemo(
    () =>
      line<{ rel: number; v: number | null }>()
        .defined((s) => s.v !== null)
        .x((s) => x(s.rel + 180))
        .y((s) => y(s.v as number))(series) ?? "",
    [series, x, y],
  );

  if (!series.length) {
    return <div className="note py-10 text-center">No daily record at this origin.</div>;
  }

  const lastObs = [...series].reverse().find((s) => s.v !== null)?.v ?? null;
  const nowFill = lastObs === null ? null : lastObs / cap;
  const originX = x(Math.min(180, series.length - 1));

  return (
    <>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-auto block"
        role="img"
        aria-label={`Storage trajectory for ${reservoir.name}: ${fmtStorageTmc(cap, 0)} TMC gross capacity, ${fmtStorageTmc(dead, 0)} TMC dead storage`}
      >
        {/* forecast window shading */}
        <rect x={originX} y={M.top} width={M.left + iw - originX} height={ih} fill="rgba(99,176,218,0.08)" />

        {/* grid */}
        <g stroke="rgba(148,178,188,0.07)" strokeWidth="0.6">
          {y.ticks(4).map((t) => (
            <line key={`g${t}`} x1={M.left} y1={y(t)} x2={M.left + iw} y2={y(t)} />
          ))}
        </g>

        {/* capacity rule */}
        <line x1={M.left} y1={y(cap)} x2={M.left + iw} y2={y(cap)} stroke="rgba(148,178,188,0.5)" strokeWidth="1" />
        <text x={M.left + iw + 6} y={y(cap) + 4} className="num" fontSize="11" fill="#93a7af">
          {fmtStorageTmc(cap, 0)}
        </text>
        <text x={M.left + iw + 6} y={y(cap) - 8} className="num" fontSize="10" fill="#6a7f87">
          gross
        </text>

        {/* dead storage rule */}
        <line
          x1={M.left}
          y1={y(dead)}
          x2={M.left + iw}
          y2={y(dead)}
          stroke="rgba(207,101,99,0.6)"
          strokeWidth="1"
          strokeDasharray="4 3"
        />
        <text x={M.left + iw + 6} y={y(dead) + 4} className="num" fontSize="11" fill="#cf6563">
          {fmtStorageTmc(dead, 0)}
        </text>

        <path d={obsPath} fill="none" stroke="var(--sky)" strokeWidth="1.8" strokeLinejoin="round" />

        {/* origin + horizon markers */}
        <line x1={originX} y1={M.top} x2={originX} y2={M.top + ih} stroke="rgba(234,247,244,0.5)" strokeWidth="1" />
        <text x={originX + 5} y={M.top - 8} className="num" fontSize="10.5" fill="#93a7af">
          origin
        </text>
        {(() => {
          const hx = x(Math.min(180 + 7 * horizon, series.length - 1));
          return (
            <g>
              <line x1={hx} y1={M.top} x2={hx} y2={M.top + ih} stroke="var(--teal)" strokeWidth="1.2" />
              <text x={hx + 5} y={M.top + ih - 6} className="num" fontSize="10.5" fill="var(--teal)">
                wk {horizon}
              </text>
            </g>
          );
        })()}

        {/* axes */}
        <g className="num" fill="#4f626a" fontSize="10.5">
          {y.ticks(4).map((t) => (
            <text key={`t${t}`} x={M.left - 8} y={y(t) + 4} textAnchor="end">
              {t}
            </text>
          ))}
          {series
            .map((s, k) => ({ s, k }))
            .filter(({ s }) => s.d.endsWith("-01"))
            .filter((_, i) => i % 2 === 0)
            .map(({ s, k }) => (
              <text key={s.d} x={x(k)} y={M.top + ih + 17} textAnchor="middle">
                {s.d.slice(0, 7)}
              </text>
            ))}
        </g>
      </svg>

      <div className="mt-1.5 flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <span className="num text-[12px] text-data-sky">
          {reservoir.name} · {fmtStorageTmc(lastObs, 1)} TMC at origin
        </span>
        <span className="num text-[12px] text-data-text">{nowFill === null ? DASH : fmtPct(nowFill)} full</span>
        <span className="label">storage moves 1–5% of capacity per week, so it is persistence-dominated</span>
      </div>
    </>
  );
}
