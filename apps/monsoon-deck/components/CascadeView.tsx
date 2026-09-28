"use client";

/**
 * Cascade: how a pulse travels down the hydraulic topology.
 *
 * Redesign changes: the outer wrapper/heading moved to the shell's Block, the
 * reach selector became a proper segmented control, and the disclaimer is now a
 * one-line eyebrow plus an expandable note instead of a paragraph competing with
 * the chart. Legend swatches are paired with text so colour is not the only cue.
 */

import React, { useMemo, useState } from "react";
import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import type { AppData, ReservoirId } from "@/lib/types";
import { DASH, fmtMagnitude, fmtMetric } from "@/lib/format";
import { weekWindow } from "@/lib/selectors";
import { useDeck } from "@/lib/store";
import { basinColor } from "./MapDeck";

const M = { top: 30, right: 20, bottom: 26, left: 20 };
const W = 1180;
const H = 210;
const iw = W - M.left - M.right;
const ih = H - M.top - M.bottom;

export default function CascadeView() {
  const { state, dispatch } = useDeck();
  const data = state.data!;
  const first = data.cascade[0]?.upstream ?? null;
  const [from, setFrom] = useState<ReservoirId | null>(state.cascadeFrom ?? first);
  const [open, setOpen] = useState(false);

  const active = from ?? first;
  const link = useMemo(() => data.cascade.find((c) => c.upstream === active) ?? null, [data, active]);
  const nameOf = (id: string) => data.reservoirs.find((r) => r.id === id)?.name ?? id;

  return (
    <div>
      {/* segmented reach selector */}
      <div className="px-4 py-3 border-b border-hair flex flex-wrap items-center gap-2">
        <span className="eyebrow mr-1">Reach</span>
        {data.cascade.map((c) => {
          const on = c.upstream === active;
          return (
            <button
              key={`${c.upstream}->${c.downstream}`}
              onClick={() => {
                setFrom(c.upstream);
                dispatch({ type: "cascadeFrom", value: c.upstream });
              }}
              className="ctl ctl-sm num transition-colors"
              style={{
                color: on ? "var(--txt)" : "var(--txt-3)",
                borderColor: on ? "rgba(62,201,182,0.5)" : "var(--hair)",
                background: on ? "rgba(62,201,182,0.09)" : "transparent",
              }}
              aria-pressed={on}
            >
              {nameOf(c.upstream)}
              <span style={{ color: "var(--teal)" }} className="mx-1.5">
                →
              </span>
              {nameOf(c.downstream)}
              <span className="text-data-faint ml-2">lag {c.lag_days}d</span>
            </button>
          );
        })}
      </div>

      {!link ? (
        <div className="px-4 py-10 text-center note">
          No cascade link for this node — it is a terminal or isolated reach.
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 border-b border-hair">
            <Stat label="Upstream" value={nameOf(link.upstream)} tone={basinColor(basinOf(data, link.upstream))} />
            <Stat label="Downstream" value={nameOf(link.downstream)} tone={basinColor(basinOf(data, link.downstream))} />
            <Stat label="Measured lag" value={`${link.lag_days} days`} tone="var(--teal)" sub="max cross-correlation" />
            <Stat
              label="Peak correlation"
              value={fmtMetric(link.peak_corr, 3)}
              sub={`${link.n_pairs.toLocaleString("en-IN")} daily pairs`}
              tone="var(--indigo)"
            />
          </div>

          <div className="px-4 pt-3">
            <PulsePlot data={data} link={link} />
          </div>

          <div className="rule px-4 py-2.5">
            <button
              onClick={() => setOpen((v) => !v)}
              className="w-full text-left label hover:text-data-dim flex items-center gap-2"
              aria-expanded={open}
            >
              <span className="text-data-faint">{open ? "▾" : "▸"}</span>
              What this lag is — and is not
            </button>
            {open && (
              <div className="mt-2 note max-w-4xl">
                <p className="mb-2">{link.cascade_note}</p>
                <p className="mb-0">
                  The downstream series is shifted by the measured lag, so a shared monsoon pulse
                  should line up if the reach genuinely couples. Almatti→Srisailam shows a multi-week
                  lag because Almatti sits far up the Krishna and its release signal reaches Srisailam
                  only after routing through the intervening reach.
                </p>
              </div>
            )}
          </div>

          <div className="rule overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-hair">
                  <th className="px-4 py-2 text-left font-medium text-data-faint" style={{ fontSize: 11.5 }}>Reach</th>
                  <th className="px-4 py-2 text-left font-medium text-data-faint" style={{ fontSize: 11.5 }}>Basin</th>
                  <th className="px-4 py-2 text-right font-medium text-data-faint" style={{ fontSize: 11.5 }}>Lag</th>
                  <th className="px-4 py-2 text-right font-medium text-data-faint" style={{ fontSize: 11.5 }}>Peak r</th>
                  <th className="px-4 py-2 text-right font-medium text-data-faint" style={{ fontSize: 11.5 }}>Pairs</th>
                </tr>
              </thead>
              <tbody>
                {data.cascade.map((c) => {
                  const on = c.upstream === active;
                  return (
                    <tr
                      key={`${c.upstream}->${c.downstream}`}
                      onClick={() => {
                        setFrom(c.upstream);
                        dispatch({ type: "cascadeFrom", value: c.upstream });
                      }}
                      className={`border-b border-hair/70 cursor-pointer transition-colors ${
                        on ? "bg-data-teal/[0.08]" : "hover:bg-white/[0.022]"
                      }`}
                    >
                      <td className="px-4 py-2 text-data-text whitespace-nowrap">
                        {nameOf(c.upstream)} <span style={{ color: "var(--teal)" }}>→</span> {nameOf(c.downstream)}
                      </td>
                      <td className="px-4 py-2 text-data-faint">{basinOf(data, c.upstream)}</td>
                      <td className="num px-4 py-2 text-right text-data-text">{c.lag_days}d</td>
                      <td className="num px-4 py-2 text-right text-data-dim">{fmtMetric(c.peak_corr, 3)}</td>
                      <td className="num px-4 py-2 text-right text-data-dim">
                        {c.n_pairs.toLocaleString("en-IN")}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function basinOf(data: AppData, id: string): string {
  return data.reservoirs.find((r) => r.id === id)?.basin ?? "";
}

function Stat({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone: string }) {
  return (
    <div className="px-4 py-3 border-r border-hair last:border-r-0">
      <div className="eyebrow">{label}</div>
      <div className="stat mt-1" style={{ color: tone }}>
        {value}
      </div>
      {sub && <div className="label mt-0.5">{sub}</div>}
    </div>
  );
}

/**
 * Upstream inflow against the lag-shifted downstream inflow. Each series is
 * normalised to its own maximum — this is a timing comparison, and the caption
 * says so rather than leaving the reader to assume shared units.
 */
function PulsePlot({ data, link }: { data: AppData; link: { upstream: string; downstream: string; lag_days: number } }) {
  const { state } = useDeck();

  const series = useMemo(() => {
    const dates = data.daily_dates;
    const idx = dates.indexOf(state.origin ?? dates[dates.length - 1]);
    if (idx < 0) return null;
    const from = Math.max(0, idx - 300);
    const to = Math.min(dates.length - 1, idx + 7 * 12);
    const up = data.daily[link.upstream]?.inflow ?? [];
    const down = data.daily[link.downstream]?.inflow ?? [];
    const out: Array<{ k: number; up: number | null; down: number | null; d: string }> = [];
    for (let i = from; i <= to; i++) {
      out.push({
        k: i - from,
        up: up[i] ?? null,
        down: i + link.lag_days < down.length ? down[i + link.lag_days] ?? null : null,
        d: dates[i],
      });
    }
    return { rows: out, originRel: idx - from };
  }, [data, link.upstream, link.downstream, link.lag_days, state.origin]);

  if (!series) return <div className="note py-8 text-center">No daily record at this origin.</div>;

  const { rows, originRel } = series;
  const upMax = Math.max(1, ...rows.map((r) => r.up ?? 0));
  const downMax = Math.max(1, ...rows.map((r) => r.down ?? 0));

  const x = scaleLinear().domain([0, Math.max(1, rows.length - 1)]).range([M.left, M.left + iw]);
  const yUp = scaleLinear().domain([0, upMax]).range([M.top + ih, M.top]);
  const yDn = scaleLinear().domain([0, downMax]).range([M.top + ih, M.top]);

  const upPath =
    line<{ k: number; up: number | null }>()
      .defined((r) => r.up !== null)
      .x((r) => x(r.k))
      .y((r) => yUp(r.up as number))(rows) ?? "";
  const dnPath =
    line<{ k: number; down: number | null }>()
      .defined((r) => r.down !== null)
      .x((r) => x(r.k))
      .y((r) => yDn(r.down as number))(rows) ?? "";

  const wSum = weekWindow(data, link.downstream, state.origin!, state.horizon);
  const originX = x(originRel);

  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block" role="img"
           aria-label={`${link.upstream} inflow versus ${link.downstream} inflow shifted by ${link.lag_days} days`}>
        <rect x={originX} y={M.top} width={M.left + iw - originX} height={ih} fill="rgba(99,176,218,0.07)" />

        <g stroke="rgba(148,178,188,0.07)" strokeWidth="0.6">
          {[0, 0.5, 1].map((f) => (
            <line key={f} x1={M.left} y1={M.top + ih * f} x2={M.left + iw} y2={M.top + ih * f} />
          ))}
        </g>

        <path d={upPath} fill="none" stroke="var(--teal)" strokeWidth="1.5" strokeLinejoin="round" />
        <path d={dnPath} fill="none" stroke="var(--amber)" strokeWidth="1.5" strokeLinejoin="round" strokeDasharray="4 2" />

        <line x1={originX} y1={M.top} x2={originX} y2={M.top + ih} stroke="rgba(234,247,244,0.5)" strokeWidth="1" />
        <text x={originX + 5} y={M.top - 8} className="num" fontSize="10.5" fill="var(--txt-3)">
          origin
        </text>

        {/* series labels sit above the plot so they never collide with the trace */}
        <g>
          <line x1={M.left} y1={M.top - 16} x2={M.left + 16} y2={M.top - 16} stroke="var(--teal)" strokeWidth="2" />
          <text x={M.left + 22} y={M.top - 12} className="num" fontSize="11" fill="var(--teal)">
            {link.upstream} inflow
          </text>
          <line x1={M.left + 175} y1={M.top - 16} x2={M.left + 191} y2={M.top - 16} stroke="var(--amber)" strokeWidth="2" strokeDasharray="4 2" />
          <text x={M.left + 197} y={M.top - 12} className="num" fontSize="11" fill="var(--amber)">
            {link.downstream} inflow, shifted +{link.lag_days}d
          </text>
        </g>

        <g className="num" fill="#3d4f56" fontSize="10.5">
          {rows
            .map((r, k) => ({ r, k }))
            .filter(({ r }) => r.d.endsWith("-01"))
            .map(({ r, k }) => (
              <text key={r.d} x={x(k)} y={M.top + ih + 17} textAnchor="middle">
                {r.d.slice(0, 7)}
              </text>
            ))}
        </g>
      </svg>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-5 gap-y-1">
        <span className="label">
          Each series is scaled to its own maximum — a timing comparison, not a volume comparison.
        </span>
        <span className="num text-2xs text-data-faint">
          week {state.horizon} downstream total {wSum.sum === null ? DASH : fmtMagnitude(wSum.sum)}
          {wSum.partial ? " (partial window)" : ""}
        </span>
      </div>
    </>
  );
}
