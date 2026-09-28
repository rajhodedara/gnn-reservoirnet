"use client";

/**
 * Robustness — rolling-origin 2020–2024, 5 folds × 12 weeks.
 *
 * The point is to make "week-1 skill holds in every year" visually obvious, so the
 * week-1 column is emphasised and the diverging scale is centred on zero.
 *
 * Redesign changes: the shell supplies the panel and heading; the cell matrix runs
 * taller with legible numerals; week-1 is separated by a rule so the emphasised
 * column reads as deliberate rather than randomly coloured; and a compact colour
 * key replaces the gradient strip that carried no numbers. Years that fail are
 * still shown — 2020 and 2024 both go negative at long horizons.
 */

import React, { useMemo, useState } from "react";
import { DASH, fmtMetric } from "@/lib/format";
import { rollingOriginRows } from "@/lib/selectors";
import { useDeck } from "@/lib/store";

/** Diverging scale around 0: rose (−1) → ink (0) → teal (+1). */
export function nseColor(v: number | null, scale = 0.8): string {
  if (v === null || !Number.isFinite(v)) return "rgba(148,178,188,0.05)";
  const t = Math.max(-1, Math.min(1, v / scale));
  if (t >= 0) return `rgba(62,201,182,${(0.10 + 0.62 * t).toFixed(3)})`;
  return `rgba(207,101,99,${(0.10 + 0.62 * -t).toFixed(3)})`;
}

export function nseTextColor(v: number | null): string {
  if (v === null || !Number.isFinite(v)) return "#4f626a";
  if (v >= 0.45) return "#03110f";
  if (v >= 0) return "#d3f0ea";
  if (v >= -0.3) return "#f2d6d5";
  return "#f7e6e5";
}

export default function RobustnessView() {
  const { state } = useDeck();
  const data = state.data!;
  const rows = useMemo(() => rollingOriginRows(data), [data]);
  const [metric, setMetric] = useState<"week" | "pooled">("week");
  const [open, setOpen] = useState(false);

  const weekKeys = Array.from({ length: 12 }, (_, i) => `week_${i + 1}`);
  const week1Vals = rows
    .map((r) => r.week_1 as number | null)
    .filter((v): v is number => v !== null);
  const week1Min = week1Vals.length ? Math.min(...week1Vals) : null;
  const week1Max = week1Vals.length ? Math.max(...week1Vals) : null;

  return (
    <div>
      <div className="px-4 py-3 border-b border-hair flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <div className="flex items-baseline gap-3">
          <span className="eyebrow">Week-1 range</span>
          <span className="stat" style={{ color: "var(--teal)" }}>
            {week1Min === null || week1Max === null
              ? DASH
              : `${week1Min.toFixed(2)} – ${week1Max.toFixed(2)}`}
          </span>
          <span className="label">across {rows.length} folds</span>
        </div>
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setMetric("week")}
            className="text-[11.5px] px-2 py-1 border transition-colors"
            style={{
              color: metric === "week" ? "var(--txt)" : "var(--txt-3)",
              borderColor: metric === "week" ? "var(--hair-2)" : "var(--hair)",
            }}
            aria-pressed={metric === "week"}
          >
            12-week matrix
          </button>
          <button
            onClick={() => setMetric("pooled")}
            className="text-[11.5px] px-2 py-1 border transition-colors"
            style={{
              color: metric === "pooled" ? "var(--txt)" : "var(--txt-3)",
              borderColor: metric === "pooled" ? "var(--hair-2)" : "var(--hair)",
            }}
            aria-pressed={metric === "pooled"}
          >
            Pooled
          </button>
        </div>
      </div>

      {metric === "week" ? (
        <div className="overflow-x-auto px-4 pt-3.5">
          <table className="w-full border-separate" style={{ borderSpacing: "3px" }}>
            <thead>
              <tr>
                <th
                  className="text-left pr-3 font-medium text-data-faint whitespace-nowrap"
                  style={{ fontSize: 11 }}
                >
                  Test year
                </th>
                {weekKeys.map((k, i) => (
                  <th
                    key={k}
                    className="text-center font-medium whitespace-nowrap"
                    style={{
                      fontSize: 11,
                      color: i === 0 ? "var(--teal)" : "var(--txt-3)",
                    }}
                    title={i === 0 ? "Week-1 skill across all folds" : undefined}
                  >
                    {i + 1}
                  </th>
                ))}
                <th
                  className="text-right pl-3 font-medium text-data-faint whitespace-nowrap"
                  style={{ fontSize: 11 }}
                >
                  Pooled
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.fold}>
                  <td className="num pr-3 text-[12px] text-data-dim whitespace-nowrap">{r.fold}</td>
                  {weekKeys.map((k, i) => {
                    const v = r[k] as number | null;
                    return (
                      <td key={k}>
                        <div
                          className="num h-8 flex items-center justify-center text-[11.5px]"
                          style={{
                            background: nseColor(v),
                            color: nseTextColor(v),
                            outline: i === 0 ? "1px solid rgba(62,201,182,0.45)" : "none",
                          }}
                          title={`${r.fold} · week ${i + 1} · NSE ${fmtMetric(v)}`}
                        >
                          {v === null ? "" : v.toFixed(2)}
                        </div>
                      </td>
                    );
                  })}
                  <td className="num pl-3 text-right text-[12px] text-data-text">
                    {fmtMetric(r.pooled_mean_NSE as number | null)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 py-3">
            <div className="flex items-center gap-2">
              <span className="num text-[10.5px] text-data-rose">−0.8</span>
              <span
                className="inline-block w-28 h-2.5"
                style={{
                  background:
                    "linear-gradient(90deg, rgba(207,101,99,0.72), rgba(148,178,188,0.06), rgba(62,201,182,0.72))",
                }}
              />
              <span className="num text-[10.5px] text-data-teal">+0.8</span>
              <span className="label ml-1">NSE, centred on zero</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="px-4 pt-3.5">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-hair">
                {["Fold", "Week 1", "Week 4", "Week 12", "Pooled mean"].map((h, i) => (
                  <th
                    key={h}
                    className={`py-2 font-medium text-data-faint ${i === 0 ? "text-left" : "text-right"}`}
                    style={{ fontSize: 11.5 }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.fold} className="border-b border-hair/70">
                  <td className="num py-2 text-data-text">{r.fold}</td>
                  <td
                    className="num py-2 text-right"
                    style={{ color: nseTextColor(r.week_1 as number | null) }}
                  >
                    {fmtMetric(r.week_1 as number | null)}
                  </td>
                  <td className="num py-2 text-right text-data-dim">
                    {fmtMetric(r.week_4 as number | null)}
                  </td>
                  <td
                    className="num py-2 text-right"
                    style={{ color: nseTextColor(r.week_12 as number | null) }}
                  >
                    {fmtMetric(r.week_12 as number | null)}
                  </td>
                  <td className="num py-2 text-right text-data-text">
                    {fmtMetric(r.pooled_mean_NSE as number | null)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="rule">
        <button
          onClick={() => setOpen((v) => !v)}
          className="w-full text-left px-4 py-2.5 label hover:text-data-dim flex items-center gap-2"
          aria-expanded={open}
        >
          <span className="text-data-faint">{open ? "▾" : "▸"}</span>
          How the folds are built
        </button>
        {open && (
          <div className="px-4 pb-3 note max-w-4xl">
            <p className="mb-2">
              Each fold refits on an expanding window: test year Y, validation Y−1, training through
              Y−2, so no fold sees its own test data.
            </p>
            <p className="mb-0">
              The 2023 fold is the El Niño onset year and is the hardest pooled year in the sweep.
              Week-1 skill holds in every fold; week-12 is honestly year-dependent and goes negative in
              several years, which is why the matrix keeps those cells visible rather than trimming the
              range.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
