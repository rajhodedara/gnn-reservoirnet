"use client";

/**
 * The signature chart: quantile fan for one reservoir.
 *
 * Redesign changes:
 *  - Series labels are direct-labelled at the end of each line instead of living
 *    in a legend block below the chart, so the eye does not have to travel.
 *  - Axis text is at the product type scale and legible; the "m³/s·day" unit is
 *    stated once, next to the y axis.
 *  - The provenance sentence moved into a small eyebrow under the plot rather
 *    than sitting in a row of legend swatches.
 *
 * Degradation contract (unchanged, and important): if the export carries real
 * model quantiles they are used. If only the median exists, the ribbon falls back
 * to the empirical residual band and the chart SAYS SO. The two are never
 * presented as the same thing.
 */

import React, { useMemo } from "react";
import { scaleLinear } from "d3-scale";
import { line, area } from "d3-shape";
import type { AppData, Reservoir } from "@/lib/types";
import { addDays, fmtDate, fmtMagnitude, isMonsoon } from "@/lib/format";
import { useDeck } from "@/lib/store";
import { horizonSlice } from "./MapDeck";

const M = { top: 22, right: 62, bottom: 32, left: 58 };

interface Props {
  data: AppData;
  reservoir: Reservoir;
  origin: string;
  horizon: number;
  height?: number;
  compact?: boolean;
}

export default function FanChart({ data, reservoir, origin, horizon, height = 268, compact = false }: Props) {
  const { dispatch } = useDeck();
  const W = 780;
  const H = height;
  const iw = W - M.left - M.right;
  const ih = H - M.top - M.bottom;

  const series = useMemo(
    () =>
      Array.from({ length: 12 }, (_, i) => {
        const w = i + 1;
        const s = horizonSlice(data, origin, reservoir.id, w);
        return {
          week: w,
          date: addDays(origin, 7 * (w - 1)),
          p10: s.p10,
          p50: s.p50,
          p90: s.p90,
          observed: s.observed,
          bandIsEmpirical: s.bandIsEmpirical,
        };
      }),
    [data, origin, reservoir.id],
  );

  const bandIsEmpirical = series.some((s) => s.bandIsEmpirical);
  const hasBand = series.some((s) => s.p10 !== null && s.p90 !== null);

  const yMax = useMemo(() => {
    const vals = series.flatMap((s) => [s.p10, s.p50, s.p90, s.observed]).filter((v): v is number => v !== null);
    return vals.length ? Math.max(...vals) * 1.08 : 1;
  }, [series]);

  const x = useMemo(() => scaleLinear().domain([1, 12]).range([M.left, M.left + iw]), [iw]);
  const y = useMemo(() => scaleLinear().domain([0, yMax]).range([M.top + ih, M.top]), [ih, yMax]);

  const p50Path = useMemo(
    () =>
      line<{ week: number; p50: number | null }>()
        .defined((d) => d.p50 !== null)
        .x((d) => x(d.week))
        .y((d) => y(d.p50 as number))(series) ?? "",
    [series, x, y],
  );
  const obsPath = useMemo(
    () =>
      line<{ week: number; observed: number | null }>()
        .defined((d) => d.observed !== null)
        .x((d) => x(d.week))
        .y((d) => y(d.observed as number))(series) ?? "",
    [series, x, y],
  );
  const bandPath = useMemo(() => {
    if (!hasBand) return "";
    return (
      area<{ week: number; p10: number | null; p90: number | null }>()
        .defined((d) => d.p10 !== null && d.p90 !== null)
        .x((d) => x(d.week))
        .y0((d) => y(Math.max(0, d.p10 as number)))
        .y1((d) => y(d.p90 as number))(series) ?? ""
    );
  }, [series, x, y, hasBand]);

  const monsoonWindows = useMemo(() => {
    const spans: Array<[number, number]> = [];
    let start: number | null = null;
    series.forEach((s, i) => {
      if (isMonsoon(s.date)) {
        if (start === null) start = i;
      } else if (start !== null) {
        spans.push([start, i - 1]);
        start = null;
      }
    });
    if (start !== null) spans.push([start, series.length - 1]);
    return spans;
  }, [series]);

  const yTicks = useMemo(() => y.ticks(4), [y]);
  const cur = series[horizon - 1];

  // End labels: where each line finishes, so the reader identifies series in place.
  const lastOf = (key: "p50" | "observed") => {
    for (let i = series.length - 1; i >= 0; i--) {
      const v = series[i][key];
      if (v !== null && Number.isFinite(v)) return { v: v as number, week: series[i].week };
    }
    return null;
  };
  const endObs = lastOf("observed");
  const endP50 = lastOf("p50");

  const onMove = (ev: React.MouseEvent<SVGSVGElement>) => {
    const rect = ev.currentTarget.getBoundingClientRect();
    const rel = ((ev.clientX - rect.left) / rect.width) * W;
    const wk = Math.round(x.invert(rel));
    if (wk >= 1 && wk <= 12) dispatch({ type: "horizon", value: wk });
  };

  return (
    <div className="w-full">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-auto block touch-none"
        onClick={onMove}
        onMouseMove={(e) => {
          if (e.buttons === 1) onMove(e);
        }}
        role="img"
        aria-label={`Quantile fan for ${reservoir.name} from ${origin}`}
      >
        <defs>
          <linearGradient id="bandFill2" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#3ec9b6" stopOpacity="0.26" />
            <stop offset="100%" stopColor="#3ec9b6" stopOpacity="0.07" />
          </linearGradient>
        </defs>

        {/* monsoon shading */}
        {monsoonWindows.map(([a, b], i) => (
          <rect
            key={`mon${i}`}
            x={x(series[a].week - 0.5)}
            y={M.top}
            width={x(series[b].week + 0.5) - x(series[a].week - 0.5)}
            height={ih}
            fill="rgba(99,176,218,0.07)"
          />
        ))}

        {/* grid + y axis */}
        <g stroke="rgba(148,178,188,0.09)" strokeWidth="0.6">
          {yTicks.map((t) => (
            <line key={`gy${t}`} x1={M.left} y1={y(t)} x2={M.left + iw} y2={y(t)} />
          ))}
        </g>
        <g className="num" fill="#4f626a" fontSize="11">
          {yTicks.map((t) => (
            <text key={`ty${t}`} x={M.left - 8} y={y(t) + 4} textAnchor="end">
              {fmtMagnitude(t, 0)}
            </text>
          ))}
          <text x={M.left - 8} y={M.top - 8} textAnchor="end" fill="#6a7f87" fontSize="10.5">
            m³/s·day
          </text>
          {series.map((s) => (
            <text
              key={`tx${s.week}`}
              x={x(s.week)}
              y={M.top + ih + 18}
              textAnchor="middle"
              fill={s.week === horizon ? "var(--teal)" : "#4f626a"}
            >
              {s.week}
            </text>
          ))}
          <text x={M.left + iw} y={M.top + ih + 18} textAnchor="end" fill="#6a7f87" fontSize="10.5">
            horizon (weeks)
          </text>
        </g>

        {/* ribbon */}
        {hasBand && <path d={bandPath} fill="url(#bandFill2)" stroke="none" />}
        {hasBand && (
          <g stroke="rgba(62,201,182,0.4)" strokeWidth="0.8" fill="none" strokeDasharray="2 3">
            <path
              d={
                line<{ week: number; p10: number | null }>()
                  .defined((d) => d.p10 !== null)
                  .x((d) => x(d.week))
                  .y((d) => y(Math.max(0, d.p10 as number)))(series) ?? ""
              }
            />
            <path
              d={
                line<{ week: number; p90: number | null }>()
                  .defined((d) => d.p90 !== null)
                  .x((d) => x(d.week))
                  .y((d) => y(d.p90 as number))(series) ?? ""
              }
            />
          </g>
        )}

        {/* observed then median, with end labels */}
        <path d={obsPath} fill="none" stroke="var(--amber)" strokeWidth="2" strokeLinejoin="round" />
        <path d={p50Path} fill="none" stroke="#eaf7f4" strokeWidth="1.9" strokeLinejoin="round" />

        {endObs && !compact && (
          <text
            x={x(endObs.week) + 6}
            y={y(endObs.v) + 4}
            className="num"
            fontSize="11"
            fill="var(--amber)"
          >
            observed
          </text>
        )}
        {endP50 && !compact && (
          <text
            x={x(endP50.week) + 6}
            y={y(endP50.v) + 4}
            className="num"
            fontSize="11"
            fill="#eaf7f4"
          >
            P50
          </text>
        )}

        {/* points */}
        {series.map((s) =>
          s.p50 === null ? null : (
            <circle key={`p${s.week}`} cx={x(s.week)} cy={y(s.p50)} r={s.week === horizon ? 3.8 : 1.9} fill="#eaf7f4" />
          ),
        )}

        {/* horizon cursor */}
        <g className="sweep">
          <line x1={x(horizon)} y1={M.top} x2={x(horizon)} y2={M.top + ih} stroke="var(--teal)" strokeWidth="1.4" />
          <circle cx={x(horizon)} cy={M.top + 5} r="2.8" fill="var(--teal)" />
        </g>

        {/* origin marker */}
        <g>
          <line
            x1={M.left}
            y1={M.top + ih}
            x2={M.left}
            y2={M.top}
            stroke="rgba(148,178,188,0.3)"
            strokeWidth="0.8"
            strokeDasharray="3 3"
          />
          <text x={M.left + 5} y={M.top + 13} className="num" fontSize="11" fill="#6a7f87">
            origin {fmtDate(origin)}
          </text>
        </g>

        {/* readout for the current horizon */}
        {cur && (
          <g transform={`translate(${Math.min(x(horizon) + 10, W - M.right - 4)},${M.top + 38})`}>
            <text className="num" fontSize="11.5" fill="#93a7af" textAnchor="end">
              wk {horizon} · {fmtDate(cur.date)}
            </text>
            <text className="num" fontSize="12" fill="#eaf7f4" textAnchor="end" dy="14">
              {fmtMagnitude(cur.p50)}
            </text>
            <text className="num" fontSize="11" fill="var(--amber)" textAnchor="end" dy="14">
              obs {fmtMagnitude(cur.observed)}
            </text>
          </g>
        )}
      </svg>

      <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1">
        <span className="eyebrow" style={{ color: bandIsEmpirical ? "var(--txt-4)" : "var(--teal)" }}>
          {bandIsEmpirical
            ? "ribbon = empirical P10–P90 of held-out residuals, not a model-issued quantile"
            : "ribbon = model-issued quantiles"}
        </span>
        {cur && cur.p50 === null && <span className="eyebrow">· week {horizon} unavailable</span>}
      </div>
    </div>
  );
}
