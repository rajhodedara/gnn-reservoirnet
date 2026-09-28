"use client";

/**
 * The map is the instrument.
 *
 * Redesign changes that matter:
 *  - Node labels no longer sit permanently under every dot. Ten always-on
 *    labels collided with the reaches and each other; only the selected and
 *    hovered nodes are labelled, which is the standard cartographic answer and
 *    immediately de-clutters the graphic.
 *  - Nodes are bigger and carry three encodings at once: ring = capacity fill,
 *    radius = gross capacity (importance), tick = forecast inflow for the
 *    current horizon.
 *  - The graticule is quieter and its degree labels are legible instead of
 *    near-invisible 10px text straddling the frame edge.
 *  - A real HTML readout card replaces the cramped SVG tooltip, so it can use
 *    the same type scale as the rest of the product.
 */

import React, { useMemo, useState } from "react";
import type { AppData, ForecastPoint, Reservoir as Res } from "@/lib/types";
import {
  PENINSULA_BOUNDS,
  edgeGeom,
  geoToPaths,
  graticule,
  makeProjector,
} from "@/lib/projection";
import { fillFraction, fmtMagnitude, fmtMetric, fmtDate, DASH } from "@/lib/format";
import { focusSet } from "@/lib/selectors";
import { useDeck } from "@/lib/store";

const VB_W = 1000;
const VB_H = 820;

export interface HorizonSlice {
  p50: number | null;
  p10: number | null;
  p90: number | null;
  observed: number | null;
  bandIsEmpirical: boolean;
}

export function horizonSlice(data: AppData, origin: string, id: string, horizon: number): HorizonSlice {
  const f: ForecastPoint | undefined = data.forecast[origin]?.[id];
  const p50 = f?.p50?.[horizon - 1] ?? null;
  const observed = (f?.observed?.[horizon - 1] ?? null) as number | null;
  const native10 = f?.p10?.[horizon - 1] ?? null;
  const native90 = f?.p90?.[horizon - 1] ?? null;
  if (native10 !== null && native90 !== null) {
    return { p50, p10: native10, p90: native90, observed, bandIsEmpirical: false };
  }
  const eb = data.forecast_residual_band?.[origin]?.[id];
  return {
    p50,
    p10: eb?.p10?.[horizon - 1] ?? null,
    p90: eb?.p90?.[horizon - 1] ?? null,
    observed,
    bandIsEmpirical: true,
  };
}

export function basinColor(basin: string): string {
  switch (basin) {
    case "krishna":
      return "#3ec9b6";
    case "cauvery":
      return "#8598e8";
    case "godavari":
      return "#63b0da";
    case "narmada":
      return "#eda941";
    case "tapi":
      return "#cf6563";
    default:
      return "#93a7af";
  }
}

/** Gross capacity → node radius. Sqrt keeps area proportional to volume. */
function nodeRadius(grossMcm: number): number {
  const t = Math.sqrt(Math.max(0, grossMcm)) / Math.sqrt(11560);
  return 9 + t * 7;
}

export default function MapDeck({ geo }: { geo: { states: any | null; outline: any | null } }) {
  const { state, dispatch } = useDeck();
  const data = state.data!;
  const { horizon, selected, hovered, origin, focusMode } = state;
  const [tip, setTip] = useState<{ id: string } | null>(null);

  const proj = useMemo(() => makeProjector(PENINSULA_BOUNDS, VB_W, VB_H, 10), []);
  const grat = useMemo(() => graticule(PENINSULA_BOUNDS, 2), []);

  const byId = useMemo(() => {
    const m = new Map<string, Res>();
    for (const r of data.reservoirs) m.set(r.id, r);
    return m;
  }, [data.reservoirs]);

  const statePaths = useMemo(
    () => (geo.states?.features ?? []).flatMap((f: { geometry: unknown }) => geoToPaths(f.geometry, proj)),
    [geo.states, proj],
  );
  const outlinePaths = useMemo(
    () => (geo.outline?.features ?? []).flatMap((f: { geometry: unknown }) => geoToPaths(f.geometry, proj)),
    [geo.outline, proj],
  );

  const physical = useMemo(
    () =>
      data.edges
        .filter((e) => e.kind === "physical")
        .map((e) => {
          const a = byId.get(e.source);
          const b = byId.get(e.target);
          if (!a || !b) return null;
          const bow = e.source === "almatti" ? 0.07 : e.source === "tungabhadra" ? -0.11 : 0.05;
          return { ...edgeGeom(a.lon, a.lat, b.lon, b.lat, e.source, e.target, proj, bow), a, b };
        })
        .filter(Boolean) as Array<ReturnType<typeof edgeGeom> & { a: Res; b: Res }>,
    [data.edges, byId, proj],
  );

  const climEdges = useMemo(
    () =>
      data.edges
        .filter((e) => e.kind === "climatological")
        .map((e) => {
          const a = byId.get(e.source);
          const b = byId.get(e.target);
          if (!a || !b) return null;
          return { ...edgeGeom(a.lon, a.lat, b.lon, b.lat, e.source, e.target, proj, 0.16), w: e.weight ?? 0.6 };
        })
        .filter(Boolean) as Array<ReturnType<typeof edgeGeom> & { w: number }>,
    [data.edges, byId, proj],
  );

  const focus = useMemo(
    () => (focusMode && selected ? focusSet(data, selected) : null),
    [focusMode, selected, data],
  );
  const focusActive = (id: string) => !focus || focus.has(id);

  const shown = hovered ?? selected;
  const tipRes = tip ? byId.get(tip.id) : null;

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className="w-full h-auto block"
        role="img"
        aria-label="Peninsular India reservoir network with forecast inflows"
      >
        <defs>
          <radialGradient id="base" cx="52%" cy="44%" r="78%">
            <stop offset="0%" stopColor="#0c171d" />
            <stop offset="72%" stopColor="#070c10" />
            <stop offset="100%" stopColor="#04070a" />
          </radialGradient>
          <linearGradient id="reach" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#2c857a" />
            <stop offset="100%" stopColor="#3ec9b6" />
          </linearGradient>
        </defs>

        <rect x="0" y="0" width={VB_W} height={VB_H} fill="url(#base)" />

        {/* quiet graticule */}
        <g stroke="rgba(148,178,188,0.055)" strokeWidth="0.5">
          {grat.lon.map((lon) => {
            const p1 = proj.project(lon, PENINSULA_BOUNDS.latMin);
            const p2 = proj.project(lon, PENINSULA_BOUNDS.latMax);
            return <line key={`lo${lon}`} x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} />;
          })}
          {grat.lat.map((lat) => {
            const p1 = proj.project(PENINSULA_BOUNDS.lonMin, lat);
            const p2 = proj.project(PENINSULA_BOUNDS.lonMax, lat);
            return <line key={`la${lat}`} x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} />;
          })}
        </g>

        {/* landmass: real state polygons */}
        <g>
          {statePaths.map((d: string, i: number) => (
            <path key={`st${i}`} d={d} fill="#0a1319" stroke="rgba(148,178,188,0.17)" strokeWidth="0.6" />
          ))}
        </g>
        <g>
          {outlinePaths.map((d: string, i: number) => (
            <path key={`ol${i}`} d={d} fill="none" stroke="rgba(148,178,188,0.30)" strokeWidth="0.9" />
          ))}
        </g>

        {/* graticule labels — placed inside the frame, readable */}
        <g className="num" fill="#3d4f56" fontSize="11">
          {grat.lon.map((lon) => {
            const p = proj.project(lon, PENINSULA_BOUNDS.latMin);
            return (
              <text key={`lt${lon}`} x={p.x} y={VB_H - 8} textAnchor="middle">
                {lon}°E
              </text>
            );
          })}
          {grat.lat.map((lat) => {
            const p = proj.project(PENINSULA_BOUNDS.lonMin, lat);
            return (
              <text key={`laT${lat}`} x={8} y={p.y + 4}>
                {lat}°N
              </text>
            );
          })}
        </g>

        {/* climatological links */}
        <g opacity={selected && focusMode ? 0.2 : 0.42}>
          {climEdges.map((e) => {
            if (focus && !(focus.has(e.from) && focus.has(e.to))) return null;
            return (
              <path
                key={e.key}
                d={e.path}
                fill="none"
                stroke="#4f7d9b"
                strokeWidth={0.5 + (e.w - 0.6) * 2}
                strokeDasharray="1 6"
              />
            );
          })}
        </g>

        {/* hydraulic reaches */}
        <g>
          {physical.map((e) => {
            const dim = !focusActive(e.from) && !focusActive(e.to);
            const active = focus?.has(e.from) && focus?.has(e.to);
            return (
              <g key={e.key} opacity={dim ? 0.16 : 1}>
                <path d={e.path} fill="none" stroke="rgba(62,201,182,0.14)" strokeWidth={active ? 9 : 6} />
                <path
                  d={e.path}
                  fill="none"
                  stroke="url(#reach)"
                  strokeWidth={active ? 2 : 1.4}
                  strokeLinecap="round"
                />
                <path
                  className="flow"
                  d={e.path}
                  fill="none"
                  stroke="#c9f5ee"
                  strokeWidth={active ? 1.6 : 1.1}
                  strokeLinecap="round"
                  opacity={active ? 0.95 : 0.55}
                />
                <g transform={`translate(${e.x2},${e.y2})`} opacity={dim ? 0.18 : 0.9}>
                  <ArrowHead from={e} />
                </g>
              </g>
            );
          })}
        </g>

        {/* nodes */}
        <g>
          {data.reservoirs.map((r) => {
            const p = proj.project(r.lon, r.lat);
            const sl = horizonSlice(data, origin!, r.id, horizon);
            const storageNow = storageAt(data, r.id, origin!);
            const frac = fillFraction(storageNow, r.gross_capacity_mcm);
            const R = nodeRadius(r.gross_capacity_mcm);
            const isSel = selected === r.id;
            const dim = !focusActive(r.id);
            const showLabel = isSel || shown === r.id;
            return (
              <g
                key={r.id}
                transform={`translate(${p.x},${p.y})`}
                opacity={dim ? 0.28 : 1}
                style={{ cursor: "pointer", transition: "opacity 200ms ease" }}
                onMouseEnter={() => {
                  dispatch({ type: "hover", value: r.id });
                  setTip({ id: r.id });
                }}
                onMouseLeave={() => {
                  dispatch({ type: "hover", value: null });
                  setTip(null);
                }}
                onClick={() => dispatch({ type: "select", value: r.id })}
                role="button"
                tabIndex={0}
                onKeyDown={(ev) => {
                  if (ev.key === "Enter" || ev.key === " ") {
                    ev.preventDefault();
                    dispatch({ type: "select", value: r.id });
                  }
                }}
                aria-label={`${r.name}, ${r.basin} basin, forecast inflow ${fmtMagnitude(sl.p50)} cubic metres per second per day`}
              >
                <circle r={R} fill="rgba(5,8,10,0.9)" stroke="rgba(148,178,188,0.20)" strokeWidth="1" />
                <FillRing radius={R} frac={frac} />
                {isSel && (
                  <circle r={R + 6} fill="none" stroke="var(--teal)" strokeWidth="1.2" opacity="0.9" />
                )}
                <circle r={isSel ? 5 : 3.8} fill={basinColor(r.basin)} stroke="#04070a" strokeWidth="0.9" />
                <InflowTick radius={R} value={sl.p50} />
                {showLabel && (
                  <g>
                    <text
                      x={0}
                      y={R + 15}
                      textAnchor="middle"
                      fontSize="12.5"
                      fontWeight={isSel ? 600 : 500}
                      fill={isSel ? "#eaf7f4" : "#a9bcc3"}
                      style={{
                        paintOrder: "stroke",
                        stroke: "rgba(4,7,10,0.92)",
                        strokeWidth: 3.5,
                        strokeLinejoin: "round",
                      }}
                    >
                      {r.name}
                    </text>
                    <text
                      x={0}
                      y={R + 29}
                      textAnchor="middle"
                      className="num"
                      fontSize="11.5"
                      fill={isSel ? "#8fd9cd" : "#7b9099"}
                      style={{
                        paintOrder: "stroke",
                        stroke: "rgba(4,7,10,0.92)",
                        strokeWidth: 3.5,
                        strokeLinejoin: "round",
                      }}
                    >
                      {fmtMagnitude(sl.p50)}
                    </text>
                  </g>
                )}
              </g>
            );
          })}
        </g>
      </svg>

      {/* HTML readout card — shares the product type scale. */}
      {tipRes && (
        <div className="pointer-events-none absolute left-3 bottom-3 panel-tight px-3.5 py-3 w-[286px] backdrop-blur-[2px]">
          <div className="text-[14px] font-semibold text-data-text">{tipRes.name}</div>
          <div className="label mb-2">
            {tipRes.river} · {tipRes.basin} basin · {tipRes.state}
          </div>
          <div className="rule pt-2.5">
            <div className="eyebrow mb-1.5">
              forecast · week {horizon} · {fmtDate(addDaysSafe(origin, 7 * (horizon - 1)))}
            </div>
            <div className="stat-lg">
              {fmtMagnitude(horizonSlice(data, origin!, tipRes.id, horizon).p50)}
            </div>
            <div className="label mb-2">m³/s·day forecast inflow</div>
            <div className="num text-[12px] text-data-indigo">
              P10–P90 {fmtMagnitude(horizonSlice(data, origin!, tipRes.id, horizon).p10)} –{" "}
              {fmtMagnitude(horizonSlice(data, origin!, tipRes.id, horizon).p90)}
            </div>
            <div className="num text-[12px]" style={{ color: "var(--amber)" }}>
              observed {fmtMagnitude(horizonSlice(data, origin!, tipRes.id, horizon).observed)}
            </div>
          </div>
          <div className="rule mt-2.5 pt-2.5">
            <div className="eyebrow mb-1.5">storage at origin</div>
            <div className="num text-[12px] text-data-text">
              {fmtStorage(storageAt(data, tipRes.id, origin!))}
              <span className="label"> / {fmtMagnitude(tipRes.gross_capacity_mcm / 28.3168, 0)} TMC cap</span>
            </div>
          </div>
        </div>
      )}

      {/* Legend. The previous pass had no key at all for the node encodings, so
          the ring, the tick and the size had to be guessed. */}
      <div className="absolute top-3 right-3 panel-tight px-3 py-2.5 w-[212px]">
        <div className="eyebrow mb-2">Legend</div>
        <ul className="space-y-1.5">
          <LegendRow glyph="ring" label="Ring = storage fill" />
          <LegendRow glyph="tick" label="Tick = forecast inflow (log)" />
          <LegendRow glyph="dot" label="Dot = basin; size = capacity" />
          <LegendRow glyph="reach" label="Solid = hydraulic reach" />
          <LegendRow glyph="clim" label="Dotted = rainfall correlation" />
        </ul>
      </div>
    </div>
  );
}

function LegendRow({ glyph, label }: { glyph: "ring" | "tick" | "dot" | "reach" | "clim"; label: string }) {
  return (
    <li className="flex items-center gap-2">
      <svg width="18" height="14" viewBox="0 0 18 14" aria-hidden className="shrink-0">
        {glyph === "ring" && (
          <>
            <circle cx="9" cy="7" r="5" fill="none" stroke="rgba(148,178,188,0.35)" strokeWidth="1.6" />
            <circle
              cx="9"
              cy="7"
              r="5"
              fill="none"
              stroke="var(--teal)"
              strokeWidth="1.6"
              strokeDasharray="23 40"
              strokeDashoffset="31"
              transform="rotate(-90 9 7)"
            />
          </>
        )}
        {glyph === "tick" && <line x1="9" y1="12" x2="9" y2="1" stroke="var(--indigo)" strokeWidth="1.6" strokeLinecap="round" />}
        {glyph === "dot" && <circle cx="9" cy="7" r="3.4" fill="var(--teal)" />}
        {glyph === "reach" && <line x1="0" y1="7" x2="18" y2="7" stroke="var(--teal)" strokeWidth="1.6" />}
        {glyph === "clim" && <line x1="0" y1="7" x2="18" y2="7" stroke="#4f7d9b" strokeWidth="1.2" strokeDasharray="1 4" />}
      </svg>
      <span className="label">{label}</span>
    </li>
  );
}

function addDaysSafe(iso: string | null, days: number): string {
  if (!iso) return "";
  const d = new Date(`${iso}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

function fmtStorage(v: number | null): string {
  return v === null ? DASH : `${v.toFixed(1)} TMC`;
}

function ArrowHead({ from }: { from: { x1: number; y1: number; x2: number; y2: number } }) {
  const ang = (Math.atan2(from.y2 - from.y1, from.x2 - from.x1) * 180) / Math.PI;
  return <path d="M0,0 L-10,-4.5 L-10,4.5 Z" transform={`rotate(${ang})`} fill="#c9f5ee" opacity="0.9" />;
}

function FillRing({ radius, frac }: { radius: number; frac: number | null }) {
  const c = 2 * Math.PI * radius;
  if (frac === null) {
    return (
      <circle r={radius} fill="none" stroke="#47595f" strokeWidth="2.6" strokeDasharray="2 4" />
    );
  }
  const f = Math.max(0, Math.min(1, frac));
  const col = frac >= 0.7 ? "#3ec9b6" : frac >= 0.35 ? "#8598e8" : frac >= 0.15 ? "#eda941" : "#cf6563";
  return (
    <>
      <circle r={radius} fill="none" stroke="rgba(148,178,188,0.14)" strokeWidth="2.6" />
      <circle
        r={radius}
        fill="none"
        stroke={col}
        strokeWidth="2.6"
        strokeDasharray={`${(c * f).toFixed(2)} ${(c * (1 - f)).toFixed(2)}`}
        strokeDashoffset={c * 0.25}
        transform="rotate(-90)"
        style={{ transition: "stroke-dasharray 220ms ease, stroke 220ms ease" }}
      />
    </>
  );
}

function InflowTick({ radius, value }: { radius: number; value: number | null }) {
  if (value === null || !Number.isFinite(value)) return null;
  const v = Math.max(0, value);
  const len = v <= 0 ? 0 : Math.min(13, 4 + Math.log10(1 + v) * 1.9);
  if (len === 0) {
    return (
      <circle
        r={radius + 3}
        fill="none"
        stroke="#cf6563"
        strokeWidth="1"
        opacity="0.8"
        strokeDasharray="1 3"
      />
    );
  }
  return (
    <line
      x1={0}
      y1={-radius - 2}
      x2={0}
      y2={-radius - 2 - len}
      stroke="#8598e8"
      strokeWidth="1.8"
      strokeLinecap="round"
    />
  );
}

function storageAt(data: AppData, id: string, iso: string): number | null {
  const dates = data.daily_dates;
  const target = iso.slice(0, 10);
  let lo = 0;
  let hi = dates.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (dates[mid] === target) return data.daily[id]?.storage?.[mid] ?? null;
    if (dates[mid] < target) lo = mid + 1;
    else hi = mid - 1;
  }
  return null;
}

export { fmtMetric };
