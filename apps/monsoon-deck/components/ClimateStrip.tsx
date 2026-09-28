"use client";

/**
 * Climate telemetry — the "why is this horizon uncertain?" evidence.
 *
 * Redesign changes: the shell now supplies the panel and its heading; the chart
 * runs taller with legible axis text; active series are direct-labelled at their
 * right edge instead of relying on a separate legend; and the explanatory footer
 * is a collapsible note. Orange/teal regime shading is paired with an explicit
 * word in the readouts so colour is never the sole carrier of meaning.
 *
 * All four indices, the El Niño / La Niña spans, the 2023 onset and the forecast
 * window are located dynamically from the real series — nothing is hard-coded.
 */

import React, { useMemo, useState } from "react";
import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import { DASH, fmtMetric, fmtMonth } from "@/lib/format";
import { climateStats, elNinoOnset, oniRegime } from "@/lib/selectors";
import { useDeck } from "@/lib/store";

type IndexKey = "oni" | "soi" | "nino34" | "iod";

const INDICES: Array<{ key: IndexKey; label: string; color: string }> = [
  { key: "oni", label: "ONI", color: "#eda941" },
  { key: "soi", label: "SOI", color: "#8598e8" },
  { key: "nino34", label: "Niño3.4", color: "#cf6563" },
  { key: "iod", label: "IOD (DMI)", color: "#3ec9b6" },
];

const M = { top: 26, right: 62, bottom: 26, left: 44 };

export default function ClimateStrip({ forecastIso }: { forecastIso: string | null }) {
  const { state } = useDeck();
  const data = state.data!;
  const [active, setActive] = useState<IndexKey[]>(["oni", "iod"]);
  const [zoom, setZoom] = useState(false);
  const [open, setOpen] = useState(false);

  const W = 1180;
  const H = 176;
  const iw = W - M.left - M.right;
  const ih = H - M.top - M.bottom;

  const view = useMemo(() => {
    const c = data.climate;
    const from = zoom ? "2000-01-01" : "1948-01-01";
    return c.dates
      .map((d: string, i: number) => ({ d, i }))
      .filter((x: { d: string }) => x.d >= from && x.d <= "2026-12-01");
  }, [data.climate, zoom]);

  const x = useMemo(
    () => scaleLinear().domain([0, Math.max(1, view.length - 1)]).range([M.left, M.left + iw]),
    [view.length, iw],
  );

  const y = useMemo(() => {
    // Fixed anomaly domain keeps ONI / SOI / Niño3.4 directly comparable.
    let lo = -3;
    let hi = 3;
    const c = data.climate;
    for (const idx of INDICES) {
      if (!active.includes(idx.key)) continue;
      for (const { i } of view) {
        const v = c[idx.key][i];
        if (v === null || !Number.isFinite(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    return scaleLinear().domain([lo, hi]).range([M.top + ih, M.top]);
  }, [active, view, data.climate, ih]);

  const paths = useMemo(
    () =>
      INDICES.filter((i) => active.includes(i.key)).map((idx) => {
        const pts = view.map((v: { d: string; i: number }, k: number) => ({
          k,
          v: data.climate[idx.key][v.i],
        }));
        const d =
          line<{ k: number; v: number | null }>()
            .defined((p: { v: number | null }) => p.v !== null && Number.isFinite(p.v as number))
            .x((p: { k: number }) => x(p.k))
            .y((p: { v: number | null }) => y(p.v as number))(pts) ?? "";
        const last = [...pts].reverse().find((p) => p.v !== null && Number.isFinite(p.v as number));
        return { ...idx, d, last };
      }),
    [active, view, x, y, data.climate],
  );

  const stats = useMemo(() => climateStats(data), [data]);
  const onset = useMemo(() => elNinoOnset(data), [data]);

  const fcStartIdx = useMemo(() => {
    if (!forecastIso) return -1;
    const m = forecastIso.slice(0, 7);
    return view.findIndex((v: { d: string }) => v.d.slice(0, 7) === m);
  }, [forecastIso, view]);

  const fcEndIdx = useMemo(() => {
    if (fcStartIdx < 0) return -1;
    const startMonth = new Date(`${forecastIso!.slice(0, 7)}-01T00:00:00Z`);
    startMonth.setUTCMonth(startMonth.getUTCMonth() + 3);
    const target = startMonth.toISOString().slice(0, 7);
    const i = view.findIndex((v: { d: string }) => v.d.slice(0, 7) === target);
    return i >= 0 ? i : Math.min(view.length - 1, fcStartIdx + 3);
  }, [fcStartIdx, forecastIso, view]);

  const spans = useMemo(() => {
    const out: Array<{ a: number; b: number; kind: "el" | "la" }> = [];
    let start = -1;
    let kind: "el" | "la" | null = null;
    view.forEach((v: { d: string; i: number }, k: number) => {
      const r = oniRegime(data.climate.oni[v.i]);
      const cur: "el" | "la" | null = r === "El Nino" ? "el" : r === "La Nina" ? "la" : null;
      if (cur !== kind) {
        if (kind && start >= 0) out.push({ a: start, b: k - 1, kind });
        kind = cur;
        start = cur ? k : -1;
      }
    });
    if (kind && start >= 0) out.push({ a: start, b: view.length - 1, kind });
    return out;
  }, [view, data.climate.oni]);

  const toggle = (k: IndexKey) =>
    setActive((prev) => (prev.includes(k) ? prev.filter((x2) => x2 !== k) : [...prev, k]));

  return (
    <div>
      {/* index toggles + range */}
      <div className="px-4 py-3 border-b border-hair flex flex-wrap items-center gap-2">
        <span className="eyebrow mr-1">Index</span>
        {INDICES.map((i) => (
          <button
            key={i.key}
            onClick={() => toggle(i.key)}
            className="ctl ctl-sm num"
            style={{
              color: active.includes(i.key) ? i.color : "var(--txt-3)",
              borderColor: active.includes(i.key) ? `${i.color}66` : "var(--hair)",
              background: active.includes(i.key) ? `${i.color}14` : "transparent",
            }}
            aria-pressed={active.includes(i.key)}
          >
            {i.label}
          </button>
        ))}
        <button
          onClick={() => setZoom((z) => !z)}
          className="ctl ctl-sm ctl-quiet ml-1"
        >
          {zoom ? "Since 2000" : "Full record"}
        </button>
      </div>

      <div className="px-3 pt-3">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full h-auto block"
          role="img"
          aria-label="Climate teleconnection indices: ONI, SOI, Niño3.4 and IOD monthly series with El Niño and La Niña periods shaded"
        >
          {/* regime spans */}
          {spans.map((s, i) => (
            <rect
              key={`sp${i}`}
              x={x(s.a)}
              y={M.top}
              width={Math.max(0.8, x(s.b) - x(s.a))}
              height={ih}
              fill={s.kind === "el" ? "rgba(237,169,65,0.11)" : "rgba(62,201,182,0.10)"}
            />
          ))}

          {/* zero + ±0.5 thresholds */}
          <line x1={M.left} y1={y(0)} x2={M.left + iw} y2={y(0)} stroke="rgba(148,178,188,0.18)" strokeWidth="0.7" />
          <g stroke="rgba(237,169,65,0.3)" strokeWidth="0.7" strokeDasharray="3 3">
            <line x1={M.left} y1={y(0.5)} x2={M.left + iw} y2={y(0.5)} />
            <line x1={M.left} y1={y(-0.5)} x2={M.left + iw} y2={y(-0.5)} />
          </g>

          {/* forecast window */}
          {fcStartIdx >= 0 && fcEndIdx >= 0 && (
            <g>
              <rect
                x={x(fcStartIdx)}
                y={M.top}
                width={Math.max(2, x(fcEndIdx) - x(fcStartIdx))}
                height={ih}
                fill="rgba(234,247,244,0.10)"
                stroke="rgba(234,247,244,0.45)"
                strokeWidth="0.8"
              />
              <text
                x={Math.min(x(fcStartIdx) + 4, M.left + iw - 4)}
                y={M.top - 8}
                className="num"
                fontSize="10.5"
                fill="#93a7af"
              >
                forecast window
              </text>
            </g>
          )}

          {/* series + direct end labels */}
          {paths.map((p) => (
            <g key={p.key}>
              <path d={p.d} fill="none" stroke={p.color} strokeWidth="1.4" strokeLinejoin="round" />
              {p.last && (
                <text
                  x={Math.min(x(p.last.k) + 6, W - 4)}
                  y={y(p.last.v as number) + 4}
                  className="num"
                  fontSize="10.5"
                  fill={p.color}
                >
                  {p.key === "nino34" ? "N3.4" : p.key === "iod" ? "IOD" : p.key.toUpperCase()}
                </text>
              )}
            </g>
          ))}

          {/* axes */}
          <g className="num" fill="#4f626a" fontSize="10.5">
            {y.ticks(4).map((t) => (
              <text key={`ct${t}`} x={M.left - 7} y={y(t) + 3.5} textAnchor="end">
                {t}
              </text>
            ))}
            {view
              .map((v: { d: string; i: number }, k: number) => ({ v, k }))
              .filter(({ v }: { v: { d: string } }) => v.d.endsWith("-01-01"))
              .filter(({ v }: { v: { d: string } }) =>
                zoom ? Number(v.d.slice(0, 4)) % 4 === 0 : Number(v.d.slice(0, 4)) % 10 === 0,
              )
              .map(({ v, k }: { v: { d: string }; k: number }) => (
                <text key={v.d} x={x(k)} y={M.top + ih + 16} textAnchor="middle">
                  {v.d.slice(0, 4)}
                </text>
              ))}
          </g>

          {/* 2023 onset, located from the real ONI series */}
          {onset.firstIso &&
            (() => {
              const k = view.findIndex(
                (v: { d: string }) => v.d.slice(0, 7) === onset.firstIso!.slice(0, 7),
              );
              if (k < 0) return null;
              return (
                <g>
                  <line
                    x1={x(k)}
                    y1={M.top}
                    x2={x(k)}
                    y2={M.top + ih}
                    stroke="rgba(237,169,65,0.75)"
                    strokeWidth="1"
                  />
                  <text x={x(k) + 4} y={M.top + ih - 5} className="num" fontSize="10.5" fill="#eda941">
                    2023 El Niño onset
                  </text>
                </g>
              );
            })()}
        </svg>
      </div>

      {/* readouts */}
      <div className="grid grid-cols-2 md:grid-cols-4 border-y border-hair">
        <Readout
          label="2023 onset · ONI ≥ 0.5"
          value={onset.firstIso ? fmtMonth(onset.firstIso) : DASH}
          tone="var(--amber)"
        />
        <Readout
          label="Peak ONI 2023–24"
          value={onset.peakOni === null ? DASH : fmtMetric(onset.peakOni, 2)}
          sub={onset.peakIso ? fmtMonth(onset.peakIso) : undefined}
          tone="var(--amber)"
        />
        <Readout
          label="El Niño months since 1990"
          value={String(stats.elNinoMonths)}
          sub={`La Niña ${stats.laNinaMonths} · neutral ${stats.neutralMonths}`}
          tone="var(--amber)"
        />
        <Readout
          label="IOD coverage"
          value={`${data.climate.iod.filter((v: number | null) => v !== null).length} / ${data.climate.iod.length}`}
          sub="Nulls are real gaps, never zero-filled"
          tone="var(--teal)"
        />
      </div>

      <div>
        <button
          onClick={() => setOpen((v) => !v)}
          className="w-full text-left px-4 py-2.5 label hover:text-data-dim flex items-center gap-2"
          aria-expanded={open}
        >
          <span className="text-data-faint">{open ? "▾" : "▸"}</span>
          Why this matters for a horizon
        </button>
        {open && (
          <div className="px-4 pb-3 note max-w-4xl">
            <p className="mb-2">
              El Niño (ONI ≥ 0.5) and La Niña (≤ −0.5) periods are shaded from the real monthly
              series. The IOD record begins in 1979 and ONI/SOI have pre-1980 gaps — those months are
              drawn as breaks, never interpolated.
            </p>
            <p className="mb-0">
              Monsoon inflow is 60–90% of the annual total, so a positive-IOD co-occurrence with
              El Niño (as in 2023) is exactly the regime where horizon uncertainty matters most. This
              is the telemetry behind the confidence ribbon widening on the map and fan chart.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function Readout({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: string;
  sub?: string;
  tone: string;
}) {
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
