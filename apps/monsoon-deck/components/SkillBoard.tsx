"use client";

/**
 * Skill scoreboard.
 *
 * Redesign changes: the duplicate header is gone (the shell's Block supplies the
 * title and the question), the table reads at the new type scale, long-form
 * caveats moved into a collapsible note, and the win/loss chips now pair colour
 * with a glyph and text so they survive colour-blindness and greyscale print.
 */

import React, { useMemo, useState } from "react";
import type { AppData } from "@/lib/types";
import { fmtMetric, fmtSigned, fmtMagnitude, DASH } from "@/lib/format";
import {
  basinRollup,
  ensoRows,
  scoreboard,
  systemSummary,
  type Outcome,
  type ScoreRow,
} from "@/lib/selectors";
import { useDeck } from "@/lib/store";

type SortKey =
  | "name"
  | "basin"
  | "nseAtHorizon"
  | "nse"
  | "kge"
  | "rmse"
  | "crps"
  | "persistence"
  | "climatology";

const COLS: Array<{ key: SortKey; label: string; align: "left" | "right"; title: string }> = [
  { key: "name", label: "Reservoir", align: "left", title: "Reservoir" },
  { key: "basin", label: "Basin", align: "left", title: "River basin" },
  { key: "nseAtHorizon", label: "NSE@wk", align: "right", title: "NSE at the selected horizon" },
  { key: "persistence", label: "Persist.", align: "right", title: "Persistence baseline NSE" },
  { key: "climatology", label: "Clim.", align: "right", title: "Seasonal climatology baseline NSE" },
  { key: "kge", label: "KGE", align: "right", title: "Kling-Gupta efficiency (pooled)" },
  { key: "crps", label: "CRPS", align: "right", title: "Continuous ranked probability score — lower is better" },
  { key: "rmse", label: "RMSE", align: "right", title: "Root mean squared error, m³/s·day — lower is better" },
];

export default function SkillBoard() {
  const { state, dispatch } = useDeck();
  const data = state.data!;
  const [sort, setSort] = useState<SortKey>("nseAtHorizon");
  const [dir, setDir] = useState<1 | -1>(-1);
  const [showBasins, setShowBasins] = useState(false);
  const [showCaveats, setShowCaveats] = useState(false);

  const rows = useMemo(() => {
    const base = scoreboard(data, state.horizon);
    return [...base].sort((a, b) => {
      const av = val(a, sort);
      const bv = val(b, sort);
      if (typeof av === "string" || typeof bv === "string") {
        return String(av).localeCompare(String(bv)) * dir;
      }
      const an = av as number | null;
      const bn = bv as number | null;
      if (an === null && bn === null) return 0;
      if (an === null) return 1;
      if (bn === null) return -1;
      return (an - bn) * dir;
    });
  }, [data, sort, dir, state.horizon]);

  const summary = systemSummary(rows);
  const basins = useMemo(() => basinRollup(rows, data), [rows, data]);
  const enso = useMemo(() => ensoRows(data), [data]);

  const click = (k: SortKey) => {
    if (k === sort) setDir((d) => (d === 1 ? -1 : 1));
    else {
      setSort(k);
      setDir(k === "name" || k === "basin" || k === "crps" || k === "rmse" ? 1 : -1);
    }
  };

  const elNino = enso.find((e) => e.condition === "El Nino");
  const neutral = enso.find((e) => e.condition === "Neutral");

  return (
    <div>
      {/* headline counts — the honest scoreboard, stated up front */}
      <div className="grid grid-cols-3 border-b border-hair">
        <Kpi
          value={`${summary.winPers}/${summary.total}`}
          label="beat persistence"
          sub="the harder baseline"
          tone="var(--teal)"
        />
        <Kpi
          value={`${summary.winClim}/${summary.total}`}
          label="beat climatology"
          sub="seasonal benchmark"
          tone="var(--amber)"
        />
        <Kpi
          value={fmtMetric(summary.mean)}
          label="mean NSE"
          sub={`${data.metrics.headline.n_seeds}-seed average`}
          tone="var(--txt)"
        />
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-hair">
              {COLS.map((c) => (
                <th
                  key={c.key}
                  title={c.title}
                  onClick={() => click(c.key)}
                  className={`cursor-pointer select-none px-3.5 py-2.5 font-medium whitespace-nowrap transition-colors ${
                    c.align === "right" ? "text-right" : "text-left"
                  } ${sort === c.key ? "text-data-teal" : "text-data-faint hover:text-data-dim"}`}
                  style={{ fontSize: 11.5 }}
                >
                  {c.label}
                  {sort === c.key && <span className="ml-1 opacity-70">{dir === 1 ? "▲" : "▼"}</span>}
                </th>
              ))}
              <th
                className="px-3.5 py-2.5 text-right font-medium text-data-faint whitespace-nowrap"
                style={{ fontSize: 11.5 }}
                title="Does the model beat each baseline on this reservoir?"
              >
                Verdict
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const isSel = state.selected === r.id;
              return (
                <tr
                  key={r.id}
                  onClick={() => dispatch({ type: "select", value: r.id })}
                  className={`border-b border-hair/70 cursor-pointer transition-colors ${
                    isSel ? "bg-data-teal/[0.08]" : "hover:bg-white/[0.022]"
                  }`}
                >
                  <td className="px-3.5 py-2 whitespace-nowrap">
                    <span className={`inline-flex items-center gap-2 ${isSel ? "text-data-teal" : "text-data-text"}`}>
                      <span
                        className="inline-block w-[3px] h-[13px]"
                        style={{ background: isSel ? "var(--teal)" : "transparent" }}
                        aria-hidden
                      />
                      {r.name}
                    </span>
                  </td>
                  <td className="px-3.5 py-2 text-data-faint whitespace-nowrap">{r.basin}</td>
                  <td className="num px-3.5 py-2 text-right text-data-text">
                    {fmtMetric(r.nseAtHorizon)}
                    {r.nseAtHorizonStd !== null && (
                      <span className="ml-1.5 text-2xs text-data-faint">
                        ±{r.nseAtHorizonStd.toFixed(2)}
                      </span>
                    )}
                  </td>
                  <td className="num px-3.5 py-2 text-right text-data-dim">{fmtMetric(r.persistence)}</td>
                  <td className="num px-3.5 py-2 text-right text-data-dim">{fmtMetric(r.climatology)}</td>
                  <td className="num px-3.5 py-2 text-right text-data-dim">{fmtMetric(r.kge)}</td>
                  <td className="num px-3.5 py-2 text-right text-data-dim">{fmtMagnitude(r.crps, 1)}</td>
                  <td className="num px-3.5 py-2 text-right text-data-dim">{fmtMagnitude(r.rmse, 1)}</td>
                  <td className="px-3.5 py-2 text-right whitespace-nowrap">
                    <Chip o={r.vsPersistence} label="P" />
                    <Chip o={r.vsClimatology} label="C" />
                  </td>
                </tr>
              );
            })}
          </tbody>
          <tfoot>
            <tr className="border-t border-hair2">
              <td className="px-3.5 py-2.5 text-data-dim font-medium">Pooled mean</td>
              <td />
              <td className="num px-3.5 py-2.5 text-right text-data-text">{fmtMetric(summary.mean)}</td>
              <td className="num px-3.5 py-2.5 text-right text-data-dim">
                {fmtMetric(data.baselines.pooled?.persistence?.NSE ?? null)}
              </td>
              <td className="num px-3.5 py-2.5 text-right text-data-dim">
                {fmtMetric(data.baselines.pooled?.climatology?.NSE ?? null)}
              </td>
              <td colSpan={4} />
            </tr>
          </tfoot>
        </table>
      </div>

      <div className="rule">
        <button
          onClick={() => setShowBasins((v) => !v)}
          className="w-full text-left px-3.5 py-2.5 label hover:text-data-dim flex items-center gap-2"
          aria-expanded={showBasins}
        >
          <span className="text-data-faint">{showBasins ? "▾" : "▸"}</span>
          Basin rollup
        </button>
        {showBasins && (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-hair">
                <th className="px-3.5 py-2 text-left font-medium text-data-faint" style={{ fontSize: 11.5 }}>Basin</th>
                <th className="px-3.5 py-2 text-right font-medium text-data-faint" style={{ fontSize: 11.5 }}>Dams</th>
                <th className="px-3.5 py-2 text-right font-medium text-data-faint" style={{ fontSize: 11.5 }}>NSE</th>
                <th className="px-3.5 py-2 text-right font-medium text-data-faint" style={{ fontSize: 11.5 }}>KGE</th>
                <th className="px-3.5 py-2 text-right font-medium text-data-faint" style={{ fontSize: 11.5 }}>RMSE</th>
                <th className="px-3.5 py-2 text-right font-medium text-data-faint" style={{ fontSize: 11.5 }}>CRPS</th>
                <th className="px-3.5 py-2 text-right font-medium text-data-faint" style={{ fontSize: 11.5 }}>Beat P / C</th>
              </tr>
            </thead>
            <tbody>
              {basins.map((b) => (
                <tr key={b.basin} className="border-b border-hair/70">
                  <td className="px-3.5 py-2 text-data-text">{b.basin}</td>
                  <td className="num px-3.5 py-2 text-right text-data-faint">{b.dams}</td>
                  <td className="num px-3.5 py-2 text-right text-data-text">{fmtMetric(b.nse)}</td>
                  <td className="num px-3.5 py-2 text-right text-data-dim">{fmtMetric(b.kge)}</td>
                  <td className="num px-3.5 py-2 text-right text-data-dim">{fmtMagnitude(b.rmse, 1)}</td>
                  <td className="num px-3.5 py-2 text-right text-data-dim">{fmtMagnitude(b.crps, 1)}</td>
                  <td className="num px-3.5 py-2 text-right">
                    <span className="text-data-teal">{b.winsVsPersistence}</span>
                    <span className="text-data-faint"> / </span>
                    <span className="text-data-amber">{b.winsVsClimatology}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* ENSO lens as a compact comparison, not another full table */}
      <div className="rule px-3.5 py-3">
        <div className="eyebrow mb-2.5">El Niño versus neutral · pooled over all horizons and reservoirs</div>
        <div className="grid grid-cols-2 gap-3">
          <EnsoCell label="El Niño" tone="var(--amber)" row={elNino} />
          <EnsoCell label="Neutral" tone="var(--teal)" row={neutral} />
        </div>
      </div>

      <div className="rule">
        <button
          onClick={() => setShowCaveats((v) => !v)}
          className="w-full text-left px-3.5 py-2.5 label hover:text-data-dim flex items-center gap-2"
          aria-expanded={showCaveats}
        >
          <span className="text-data-faint">{showCaveats ? "▾" : "▸"}</span>
          Reading these numbers honestly
        </button>
        {showCaveats && (
          <div className="px-3.5 pb-3 note max-w-4xl">
            <p className="mb-2">
              The El Niño row is genuinely negative on NSE. That is the model&apos;s real behaviour on
              the few El Niño origins inside the held-out year, pooled into a small sample. It is shown
              rather than smoothed, because hiding it would misstate the operating envelope.
            </p>
            <p className="mb-2">
              LOSS markers are rendered at exactly the same weight as WIN markers. Several reservoirs
              lose to seasonal climatology; those losses are not filtered out, re-ranked, or explained
              away by the layout.
            </p>
            <p className="mb-0">
              Primary metrics are the mean over {data.metrics.headline.n_seeds} training seeds, with the
              seed standard deviation shown beside the horizon NSE as ±. A single-seed result would be
              materially less stable than it appears.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function Kpi({ value, label, sub, tone }: { value: string; label: string; sub: string; tone: string }) {
  return (
    <div className="px-3.5 py-3 border-r border-hair last:border-r-0">
      <div className="stat-lg" style={{ color: tone }}>
        {value}
      </div>
      <div className="label mt-1">{label}</div>
      <div className="text-2xs text-data-faint">{sub}</div>
    </div>
  );
}

function EnsoCell({
  label,
  tone,
  row,
}: {
  label: string;
  tone: string;
  row?: { nse: number | null; kge: number | null; rmse: number | null; crps: number | null };
}) {
  return (
    <div className="panel-tight px-3 py-2.5">
      <div className="text-[12.5px] font-medium mb-1.5" style={{ color: tone }}>
        {label}
      </div>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-2xs">
        <dt className="text-data-faint">NSE</dt>
        <dd className="num text-right text-data-text">{fmtMetric(row?.nse ?? null)}</dd>
        <dt className="text-data-faint">KGE</dt>
        <dd className="num text-right text-data-dim">{fmtMetric(row?.kge ?? null)}</dd>
        <dt className="text-data-faint">RMSE</dt>
        <dd className="num text-right text-data-dim">{fmtMagnitude(row?.rmse ?? null, 1)}</dd>
        <dt className="text-data-faint">CRPS</dt>
        <dd className="num text-right text-data-dim">{fmtMagnitude(row?.crps ?? null, 1)}</dd>
      </dl>
    </div>
  );
}

function Chip({ o, label }: { o: Outcome; label: string }) {
  if (o === "tie") {
    return <span className="num text-2xs text-data-faint mr-1.5 last:mr-0">{DASH}</span>;
  }
  const win = o === "win";
  return (
    <span
      className="ctl ctl-xs num inline-flex gap-1 mr-1.5 last:mr-0"
      style={{
        color: win ? "var(--teal)" : "var(--rose)",
        borderColor: win ? "rgba(62,201,182,0.45)" : "rgba(207,101,99,0.45)",
        background: win ? "rgba(62,201,182,0.08)" : "rgba(207,101,99,0.08)",
      }}
      title={
        win
          ? `Model beats ${label === "P" ? "persistence" : "climatology"} on this reservoir`
          : `Model loses to ${label === "P" ? "persistence" : "climatology"} on this reservoir`
      }
    >
      <span aria-hidden>{win ? "▲" : "▼"}</span>
      {label}
    </span>
  );
}

function val(r: ScoreRow, k: SortKey): string | number | null {
  switch (k) {
    case "name":
      return r.name;
    case "basin":
      return r.basin;
    case "nseAtHorizon":
      return r.nseAtHorizon;
    case "nse":
      return r.nse;
    case "kge":
      return r.kge;
    case "rmse":
      return r.rmse;
    case "crps":
      return r.crps;
    case "persistence":
      return r.persistence;
    case "climatology":
      return r.climatology;
    default:
      return null;
  }
}

export { fmtSigned, DASH };
