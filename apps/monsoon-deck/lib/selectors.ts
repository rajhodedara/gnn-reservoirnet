/**
 * Derived views over AppData. Pure functions only — no React, no side effects, so
 * the numbers can be unit-checked and reused by any component.
 *
 * Honesty rule that runs through this file: the GNN LOSES to climatology on
 * several dams. Those losses are computed and surfaced, never filtered out.
 */

import type {
  AppData,
  EnsoLens,
  MetricsRow,
  ReservoirId,
  RollingOriginRow,
} from "./types";

export type Outcome = "win" | "loss" | "tie";

/** El Niño when ONI >= 0.5, La Niña when ONI <= -0.5, else Neutral. */
export function oniRegime(oni: number | null | undefined): "El Nino" | "La Nina" | "Neutral" {
  if (oni === null || oni === undefined || !Number.isFinite(oni)) return "Neutral";
  if (oni >= 0.5) return "El Nino";
  if (oni <= -0.5) return "La Nina";
  return "Neutral";
}

/** Map each forecast origin date to its ONI regime, using the real monthly ONI series. */
export function originRegimes(data: AppData): Record<string, "El Nino" | "La Nina" | "Neutral"> {
  const byMonth = new Map<string, number | null>();
  data.climate.dates.forEach((d, i) => {
    byMonth.set(d.slice(0, 7), data.climate.oni[i] ?? null);
  });
  const out: Record<string, "El Nino" | "La Nina" | "Neutral"> = {};
  for (const iso of Object.keys(data.forecast)) {
    out[iso] = oniRegime(byMonth.get(iso.slice(0, 7)));
  }
  return out;
}

export interface ScoreRow {
  id: ReservoirId;
  name: string;
  basin: string;
  river: string;
  state: string;
  tribunal: string;
  nse: number | null;
  nseStd: number | null;
  kge: number | null;
  rmse: number | null;
  crps: number | null;
  eventNse: number | null;
  persistence: number | null;
  climatology: number | null;
  vsPersistence: Outcome;
  vsClimatology: Outcome;
  /** NSE at the currently-selected horizon, from the per-week table. */
  nseAtHorizon: number | null;
  nseAtHorizonStd: number | null;
}

/**
 * Prefer the 5-seed mean columns when present, and fall back to the single-seed
 * fold values otherwise, so the deck still renders against an older export.
 */
function pickNse(row: MetricsRow | undefined): number | null {
  if (!row) return null;
  return row.NSE_mean ?? row.NSE ?? null;
}

function pickStd(row: MetricsRow | undefined): number | null {
  if (!row) return null;
  return row.NSE_std ?? null;
}

function pickKge(row: MetricsRow | undefined): number | null {
  if (!row) return null;
  return row.KGE_mean ?? row.KGE ?? null;
}

function pickRmse(row: MetricsRow | undefined): number | null {
  if (!row) return null;
  return row.RMSE_mean ?? row.RMSE ?? null;
}

function pickCrps(row: MetricsRow | undefined): number | null {
  if (!row) return null;
  return row.CRPS_mean ?? row.CRPS ?? null;
}

function pickEvent(row: MetricsRow | undefined): number | null {
  if (!row) return null;
  return row.Event_NSE_mean ?? row.Event_NSE ?? null;
}

function outcome(gnn: number | null, base: number | null): Outcome {
  if (gnn === null || base === null) return "tie";
  const d = gnn - base;
  if (Math.abs(d) < 1e-6) return "tie";
  return d > 0 ? "win" : "loss";
}

export function scoreboard(data: AppData, horizon: number): ScoreRow[] {
  const perRes = new Map<string, MetricsRow>();
  for (const r of data.metrics.per_reservoir) {
    const id = r.reservoir_id ?? labelToId(data, r.Reservoir ?? "");
    if (id) perRes.set(id, r);
  }
  const perWeekHorizon = new Map<string, MetricsRow>();
  for (const r of data.metrics.per_week) {
    if (r.Week !== horizon) continue;
    const id = r.reservoir_id ?? labelToId(data, r.Reservoir ?? "");
    if (id) perWeekHorizon.set(id, r);
  }

  return data.reservoirs.map((res) => {
    const m = perRes.get(res.id);
    const wk = perWeekHorizon.get(res.id);
    const p = data.baselines.persistence[res.id]?.NSE ?? null;
    const c = data.baselines.climatology[res.id]?.NSE ?? null;
    const nse = pickNse(m);
    return {
      id: res.id,
      name: res.name,
      basin: res.basin,
      river: res.river,
      state: res.state,
      tribunal: res.tribunal,
      nse,
      nseStd: pickStd(m),
      kge: pickKge(m),
      rmse: pickRmse(m),
      crps: pickCrps(m),
      eventNse: pickEvent(m),
      persistence: p,
      climatology: c,
      vsPersistence: outcome(nse, p),
      vsClimatology: outcome(nse, c),
      nseAtHorizon: pickNse(wk),
      nseAtHorizonStd: pickStd(wk),
    };
  });
}

function labelToId(data: AppData, label: string): string | null {
  const norm = label.trim().toLowerCase();
  for (const r of data.reservoirs) {
    if (r.name.toLowerCase() === norm) return r.id;
    if (r.id.toLowerCase() === norm) return r.id;
  }
  return null;
}

export interface BasinRollup {
  basin: string;
  nse: number | null;
  kge: number | null;
  rmse: number | null;
  crps: number | null;
  eventNse: number | null;
  dams: number;
  winsVsPersistence: number;
  winsVsClimatology: number;
}

export function basinRollup(rows: ScoreRow[], data: AppData): BasinRollup[] {
  const byBasin = new Map<string, MetricsRow>();
  for (const r of data.metrics.per_basin) {
    if (r.Basin) byBasin.set(r.Basin, r);
  }
  const grouped = new Map<string, ScoreRow[]>();
  for (const r of rows) {
    const list = grouped.get(r.basin) ?? [];
    list.push(r);
    grouped.set(r.basin, list);
  }
  return [...grouped.entries()]
    .map(([basin, dams]) => {
      // Prefer the model's own pooled per-basin row; fall back to a mean of the dams.
      const m = byBasin.get(basin);
      const mean = (pick: (r: ScoreRow) => number | null) => {
        const vals = dams.map(pick).filter((v): v is number => v !== null && Number.isFinite(v));
        return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
      };
      return {
        basin,
        nse: pickNse(m) ?? mean((d) => d.nse),
        kge: pickKge(m) ?? mean((d) => d.kge),
        rmse: pickRmse(m) ?? mean((d) => d.rmse),
        crps: pickCrps(m) ?? mean((d) => d.crps),
        eventNse: pickEvent(m) ?? mean((d) => d.eventNse),
        dams: dams.length,
        winsVsPersistence: dams.filter((d) => d.vsPersistence === "win").length,
        winsVsClimatology: dams.filter((d) => d.vsClimatology === "win").length,
      };
    })
    .sort((a, b) => (b.nse ?? -Infinity) - (a.nse ?? -Infinity));
}

/** Headline system totals, counted straight off the scoreboard. */
export function systemSummary(rows: ScoreRow[]) {
  const total = rows.length;
  const winPers = rows.filter((r) => r.vsPersistence === "win").length;
  const winClim = rows.filter((r) => r.vsClimatology === "win").length;
  const positive = rows.filter((r) => (r.nse ?? -1) > 0).length;
  const vals = rows.map((r) => r.nse).filter((v): v is number => v !== null && Number.isFinite(v));
  const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  return { total, winPers, winClim, positive, mean };
}

/** The pooled El Niño / Neutral rows from the export (both real, 5-seed means). */
export function ensoRows(data: AppData) {
  return data.metrics.enso.map((r) => ({
    condition: r.Condition ?? "",
    nse: pickNse(r),
    kge: pickKge(r),
    rmse: pickRmse(r),
    crps: pickCrps(r),
    eventNse: pickEvent(r),
  }));
}

/** Weekly NSE curve per reservoir across the 12 horizons. */
export function weeklyNse(data: AppData): Record<string, (number | null)[]> {
  const out: Record<string, (number | null)[]> = {};
  for (const r of data.reservoirs) out[r.id] = Array.from({ length: 12 }, () => null);
  for (const row of data.metrics.per_week) {
    const id = row.reservoir_id ?? labelToId(data, row.Reservoir ?? "");
    if (!id || !row.Week) continue;
    const arr = out[id];
    if (!arr) continue;
    arr[row.Week - 1] = pickNse(row);
  }
  return out;
}

/** Weekly NSE standard deviation, for error bars on the seed spread. */
export function weeklyNseStd(data: AppData): Record<string, (number | null)[]> {
  const out: Record<string, (number | null)[]> = {};
  for (const r of data.reservoirs) out[r.id] = Array.from({ length: 12 }, () => null);
  for (const row of data.metrics.per_week) {
    const id = row.reservoir_id ?? labelToId(data, row.Reservoir ?? "");
    if (!id || !row.Week) continue;
    const arr = out[id];
    if (!arr) continue;
    arr[row.Week - 1] = pickStd(row);
  }
  return out;
}

/** Mean NSE at each horizon across all reservoirs — the "skill decay" spine. */
export function meanNseByHorizon(data: AppData): (number | null)[] {
  const wk = weeklyNse(data);
  return Array.from({ length: 12 }, (_, h) => {
    const vals = Object.values(wk)
      .map((a) => a[h])
      .filter((v): v is number => v !== null && Number.isFinite(v));
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  });
}

export function rollingOriginRows(data: AppData): RollingOriginRow[] {
  return [...data.rolling_origin].sort((a, b) => a.fold - b.fold);
}

/** Level rows for one split, at a given horizon. */
export function levelsAt(data: AppData, which: "levels" | "physics_levels", week: number) {
  return data[which]
    .filter((r) => r.week === week)
    .map((r) => ({
      ...r,
      name: data.reservoirs.find((x) => x.id === r.dam)?.name ?? r.dam,
      beatsPersistence: r.level_nse !== null && r.pers_nse !== null ? r.level_nse > r.pers_nse : null,
      beatsClimatology: r.level_nse !== null && r.clim_nse !== null ? r.level_nse > r.clim_nse : null,
    }))
    .sort((a, b) => (b.level_nse ?? -Infinity) - (a.level_nse ?? -Infinity));
}

/**
 * Which downstream node receives a given upstream node, and with what measured lag.
 * Uses the real cascade table; returns null when the node is terminal.
 */
export function cascadeFor(data: AppData, upstream: ReservoirId) {
  return data.cascade.find((c) => c.upstream === upstream) ?? null;
}

export function upstreamsOf(data: AppData, id: ReservoirId): ReservoirId[] {
  return data.cascade.filter((c) => c.downstream === id).map((c) => c.upstream);
}

/**
 * The set of nodes the map should keep bright when one node is focused:
 * the selection, its upstreams, and everything downstream of it, transitively.
 */
export function focusSet(data: AppData, id: ReservoirId | null): Set<string> {
  const set = new Set<string>();
  if (!id) return set;
  set.add(id);
  const addUp = (x: string) => {
    for (const u of upstreamsOf(data, x)) {
      if (!set.has(u)) {
        set.add(u);
        addUp(u);
      }
    }
  };
  const addDown = (x: string) => {
    for (const c of data.cascade.filter((cc) => cc.upstream === x)) {
      if (!set.has(c.downstream)) {
        set.add(c.downstream);
        addDown(c.downstream);
      }
    }
  };
  addUp(id);
  addDown(id);
  return set;
}

/** Does a reservoir take part in any hydraulic cascade at all? */
export function isCascaded(data: AppData, id: ReservoirId): boolean {
  return data.cascade.some((c) => c.upstream === id || c.downstream === id);
}

/** Index of a date inside the shared daily axis, or -1. */
export function dailyIndex(dates: string[], iso: string): number {
  const target = iso.slice(0, 10);
  let lo = 0;
  let hi = dates.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (dates[mid] === target) return mid;
    if (dates[mid] < target) lo = mid + 1;
    else hi = mid - 1;
  }
  return -1;
}

/**
 * Observed daily inflow for the 7 days of a forecast week, plus the weekly sum the
 * model was actually scored on. Both come from the real record.
 */
export function weekWindow(data: AppData, id: ReservoirId, origin: string, week: number, offsetDays = 0) {
  const startIdx = dailyIndex(data.daily_dates, origin);
  if (startIdx < 0) return { days: [], dates: [], sum: null, partial: true };
  const series = data.daily[id]?.inflow ?? [];
  const a = startIdx + 7 * (week - 1) + offsetDays;
  const b = a + 6;
  const days: (number | null)[] = [];
  const dates: string[] = [];
  for (let i = a; i <= b; i++) {
    if (i < 0 || i >= series.length) {
      days.push(null);
      dates.push("");
      continue;
    }
    days.push(series[i] ?? null);
    dates.push(data.daily_dates[i]);
  }
  const present = days.filter((v): v is number => v !== null);
  const partial = present.length !== 7;
  return {
    days,
    dates,
    sum: present.length ? present.reduce((x, y) => x + y, 0) : null,
    partial,
  };
}

/**
 * ENSO lens filter: which origins to include. Uses each origin's real ONI regime.
 * `all` keeps everything.
 */
export function lensOrigins(data: AppData, lens: EnsoLens): string[] {
  const all = Object.keys(data.forecast).sort();
  if (lens === "all") return all;
  const regimes = originRegimes(data);
  return all.filter((o) => regimes[o] === lens);
}

/** Climate index summary for a lens, computed from the real monthly series. */
export function climateStats(data: AppData) {
  const c = data.climate;
  const count = (cond: "El Nino" | "La Nina" | "Neutral") =>
    c.oni.filter((v, i) => oniRegime(v) === cond && c.dates[i] >= "1990-01-01").length;
  return {
    elNinoMonths: count("El Nino"),
    laNinaMonths: count("La Nina"),
    neutralMonths: count("Neutral"),
  };
}

/** The 2023 El Niño onset window, located in the real ONI series (not hard-coded fiction). */
export function elNinoOnset(data: AppData): { firstIso: string | null; peakIso: string | null; peakOni: number | null } {
  const c = data.climate;
  let firstIdx = -1;
  for (let i = 0; i < c.dates.length; i++) {
    if (c.dates[i] < "2023-01-01" || c.dates[i] > "2024-06-01") continue;
    const oni = c.oni[i];
    if (oni !== null && oni >= 0.5) {
      if (firstIdx < 0) firstIdx = i;
    } else if (firstIdx >= 0 && oni !== null && oni < 0.5) {
      break;
    }
  }
  let peakIdx = -1;
  let peak = -Infinity;
  for (let i = 0; i < c.dates.length; i++) {
    if (c.dates[i] < "2023-01-01" || c.dates[i] > "2024-12-31") continue;
    const oni = c.oni[i];
    if (oni !== null && oni > peak) {
      peak = oni;
      peakIdx = i;
    }
  }
  return {
    firstIso: firstIdx >= 0 ? c.dates[firstIdx] : null,
    peakIso: peakIdx >= 0 ? c.dates[peakIdx] : null,
    peakOni: peakIdx >= 0 ? c.oni[peakIdx] ?? null : null,
  };
}
