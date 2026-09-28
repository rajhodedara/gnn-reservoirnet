/**
 * Number formatting.
 *
 * Inflow is a weekly SUM in m3/s·day and runs into the 10^5 range, so the whole
 * product formats with K/M suffixes. All numerals render in a monospaced,
 * tabular-figure stack so columns of digits line up.
 *
 * A null/missing value renders as an em dash (—), never as 0. Several reservoirs
 * are genuinely dry for 46-67% of days; that is "no flow", which is different from
 * "unknown", and the two must never be conflated.
 */

export const DASH = "—";

/** Compact K/M magnitude for large inflows. */
export function fmtMagnitude(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return DASH;
  const a = Math.abs(v);
  if (a >= 1e6) return `${(v / 1e6).toFixed(digits)}M`;
  if (a >= 1e4) return `${(v / 1e3).toFixed(digits)}K`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(2)}K`;
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(digits);
  if (a === 0) return "0";
  return v.toFixed(2);
}

/** Full grouped integer, for tooltips and detail readouts. */
export function fmtFull(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return DASH;
  return v.toLocaleString("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/** Signed metric with fixed decimals (NSE, KGE, correlation). */
export function fmtMetric(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return DASH;
  return v.toFixed(digits);
}

/** Signed metric, always showing the sign — used for skill deltas. */
export function fmtSigned(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return DASH;
  const s = v.toFixed(digits);
  return v > 0 ? `+${s}` : s;
}

export function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return DASH;
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtStorageTmc(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return DASH;
  return v.toFixed(digits);
}

export function fmtDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}

export function fmtMonth(iso: string): string {
  const d = new Date(`${iso}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-GB", {
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}

export function addDays(iso: string, days: number): string {
  const d = new Date(`${iso}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

/** 1 TMC = 28.3168 MCM */
export const TMC_IN_MCM = 28.3168;

export function capacityTmc(grossMcm: number): number {
  return grossMcm / TMC_IN_MCM;
}

export function fillFraction(storageTmc: number | null, grossMcm: number): number | null {
  if (storageTmc === null || !Number.isFinite(storageTmc)) return null;
  const cap = capacityTmc(grossMcm);
  if (!cap) return null;
  return Math.max(0, Math.min(1.5, storageTmc / cap));
}

export function deadFraction(deadMcm: number, grossMcm: number): number {
  const cap = capacityTmc(grossMcm);
  if (!cap) return 0;
  return Math.max(0, Math.min(1, deadMcm / TMC_IN_MCM / cap));
}

/** True during the south-west monsoon window (Jun-Sep), the 60-90% inflow season. */
export function isMonsoon(iso: string): boolean {
  const m = new Date(`${iso}T00:00:00Z`).getUTCMonth() + 1;
  return m >= 6 && m <= 9;
}
