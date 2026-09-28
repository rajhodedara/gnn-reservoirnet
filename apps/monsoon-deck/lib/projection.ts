/**
 * Map projection: plain equirectangular (Plate Carrée), which is the right choice
 * for a regional-scale operational map — it keeps lat/lon graticule lines honest
 * and, at Peninsular-India latitudes, the shape distortion is small.
 *
 * The map is intentionally locked to a Peninsular-India window rather than fitting
 * the full national bounding box, so the 10 reservoirs and their river topology
 * occupy the frame instead of being a cluster of dots in the middle of a lot of
 * Himalaya. The wider national outline is still available for context.
 */

import type { ReservoirId } from "./types";

export interface Bounds {
  lonMin: number;
  lonMax: number;
  latMin: number;
  latMax: number;
}

/** Peninsular window: covers all 10 nodes plus their catchments with margin. */
export const PENINSULA_BOUNDS: Bounds = {
  lonMin: 69.5,
  lonMax: 85.5,
  latMin: 7.0,
  latMax: 24.5,
};

/** Full national extent, used only if a caller opts into context mode. */
export const INDIA_BOUNDS: Bounds = {
  lonMin: 67.0,
  lonMax: 98.5,
  latMin: 6.0,
  latMax: 37.5,
};

export interface Projected {
  x: number;
  y: number;
}

export interface Projector {
  project(lon: number, lat: number): Projected;
  width: number;
  height: number;
  bounds: Bounds;
}

/**
 * Build a projector into a fixed viewBox. Aspect is preserved by fitting the
 * bounds box into the target box and letterboxing the remainder.
 */
export function makeProjector(
  bounds: Bounds,
  width: number,
  height: number,
  padding = 0,
): Projector {
  const b = {
    lonMin: bounds.lonMin,
    lonMax: bounds.lonMax,
    latMin: bounds.latMin,
    latMax: bounds.latMax,
  };
  const innerW = Math.max(1, width - padding * 2);
  const innerH = Math.max(1, height - padding * 2);

  const spanLon = b.lonMax - b.lonMin;
  const spanLat = b.latMax - b.latMin;
  const scale = Math.min(innerW / spanLon, innerH / spanLat);

  const drawnW = spanLon * scale;
  const drawnH = spanLat * scale;
  const offX = padding + (innerW - drawnW) / 2;
  const offY = padding + (innerH - drawnH) / 2;

  return {
    width,
    height,
    bounds: b,
    project(lon: number, lat: number) {
      return {
        x: offX + (lon - b.lonMin) * scale,
        y: offY + (b.latMax - lat) * scale,
      };
    },
  };
}

/** A graticule at a sensible density for the window; used for the cartographic rules. */
export function graticule(bounds: Bounds, step = 2): { lon: number[]; lat: number[] } {
  const lons: number[] = [];
  const lats: number[] = [];
  for (let v = Math.ceil(bounds.lonMin / step) * step; v <= bounds.lonMax; v += step) lons.push(v);
  for (let v = Math.ceil(bounds.latMin / step) * step; v <= bounds.latMax; v += step) lats.push(v);
  return { lon: lons, lat: lats };
}

/**
 * Convert a GeoJSON MultiPolygon/Polygon into SVG path `d` strings.
 * Coordinates outside the window are kept (cheap) so coastline strokes run off
 * frame naturally instead of being visibly clipped mid-run.
 */
export function geoToPaths(geometry: any, proj: Projector): string[] {
  if (!geometry) return [];
  const type = geometry.type;
  const polys: number[][][][] =
    type === "Polygon" ? [geometry.coordinates] : type === "MultiPolygon" ? geometry.coordinates : [];
  const out: string[] = [];
  for (const poly of polys) {
    for (const ring of poly) {
      if (!ring || ring.length < 3) continue;
      let d = "";
      for (let i = 0; i < ring.length; i++) {
        const [lon, lat] = ring[i];
        const p = proj.project(lon, lat);
        d += `${i === 0 ? "M" : "L"}${p.x.toFixed(2)},${p.y.toFixed(2)}`;
      }
      d += "Z";
      out.push(d);
    }
  }
  return out;
}

/** Handles the 4 directed hydraulic edges; used for the animated flow overlay. */
export interface EdgeGeom {
  key: string;
  from: ReservoirId;
  to: ReservoirId;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  path: string;
}

/**
 * A slight quadratic bow makes parallel/collinear reaches (Almatti→Srisailam and
 * Tungabhadra→Srisailam converge) readable instead of overlapping.
 */
export function edgeGeom(
  fromLon: number,
  fromLat: number,
  toLon: number,
  toLat: number,
  from: ReservoirId,
  to: ReservoirId,
  proj: Projector,
  bow = 0.08,
): EdgeGeom {
  const a = proj.project(fromLon, fromLat);
  const b = proj.project(toLon, toLat);
  const mx = (a.x + b.x) / 2;
  const my = (a.y + b.y) / 2;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  // Perpendicular offset for the control point.
  const cx = mx - dy * bow;
  const cy = my + dx * bow;
  return {
    key: `${from}->${to}`,
    from,
    to,
    x1: a.x,
    y1: a.y,
    x2: b.x,
    y2: b.y,
    path: `M${a.x.toFixed(2)},${a.y.toFixed(2)}Q${cx.toFixed(2)},${cy.toFixed(2)} ${b.x.toFixed(2)},${b.y.toFixed(2)}`,
  };
}

/** Resize a bucket of values into a small sparkline path. */
export function sparkPath(
  values: (number | null)[],
  w: number,
  h: number,
  pad = 1,
): { line: string; area: string; min: number; max: number } | null {
  const finite = values.filter((v): v is number => v !== null && Number.isFinite(v));
  if (finite.length < 2) return null;
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const span = max - min || 1;
  const innerW = Math.max(1, w - pad * 2);
  const innerH = Math.max(1, h - pad * 2);
  let line = "";
  let started = false;
  const pts: Array<[number, number]> = [];
  values.forEach((v, i) => {
    if (v === null || !Number.isFinite(v)) return;
    const x = pad + (i / (values.length - 1)) * innerW;
    const y = pad + innerH - ((v - min) / span) * innerH;
    pts.push([x, y]);
    line += `${started ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
    started = true;
  });
  if (!pts.length) return null;
  const area = `${line}L${pts[pts.length - 1][0].toFixed(2)},${(pad + innerH).toFixed(2)}L${pts[0][0].toFixed(2)},${(pad + innerH).toFixed(2)}Z`;
  return { line, area, min, max };
}

/** Insert a lon/lat series into the shared daily axis by calendar date. */
export function dailySlice(
  dates: string[],
  values: (number | null)[],
  fromIso: string,
  toIso: string,
): (number | null)[] {
  const out: (number | null)[] = [];
  for (let i = 0; i < dates.length; i++) {
    const d = dates[i];
    if (d >= fromIso && d <= toIso) out.push(values[i] ?? null);
  }
  return out;
}
