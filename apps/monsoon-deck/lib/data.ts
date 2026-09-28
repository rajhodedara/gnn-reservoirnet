/**
 * Data access. app_data.json is a single static artifact loaded once per route
 * load, exactly as specified in the contract — no backend, no incremental fetching.
 *
 * Everything downstream is 100% data-driven: no result is hard-coded anywhere in
 * the UI. Retraining changes the JSON and the whole deck follows.
 */

import type { AppData } from "./types";

let cache: AppData | null = null;
let inflight: Promise<AppData> | null = null;

/** Base path so the static export works when served from a sub-path. */
const BASE = "";

export async function loadAppData(): Promise<AppData> {
  if (cache) return cache;
  if (inflight) return inflight;
  inflight = (async () => {
    const res = await fetch(`${BASE}/data/app_data.json`, { cache: "force-cache" });
    if (!res.ok) {
      throw new Error(`Failed to load app_data.json (HTTP ${res.status}). Run: python scripts/export_app_data.py`);
    }
    const json = (await res.json()) as AppData;
    cache = json;
    return json;
  })();
  try {
    return await inflight;
  } finally {
    inflight = null;
  }
}

export function getCached(): AppData | null {
  return cache;
}

export interface GeoAssets {
  states: any | null;
  outline: any | null;
}

let geoCache: GeoAssets | null = null;

export async function loadGeo(): Promise<GeoAssets> {
  if (geoCache) return geoCache;
  const [states, outline] = await Promise.all([
    fetch(`${BASE}/data/geo/india_states_simplified.geojson`, { cache: "force-cache" })
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null),
    fetch(`${BASE}/data/geo/india_outline_simplified.geojson`, { cache: "force-cache" })
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null),
  ]);
  geoCache = { states, outline };
  return geoCache;
}
