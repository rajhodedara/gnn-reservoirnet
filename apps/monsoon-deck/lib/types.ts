/** Shared domain types for the Monsoon Command Deck, mirroring app_data.json. */

export type ReservoirId = string;

export interface Reservoir {
  id: ReservoirId;
  name: string;
  basin: string;
  river: string;
  state: string;
  lat: number;
  lon: number;
  gross_capacity_mcm: number;
  dead_storage_mcm: number;
  catchment_area_km2: number;
  tribunal: string;
  elevation_m: number | null;
  upstream: ReservoirId[];
}

export interface GraphEdge {
  source: ReservoirId;
  target: ReservoirId;
  kind: "physical" | "climatological";
  weight?: number | null;
}

export interface ForecastPoint {
  p10: number[] | null;
  p50: number[] | null;
  p90: number[] | null;
  observed: (number | null)[];
}

export interface BandPoint {
  p10: (number | null)[];
  p90: (number | null)[];
}

export interface BandSkew {
  n_residuals: number;
  median: number | null;
  q10: number | null;
  q90: number | null;
  weeks_with_positive_q10: number[];
  negative_median_cells: number;
  negative_median_weeks: number[];
}

export interface MetricsRow {
  CRPS: number | null;
  RMSE: number | null;
  MAE: number | null;
  NSE: number | null;
  Event_NSE: number | null;
  Log_NSE: number | null;
  KGE: number | null;
  Reservoir?: string;
  Basin?: string;
  Week?: number;
  Condition?: string;
  reservoir_id?: string | null;
  /** Present on 5-seed mean rows. */
  NSE_mean?: number | null;
  NSE_std?: number | null;
  KGE_mean?: number | null;
  RMSE_mean?: number | null;
  CRPS_mean?: number | null;
  Event_NSE_mean?: number | null;
  Log_NSE_mean?: number | null;
  MAE_mean?: number | null;
  n_seeds?: number;
}

export interface BaselineRow {
  NSE: number | null;
  RMSE: number | null;
  MAE: number | null;
  n?: number | null;
}

export interface LevelRow {
  dam: ReservoirId;
  split: string;
  week: number;
  level_nse: number | null;
  pers_nse: number | null;
  clim_nse: number | null;
  ds_nse?: number | null;
  n_samples?: number | null;
  gross_tmc?: number | null;
}

export interface RollingOriginRow {
  fold: number;
  pooled_mean_NSE: number | null;
  [weekKey: string]: number | null;
}

export interface CascadeRow {
  upstream: ReservoirId;
  downstream: ReservoirId;
  lag_days: number;
  peak_corr: number | null;
  n_pairs: number;
  cascade_note: string;
}

export interface Climate {
  dates: string[];
  oni: (number | null)[];
  soi: (number | null)[];
  nino34: (number | null)[];
  iod: (number | null)[];
}

export interface DailySeries {
  inflow: (number | null)[];
  storage: (number | null)[];
}

export interface Era5Series {
  rainfall: (number | null)[];
  runoff: (number | null)[];
  evap: (number | null)[];
  soil_moisture: (number | null)[];
}

export interface DeckMeta {
  generated_at: string;
  test_year: number;
  val_year: number;
  train_end: string;
  units_inflow: string;
  units_storage: string;
  band_source: string;
  band_note: string;
  units_capacity_note: string;
}

export interface IntegrityCheck {
  name: string;
  status: "pass" | "fail";
  detail: string;
}

export interface MetricsBlock {
  per_reservoir: MetricsRow[];
  per_week: MetricsRow[];
  per_basin: MetricsRow[];
  enso: MetricsRow[];
  per_seed: MetricsRow[];
  headline: {
    source: string;
    n_seeds: number;
    mean_nse_pooled: number | null;
    mean_nse_week1: number | null;
  };
  fold_per_reservoir: MetricsRow[];
  fold_per_week: MetricsRow[];
  fold_per_basin: MetricsRow[];
  fold_enso: MetricsRow[];
  note: string;
}

export interface AppData {
  meta: DeckMeta;
  reservoirs: Reservoir[];
  edges: GraphEdge[];
  forecast: Record<string, Record<ReservoirId, ForecastPoint>>;
  forecast_residual_band: Record<string, Record<ReservoirId, BandPoint>>;
  band_skew: Record<ReservoirId, BandSkew>;
  metrics: MetricsBlock;
  integrity: { checked: IntegrityCheck[]; note: string };
  baselines: {
    persistence: Record<ReservoirId, BaselineRow>;
    climatology: Record<ReservoirId, BaselineRow>;
    pooled: Record<string, BaselineRow>;
  };
  levels: LevelRow[];
  physics_levels: LevelRow[];
  rolling_origin: RollingOriginRow[];
  daily_dates: string[];
  daily: Record<ReservoirId, DailySeries>;
  climate: Climate;
  era5: Record<ReservoirId, Era5Series>;
  cascade: CascadeRow[];
}

export type EnsoLens = "all" | "El Nino" | "Neutral";
