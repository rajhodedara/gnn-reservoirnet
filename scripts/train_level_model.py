import os
import yaml
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

MCM_TO_TMC = 1 / 28.3168466

def nse(y_true, y_pred):
    return r2_score(y_true, y_pred)

def main():
    with open("configs/reservoirs.yaml", "r") as f:
        res_config = yaml.safe_load(f)["reservoirs"]

    enso = pd.read_csv("data/raw/enso/combined_climate_indices.csv")
    enso = enso.dropna(subset=['Year', 'Month'])
    enso['Date'] = pd.to_datetime(enso[['Year', 'Month']].assign(DAY=1))
    enso = enso.set_index('Date')[['oni', 'iod']]
    
    full_dates = pd.date_range("2010-01-01", "2024-12-31", freq='D')
    enso_daily = enso.reindex(full_dates, method='ffill')
    enso_daily.index.name = 'Date'

    era5 = pd.read_csv("data/raw/era5/reservoir_era5_daily.csv")
    era5['Date'] = pd.to_datetime(era5['Date'])
    era5 = era5.set_index('Date')

    out_dir = "outputs/level_results"
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []
    
    for res in res_config:
        slug = res['id']
        gross_mcm = res['gross_capacity_mcm']
        dead_mcm = res['dead_storage_mcm']
        
        gross_tmc = gross_mcm * MCM_TO_TMC
        dead_tmc = dead_mcm * MCM_TO_TMC
        live_tmc = gross_tmc - dead_tmc

        df = pd.read_csv(f"data/raw/wris_v2/{slug}.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()

        S = df['Storage (TMC/MCM)']
        inflow = df['Inflow (cusecs/cumecs)']
        
        targets = []
        for w in range(1, 13):
            # FIXED: Target is the weekly change, not cumulative
            dS_w = S.shift(-7*w) - S.shift(-7*(w-1))
            targets.append(dS_w.rename(f'target_w{w}'))
            
        target_df = pd.concat(targets, axis=1)

        feats = []
        
        month_dummies = pd.get_dummies(df.index.month, prefix='month', drop_first=True)
        month_dummies.index = df.index
        feats.append(month_dummies)
        
        S_live = np.clip(S - dead_tmc, 0, live_tmc)
        S_pct = np.clip(S_live / live_tmc, 0, 1)
        feats.append(S_pct.rename('S_pct'))
        feats.append((S_pct**2).rename('S_pct_sq'))
        
        inflow_1w = inflow.rolling(7, min_periods=1).sum()
        inflow_2w = inflow.rolling(14, min_periods=1).sum()
        inflow_4w = inflow.rolling(28, min_periods=1).sum()
        feats.append(inflow_1w.rename('inflow_1w'))
        feats.append(inflow_2w.rename('inflow_2w'))
        feats.append(inflow_4w.rename('inflow_4w'))
        
        era5_slug = era5[[f'{slug}_runoff', f'{slug}_evap', f'{slug}_soil_moisture']].rolling(7, min_periods=1).mean()
        feats.append(era5_slug)
        
        feats.append(enso_daily[['oni', 'iod']])
        
        dS_trail_1w = S - S.shift(7)
        dS_trail_4w = S - S.shift(28)
        feats.append(dS_trail_1w.rename('dS_trail_1w'))
        feats.append(dS_trail_4w.rename('dS_trail_4w'))
        
        X = pd.concat(feats, axis=1)
        y = target_df
        
        data = pd.concat([X, y, S.rename('S_current')], axis=1)
        
        train_mask = (data.index.year >= 2010) & (data.index.year <= 2022)
        val_mask = (data.index.year == 2023)
        test_mask = (data.index.year == 2024)
        
        data_train = data[train_mask].dropna()
        X_train = data_train[X.columns]
        y_train = data_train[y.columns]
        
        S_clim_map = data_train.groupby(data_train.index.dayofyear)['S_current'].mean()
        
        print(f"Training {slug}...")
        model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42))
        model.fit(X_train, y_train)
        
        importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
        top_10_feats = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(10)
        print(f"--- {slug} Top 10 Features ---")
        print(top_10_feats)
        
        for split_name, mask in [('val', val_mask), ('test', test_mask)]:
            data_split = data[mask]
            
            X_split = data_split[X.columns]
            
            X_split_filled = X_split.fillna(0)
            y_pred_dS = model.predict(X_split_filled)
            y_pred_dS = pd.DataFrame(y_pred_dS, index=X_split.index, columns=y.columns)
            
            S_current = data_split['S_current']
            
            for w in [1, 4, 12]:
                target_col = f'target_w{w}'
                
                # FIXED: Actual level is current + cumulative weekly change up to w
                # We can just look at S(t+7w) from the raw data or sum the target columns
                S_actual = S_current + data_split[[f'target_w{k}' for k in range(1, w+1)]].sum(axis=1)
                
                # FIXED: Predicted level is current + sum of predicted weekly changes
                S_pred = np.clip(S_current + y_pred_dS[[f'target_w{k}' for k in range(1, w+1)]].sum(axis=1), 0, gross_tmc)
                S_pers = np.clip(S_current, 0, gross_tmc)
                
                target_dates = data_split.index + pd.Timedelta(days=7*w)
                S_clim = target_dates.dayofyear.map(S_clim_map)
                S_clim = np.clip(S_clim, 0, gross_tmc)
                S_clim = S_clim.fillna(data_train['S_current'].mean())
                
                # Check for NaNs in the actual cumulative change
                # A row is valid if all targets up to w are non-NaN
                valid_mask = ~data_split[[f'target_w{k}' for k in range(1, w+1)]].isna().any(axis=1)
                
                if valid_mask.sum() == 0:
                    continue
                    
                y_t = S_actual[valid_mask]
                y_p = S_pred[valid_mask]
                y_pers = S_pers[valid_mask]
                y_c = S_clim.values[valid_mask]
                
                ds_true = data_split[target_col][valid_mask]
                ds_pred = y_pred_dS[target_col][valid_mask]
                
                val_nse = nse(y_t, y_p)
                val_pers_nse = nse(y_t, y_pers)
                val_clim_nse = nse(y_t, y_c)
                val_ds_nse = nse(ds_true, ds_pred)
                
                summary_rows.append({
                    'dam': slug,
                    'split': split_name,
                    'week': w,
                    'level_nse': val_nse,
                    'pers_nse': val_pers_nse,
                    'clim_nse': val_clim_nse,
                    'ds_nse': val_ds_nse
                })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{out_dir}/summary.csv", index=False)
    
    for dam in res_config:
        slug = dam['id']
        dam_df = summary_df[summary_df['dam'] == slug]
        dam_df.to_csv(f"{out_dir}/{slug}.csv", index=False)

    print("\n=== SUMMARY (TEST SPLIT) ===")
    test_summary = summary_df[summary_df['split'] == 'test']
    for w in [1, 4, 12]:
        w_df = test_summary[test_summary['week'] == w]
        print(f"\nWeek {w} Ahead:")
        print(w_df[['dam', 'level_nse', 'pers_nse', 'clim_nse', 'ds_nse']].to_string(index=False))
        
        beats_pers = (w_df['level_nse'] > w_df['pers_nse']).sum()
        print(f"Beats persistence: {beats_pers}/{len(w_df)}")

if __name__ == "__main__":
    main()

