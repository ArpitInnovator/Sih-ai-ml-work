import pandas as pd
import numpy as np

print("="*70)
print("FINAL VERIFICATION CHECK")
print("="*70)

# Load datasets
df_orig = pd.read_csv('master_site1_final.csv', nrows=0)
df_clean = pd.read_csv('master_site1_final_cleaned.csv', nrows=0)

print(f"\nOriginal columns: {len(df_orig.columns)}")
print(f"Cleaned columns: {len(df_clean.columns)}")
print(f"Columns removed: {len(df_orig.columns) - len(df_clean.columns)}")

print("\n" + "="*70)
print("\n1. TARGET VARIABLES CHECK:")
targets = ['NO2_target', 'O3_target', 'co', 'hcho', 'HCHO_target']
for t in targets:
    if t in df_orig.columns:
        status = "KEPT" if t in df_clean.columns else "MISSING"
        print(f"   [{status}] {t}")

print("\n" + "="*70)
print("\n2. FORECAST VARIABLES CHECK (Should all be REMOVED):")
forecast_cols = [c for c in df_orig.columns if 'forecast' in c.lower() or 
                 'pred' in c.lower() or 'lead' in c.lower() or 
                 c in ['go3', 'gtco3', 'tcco', 'tcno2', 'tchcho', 'tcso2', 'tc_no']]
forecast_still_present = [c for c in forecast_cols if c in df_clean.columns]
if forecast_still_present:
    print(f"   ERROR: {len(forecast_still_present)} forecast variables still present!")
    for c in forecast_still_present[:10]:
        print(f"   - {c}")
else:
    print(f"   ✓ All {len(forecast_cols)} forecast variables REMOVED")

print("\n" + "="*70)
print("\n3. TRIGONOMETRIC FEATURES CHECK (Should all be REMOVED):")
trig_cols = [c for c in df_orig.columns if 'sin_' in c.lower() or 
             'cos_' in c.lower() or 'cosSZA' in c or 
             'hour_sin' in c.lower() or 'hour_cos' in c.lower()]
trig_still_present = [c for c in trig_cols if c in df_clean.columns]
if trig_still_present:
    print(f"   ERROR: {len(trig_still_present)} trigonometric features still present!")
    for c in trig_still_present:
        print(f"   - {c}")
else:
    print(f"   ✓ All {len(trig_cols)} trigonometric features REMOVED")

print("\n" + "="*70)
print("\n4. ERA5 DUPLICATES CHECK (Should only have ERA5, not base):")
era5_pairs = [('t2m_era5', 't2m'), ('d2m_era5', 'd2m'), ('tcc_era5', 'tcc'),
              ('u10_era5', 'u10'), ('v10_era5', 'v10'), ('blh_era5', 'blh')]
has_duplicates = False
for era5, base in era5_pairs:
    era5_status = "KEPT" if era5 in df_clean.columns else "MISSING"
    base_status = "STILL PRESENT!" if base in df_clean.columns else "REMOVED"
    if base in df_clean.columns:
        has_duplicates = True
        print(f"   ERROR: {era5}: {era5_status} | {base}: {base_status}")
    else:
        print(f"   ✓ {era5}: {era5_status} | {base}: {base_status}")
if not has_duplicates:
    print("   ✓ No duplicate base/ERA5 versions found")

print("\n" + "="*70)
print("\n5. DAILY SATELLITE LAG CHECK (Should all be REMOVED):")
daily_lags = [c for c in df_orig.columns if 'satellite_daily_lag' in c.lower() or 
              ('daily' in c.lower() and 'lag' in c.lower())]
daily_lags_present = [c for c in daily_lags if c in df_clean.columns]
if daily_lags_present:
    print(f"   ERROR: {len(daily_lags_present)} daily lag columns still present!")
    for c in daily_lags_present:
        print(f"   - {c}")
else:
    print(f"   ✓ All {len(daily_lags)} daily lag columns REMOVED")

print("\n" + "="*70)
print("\n6. LAG VARIABLES > 6h CHECK (Should all be REMOVED):")
long_lags = [c for c in df_orig.columns if '_lag_12h' in c or '_lag_24h' in c]
long_lags_present = [c for c in long_lags if c in df_clean.columns]
if long_lags_present:
    print(f"   ERROR: {len(long_lags_present)} lag > 6h columns still present!")
    for c in long_lags_present:
        print(f"   - {c}")
else:
    print(f"   ✓ All {len(long_lags)} lag > 6h columns REMOVED")

print("\n" + "="*70)
print("\n7. KEPT FEATURES SUMMARY:")
target_count = len([c for c in df_clean.columns if 'target' in c.lower()])
meteo_count = len([c for c in df_clean.columns if any(k in c.lower() for k in 
                 ['temp', 'wind', 'pressure', 'sp', 'solar', 'era5', 't2m', 
                  'd2m', 'tcc', 'u10', 'v10', 'blh', 'sza'])])
satellite_count = len([c for c in df_clean.columns if 'satellite' in c.lower()])
pollutant_count = len([c for c in df_clean.columns if any(k in c.lower() for k in 
                       ['pm', 'so2', 'aod', 'aluv', 'no', 'hcho', 'co'])])
time_count = len([c for c in df_clean.columns if c == 'datetime' or 'time' in c.lower()])

print(f"   Targets: {target_count}")
print(f"   Meteorological: {meteo_count}")
print(f"   Satellite observations: {satellite_count}")
print(f"   Local pollutants: {pollutant_count}")
print(f"   Time features: {time_count}")
print(f"   Total: {len(df_clean.columns)}")

print("\n" + "="*70)
print("\n8. FINAL CLEANED DATASET COLUMNS:")
print("\n" + "\n".join([f"   {i+1:2d}. {col}" for i, col in enumerate(sorted(df_clean.columns))]))

print("\n" + "="*70)
print("\nVERIFICATION COMPLETE!")
print("="*70)


