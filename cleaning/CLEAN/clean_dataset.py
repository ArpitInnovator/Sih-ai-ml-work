import pandas as pd
import numpy as np
from datetime import datetime

# Read the dataset
print("Reading dataset...")
df = pd.read_csv('master_site1_final.csv')

print(f"Original dataset shape: {df.shape}")
print(f"Total columns: {len(df.columns)}")

# Lists to track what we keep and remove
columns_to_keep = []
columns_removed = []
removal_reasons = {}

# ==================== TARGETS - KEEP ====================
targets_to_keep = ['NO2_target', 'O3_target', 'co']
# Check for HCHO target
if 'HCHO_target' in df.columns:
    targets_to_keep.append('HCHO_target')
elif 'hcho' in df.columns:
    targets_to_keep.append('hcho')
else:
    # Check if there's a surface HCHO column
    hcho_cols = [col for col in df.columns if 'hcho' in col.lower() and 'target' in col.lower()]
    if hcho_cols:
        targets_to_keep.extend(hcho_cols)

for col in targets_to_keep:
    if col in df.columns:
        columns_to_keep.append(col)
        removal_reasons[col] = "KEPT - Target variable"

# ==================== REMOVE FUTURE LEAKAGE ====================
forecast_patterns = ['_forecast', '_pred', '_lead', 'forecast', 'pred', 'lead']
for col in df.columns:
    if col in columns_to_keep:
        continue
    if any(pattern in col.lower() for pattern in forecast_patterns):
        columns_removed.append(col)
        removal_reasons[col] = "REMOVED - Future leakage (forecast/prediction variable)"
        
# Remove specific forecast columns mentioned
forecast_cols_to_remove = ['O3_forecast', 'NO2_forecast', 'T_forecast', 'q_forecast', 
                           'u_forecast', 'v_forecast', 'w_forecast', 'go3', 'gtco3', 
                           'tcco', 'O3_forecast_lag_1h', 'O3_forecast_lag_3h', 
                           'O3_forecast_lag_6h', 'O3_forecast_lag_12h', 'O3_forecast_lag_24h',
                           'NO2_forecast_lag_1h', 'NO2_forecast_lag_3h', 
                           'NO2_forecast_lag_6h', 'NO2_forecast_lag_12h', 'NO2_forecast_lag_24h']

for col in forecast_cols_to_remove:
    if col in df.columns and col not in columns_removed:
        columns_removed.append(col)
        removal_reasons[col] = "REMOVED - Future leakage (forecast variable)"

# ==================== REMOVE DERIVED/TRANSFORMED DUPLICATES ====================
trig_cols = ['sin_hour', 'cos_hour', 'cosSZA', 'hour_sin', 'hour_cos']
for col in trig_cols:
    if col in df.columns and col not in columns_removed:
        columns_removed.append(col)
        removal_reasons[col] = "REMOVED - Derived/transformed duplicate (trig transformation)"

# ==================== REMOVE EXTREMELY SPARSE COLUMNS ====================
print("\nChecking for sparse columns (>80% missing)...")
sparse_threshold = 0.80  # 80% missing
for col in df.columns:
    if col in columns_to_keep or col in columns_removed:
        continue
    missing_pct = df[col].isna().sum() / len(df)
    if missing_pct > sparse_threshold:
        columns_removed.append(col)
        removal_reasons[col] = f"REMOVED - Extremely sparse ({missing_pct*100:.1f}% missing)"

# ==================== REMOVE DAILY SATELLITE LAG FIELDS ====================
daily_lag_patterns = ['_daily_lag_', 'satellite_daily_lag', 'daily_lag']
for col in df.columns:
    if col in columns_to_keep or col in columns_removed:
        continue
    if any(pattern in col.lower() for pattern in daily_lag_patterns):
        columns_removed.append(col)
        removal_reasons[col] = "REMOVED - Daily satellite lag field (temporally misaligned)"

# Remove satellite daily lag columns
satellite_daily_lag_cols = [col for col in df.columns 
                           if 'NO2_satellite_daily_lag' in col or 
                              'NO2_satellite_flag_lag' in col or
                              ('satellite_daily' in col and 'lag' in col)]
for col in satellite_daily_lag_cols:
    if col not in columns_removed:
        columns_removed.append(col)
        removal_reasons[col] = "REMOVED - Daily satellite lag field"

# ==================== REMOVE IDENTIFIERS / MEANINGLESS COLUMNS ====================
identifier_cols = ['file_year', 'file_month', 'year_cams', 'month_cams', 'day_cams', 
                   'hour_cams', 'trainable']
for col in identifier_cols:
    if col in df.columns and col not in columns_removed:
        columns_removed.append(col)
        removal_reasons[col] = "REMOVED - Identifier/metadata column"

# Check for constant columns
for col in df.columns:
    if col in columns_to_keep or col in columns_removed:
        continue
    if df[col].nunique() <= 1:
        columns_removed.append(col)
        removal_reasons[col] = "REMOVED - Constant column (no variance)"

# ==================== KEEP METEOROLOGICAL FEATURES ====================
# First, identify ERA5 columns to prefer them over base versions
era5_cols = ['t2m_era5', 'd2m_era5', 'tcc_era5', 'u10_era5', 'v10_era5', 'blh_era5']
era5_base_mapping = {col.replace('_era5', ''): col for col in era5_cols if col in df.columns}

meteo_keywords = ['temp', 'temperature', 't2m', 'd2m', 'rh', 'humidity', 
                  'wind_speed', 'wind_dir', 'ws', 'wd', 'pressure', 'pres', 'sp',
                  'pbl', 'blh', 'solar', 'sw', 'lw', 'precip', 'rain', 'tcc',
                  'u10', 'v10', 'solar_elevation', 'SZA_deg']
for col in df.columns:
    if col in columns_to_keep or col in columns_removed:
        continue
    col_lower = col.lower()
    
    # Skip if this is an ERA5 column (we'll handle it separately)
    if col in era5_cols:
        continue
    
    # Skip base version if ERA5 version exists
    if col in era5_base_mapping:
        continue
    
    if any(keyword in col_lower for keyword in meteo_keywords):
        columns_to_keep.append(col)
        removal_reasons[col] = "KEPT - Meteorological feature"

# Now add ERA5 versions (prefer ERA5 over original)
for era5_col in era5_cols:
    if era5_col in df.columns:
        base_col = era5_col.replace('_era5', '')
        # Remove base version if it was added
        if base_col in columns_to_keep:
            columns_to_keep.remove(base_col)
            removal_reasons[base_col] = "REMOVED - Replaced by ERA5 version"
        # Add ERA5 version
        if era5_col not in columns_to_keep:
            columns_to_keep.append(era5_col)
            removal_reasons[era5_col] = "KEPT - Meteorological feature (ERA5)"

# ==================== KEEP SATELLITE CURRENT OBSERVATIONS ====================
satellite_current = ['NO2_satellite', 'HCHO_satellite', 'ratio_satellite',
                     'NO2_satellite_daily', 'HCHO_satellite_daily', 
                     'NO2_satellite_flag', 'HCHO_satellite_flag',
                     'ratio_satellite_daily', 'ratio_satellite_flag']
for col in satellite_current:
    if col in df.columns and col not in columns_removed:
        columns_to_keep.append(col)
        removal_reasons[col] = "KEPT - Satellite current observation"

# Keep satellite lag columns with <= 6h lag only
for col in df.columns:
    if col in columns_to_keep or col in columns_removed:
        continue
    if 'satellite_lag' in col.lower() and '_lag_1h' in col.lower():
        columns_to_keep.append(col)
        removal_reasons[col] = "KEPT - Satellite observation (1h lag - acceptable)"
    elif 'satellite_lag' in col.lower() and '_lag_3h' in col.lower():
        columns_to_keep.append(col)
        removal_reasons[col] = "KEPT - Satellite observation (3h lag - acceptable)"

# ==================== REMOVE GLOBAL FORECAST FIELDS FIRST ====================
global_forecast = ['go3', 'gtco3', 'tcco', 'tcno2', 'tchcho', 'tc_no', 'tcso2']
for col in global_forecast:
    if col in df.columns and col not in columns_removed:
        columns_removed.append(col)
        removal_reasons[col] = "REMOVED - Global forecast field"

# Also remove their lag versions
for col in df.columns:
    if col in columns_to_keep or col in columns_removed:
        continue
    if any(gf in col for gf in ['go3_lag', 'gtco3_lag', 'tcco_lag', 'tcno2_lag', 
                                 'tchcho_lag', 'tc_no_lag', 'tcso2_lag']):
        columns_removed.append(col)
        removal_reasons[col] = "REMOVED - Global forecast field lag"

# ==================== KEEP LOCAL POLLUTION FEATURES ====================
pollution_keywords = ['pm2.5', 'pm2p5', 'pm10', 'pm1', 'so2', 'nh3', 'voc', 
                     'aod', 'bcaod', 'aluvd', 'aluvp']
# Exclude global forecast patterns
exclude_patterns = ['tcso2', 'tcno2', 'tchcho', 'tcco', 'gtco3', 'go3', 'tc_']
for col in df.columns:
    if col in columns_to_keep or col in columns_removed:
        continue
    col_lower = col.lower()
    # Check if it matches pollution keywords but is NOT a global forecast
    if any(keyword in col_lower for keyword in pollution_keywords):
        if not any(exclude in col_lower for exclude in exclude_patterns):
            columns_to_keep.append(col)
            removal_reasons[col] = "KEPT - Local pollution feature"

# ==================== REMOVE ALL LAG VARIABLES > 6h ====================
lag_cols_to_remove = []
for col in df.columns:
    if col in columns_to_keep or col in columns_removed:
        continue
    # Remove lag variables with > 6h
    if '_lag_12h' in col or '_lag_24h' in col:
        lag_cols_to_remove.append(col)
        removal_reasons[col] = "REMOVED - Lag > 6h (temporal leakage risk)"
    # Remove complex nested lag patterns
    if '_lag' in col and ('_lag_1h' in col or '_lag_3h' in col or '_lag_6h' in col):
        if col.count('_lag') > 1:  # Nested lags like NO2_target_lag1_lag_1h
            lag_cols_to_remove.append(col)
            removal_reasons[col] = "REMOVED - Nested lag pattern"

# Remove target lag variables (except we might want to keep some as features)
# But user said remove all future leakage, so let's be conservative
target_lag_patterns = ['_target_lag', '_lag1', '_lag24']
for col in df.columns:
    if col in columns_to_keep or col in columns_removed or col in lag_cols_to_remove:
        continue
    if any(pattern in col for pattern in target_lag_patterns) and col not in targets_to_keep:
        lag_cols_to_remove.append(col)
        removal_reasons[col] = "REMOVED - Target lag variable (potential leakage)"

columns_removed.extend(lag_cols_to_remove)

# ==================== KEEP TIME FEATURES (will regenerate later, but keep datetime) ====================
time_features_to_keep = ['datetime']
for col in time_features_to_keep:
    if col in df.columns:
        columns_to_keep.append(col)
        removal_reasons[col] = "KEPT - Time feature (for temporal feature generation)"

# Remove old time features that we'll regenerate
old_time_features = ['sin_hour', 'cos_hour', 'hour_sin', 'hour_cos', 'dayofweek', 
                     'day_of_week', 'is_weekend', 'day_of_year', 'year', 'month', 
                     'day', 'hour']
for col in old_time_features:
    if col in df.columns and col not in columns_removed:
        if col not in columns_to_keep:
            columns_removed.append(col)
            removal_reasons[col] = "REMOVED - Old time feature (will regenerate)"

# ==================== KEEP OTHER VALID FEATURES ====================
# Keep co, no2, no, hcho (if not already kept as targets)
other_pollutants = ['no2', 'no']
for col in other_pollutants:
    if col in df.columns and col not in columns_removed and col not in columns_to_keep:
        columns_to_keep.append(col)
        removal_reasons[col] = "KEPT - Local pollutant measurement"

# ==================== FINALIZE COLUMN LISTS ====================
# Make sure we have unique lists
columns_to_keep = list(set(columns_to_keep))
columns_removed = list(set(columns_removed))

# Find any columns we haven't classified yet
all_classified = set(columns_to_keep) | set(columns_removed)
unclassified = [col for col in df.columns if col not in all_classified]

# For unclassified columns, check if they should be kept (might be valid features)
# or removed (likely noise)
print(f"\nUnclassified columns: {len(unclassified)}")
for col in unclassified:
    # Check if it's a valid measurement/observation
    if 'lag' not in col.lower() and 'forecast' not in col.lower():
        # Check missing data
        missing_pct = df[col].isna().sum() / len(df)
        if missing_pct < 0.50:  # Less than 50% missing
            columns_to_keep.append(col)
            removal_reasons[col] = "KEPT - Valid feature with <50% missing"
        else:
            columns_removed.append(col)
            removal_reasons[col] = f"REMOVED - High missing data ({missing_pct*100:.1f}%)"
    else:
        columns_removed.append(col)
        removal_reasons[col] = "REMOVED - Unclassified lag/forecast variable"

# ==================== FINAL CLEANUP: REMOVE BASE VERSIONS WHEN ERA5 EXISTS ====================
# Final check: if both ERA5 and base version are kept, remove base version
era5_base_pairs = [
    ('t2m_era5', 't2m'), ('d2m_era5', 'd2m'), ('tcc_era5', 'tcc'),
    ('u10_era5', 'u10'), ('v10_era5', 'v10'), ('blh_era5', 'blh')
]
for era5_col, base_col in era5_base_pairs:
    if era5_col in columns_to_keep and base_col in columns_to_keep:
        columns_to_keep.remove(base_col)
        if base_col not in columns_removed:
            removal_reasons[base_col] = "REMOVED - Replaced by ERA5 version"

# ==================== CREATE CLEANED DATASET ====================
print(f"\nColumns to KEEP: {len(columns_to_keep)}")
print(f"Columns to REMOVE: {len(columns_removed)}")

# Create cleaned dataframe
df_cleaned = df[columns_to_keep].copy()

# Save cleaned dataset
output_file = 'master_site1_final_cleaned.csv'
df_cleaned.to_csv(output_file, index=False)
print(f"\nCleaned dataset saved to: {output_file}")
print(f"Cleaned dataset shape: {df_cleaned.shape}")

# ==================== CREATE DOCUMENTATION ====================
doc_lines = []
doc_lines.append("# Dataset Cleaning Documentation")
doc_lines.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
doc_lines.append(f"\n**Original Dataset:** master_site1_final.csv")
doc_lines.append(f"**Cleaned Dataset:** master_site1_final_cleaned.csv")
doc_lines.append(f"\n**Original Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
doc_lines.append(f"**Cleaned Shape:** {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
doc_lines.append(f"\n**Columns Removed:** {len(columns_removed)}")
doc_lines.append(f"**Columns Kept:** {len(columns_to_keep)}")

doc_lines.append("\n---\n")
doc_lines.append("\n## ✅ FEATURES KEPT")
doc_lines.append(f"\n### Total: {len(columns_to_keep)} features\n")

# Group kept features by category
kept_categories = {
    'Targets': [],
    'Meteorological': [],
    'Satellite Observations': [],
    'Local Pollutants': [],
    'Time Features': [],
    'Other': []
}

for col in sorted(columns_to_keep):
    reason = removal_reasons.get(col, "KEPT")
    if 'target' in reason.lower() or col in targets_to_keep:
        kept_categories['Targets'].append((col, reason))
    elif 'meteorological' in reason.lower() or any(mk in col.lower() for mk in meteo_keywords):
        kept_categories['Meteorological'].append((col, reason))
    elif 'satellite' in reason.lower() or 'satellite' in col.lower():
        kept_categories['Satellite Observations'].append((col, reason))
    elif 'pollution' in reason.lower() or any(pk in col.lower() for pk in pollution_keywords):
        kept_categories['Local Pollutants'].append((col, reason))
    elif 'time' in reason.lower() or col == 'datetime':
        kept_categories['Time Features'].append((col, reason))
    else:
        kept_categories['Other'].append((col, reason))

for category, items in kept_categories.items():
    if items:
        doc_lines.append(f"\n### {category} ({len(items)} features)")
        for col, reason in sorted(items):
            doc_lines.append(f"- `{col}` - {reason}")

doc_lines.append("\n---\n")
doc_lines.append("\n## ❌ FEATURES REMOVED")
doc_lines.append(f"\n### Total: {len(columns_removed)} features\n")

# Group removed features by reason
removed_categories = {}
for col in columns_removed:
    reason = removal_reasons.get(col, "REMOVED")
    category = reason.split(' - ')[1] if ' - ' in reason else "Other"
    if category not in removed_categories:
        removed_categories[category] = []
    removed_categories[category].append(col)

for category in sorted(removed_categories.keys()):
    items = removed_categories[category]
    doc_lines.append(f"\n### {category} ({len(items)} features)")
    for col in sorted(items)[:20]:  # Limit to first 20 per category
        doc_lines.append(f"- `{col}`")
    if len(items) > 20:
        doc_lines.append(f"- ... and {len(items) - 20} more")

doc_lines.append("\n---\n")
doc_lines.append("\n## 📊 Summary Statistics")

doc_lines.append("\n### Missing Data in Kept Features")
for col in sorted(columns_to_keep):
    missing_pct = df_cleaned[col].isna().sum() / len(df_cleaned) * 100
    if missing_pct > 0:
        doc_lines.append(f"- `{col}`: {missing_pct:.1f}% missing")

doc_lines.append("\n---\n")
doc_lines.append("\n## 🔍 Cleaning Rules Applied")
doc_lines.append("\n1. **Targets Kept:** NO2_target, O3_target, CO, HCHO (surface measurement)")
doc_lines.append("\n2. **Removed Future Leakage:** All forecast, prediction, and lead variables")
doc_lines.append("\n3. **Removed Derived Features:** Trigonometric transformations (sin_hour, cos_hour, etc.)")
doc_lines.append("\n4. **Removed Sparse Features:** Columns with >80% missing data")
doc_lines.append("\n5. **Removed Daily Lag Fields:** Satellite daily products with lag times")
doc_lines.append("\n6. **Removed Identifiers:** File metadata and constant columns")
doc_lines.append("\n7. **Kept Meteorological Features:** Temperature, humidity, wind, pressure, etc.")
doc_lines.append("\n8. **Kept Satellite Observations:** Current observations and short lags (≤6h)")
doc_lines.append("\n9. **Kept Local Pollutants:** PM2.5, PM10, SO2, AOD, etc.")
doc_lines.append("\n10. **Removed Global Forecasts:** go3, gtco3, tcco, and similar fields")
doc_lines.append("\n11. **Time Features:** Kept datetime only (will regenerate cyclical features)")

# Save documentation
doc_file = 'CLEANING_DOCUMENTATION.md'
with open(doc_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(doc_lines))

print(f"Documentation saved to: {doc_file}")

print("\nCleaning complete!")
print(f"\nFinal kept columns ({len(columns_to_keep)}):")
for col in sorted(columns_to_keep):
    print(f"  - {col}")
