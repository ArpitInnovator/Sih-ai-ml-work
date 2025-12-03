# Guide: Converting Cleaned Dataset to Engineered Dataset

This guide explains how to convert `master_site1_final_cleaned.csv` to `master_site1_final_engineered.csv` by applying feature engineering transformations.

## 📋 Overview

The feature engineering process transforms the cleaned dataset (44 columns) into a machine learning-ready dataset (100+ columns) by adding:
- Time-based features
- Weather interaction features
- Wind features
- Air quality ratios
- Lag features
- Rolling statistics
- And more...

## 🔧 Prerequisites

### Required Python Packages
pip install pandas numpy matplotlib seabornOr install from requirements.txt:
pip install -r requirements.txt
### Required Files
- `master_site1_final_cleaned.csv` - The cleaned input dataset

## 🚀 Quick Start

### Option 1: Use the Existing Script
python feature_engineering_analysis.pyThis will automatically:
1. Load `master_site1_final_cleaned.csv`
2. Apply all feature engineering transformations
3. Generate visualizations and reports
4. Save `master_site1_final_engineered.csv`

### Option 2: Manual Step-by-Step Process

Follow the detailed steps below to understand and customize the process.

## 📝 Step-by-Step Feature Engineering Process

### Step 1: Load and Prepare Data

import pandas as pd
import numpy as np

# Load the cleaned dataset
df = pd.read_csv('master_site1_final_cleaned.csv')

# Convert datetime column
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)### Step 2: Time-Based Features

Extract temporal components and create cyclical encodings:

# Basic time components
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
df['day_of_year'] = df['datetime'].dt.dayofyear
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Cyclical encoding (captures periodic patterns)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)**Why?** Cyclical encoding helps models understand that hour 23 is close to hour 0, and month 12 is close to month 1.

### Step 3: Weather Interaction Features

Create derived weather features:
n
# Dewpoint depression (temperature - dewpoint)
df['dewpoint_depression'] = df['t2m_era5'] - df['d2m_era5']

# Approximate relative humidity using Magnus formula
df['relative_humidity_approx'] = 100 * (
    np.exp((17.27 * df['d2m_era5']) / (237.7 + df['d2m_era5'])) / 
    np.exp((17.27 * df['t2m_era5']) / (237.7 + df['t2m_era5']))
)**Why?** These features capture atmospheric moisture conditions that affect air quality.

### Step 4: Wind Features

Transform wind components into meaningful features:

# Wind magnitude (speed)
df['wind_magnitude'] = np.sqrt(df['u10_era5']**2 + df['v10_era5']**2)

# Wind direction in radians
df['wind_direction_rad'] = np.arctan2(df['v10_era5'], df['u10_era5'])

# Squared components (captures non-linear relationships)
df['wind_u_squared'] = df['u10_era5']**2
df['wind_v_squared'] = df['v10_era5']**2**Why?** Wind magnitude and direction are more interpretable than raw u/v components.

### Step 5: Air Quality Ratios

Create ratios between pollutants:

# NO2 to PM2.5 ratio (with small epsilon to avoid division by zero)
df['no2_pm25_ratio'] = df['no2'] / (df['pm2p5'] + 1e-10)

# PM2.5 to PM10 ratio
df['pm25_pm10_ratio'] = df['pm2p5'] / (df['pm10'] + 1e-10)**Why?** Ratios capture relative pollutant concentrations, which can be more informative than absolute values.

### Step 6: Satellite Features

Create satellite data interactions:

# Satellite ratio (NO2 to HCHO)
df['satellite_ratio'] = df['NO2_satellite'] / (df['HCHO_satellite'] + 1e-10)**Why?** Ratios between satellite measurements can indicate pollution sources.

### Step 7: AOD (Aerosol Optical Depth) Features
ython
# AOD ratio
df['aod_ratio'] = df['bcaod550'] / (df['aod550'] + 1e-10)**Why?** Black carbon AOD to total AOD ratio indicates pollution type.

### Step 8: Boundary Layer Features
on
# Interaction between boundary layer height and wind speed
df['blh_wind_interaction'] = df['blh_era5'] * df['wind_speed']**Why?** Boundary layer height affects pollutant dispersion, especially with wind.

### Step 9: Solar Features

# Solar elevation squared (captures non-linear solar effects)
df['solar_elevation_squared'] = df['solar_elevation']**2

# Daytime indicator
df['is_daytime'] = (df['solar_elevation'] > 0).astype(int)**Why?** Solar radiation affects photochemical reactions and pollutant formation.

### Step 10: Pressure Features
on
# Normalized pressure (z-score normalization)
df['pressure_normalized'] = (df['sp'] - df['sp'].mean()) / df['sp'].std()**Why?** Normalization helps with model training and captures pressure anomalies.

### Step 11: Lag Features

Create time-lagged features (previous hour values):
ython
# Key features for lagging
key_features = ['no2', 'pm2p5', 'pm10', 'so2', 'co', 't2m_era5', 'wind_speed']

# Create 1-hour and 3-hour lags
for col in key_features:
    if col in df.columns:
        df[f'{col}_lag_1h'] = df[col].shift(1)  # Previous hour
        df[f'{col}_lag_3h'] = df[col].shift(3)   # 3 hours ago**Why?** Lag features capture temporal dependencies and persistence in air quality.

**Note:** The first few rows will have NaN values for lag features (no previous data).

### Step 12: Rolling Statistics

Create rolling window statistics:

# Key features for rolling statistics
key_features = ['no2', 'pm2p5', 'NO2_target', 'O3_target']

# Create 6-hour rolling mean and standard deviation
for col in key_features:
    if col in df.columns:
        df[f'{col}_rolling_mean_6h'] = df[col].rolling(window=6, min_periods=1).mean()
        df[f'{col}_rolling_std_6h'] = df[col].rolling(window=6, min_periods=1).std()**Why?** Rolling statistics capture short-term trends and variability.

**Note:** `min_periods=1` means it uses available data even if less than 6 hours.

### Step 13: Save the Engineered Dataset
n
# Save to CSV
df.to_csv('master_site1_final_engineered.csv', index=False)
print(f"Saved engineered dataset: {df.shape[0]} rows, {df.shape[1]} columns")## 📊 Complete Feature List

### Original Features (44)
All columns from `master_site1_final_cleaned.csv` are preserved.

### Engineered Features (~60+)

#### Time Features (13)
- `year`, `month`, `day`, `hour`, `day_of_week`, `day_of_year`, `is_weekend`
- `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `day_of_year_sin`, `day_of_year_cos`

#### Weather Features (2)
- `dewpoint_depression`, `relative_humidity_approx`

#### Wind Features (4)
- `wind_magnitude`, `wind_direction_rad`, `wind_u_squared`, `wind_v_squared`

#### Air Quality Ratios (2)
- `no2_pm25_ratio`, `pm25_pm10_ratio`

#### Satellite Features (1)
- `satellite_ratio`

#### AOD Features (1)
- `aod_ratio`

#### Interaction Features (3)
- `blh_wind_interaction`, `solar_elevation_squared`, `is_daytime`

#### Pressure Features (1)
- `pressure_normalized`

#### Lag Features (14)
- `no2_lag_1h`, `no2_lag_3h`
- `pm2p5_lag_1h`, `pm2p5_lag_3h`
- `pm10_lag_1h`, `pm10_lag_3h`
- `so2_lag_1h`, `so2_lag_3h`
- `co_lag_1h`, `co_lag_3h`
- `t2m_era5_lag_1h`, `t2m_era5_lag_3h`
- `wind_speed_lag_1h`, `wind_speed_lag_3h`

#### Rolling Statistics (8)
- `no2_rolling_mean_6h`, `no2_rolling_std_6h`
- `pm2p5_rolling_mean_6h`, `pm2p5_rolling_std_6h`
- `NO2_target_rolling_mean_6h`, `NO2_target_rolling_std_6h`
- `O3_target_rolling_mean_6h`, `O3_target_rolling_std_6h`

## 🔍 Verification

After feature engineering, verify the results:

# Check shape
print(f"Original shape: (27410, 44)")
print(f"Engineered shape: {df.shape}")

# Check for missing values
print(f"Missing values: {df.isnull().sum().sum()}")

# Check feature types
print(f"Numeric columns: {df.select_dtypes(include=[np.number]).shape[1]}")
print(f"Total columns: {df.shape[1]}")## ⚠️ Important Notes

### 1. Missing Values
- Lag features will have NaN in the first few rows (no previous data)
- Rolling statistics use `min_periods=1`, so they have fewer NaNs
- Consider forward-filling or dropping rows with critical missing values for ML models

### 2. Data Order
- The dataset must be sorted by `datetime` before creating lag and rolling features
- Ensure chronological order for time-based features to work correctly

### 3. Division by Zero
- All ratio calculations use `+ 1e-10` to prevent division by zero
- This is a small epsilon that doesn't significantly affect results

### 4. Target Variables
- Target variables (`NO2_target`, `O3_target`, `co`, `hcho`) are NOT used in lag features
- This prevents data leakage in machine learning models

## 🐛 Troubleshooting

### Issue: "KeyError: 'datetime'"
**Solution:** Ensure the cleaned dataset has a `datetime` column.

### Issue: Lag features are all NaN
**Solution:** Check that data is sorted by datetime and has no gaps.

### Issue: Memory errors
**Solution:** Process in chunks or use `dtype` optimization:thon
# Optimize data types
for col in df.select_dtypes(include=['int64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='integer')
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='float')### Issue: Division by zero warnings
**Solution:** The code already handles this with `+ 1e-10`, but ensure source columns don't have all-zero values.

## 📈 Expected Output

After running the feature engineering:

- **Input:** `master_site1_final_cleaned.csv` (44 columns, 27,410 rows)
- **Output:** `master_site1_final_engineered.csv` (~100+ columns, 27,410 rows)

## 🎯 Next Steps

After creating the engineered dataset:

1. **Exploratory Data Analysis:** Use the generated visualizations
2. **Feature Selection:** Review correlation matrices and feature importance
3. **Model Training:** Use the engineered features for machine learning
4. **Validation:** Ensure no data leakage and proper train/test splits

## 📚 Additional Resources

- `FEATURE_ENGINEERING_README.md` - Overview of the feature engineering script
- `feature_engineering_report.txt` - Detailed statistics and analysis
- `feature_engineering_analysis.py` - Complete implementation

## 💡 Customization Tips

### Add Custom Features
# Example: Add temperature difference
df['temp_diff_24h'] = df['t2m_era5'] - df['t2m_era5'].shift(24)

# Example: Add custom ratio
df['custom_ratio'] = df['no2'] / (df['o3'] + 1e-10)### Modify Rolling Window
# Change from 6-hour to 12-hour window
df['no2_rolling_mean_12h'] = df['no2'].rolling(window=12, min_periods=1).mean()### Add More Lag Periods
# Add 6-hour and 24-hour lags
df['no2_lag_6h'] = df['no2'].shift(6)
df['no2_lag_24h'] = df['no2'].shift(24)---

**Last Updated:** Based on `feature_engineering_analysis.py` implementation
**Version:** 1.0
