"""
Enhanced NO2 and O3 Model Training
- NO2: Extended lags (12h, 24h), PM interactions, holiday/weekend flags, wind components
- O3: Sun elevation, photochemical interactions, season-wise models, PBL × solar interactions
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import pickle
import os
from datetime import datetime

warnings.filterwarnings('ignore')

print("="*80)
print("ENHANCED NO2 AND O3 MODEL TRAINING")
print("="*80)

# ==================== LOAD DATA ====================
print("\n1. Loading data...")
df = pd.read_csv('master_site1_final_cleaned.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# ==================== FEATURE ENGINEERING ====================
print("\n2. Creating enhanced features...")

# Time features
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['day_of_year'] = df['datetime'].dt.dayofyear
df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
df['is_weekday'] = (df['datetime'].dt.dayofweek < 5).astype(int)

# Holiday flags (approximate - you can add actual holiday dates)
# Common Indian holidays: Republic Day (Jan 26), Independence Day (Aug 15), Diwali (varies), etc.
df['is_holiday'] = 0  # Placeholder - add actual holiday dates if available
df['is_weekend_or_holiday'] = ((df['is_weekend'] == 1) | (df['is_holiday'] == 1)).astype(int)

# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

# Season
def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5, 6]:
        return 'summer'
    elif month in [7, 8, 9]:
        return 'monsoon'
    else:
        return 'post_monsoon'

df['season'] = df['month'].apply(get_season)
df['is_winter'] = (df['season'] == 'winter').astype(int)
df['is_summer'] = (df['season'] == 'summer').astype(int)
df['is_monsoon'] = (df['season'] == 'monsoon').astype(int)
df['is_post_monsoon'] = (df['season'] == 'post_monsoon').astype(int)

# Derived meteorological features
if 'wind_speed' not in df.columns and 'u10_era5' in df.columns and 'v10_era5' in df.columns:
    df['wind_speed'] = np.sqrt(df['u10_era5']**2 + df['v10_era5']**2)
if 'wind_direction_rad' not in df.columns and 'u10_era5' in df.columns and 'v10_era5' in df.columns:
    df['wind_direction_rad'] = np.arctan2(df['v10_era5'], df['u10_era5'])
if 'dewpoint_depression' not in df.columns and 't2m_era5' in df.columns and 'd2m_era5' in df.columns:
    df['dewpoint_depression'] = df['t2m_era5'] - df['d2m_era5']
if 'relative_humidity_approx' not in df.columns and 'dewpoint_depression' in df.columns:
    df['relative_humidity_approx'] = 100 * np.exp(-df['dewpoint_depression'] / 5)

# ==================== NO2 SPECIFIC FEATURES ====================
print("   Creating NO2-specific features...")

# Extended lag features for NO2 (1h, 3h, 6h, 12h, 24h)
no2_lag_features = ['no2', 'pm2p5', 'pm10', 'so2', 't2m_era5', 'wind_speed']
for col in no2_lag_features:
    if col in df.columns:
        for lag in [1, 3, 6, 12, 24]:
            if f'{col}_lag_{lag}h' not in df.columns:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)

# PM interactions for NO2
if 'pm2p5' in df.columns and 'pm10' in df.columns:
    df['pm25_pm10_ratio'] = df['pm2p5'] / (df['pm10'] + 1e-10)
    df['pm25_pm10_product'] = df['pm2p5'] * df['pm10']
    df['pm25_pm10_sum'] = df['pm2p5'] + df['pm10']
if 'no2' in df.columns and 'pm2p5' in df.columns:
    df['no2_pm25_ratio'] = df['no2'] / (df['pm2p5'] + 1e-10)
    df['no2_pm25_product'] = df['no2'] * df['pm2p5']
if 'no2' in df.columns and 'pm10' in df.columns:
    df['no2_pm10_ratio'] = df['no2'] / (df['pm10'] + 1e-10)

# Wind components (u, v) for NO2
if 'u10_era5' in df.columns:
    df['wind_u'] = df['u10_era5']
    df['wind_u_abs'] = np.abs(df['u10_era5'])
    df['wind_u_squared'] = df['u10_era5']**2
if 'v10_era5' in df.columns:
    df['wind_v'] = df['v10_era5']
    df['wind_v_abs'] = np.abs(df['v10_era5'])
    df['wind_v_squared'] = df['v10_era5']**2
if 'u10_era5' in df.columns and 'v10_era5' in df.columns:
    df['wind_uv_product'] = df['u10_era5'] * df['v10_era5']
    df['wind_uv_ratio'] = df['u10_era5'] / (df['v10_era5'] + 1e-10)

# Rolling means for NO2
for window in [3, 6, 12, 24]:
    for feat in ['no2', 'pm2p5', 'pm10']:
        if feat in df.columns:
            df[f'{feat}_rolling_mean_{window}h'] = df[feat].rolling(window=window, min_periods=1).mean()

# ==================== O3 SPECIFIC FEATURES ====================
print("   Creating O3-specific photochemical features...")

# Sun elevation angle (critical for O3)
if 'solar_elevation' in df.columns:
    df['solar_elevation_abs'] = np.abs(df['solar_elevation'])
    df['solar_elevation_squared'] = df['solar_elevation']**2
    df['is_daytime'] = (df['solar_elevation'] > 0).astype(int)
    df['solar_elevation_positive'] = np.maximum(0, df['solar_elevation'])

# SZA (Solar Zenith Angle) - inverse of elevation
if 'SZA_deg' in df.columns:
    df['sza_rad'] = np.radians(df['SZA_deg'])
    df['cos_sza'] = np.cos(df['sza_rad'])
    df['photolysis_rate_approx'] = np.maximum(0, df['cos_sza'])

# Photochemical interaction features for O3
if 't2m_era5' in df.columns and 'solar_elevation' in df.columns:
    df['temp_solar_elevation'] = df['t2m_era5'] * df['solar_elevation_abs']
    df['temp_solar_elevation_squared'] = df['t2m_era5'] * df['solar_elevation_squared']
if 't2m_era5' in df.columns and 'photolysis_rate_approx' in df.columns:
    df['temp_photolysis'] = df['t2m_era5'] * df['photolysis_rate_approx']
if 't2m_era5' in df.columns and 'cos_sza' in df.columns:
    df['temp_cos_sza'] = df['t2m_era5'] * df['cos_sza']

# PBL (Planetary Boundary Layer) × Solar interactions
if 'blh_era5' in df.columns:
    if 'solar_elevation' in df.columns:
        df['pbl_solar_elevation'] = df['blh_era5'] * df['solar_elevation_abs']
        df['pbl_solar_elevation_squared'] = df['blh_era5'] * df['solar_elevation_squared']
    if 'photolysis_rate_approx' in df.columns:
        df['pbl_photolysis'] = df['blh_era5'] * df['photolysis_rate_approx']
    if 'cos_sza' in df.columns:
        df['pbl_cos_sza'] = df['blh_era5'] * df['cos_sza']
    # PBL × Temperature (already have ventilation_rate, but add more)
    if 't2m_era5' in df.columns:
        df['pbl_temp'] = df['blh_era5'] * df['t2m_era5']

# Additional O3 interactions
if 'wind_speed' in df.columns and 'blh_era5' in df.columns:
    df['ventilation_rate'] = df['blh_era5'] / (df['wind_speed'] + 1e-6)
    df['pbl_wind_product'] = df['blh_era5'] * df['wind_speed']
if 'relative_humidity_approx' in df.columns and 't2m_era5' in df.columns:
    df['rh_temp_interaction'] = df['relative_humidity_approx'] * df['t2m_era5']
if 'is_weekend' in df.columns:
    df['weekend_solar'] = df['is_weekend'] * df.get('solar_elevation_abs', 0)

# O3 lag features (shorter lags, O3 is more reactive)
for col in ['o3', 'O3_target', 'no2', 't2m_era5', 'solar_elevation']:
    if col in df.columns:
        for lag in [1, 3, 6]:
            if f'{col}_lag_{lag}h' not in df.columns:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)

# O3 rolling means
for window in [3, 6, 12]:
    for feat in ['O3_target', 'no2', 't2m_era5']:
        if feat in df.columns:
            df[f'{feat}_rolling_mean_{window}h'] = df[feat].rolling(window=window, min_periods=1).mean()

print("   ✓ Features created")

# ==================== FEATURE SELECTION ====================
def get_no2_features(df):
    """NO2 feature set with extended lags and PM interactions"""
    features = []
    
    # Core pollutants
    features.extend([f for f in ['pm2p5', 'pm10', 'so2', 'no2', 'pm1', 'no'] if f in df.columns])
    
    # Meteorology
    meteo = ['blh_era5', 't2m_era5', 'd2m_era5', 'wind_speed', 'u10_era5', 'v10_era5',
             'relative_humidity_approx', 'tcc_era5', 'sp', 'dewpoint_depression']
    features.extend([f for f in meteo if f in df.columns])
    
    # Extended lag features (1h, 3h, 6h, 12h, 24h)
    for lag in [1, 3, 6, 12, 24]:
        for feat in ['no2', 'pm2p5', 'pm10', 'so2', 't2m_era5', 'wind_speed']:
            if f'{feat}_lag_{lag}h' in df.columns:
                features.append(f'{feat}_lag_{lag}h')
    
    # Rolling means
    for window in [3, 6, 12, 24]:
        for feat in ['no2', 'pm2p5', 'pm10']:
            if f'{feat}_rolling_mean_{window}h' in df.columns:
                features.append(f'{feat}_rolling_mean_{window}h')
    
    # PM interactions
    pm_interactions = ['pm25_pm10_ratio', 'pm25_pm10_product', 'pm25_pm10_sum',
                      'no2_pm25_ratio', 'no2_pm25_product', 'no2_pm10_ratio']
    features.extend([f for f in pm_interactions if f in df.columns])
    
    # Wind components
    wind_features = ['wind_u', 'wind_v', 'wind_u_abs', 'wind_v_abs', 
                    'wind_u_squared', 'wind_v_squared', 'wind_uv_product', 'wind_uv_ratio']
    features.extend([f for f in wind_features if f in df.columns])
    
    # Time features
    time_features = ['month', 'hour', 'day_of_week', 'is_weekend', 'is_weekday',
                    'is_weekend_or_holiday', 'hour_sin', 'hour_cos', 
                    'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos']
    features.extend([f for f in time_features if f in df.columns])
    
    # Season
    season_features = ['is_winter', 'is_summer', 'is_monsoon', 'is_post_monsoon']
    features.extend([f for f in season_features if f in df.columns])
    
    # Solar (for NO2, less critical but still useful)
    if 'solar_elevation' in df.columns:
        features.append('solar_elevation')
    
    return [f for f in features if f in df.columns]

def get_o3_features(df):
    """O3 feature set with photochemical features"""
    features = []
    
    # Core pollutants
    features.extend([f for f in ['pm2p5', 'pm10', 'so2', 'no2'] if f in df.columns])
    
    # Meteorology
    meteo = ['blh_era5', 't2m_era5', 'd2m_era5', 'wind_speed', 'u10_era5', 'v10_era5',
             'relative_humidity_approx', 'tcc_era5', 'sp', 'dewpoint_depression']
    features.extend([f for f in meteo if f in df.columns])
    
    # Sun elevation and photochemical features (CRITICAL for O3)
    solar_features = ['solar_elevation', 'solar_elevation_abs', 'solar_elevation_squared',
                     'solar_elevation_positive', 'is_daytime', 'SZA_deg', 'sza_rad',
                     'cos_sza', 'photolysis_rate_approx']
    features.extend([f for f in solar_features if f in df.columns])
    
    # Photochemical interactions
    photo_interactions = ['temp_solar_elevation', 'temp_solar_elevation_squared',
                         'temp_photolysis', 'temp_cos_sza']
    features.extend([f for f in photo_interactions if f in df.columns])
    
    # PBL × Solar interactions (CRITICAL)
    pbl_solar_features = ['pbl_solar_elevation', 'pbl_solar_elevation_squared',
                         'pbl_photolysis', 'pbl_cos_sza', 'pbl_temp']
    features.extend([f for f in pbl_solar_features if f in df.columns])
    
    # Other interactions
    other_interactions = ['ventilation_rate', 'pbl_wind_product', 'rh_temp_interaction',
                         'weekend_solar']
    features.extend([f for f in other_interactions if f in df.columns])
    
    # O3 lags (shorter)
    for lag in [1, 3, 6]:
        for feat in ['O3_target', 'no2', 't2m_era5', 'solar_elevation']:
            if f'{feat}_lag_{lag}h' in df.columns:
                features.append(f'{feat}_lag_{lag}h')
    
    # O3 rolling means
    for window in [3, 6, 12]:
        for feat in ['O3_target', 'no2', 't2m_era5']:
            if f'{feat}_rolling_mean_{window}h' in df.columns:
                features.append(f'{feat}_rolling_mean_{window}h')
    
    # Time features
    time_features = ['month', 'hour', 'day_of_week', 'is_weekend', 'is_weekday',
                    'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    features.extend([f for f in time_features if f in df.columns])
    
    # Season
    season_features = ['is_winter', 'is_summer', 'is_monsoon', 'is_post_monsoon']
    features.extend([f for f in season_features if f in df.columns])
    
    return [f for f in features if f in df.columns]

# ==================== DATA PREPARATION ====================
def prepare_data(df, target_col, features, train_mask, val_mask, test_mask):
    """Prepare data"""
    valid_mask = ~df[target_col].isna()
    
    train_idx = valid_mask & train_mask
    val_idx = valid_mask & val_mask
    test_idx = valid_mask & test_mask
    
    X_train = df[train_idx][features].copy()
    y_train = df[train_idx][target_col].copy()
    X_val = df[val_idx][features].copy()
    y_val = df[val_idx][target_col].copy()
    X_test = df[test_idx][features].copy()
    y_test = df[test_idx][target_col].copy()
    
    # Convert to numeric
    for col in features:
        if col in X_train.columns:
            if X_train[col].dtype == 'object':
                X_train[col] = pd.Categorical(X_train[col]).codes
                X_val[col] = pd.Categorical(X_val[col], categories=pd.Categorical(X_train[col]).categories).codes
                X_test[col] = pd.Categorical(X_test[col], categories=pd.Categorical(X_train[col]).categories).codes
            elif X_train[col].dtype == 'bool':
                X_train[col] = X_train[col].astype(int)
                X_val[col] = X_val[col].astype(int)
                X_test[col] = X_test[col].astype(int)
    
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    
    features = [f for f in features if f in X_train.columns]
    
    # Fill NaN
    for col in X_train.columns:
        if X_train[col].isnull().sum() > 0:
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)
            X_val[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, features

# ==================== TRAIN MODEL ====================
def train_model(df, target_col, target_name, train_mask, val_mask, test_mask, features_func):
    """Train model"""
    print(f"\n{'='*80}")
    print(f"TRAINING: {target_name}")
    print(f"{'='*80}")
    
    features = features_func(df)
    print(f"   Features: {len(features)}")
    
    X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_data(
        df, target_col, features, train_mask, val_mask, test_mask
    )
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Conservative hyperparameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'max_depth': 5,
        'learning_rate': 0.03,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_data_in_leaf': 50,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'verbose': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        num_boost_round=200,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # Predictions
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Metrics
    train_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'MAE': mean_absolute_error(y_train, y_train_pred),
        'R2': r2_score(y_train, y_train_pred)
    }
    val_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'MAE': mean_absolute_error(y_val, y_val_pred),
        'R2': r2_score(y_val, y_val_pred)
    }
    test_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred),
        'R2': r2_score(y_test, y_test_pred)
    }
    
    baseline_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean())))
    train_test_gap = train_metrics['R2'] - test_metrics['R2']
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print(f"\n   Results:")
    print(f"   Train RMSE: {train_metrics['RMSE']:.4f}, R²: {train_metrics['R2']:.4f}")
    print(f"   Val RMSE:   {val_metrics['RMSE']:.4f}, R²: {val_metrics['R2']:.4f}")
    print(f"   Test RMSE:  {test_metrics['RMSE']:.4f}, R²: {test_metrics['R2']:.4f}")
    print(f"   Train-Test R² gap: {train_test_gap:.4f}")
    print(f"   Baseline RMSE: {baseline_rmse:.4f}")
    print(f"   Improvement: {((baseline_rmse - test_metrics['RMSE']) / baseline_rmse * 100):.2f}%")
    
    return model, train_metrics, val_metrics, test_metrics, feature_importance, baseline_rmse

# ==================== TRAIN SEASON-WISE O3 MODELS ====================
def train_seasonwise_o3(df, train_mask, val_mask, test_mask):
    """Train separate O3 models for each season"""
    print(f"\n{'='*80}")
    print("TRAINING SEASON-WISE O3 MODELS")
    print(f"{'='*80}")
    
    seasons = {
        'winter': [12, 1, 2],
        'summer': [3, 4, 5, 6],
        'monsoon': [7, 8, 9],
        'post_monsoon': [10, 11]
    }
    
    season_models = {}
    season_results = {}
    
    for season_name, months in seasons.items():
        print(f"\n   Training {season_name} O3 model...")
        
        # Filter data for this season
        season_train_mask = train_mask & df['month'].isin(months)
        season_val_mask = val_mask & df['month'].isin(months)
        season_test_mask = test_mask & df['month'].isin(months)
        
        if season_train_mask.sum() < 100:
            print(f"      Skipping {season_name} - insufficient data")
            continue
        
        features = get_o3_features(df)
        X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_data(
            df, 'O3_target', features, season_train_mask, season_val_mask, season_test_mask
        )
        
        if len(X_train) < 50:
            print(f"      Skipping {season_name} - insufficient training data")
            continue
        
        print(f"      Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Handle missing validation or test data
        if len(X_val) == 0:
            print(f"      Warning: No validation data for {season_name}")
            print(f"      Using last 20% of training data as validation...")
            # Use last 20% of training as validation
            val_size = int(0.2 * len(X_train))
            X_val = X_train.iloc[-val_size:].copy()
            y_val = y_train.iloc[-val_size:].copy()
            X_train = X_train.iloc[:-val_size].copy()
            y_train = y_train.iloc[:-val_size].copy()
            print(f"      Adjusted - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        if len(X_test) == 0:
            print(f"      Warning: No test data for {season_name} in test period (2022 H2)")
            print(f"      Using validation set as test for metrics...")
            # Use validation set as test for metrics
            X_test = X_val.copy()
            y_test = y_val.copy()
        
        # Train model
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 15,
            'max_depth': 5,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'min_data_in_leaf': 30,  # Smaller for seasonal models
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'verbose': -1,
            'random_state': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Only add validation if it exists
        if len(X_val) > 0:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=200,
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
        else:
            # Train without validation set
            model = lgb.train(
                params,
                train_data,
                num_boost_round=150,  # Fixed number of rounds
                callbacks=[lgb.log_evaluation(period=50)]
            )
        
        # Predictions
        if len(X_test) > 0:
            y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
            test_metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'MAE': mean_absolute_error(y_test, y_test_pred),
                'R2': r2_score(y_test, y_test_pred)
            }
            
            baseline_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean())))
            
            season_models[season_name] = model
            season_results[season_name] = {
                'test_rmse': test_metrics['RMSE'],
                'test_mae': test_metrics['MAE'],
                'test_r2': test_metrics['R2'],
                'baseline_rmse': baseline_rmse,
                'note': 'Used val set as test' if len(X_test) == len(X_val) else 'Normal test'
            }
            
            print(f"      Test RMSE: {test_metrics['RMSE']:.4f}, R²: {test_metrics['R2']:.4f}")
        else:
            # No test data, just save model
            season_models[season_name] = model
            season_results[season_name] = {
                'test_rmse': np.nan,
                'test_mae': np.nan,
                'test_r2': np.nan,
                'baseline_rmse': np.nan,
                'note': 'No test data in test period'
            }
            print(f"      Model trained but no test data available")
    
    return season_models, season_results

# ==================== MAIN ====================
print("\n3. Training models...")

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Use more recent training period
train_mask = (df['datetime'] >= '2020-01-01') & (df['datetime'] <= '2021-12-31')
val_mask = (df['datetime'] >= '2022-01-01') & (df['datetime'] <= '2022-03-31')
test_mask = (df['datetime'] >= '2022-07-01') & (df['datetime'] <= '2022-12-31')

results_summary = []

# Train NO2 model
print("\n" + "="*80)
print("TRAINING NO2 MODEL")
print("="*80)
model_no2, train_metrics_no2, val_metrics_no2, test_metrics_no2, importance_no2, baseline_no2 = train_model(
    df, 'NO2_target', 'NO2_target', train_mask, val_mask, test_mask, get_no2_features
)

model_no2.save_model('models/enhanced_no2_model.txt')
with open('models/enhanced_no2_model.pkl', 'wb') as f:
    pickle.dump(model_no2, f)
importance_no2.to_csv('results/enhanced_no2_feature_importance.csv', index=False)

results_summary.append({
    'Model': 'NO2_target',
    'Train_RMSE': train_metrics_no2['RMSE'],
    'Train_R2': train_metrics_no2['R2'],
    'Val_RMSE': val_metrics_no2['RMSE'],
    'Val_R2': val_metrics_no2['R2'],
    'Test_RMSE': test_metrics_no2['RMSE'],
    'Test_MAE': test_metrics_no2['MAE'],
    'Test_R2': test_metrics_no2['R2'],
    'Baseline_RMSE': baseline_no2,
    'Improvement_%': ((baseline_no2 - test_metrics_no2['RMSE']) / baseline_no2 * 100)
})

# Train O3 model (all-season)
print("\n" + "="*80)
print("TRAINING O3 MODEL (ALL-SEASON)")
print("="*80)
model_o3, train_metrics_o3, val_metrics_o3, test_metrics_o3, importance_o3, baseline_o3 = train_model(
    df, 'O3_target', 'O3_target', train_mask, val_mask, test_mask, get_o3_features
)

model_o3.save_model('models/enhanced_o3_model.txt')
with open('models/enhanced_o3_model.pkl', 'wb') as f:
    pickle.dump(model_o3, f)
importance_o3.to_csv('results/enhanced_o3_feature_importance.csv', index=False)

results_summary.append({
    'Model': 'O3_target (all-season)',
    'Train_RMSE': train_metrics_o3['RMSE'],
    'Train_R2': train_metrics_o3['R2'],
    'Val_RMSE': val_metrics_o3['RMSE'],
    'Val_R2': val_metrics_o3['R2'],
    'Test_RMSE': test_metrics_o3['RMSE'],
    'Test_MAE': test_metrics_o3['MAE'],
    'Test_R2': test_metrics_o3['R2'],
    'Baseline_RMSE': baseline_o3,
    'Improvement_%': ((baseline_o3 - test_metrics_o3['RMSE']) / baseline_o3 * 100)
})

# Train season-wise O3 models
season_models, season_results = train_seasonwise_o3(df, train_mask, val_mask, test_mask)

# Save season-wise models
for season_name, model in season_models.items():
    model.save_model(f'models/enhanced_o3_{season_name}_model.txt')
    with open(f'models/enhanced_o3_{season_name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Save results
results_df = pd.DataFrame(results_summary)
results_df.to_csv('results/enhanced_no2_o3_performance_summary.csv', index=False)

# Save season-wise results
season_df = pd.DataFrame(season_results).T
season_df.to_csv('results/enhanced_o3_seasonwise_performance.csv')

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))
print("\n\nSeason-wise O3 Models:")
print(season_df.to_string())
print("\n" + "="*80)

