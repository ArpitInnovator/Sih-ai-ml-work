"""
Seasonal NO2 Specialist Training with O3 Model
- NO2: Seasonal specialists (winter, summer, monsoon, post-monsoon) with blending
- O3: Keep existing model (unchanged)
- CO: Dropped
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
print("SEASONAL NO2 SPECIALIST TRAINING + O3 MODEL")
print("="*80)

# ==================== LOAD DATA ====================
print("\n1. Loading data...")
df = pd.read_csv('master_site1_final_cleaned.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# ==================== FEATURE ENGINEERING ====================
print("\n2. Creating features...")

# Time features
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['day_of_year'] = df['datetime'].dt.dayofyear
df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
df['is_weekday'] = (df['datetime'].dt.dayofweek < 5).astype(int)
df['is_weekend_or_holiday'] = df['is_weekend'].copy()  # Add actual holidays if available

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

# NO2 specific features
print("   Creating NO2-specific features...")

# Extended lag features (1h, 3h, 6h, 12h, 24h)
no2_lag_features = ['no2', 'pm2p5', 'pm10', 'so2', 't2m_era5', 'wind_speed']
for col in no2_lag_features:
    if col in df.columns:
        for lag in [1, 3, 6, 12, 24]:
            if f'{col}_lag_{lag}h' not in df.columns:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)

# PM interactions
if 'pm2p5' in df.columns and 'pm10' in df.columns:
    df['pm25_pm10_ratio'] = df['pm2p5'] / (df['pm10'] + 1e-10)
    df['pm25_pm10_product'] = df['pm2p5'] * df['pm10']
    df['pm25_pm10_sum'] = df['pm2p5'] + df['pm10']
if 'no2' in df.columns and 'pm2p5' in df.columns:
    df['no2_pm25_ratio'] = df['no2'] / (df['pm2p5'] + 1e-10)
    df['no2_pm25_product'] = df['no2'] * df['pm2p5']
if 'no2' in df.columns and 'pm10' in df.columns:
    df['no2_pm10_ratio'] = df['no2'] / (df['pm10'] + 1e-10)

# Wind components
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

# Rolling means
for window in [3, 6, 12, 24]:
    for feat in ['no2', 'pm2p5', 'pm10']:
        if feat in df.columns:
            df[f'{feat}_rolling_mean_{window}h'] = df[feat].rolling(window=window, min_periods=1).mean()

# Interactions
if 'blh_era5' in df.columns and 'wind_speed' in df.columns:
    df['ventilation_rate'] = df['blh_era5'] / (df['wind_speed'] + 1e-6)
    df['blh_wind_interaction'] = df['blh_era5'] * df['wind_speed']
if 'hour' in df.columns and 't2m_era5' in df.columns:
    df['hour_temp_interaction'] = df['hour'] * df['t2m_era5']

# O3 features (keep same as before)
print("   Creating O3-specific photochemical features...")
if 'solar_elevation' in df.columns:
    df['solar_elevation_abs'] = np.abs(df['solar_elevation'])
    df['solar_elevation_squared'] = df['solar_elevation']**2
    df['is_daytime'] = (df['solar_elevation'] > 0).astype(int)
    df['solar_elevation_positive'] = np.maximum(0, df['solar_elevation'])
if 'SZA_deg' in df.columns:
    df['sza_rad'] = np.radians(df['SZA_deg'])
    df['cos_sza'] = np.cos(df['sza_rad'])
    df['photolysis_rate_approx'] = np.maximum(0, df['cos_sza'])
if 't2m_era5' in df.columns and 'solar_elevation' in df.columns:
    df['temp_solar_interaction'] = df['t2m_era5'] * np.abs(df['solar_elevation'])
    df['temp_solar_elevation'] = df['t2m_era5'] * df['solar_elevation_abs']
    df['temp_solar_elevation_squared'] = df['t2m_era5'] * df['solar_elevation_squared']
if 't2m_era5' in df.columns and 'photolysis_rate_approx' in df.columns:
    df['temp_photolysis'] = df['t2m_era5'] * df['photolysis_rate_approx']
if 't2m_era5' in df.columns and 'cos_sza' in df.columns:
    df['temp_cos_sza'] = df['t2m_era5'] * df['cos_sza']
if 'blh_era5' in df.columns:
    if 'solar_elevation' in df.columns:
        df['pbl_solar_elevation'] = df['blh_era5'] * df['solar_elevation_abs']
        df['pbl_solar_elevation_squared'] = df['blh_era5'] * df['solar_elevation_squared']
    if 'photolysis_rate_approx' in df.columns:
        df['pbl_photolysis'] = df['blh_era5'] * df['photolysis_rate_approx']
    if 'cos_sza' in df.columns:
        df['pbl_cos_sza'] = df['blh_era5'] * df['cos_sza']
    if 't2m_era5' in df.columns:
        df['pbl_temp'] = df['blh_era5'] * df['t2m_era5']
if 'wind_speed' in df.columns and 'blh_era5' in df.columns:
    df['pbl_wind_product'] = df['blh_era5'] * df['wind_speed']
if 'relative_humidity_approx' in df.columns and 't2m_era5' in df.columns:
    df['rh_temp_interaction'] = df['relative_humidity_approx'] * df['t2m_era5']
if 'is_weekend' in df.columns and 'solar_elevation_abs' in df.columns:
    df['weekend_solar'] = df['is_weekend'] * df['solar_elevation_abs']
for col in ['O3_target', 'no2', 't2m_era5', 'solar_elevation']:
    if col in df.columns:
        for lag in [1, 3, 6]:
            if f'{col}_lag_{lag}h' not in df.columns:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
for window in [3, 6, 12]:
    for feat in ['O3_target', 'no2', 't2m_era5']:
        if feat in df.columns:
            df[f'{feat}_rolling_mean_{window}h'] = df[feat].rolling(window=window, min_periods=1).mean()

print("   ✓ Features created")

# ==================== FEATURE SELECTION ====================
def get_no2_features(df):
    """NO2 feature set"""
    features = []
    features.extend([f for f in ['pm2p5', 'pm10', 'so2', 'no2', 'pm1', 'no'] if f in df.columns])
    meteo = ['blh_era5', 't2m_era5', 'd2m_era5', 'wind_speed', 'u10_era5', 'v10_era5',
             'relative_humidity_approx', 'tcc_era5', 'sp', 'dewpoint_depression']
    features.extend([f for f in meteo if f in df.columns])
    for lag in [1, 3, 6, 12, 24]:
        for feat in ['no2', 'pm2p5', 'pm10', 'so2', 't2m_era5', 'wind_speed']:
            if f'{feat}_lag_{lag}h' in df.columns:
                features.append(f'{feat}_lag_{lag}h')
    for window in [3, 6, 12, 24]:
        for feat in ['no2', 'pm2p5', 'pm10']:
            if f'{feat}_rolling_mean_{window}h' in df.columns:
                features.append(f'{feat}_rolling_mean_{window}h')
    pm_interactions = ['pm25_pm10_ratio', 'pm25_pm10_product', 'pm25_pm10_sum',
                      'no2_pm25_ratio', 'no2_pm25_product', 'no2_pm10_ratio']
    features.extend([f for f in pm_interactions if f in df.columns])
    wind_features = ['wind_u', 'wind_v', 'wind_u_abs', 'wind_v_abs', 
                    'wind_u_squared', 'wind_v_squared', 'wind_uv_product', 'wind_uv_ratio']
    features.extend([f for f in wind_features if f in df.columns])
    time_features = ['month', 'hour', 'day_of_week', 'is_weekend', 'is_weekday',
                    'is_weekend_or_holiday', 'hour_sin', 'hour_cos', 
                    'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos']
    features.extend([f for f in time_features if f in df.columns])
    season_features = ['is_winter', 'is_summer', 'is_monsoon', 'is_post_monsoon']
    features.extend([f for f in season_features if f in df.columns])
    if 'solar_elevation' in df.columns:
        features.append('solar_elevation')
    if 'ventilation_rate' in df.columns:
        features.append('ventilation_rate')
    if 'blh_wind_interaction' in df.columns:
        features.append('blh_wind_interaction')
    if 'hour_temp_interaction' in df.columns:
        features.append('hour_temp_interaction')
    return [f for f in features if f in df.columns]

def get_o3_features(df):
    """O3 feature set (same as before)"""
    features = []
    features.extend([f for f in ['pm2p5', 'pm10', 'so2', 'no2'] if f in df.columns])
    meteo = ['blh_era5', 't2m_era5', 'd2m_era5', 'wind_speed', 'u10_era5', 'v10_era5',
             'relative_humidity_approx', 'tcc_era5', 'sp', 'dewpoint_depression']
    features.extend([f for f in meteo if f in df.columns])
    solar_features = ['solar_elevation', 'solar_elevation_abs', 'solar_elevation_squared',
                     'solar_elevation_positive', 'is_daytime', 'SZA_deg', 'sza_rad',
                     'cos_sza', 'photolysis_rate_approx']
    features.extend([f for f in solar_features if f in df.columns])
    photo_interactions = ['temp_solar_elevation', 'temp_solar_elevation_squared',
                         'temp_photolysis', 'temp_cos_sza']
    features.extend([f for f in photo_interactions if f in df.columns])
    pbl_solar_features = ['pbl_solar_elevation', 'pbl_solar_elevation_squared',
                         'pbl_photolysis', 'pbl_cos_sza', 'pbl_temp']
    features.extend([f for f in pbl_solar_features if f in df.columns])
    other_interactions = ['ventilation_rate', 'pbl_wind_product', 'rh_temp_interaction',
                         'weekend_solar']
    features.extend([f for f in other_interactions if f in df.columns])
    for lag in [1, 3, 6]:
        for feat in ['O3_target', 'no2', 't2m_era5', 'solar_elevation']:
            if f'{feat}_lag_{lag}h' in df.columns:
                features.append(f'{feat}_lag_{lag}h')
    for window in [3, 6, 12]:
        for feat in ['O3_target', 'no2', 't2m_era5']:
            if f'{feat}_rolling_mean_{window}h' in df.columns:
                features.append(f'{feat}_rolling_mean_{window}h')
    time_features = ['month', 'hour', 'day_of_week', 'is_weekend', 'is_weekday',
                    'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    features.extend([f for f in time_features if f in df.columns])
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
    
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
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
    
    print(f"\n   Results:")
    print(f"   Train RMSE: {train_metrics['RMSE']:.4f}, R²: {train_metrics['R2']:.4f}")
    print(f"   Val RMSE:   {val_metrics['RMSE']:.4f}, R²: {val_metrics['R2']:.4f}")
    print(f"   Test RMSE:  {test_metrics['RMSE']:.4f}, R²: {test_metrics['R2']:.4f}")
    print(f"   Baseline RMSE: {baseline_rmse:.4f}")
    print(f"   Improvement: {((baseline_rmse - test_metrics['RMSE']) / baseline_rmse * 100):.2f}%")
    
    return model, train_metrics, val_metrics, test_metrics, baseline_rmse

# ==================== TRAIN SEASONAL NO2 SPECIALISTS ====================
def train_seasonal_no2_specialists(df, train_mask, val_mask, test_mask):
    """Train seasonal NO2 specialists with blending"""
    print(f"\n{'='*80}")
    print("TRAINING SEASONAL NO2 SPECIALISTS")
    print(f"{'='*80}")
    
    seasons = {
        'winter': [12, 1, 2],
        'summer': [3, 4, 5, 6],
        'monsoon': [7, 8, 9],
        'post_monsoon': [10, 11]
    }
    
    seasonal_models = {}
    seasonal_results = {}
    features = get_no2_features(df)
    
    # First train global model as fallback
    print("\n   Training global NO2 model (fallback)...")
    global_model, _, _, global_test_metrics, global_baseline = train_model(
        df, 'NO2_target', 'NO2_target (global)', train_mask, val_mask, test_mask, get_no2_features
    )
    seasonal_models['global'] = global_model
    
    # Train seasonal specialists
    for season_name, months in seasons.items():
        print(f"\n   Training {season_name} NO2 specialist...")
        
        # Filter data for this season
        season_train_mask = train_mask & df['month'].isin(months)
        season_val_mask = val_mask & df['month'].isin(months)
        season_test_mask = test_mask & df['month'].isin(months)
        
        if season_train_mask.sum() < 100:
            print(f"      Skipping {season_name} - insufficient training data ({season_train_mask.sum()} samples)")
            continue
        
        X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_data(
            df, 'NO2_target', features, season_train_mask, season_val_mask, season_test_mask
        )
        
        if len(X_train) < 50:
            print(f"      Skipping {season_name} - insufficient training data after filtering")
            continue
        
        # Handle missing validation
        if len(X_val) == 0:
            print(f"      Using last 20% of training as validation...")
            val_size = int(0.2 * len(X_train))
            X_val = X_train.iloc[-val_size:].copy()
            y_val = y_train.iloc[-val_size:].copy()
            X_train = X_train.iloc[:-val_size].copy()
            y_train = y_train.iloc[:-val_size].copy()
        
        print(f"      Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train seasonal model
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
            model = lgb.train(
                params,
                train_data,
                num_boost_round=150
            )
        
        # Evaluate on test set for this season
        if len(X_test) > 0:
            y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
            test_metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'MAE': mean_absolute_error(y_test, y_test_pred),
                'R2': r2_score(y_test, y_test_pred)
            }
            baseline_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean())))
            
            seasonal_models[season_name] = model
            seasonal_results[season_name] = {
                'test_rmse': test_metrics['RMSE'],
                'test_mae': test_metrics['MAE'],
                'test_r2': test_metrics['R2'],
                'baseline_rmse': baseline_rmse,
                'test_samples': len(X_test)
            }
            
            print(f"      Test RMSE: {test_metrics['RMSE']:.4f}, R²: {test_metrics['R2']:.4f}")
        else:
            # No test data for this season, but save model anyway
            seasonal_models[season_name] = model
            seasonal_results[season_name] = {
                'test_rmse': np.nan,
                'test_mae': np.nan,
                'test_r2': np.nan,
                'baseline_rmse': np.nan,
                'test_samples': 0,
                'note': 'No test data in test period'
            }
            print(f"      Model trained (no test data in test period)")
    
    return seasonal_models, seasonal_results, global_model, global_test_metrics

# ==================== BLENDED PREDICTION ====================
def evaluate_blended_predictions(df, seasonal_models, test_mask, features):
    """Evaluate blended predictions using seasonal specialists"""
    print(f"\n{'='*80}")
    print("EVALUATING BLENDED PREDICTIONS")
    print(f"{'='*80}")
    
    # Prepare test data
    valid_mask = ~df['NO2_target'].isna()
    test_idx = valid_mask & test_mask
    X_test = df[test_idx][features].copy()
    y_test = df[test_idx]['NO2_target'].copy()
    
    # Convert to numeric
    for col in features:
        if col in X_test.columns:
            if X_test[col].dtype == 'object':
                X_test[col] = pd.Categorical(X_test[col]).codes
            elif X_test[col].dtype == 'bool':
                X_test[col] = X_test[col].astype(int)
    
    X_test = X_test.select_dtypes(include=[np.number])
    features = [f for f in features if f in X_test.columns]
    
    # Fill NaN
    for col in X_test.columns:
        if X_test[col].isnull().sum() > 0:
            median_val = X_test[col].median()
            X_test[col].fillna(median_val, inplace=True)
    
    # Get test months
    test_months = df[test_idx]['month'].values
    test_datetimes = df[test_idx]['datetime'].values
    
    # Predict using seasonal specialists
    predictions = []
    season_used = []
    
    for i, month in enumerate(test_months):
        # Determine season
        if month in [12, 1, 2]:
            season = 'winter'
        elif month in [3, 4, 5, 6]:
            season = 'summer'
        elif month in [7, 8, 9]:
            season = 'monsoon'
        else:
            season = 'post_monsoon'
        
        # Use seasonal model if available, else global
        if season in seasonal_models:
            pred = seasonal_models[season].predict(
                X_test.iloc[[i]], num_iteration=seasonal_models[season].best_iteration
            )[0]
            season_used.append(season)
        else:
            pred = seasonal_models['global'].predict(
                X_test.iloc[[i]], num_iteration=seasonal_models['global'].best_iteration
            )[0]
            season_used.append('global')
        
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    blended_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
        'MAE': mean_absolute_error(y_test, predictions),
        'R2': r2_score(y_test, predictions)
    }
    
    # Calculate metrics by season
    season_metrics = {}
    for season in ['winter', 'summer', 'monsoon', 'post_monsoon']:
        season_mask = df[test_idx]['month'].isin(
            [12, 1, 2] if season == 'winter' else
            [3, 4, 5, 6] if season == 'summer' else
            [7, 8, 9] if season == 'monsoon' else [10, 11]
        )
        if season_mask.sum() > 0:
            season_y = y_test[season_mask]
            season_pred = predictions[season_mask]
            season_metrics[season] = {
                'RMSE': np.sqrt(mean_squared_error(season_y, season_pred)),
                'MAE': mean_absolute_error(season_y, season_pred),
                'R2': r2_score(season_y, season_pred),
                'samples': season_mask.sum()
            }
    
    baseline_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_test.mean())))
    
    print(f"\n   Blended Model Results:")
    print(f"   Overall Test RMSE: {blended_metrics['RMSE']:.4f}")
    print(f"   Overall Test MAE:  {blended_metrics['MAE']:.4f}")
    print(f"   Overall Test R²:   {blended_metrics['R2']:.4f}")
    print(f"   Baseline RMSE:     {baseline_rmse:.4f}")
    print(f"   Improvement:       {((baseline_rmse - blended_metrics['RMSE']) / baseline_rmse * 100):.2f}%")
    
    print(f"\n   Season-wise Performance:")
    for season, metrics in season_metrics.items():
        print(f"   {season.capitalize()}: RMSE={metrics['RMSE']:.4f}, R²={metrics['R2']:.4f}, N={metrics['samples']}")
    
    # Compare with global model
    global_pred = seasonal_models['global'].predict(
        X_test, num_iteration=seasonal_models['global'].best_iteration
    )
    global_metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, global_pred)),
        'MAE': mean_absolute_error(y_test, global_pred),
        'R2': r2_score(y_test, global_pred)
    }
    
    print(f"\n   Comparison with Global Model:")
    print(f"   Global RMSE:  {global_metrics['RMSE']:.4f}, R²: {global_metrics['R2']:.4f}")
    print(f"   Blended RMSE: {blended_metrics['RMSE']:.4f}, R²: {blended_metrics['R2']:.4f}")
    improvement = ((global_metrics['RMSE'] - blended_metrics['RMSE']) / global_metrics['RMSE'] * 100)
    print(f"   Improvement:  {improvement:.2f}%")
    
    return blended_metrics, season_metrics, global_metrics

# ==================== MAIN ====================
print("\n3. Training models...")

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Data splits
train_mask = (df['datetime'] >= '2020-01-01') & (df['datetime'] <= '2021-12-31')
val_mask = (df['datetime'] >= '2022-01-01') & (df['datetime'] <= '2022-03-31')
test_mask = (df['datetime'] >= '2022-07-01') & (df['datetime'] <= '2022-12-31')

results_summary = []

# Train seasonal NO2 specialists
seasonal_no2_models, seasonal_no2_results, global_no2_model, global_no2_metrics = train_seasonal_no2_specialists(
    df, train_mask, val_mask, test_mask
)

# Save seasonal NO2 models
for season_name, model in seasonal_no2_models.items():
    model.save_model(f'models/no2_{season_name}_specialist.txt')
    with open(f'models/no2_{season_name}_specialist.pkl', 'wb') as f:
        pickle.dump(model, f)

# Evaluate blended predictions
no2_features = get_no2_features(df)
blended_metrics, season_metrics, global_metrics = evaluate_blended_predictions(
    df, seasonal_no2_models, test_mask, no2_features
)

# Train O3 model (keep existing, unchanged)
print("\n" + "="*80)
print("TRAINING O3 MODEL (UNCHANGED)")
print("="*80)
o3_model, o3_train_metrics, o3_val_metrics, o3_test_metrics, o3_baseline = train_model(
    df, 'O3_target', 'O3_target', train_mask, val_mask, test_mask, get_o3_features
)

o3_model.save_model('models/enhanced_o3_model.txt')
with open('models/enhanced_o3_model.pkl', 'wb') as f:
    pickle.dump(o3_model, f)

# Save results
results_summary.append({
    'Model': 'NO2_target (blended seasonal)',
    'Train_RMSE': np.nan,
    'Train_R2': np.nan,
    'Val_RMSE': np.nan,
    'Val_R2': np.nan,
    'Test_RMSE': blended_metrics['RMSE'],
    'Test_MAE': blended_metrics['MAE'],
    'Test_R2': blended_metrics['R2'],
    'Baseline_RMSE': np.nan,
    'Improvement_%': np.nan
})

results_summary.append({
    'Model': 'NO2_target (global fallback)',
    'Train_RMSE': np.nan,
    'Train_R2': np.nan,
    'Val_RMSE': np.nan,
    'Val_R2': np.nan,
    'Test_RMSE': global_metrics['RMSE'],
    'Test_MAE': global_metrics['MAE'],
    'Test_R2': global_metrics['R2'],
    'Baseline_RMSE': np.nan,
    'Improvement_%': np.nan
})

results_summary.append({
    'Model': 'O3_target',
    'Train_RMSE': o3_train_metrics['RMSE'],
    'Train_R2': o3_train_metrics['R2'],
    'Val_RMSE': o3_val_metrics['RMSE'],
    'Val_R2': o3_val_metrics['R2'],
    'Test_RMSE': o3_test_metrics['RMSE'],
    'Test_MAE': o3_test_metrics['MAE'],
    'Test_R2': o3_test_metrics['R2'],
    'Baseline_RMSE': o3_baseline,
    'Improvement_%': ((o3_baseline - o3_test_metrics['RMSE']) / o3_baseline * 100)
})

results_df = pd.DataFrame(results_summary)
results_df.to_csv('results/seasonal_no2_o3_performance_summary.csv', index=False)

# Save seasonal NO2 results
seasonal_df = pd.DataFrame(seasonal_no2_results).T
seasonal_df.to_csv('results/seasonal_no2_specialists_performance.csv')

# Save season-wise blended metrics
season_metrics_df = pd.DataFrame(season_metrics).T
season_metrics_df.to_csv('results/seasonal_no2_blended_metrics.csv')

# Create blending documentation
with open('results/NO2_SEASONAL_BLENDING_GUIDE.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("NO2 SEASONAL SPECIALIST BLENDING GUIDE\n")
    f.write("="*80 + "\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("BLENDING STRATEGY:\n")
    f.write("-"*80 + "\n")
    f.write("1. Determine season from month:\n")
    f.write("   - Winter: December (12), January (1), February (2)\n")
    f.write("   - Summer: March (3), April (4), May (5), June (6)\n")
    f.write("   - Monsoon: July (7), August (8), September (9)\n")
    f.write("   - Post-monsoon: October (10), November (11)\n\n")
    
    f.write("2. Use seasonal specialist if available:\n")
    f.write("   - If month matches season, use that season's model\n")
    f.write("   - Example: January → use winter specialist\n")
    f.write("   - Example: July → use monsoon specialist\n\n")
    
    f.write("3. Fallback to global model:\n")
    f.write("   - If seasonal model not available (insufficient data)\n")
    f.write("   - Use global NO2 model as fallback\n\n")
    
    f.write("MODELS AVAILABLE:\n")
    f.write("-"*80 + "\n")
    for season in seasonal_no2_models.keys():
        f.write(f"✓ {season.capitalize()} specialist: models/no2_{season}_specialist.txt/.pkl\n")
    f.write("\n")
    
    f.write("PERFORMANCE SUMMARY:\n")
    f.write("-"*80 + "\n")
    f.write(f"Blended Model RMSE: {blended_metrics['RMSE']:.6f}\n")
    f.write(f"Blended Model R²:   {blended_metrics['R2']:.6f}\n")
    f.write(f"Global Model RMSE:  {global_metrics['RMSE']:.6f}\n")
    f.write(f"Global Model R²:    {global_metrics['R2']:.6f}\n")
    improvement = ((global_metrics['RMSE'] - blended_metrics['RMSE']) / global_metrics['RMSE'] * 100)
    f.write(f"Improvement:        {improvement:.2f}%\n\n")
    
    f.write("SEASON-WISE PERFORMANCE:\n")
    f.write("-"*80 + "\n")
    for season, metrics in season_metrics.items():
        f.write(f"{season.capitalize()}:\n")
        f.write(f"  RMSE: {metrics['RMSE']:.6f}\n")
        f.write(f"  MAE:  {metrics['MAE']:.6f}\n")
        f.write(f"  R²:   {metrics['R2']:.6f}\n")
        f.write(f"  Samples: {metrics['samples']}\n\n")
    
    f.write("INFERENCE CODE EXAMPLE:\n")
    f.write("-"*80 + "\n")
    f.write("```python\n")
    f.write("import lightgbm as lgb\n")
    f.write("import pandas as pd\n\n")
    f.write("# Load models\n")
    f.write("winter_model = lgb.Booster(model_file='models/no2_winter_specialist.txt')\n")
    f.write("summer_model = lgb.Booster(model_file='models/no2_summer_specialist.txt')\n")
    f.write("monsoon_model = lgb.Booster(model_file='models/no2_monsoon_specialist.txt')\n")
    f.write("post_monsoon_model = lgb.Booster(model_file='models/no2_post_monsoon_specialist.txt')\n")
    f.write("global_model = lgb.Booster(model_file='models/no2_global_specialist.txt')\n\n")
    f.write("# Predict function\n")
    f.write("def predict_no2(X, month):\n")
    f.write("    if month in [12, 1, 2]:\n")
    f.write("        return winter_model.predict(X)\n")
    f.write("    elif month in [3, 4, 5, 6]:\n")
    f.write("        return summer_model.predict(X)\n")
    f.write("    elif month in [7, 8, 9]:\n")
    f.write("        return monsoon_model.predict(X)\n")
    f.write("    elif month in [10, 11]:\n")
    f.write("        return post_monsoon_model.predict(X)\n")
    f.write("    else:\n")
    f.write("        return global_model.predict(X)  # Fallback\n")
    f.write("```\n\n")
    
    f.write("="*80 + "\n")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))
print("\n\nSeasonal NO2 Specialists:")
print(seasonal_df.to_string())
print("\n\nSeason-wise Blended Metrics:")
print(season_metrics_df.to_string())
print("\n" + "="*80)

