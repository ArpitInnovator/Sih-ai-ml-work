"""
Advanced Multi-Model Training Script with All Critical Improvements
1. Per-season MODELS (winter, summer, monsoon, post-monsoon)
2. Rolling window training (180-270 days)
3. Photochemical features for O3
4. Quantile LightGBM (50%, 75%, 90%)
5. Model stacking (LightGBM, XGBoost, CatBoost, Neural Network)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
import pickle
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Optional imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import optuna
    from optuna.integration import LightGBMPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings('ignore')

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")

print("="*80)
print("ADVANCED MULTI-MODEL TRAINING WITH ALL IMPROVEMENTS")
print("="*80)

# ==================== LOAD DATA ====================
print("\n1. Loading cleaned dataset...")
df = pd.read_csv('master_site1_final_cleaned.csv')
print(f"   Dataset shape: {df.shape}")

# Convert datetime
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# ==================== FEATURE ENGINEERING ====================
print("\n2. Creating comprehensive features...")

# Time features
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['day_of_year'] = df['datetime'].dt.dayofyear
df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
df['is_weekday'] = (df['datetime'].dt.dayofweek < 5).astype(int)

# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Season grouping
def get_season(month):
    """Indian seasons: Winter (Dec-Feb), Summer (Mar-Jun), Monsoon (Jul-Sept), Post-Monsoon (Oct-Nov)"""
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

# Traffic proxies
df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                      ((df['hour'] >= 17) & (df['hour'] <= 20))).astype(int)
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

# Derived meteorological features
if 'wind_speed' not in df.columns and 'u10_era5' in df.columns and 'v10_era5' in df.columns:
    df['wind_speed'] = np.sqrt(df['u10_era5']**2 + df['v10_era5']**2)
if 'wind_direction_rad' not in df.columns and 'u10_era5' in df.columns and 'v10_era5' in df.columns:
    df['wind_direction_rad'] = np.arctan2(df['v10_era5'], df['u10_era5'])
if 'dewpoint_depression' not in df.columns and 't2m_era5' in df.columns and 'd2m_era5' in df.columns:
    df['dewpoint_depression'] = df['t2m_era5'] - df['d2m_era5']
if 'relative_humidity_approx' not in df.columns and 'dewpoint_depression' in df.columns:
    df['relative_humidity_approx'] = 100 * np.exp(-df['dewpoint_depression'] / 5)

# Interaction features
if 'hour' in df.columns and 't2m_era5' in df.columns:
    df['hour_temp_interaction'] = df['hour'] * df['t2m_era5']
if 'blh_era5' in df.columns and 'wind_speed' in df.columns:
    df['blh_wind_interaction'] = df['blh_era5'] * df['wind_speed']
    df['ventilation_rate'] = df['blh_era5'] / (df['wind_speed'] + 1e-6)
if 't2m_era5' in df.columns and 'relative_humidity_approx' in df.columns:
    df['temp_humidity_interaction'] = df['t2m_era5'] * df['relative_humidity_approx']

# STEP 3: PHOTOCHEMICAL FEATURES FOR O3
print("   Creating photochemical features for O3...")
if 'solar_elevation' in df.columns:
    # Solar elevation angle (already exists)
    df['solar_elevation_abs'] = np.abs(df['solar_elevation'])
    
    # Approximate photolysis rate from SZA (Solar Zenith Angle)
    if 'SZA_deg' in df.columns:
        # Photolysis rate approximation: higher when SZA is lower (more direct sunlight)
        df['photolysis_rate_approx'] = np.maximum(0, np.cos(np.radians(df['SZA_deg'])))
    else:
        df['photolysis_rate_approx'] = np.maximum(0, np.sin(np.radians(df['solar_elevation'])))
    
    # Temperature × sunlight interaction (critical for O3 formation)
    if 't2m_era5' in df.columns:
        df['temp_sunlight_interaction'] = df['t2m_era5'] * df['photolysis_rate_approx']
        df['temp_solar_elevation'] = df['t2m_era5'] * df['solar_elevation_abs']
    
    # BLH × wind speed (ventilation - already created above)
    # Weekend indicator (traffic reduction affects O3)
    df['weekend_traffic_reduction'] = df['is_weekend'] * (1 - df['is_rush_hour'])

# Lag features (1h only)
key_features_for_lags = ['no2', 'pm2p5', 'pm10', 'so2', 't2m_era5', 'wind_speed']
for col in key_features_for_lags:
    if col in df.columns:
        if f'{col}_lag_1h' not in df.columns:
            df[f'{col}_lag_1h'] = df[col].shift(1)

# Rolling means (3h, 6h)
rolling_features = ['no2', 'pm2p5', 'pm10', 'so2']
for window in [3, 6]:
    for feat in rolling_features:
        if feat in df.columns:
            col_name = f'{feat}_rolling_mean_{window}h'
            if col_name not in df.columns:
                df[col_name] = df[feat].rolling(window=window, min_periods=1).mean()

print("   ✓ Features created")

# ==================== FEATURE SELECTION ====================
print("\n3. Selecting features...")

def get_feature_list(df, target_name='NO2_target'):
    """Get feature list, with O3-specific photochemical features"""
    features = []
    
    # Basic pollutants
    pollutants = ['pm2p5', 'pm10', 'so2', 'no2', 'pm1', 'no']
    features.extend([f for f in pollutants if f in df.columns])
    
    # Satellite (only current, no daily/flags)
    if 'NO2_satellite' in df.columns:
        features.append('NO2_satellite')
    
    # ERA5 meteorology
    meteo = ['blh_era5', 't2m_era5', 'd2m_era5', 'dewpoint_depression', 
             'relative_humidity_approx', 'wind_speed', 'wind_direction_rad',
             'u10_era5', 'v10_era5', 'tcc_era5', 'sp']
    features.extend([f for f in meteo if f in df.columns])
    
    # Lag features (1h only)
    for feat in ['no2', 'pm2p5', 'pm10', 'so2', 't2m_era5', 'wind_speed']:
        col_name = f'{feat}_lag_1h'
        if col_name in df.columns:
            features.append(col_name)
    
    # Rolling means
    for window in [3, 6]:
        for feat in ['no2', 'pm2p5', 'pm10', 'so2']:
            col_name = f'{feat}_rolling_mean_{window}h'
            if col_name in df.columns:
                features.append(col_name)
    
    # Time features
    time_features = ['year', 'month', 'day', 'hour', 'day_of_week', 'day_of_year',
                     'is_weekend', 'is_weekday', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    features.extend([f for f in time_features if f in df.columns])
    
    # Season features (one-hot encoded)
    season_features = ['is_winter', 'is_summer', 'is_monsoon', 'is_post_monsoon']
    features.extend([f for f in season_features if f in df.columns])
    
    # Traffic/time proxies
    traffic_features = ['is_rush_hour', 'is_night']
    features.extend([f for f in traffic_features if f in df.columns])
    
    # Solar features
    solar_features = ['solar_elevation', 'SZA_deg']
    features.extend([f for f in solar_features if f in df.columns])
    
    # AOD features
    aod_features = ['aod550', 'bcaod550']
    features.extend([f for f in aod_features if f in df.columns])
    
    # Interaction features
    interaction_features = ['hour_temp_interaction', 'blh_wind_interaction',
                           'ventilation_rate', 'temp_humidity_interaction']
    features.extend([f for f in interaction_features if f in df.columns])
    
    # O3-specific photochemical features
    if target_name == 'O3_target':
        o3_features = ['solar_elevation_abs', 'photolysis_rate_approx', 
                      'temp_sunlight_interaction', 'temp_solar_elevation',
                      'weekend_traffic_reduction']
        features.extend([f for f in o3_features if f in df.columns])
    
    # Other important
    other_features = ['wind_dir_deg', 'aluvp', 'aluvd']
    features.extend([f for f in other_features if f in df.columns])
    
    # Remove duplicates and ensure they exist
    features = list(set(features))
    features = [f for f in features if f in df.columns]
    
    return features

# ==================== HELPER FUNCTIONS ====================
def calculate_metrics(y_true, y_pred):
    """Calculate metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def prepare_data_for_training(df, target_col, features, train_mask, val_mask, test_mask):
    """Prepare data with proper type conversion"""
    # Filter to valid rows first
    valid_mask = ~df[target_col].isna()
    
    # Combine masks: valid AND split mask
    train_idx = valid_mask & train_mask
    val_idx = valid_mask & val_mask
    test_idx = valid_mask & test_mask
    
    # Split
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
    
    # Select only numeric
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    
    # Fill NaN
    for col in X_train.columns:
        if X_train[col].isnull().sum() > 0:
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)
            X_val[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ==================== STEP 1: PER-SEASON MODELS ====================
def train_seasonal_model(df, target_col, target_name, season_name, season_mask, 
                         features, use_rolling_window=False, window_days=180):
    """Train a model for a specific season"""
    print(f"\n   Training {season_name} model for {target_name}...")
    
    # Filter data for this season
    df_season = df[season_mask].copy()
    
    if len(df_season) < 100:
        print(f"      Skipping {season_name} - insufficient data ({len(df_season)} rows)")
        return None, None, None, None
    
    # Rolling window: use last N days
    if use_rolling_window:
        cutoff_date = df_season['datetime'].max() - timedelta(days=window_days)
        df_season = df_season[df_season['datetime'] >= cutoff_date].copy()
    
    # Split: 80% train, 10% val, 10% test (temporal)
    df_season = df_season.sort_values('datetime').reset_index(drop=True)
    n = len(df_season)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    # Create masks
    train_mask_season = pd.Series([True] * train_end + [False] * (n - train_end), index=df_season.index)
    val_mask_season = pd.Series([False] * train_end + [True] * (val_end - train_end) + [False] * (n - val_end), index=df_season.index)
    test_mask_season = pd.Series([False] * val_end + [True] * (n - val_end), index=df_season.index)
    
    # Filter to valid rows
    valid_mask = ~df_season[target_col].isna()
    train_idx = valid_mask & train_mask_season
    val_idx = valid_mask & val_mask_season
    test_idx = valid_mask & test_mask_season
    
    # Split
    X_train = df_season[train_idx][features].copy()
    y_train = df_season[train_idx][target_col].copy()
    X_val = df_season[val_idx][features].copy()
    y_val = df_season[val_idx][target_col].copy()
    X_test = df_season[test_idx][features].copy()
    y_test = df_season[test_idx][target_col].copy()
    
    if len(X_train) < 50:
        print(f"      Skipping {season_name} - insufficient training data")
        return None, None, None, None
    
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
    
    # Select only numeric
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    
    # Fill NaN
    for col in X_train.columns:
        if X_train[col].isnull().sum() > 0:
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)
            X_val[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)
    
    print(f"      Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train LightGBM
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 7,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    # Predictions
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    return model, test_metrics, X_test, y_test

# ==================== STEP 4: QUANTILE LIGHTGBM ====================
def train_quantile_models(X_train, y_train, X_val, y_val, quantiles=[0.5, 0.75, 0.9]):
    """Train quantile regression models"""
    models = {}
    
    for q in quantiles:
        params = {
            'objective': 'quantile',
            'alpha': q,
            'metric': 'quantile',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 7,
            'learning_rate': 0.05,
            'verbose': -1,
            'random_state': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=300,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        
        models[q] = model
    
    return models

# ==================== STEP 5: MODEL STACKING ====================
def train_stacked_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train multiple models and stack them"""
    predictions = {}
    models = {}
    
    # LightGBM
    print("      Training LightGBM...")
    params_lgb = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 7,
        'learning_rate': 0.05,
        'verbose': -1,
        'random_state': 42
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    model_lgb = lgb.train(params_lgb, train_data, valid_sets=[val_data],
                         num_boost_round=300, callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
    predictions['lgb'] = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)
    models['lgb'] = model_lgb
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        print("      Training XGBoost...")
        # XGBoost 2.0+ uses early_stopping_rounds in constructor, not fit()
        try:
            model_xgb = xgb.XGBRegressor(
                n_estimators=300, 
                max_depth=7, 
                learning_rate=0.05, 
                random_state=42,
                early_stopping_rounds=30,
                eval_metric='rmse'
            )
            model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        except (TypeError, ValueError):
            # Fallback: train without early stopping
            model_xgb = xgb.XGBRegressor(
                n_estimators=200,  # Reduced since no early stopping
                max_depth=7, 
                learning_rate=0.05, 
                random_state=42
            )
            model_xgb.fit(X_train, y_train, verbose=False)
        predictions['xgb'] = model_xgb.predict(X_test)
        models['xgb'] = model_xgb
    
    # CatBoost
    if CATBOOST_AVAILABLE:
        print("      Training CatBoost...")
        model_cb = cb.CatBoostRegressor(iterations=300, depth=7, learning_rate=0.05, random_seed=42, verbose=False)
        model_cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30)
        predictions['catboost'] = model_cb.predict(X_test)
        models['catboost'] = model_cb
    
    # Neural Network
    print("      Training Neural Network...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    model_nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True)
    model_nn.fit(X_train_scaled, y_train)
    predictions['nn'] = model_nn.predict(X_test_scaled)
    models['nn'] = (model_nn, scaler)
    
    # Stack: simple average
    stacked_pred = np.mean(list(predictions.values()), axis=0)
    
    return models, predictions, stacked_pred

# ==================== MAIN TRAINING ====================
print("\n4. Training per-season models with stacking...")

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Seasonal splits
train_mask = (df['datetime'] >= '2019-06-01') & (df['datetime'] <= '2021-12-31')
val_mask = (df['datetime'] >= '2022-01-01') & (df['datetime'] <= '2022-06-30')
test_mask = (df['datetime'] >= '2022-07-01') & (df['datetime'] <= '2022-12-31')

results_summary = []

# Train models for each target
for target_name, target_col in [('NO2_target', 'NO2_target'), ('O3_target', 'O3_target'), ('CO', 'co')]:
    print(f"\n{'='*80}")
    print(f"TRAINING MODELS FOR: {target_name}")
    print(f"{'='*80}")
    
    # Get features (with O3-specific photochemical features)
    features = get_feature_list(df, target_name)
    print(f"   Features: {len(features)}")
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_for_training(
        df, target_col, features, train_mask, val_mask, test_mask
    )
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # STEP 1: Train per-season models
    seasonal_models = {}
    for season_name, season_months in [('winter', [12, 1, 2]), ('summer', [3, 4, 5, 6]), 
                                        ('monsoon', [7, 8, 9]), ('post_monsoon', [10, 11])]:
        season_mask = df['month'].isin(season_months)
        model, metrics, X_test_season, y_test_season = train_seasonal_model(
            df, target_col, target_name, season_name, season_mask, features,
            use_rolling_window=True, window_days=180
        )
        if model is not None:
            seasonal_models[season_name] = model
            # Save model
            model.save_model(f'models/lgbm_{target_name}_{season_name}.txt')
            with open(f'models/lgbm_{target_name}_{season_name}.pkl', 'wb') as f:
                pickle.dump(model, f)
    
    # STEP 5: Train stacked models (on full dataset)
    print(f"\n   Training stacked models for {target_name}...")
    stacked_models, predictions, stacked_pred = train_stacked_models(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Evaluate stacked model
    stacked_metrics = calculate_metrics(y_test, stacked_pred)
    
    # Save stacked models
    for name, model in stacked_models.items():
        if name == 'nn':
            # NN has scaler
            with open(f'models/{name}_{target_name}.pkl', 'wb') as f:
                pickle.dump(model, f)
        else:
            if hasattr(model, 'save_model'):
                model.save_model(f'models/{name}_{target_name}.txt')
            with open(f'models/{name}_{target_name}.pkl', 'wb') as f:
                pickle.dump(model, f)
    
    # STEP 4: Train quantile models
    print(f"\n   Training quantile models for {target_name}...")
    quantile_models = train_quantile_models(X_train, y_train, X_val, y_val)
    for q, model in quantile_models.items():
        model.save_model(f'models/quantile_{q}_{target_name}.txt')
        with open(f'models/quantile_{q}_{target_name}.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    # Baseline
    baseline_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean())))
    
    results_summary.append({
        'Model': target_name,
        'Stacked_RMSE': stacked_metrics['RMSE'],
        'Stacked_MAE': stacked_metrics['MAE'],
        'Stacked_R2': stacked_metrics['R2'],
        'Baseline_RMSE': baseline_rmse,
        'Improvement_%': ((baseline_rmse - stacked_metrics['RMSE']) / baseline_rmse * 100),
        'Seasonal_Models': len(seasonal_models),
        'Quantile_Models': len(quantile_models)
    })
    
    print(f"\n   Stacked Model Results:")
    print(f"   RMSE: {stacked_metrics['RMSE']:.4f}")
    print(f"   MAE:  {stacked_metrics['MAE']:.4f}")
    print(f"   R²:   {stacked_metrics['R2']:.4f}")

# ==================== SAVE RESULTS ====================
print("\n5. Saving results...")

results_df = pd.DataFrame(results_summary)
results_df.to_csv('results/advanced_model_performance_summary.csv', index=False)

with open('results/advanced_metrics_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("ADVANCED MULTI-MODEL TRAINING RESULTS\n")
    f.write("="*80 + "\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("IMPROVEMENTS IMPLEMENTED:\n")
    f.write("1. [OK] Per-season MODELS (winter, summer, monsoon, post-monsoon)\n")
    f.write("2. [OK] Rolling window training (180 days)\n")
    f.write("3. [OK] Photochemical features for O3\n")
    f.write("4. [OK] Quantile LightGBM (50%, 75%, 90%)\n")
    f.write("5. [OK] Model stacking (LightGBM, XGBoost, CatBoost, Neural Network)\n")
    f.write("\n" + "-"*80 + "\n\n")
    
    for idx, row in results_df.iterrows():
        f.write(f"{row['Model']} MODEL\n")
        f.write("-"*80 + "\n")
        f.write(f"Stacked RMSE: {row['Stacked_RMSE']:.6f}\n")
        f.write(f"Stacked MAE:  {row['Stacked_MAE']:.6f}\n")
        f.write(f"Stacked R²:   {row['Stacked_R2']:.6f}\n")
        f.write(f"Baseline RMSE: {row['Baseline_RMSE']:.6f}\n")
        f.write(f"Improvement: {row['Improvement_%']:.2f}%\n")
        f.write(f"Seasonal Models: {row['Seasonal_Models']}\n")
        f.write(f"Quantile Models: {row['Quantile_Models']}\n\n")

print("   ✓ Results saved")
print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))
print("\n" + "="*80)

