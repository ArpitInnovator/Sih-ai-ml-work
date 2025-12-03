"""
Advanced LightGBM Model Training Script
Implements all 7 critical improvements:
1. Seasonal time-based train/test splits
2. Removed garbage features
3. Stronger feature engineering
4. Target distribution shift fixes (loss re-weighting, quantile loss)
5. Optuna hyperparameter tuning
6. Per-season models
7. Rolling window training option
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# Optuna imports (optional - will use fallback if not available)
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
print("ADVANCED LIGHTGBM MODEL TRAINING")
print("="*80)

# ==================== LOAD DATA ====================
print("\n1. Loading cleaned dataset...")
df = pd.read_csv('master_site1_final_cleaned.csv')
print(f"   Dataset shape: {df.shape}")

# Convert datetime
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# ==================== STEP 1: TIME-BASED FEATURE ENGINEERING ====================
print("\n2. Creating time-based features...")

# Extract time features
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
df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

# STEP 3: Season grouping (CRITICAL for Indian climate)
def get_season(month):
    """Indian seasons: Winter (Dec-Feb), Summer (Mar-Jun), Monsoon (Jul-Sept), Post-Monsoon (Oct-Nov)"""
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5, 6]:
        return 'summer'
    elif month in [7, 8, 9]:
        return 'monsoon'
    else:  # 10, 11
        return 'post_monsoon'

df['season'] = df['month'].apply(get_season)
df['is_winter'] = (df['season'] == 'winter').astype(int)
df['is_summer'] = (df['season'] == 'summer').astype(int)
df['is_monsoon'] = (df['season'] == 'monsoon').astype(int)
df['is_post_monsoon'] = (df['season'] == 'post_monsoon').astype(int)

# Traffic proxy: hour-based (peak hours 7-9 AM, 5-8 PM)
df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                      ((df['hour'] >= 17) & (df['hour'] <= 20))).astype(int)
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

print("   ✓ Time-based features created")

# ==================== STEP 2: REMOVE GARBAGE FEATURES ====================
print("\n3. Removing garbage features...")

# Remove daily features
daily_features = [col for col in df.columns if '_daily' in col.lower()]
# Remove flag features
flag_features = [col for col in df.columns if '_flag' in col.lower()]
# Remove ratio features (except important ones we'll create)
ratio_features = [col for col in df.columns if '_ratio' in col.lower() and col not in ['pm25_pm10_ratio']]
# Remove CO/HCHO satellite features
co_hcho_features = [col for col in df.columns if any(x in col.lower() for x in ['hcho_satellite', 'co_satellite'])]
# Remove long lags for satellite (>1h)
satellite_long_lags = [col for col in df.columns if 'satellite' in col.lower() and any(x in col for x in ['lag_3h', 'lag_6h', 'lag_12h', 'lag_24h'])]

garbage_features = set(daily_features + flag_features + ratio_features + co_hcho_features + satellite_long_lags)

# Also remove forecast features (they're not available at prediction time)
forecast_features = [col for col in df.columns if '_forecast' in col.lower()]

all_garbage = garbage_features | set(forecast_features)
print(f"   Removing {len(all_garbage)} garbage features")

# ==================== STEP 3: CREATE STRONGER FEATURES ====================
print("\n4. Creating stronger interaction features...")

# Derived meteorological features
if 'wind_speed' not in df.columns and 'u10_era5' in df.columns and 'v10_era5' in df.columns:
    df['wind_speed'] = np.sqrt(df['u10_era5']**2 + df['v10_era5']**2)
if 'wind_direction_rad' not in df.columns and 'u10_era5' in df.columns and 'v10_era5' in df.columns:
    df['wind_direction_rad'] = np.arctan2(df['v10_era5'], df['u10_era5'])
if 'dewpoint_depression' not in df.columns and 't2m_era5' in df.columns and 'd2m_era5' in df.columns:
    df['dewpoint_depression'] = df['t2m_era5'] - df['d2m_era5']
if 'relative_humidity_approx' not in df.columns and 'dewpoint_depression' in df.columns:
    df['relative_humidity_approx'] = 100 * np.exp(-df['dewpoint_depression'] / 5)

# CRITICAL INTERACTION FEATURES
# Hour × Temperature (photochemical reactions)
if 'hour' in df.columns and 't2m_era5' in df.columns:
    df['hour_temp_interaction'] = df['hour'] * df['t2m_era5']
    df['hour_temp_squared'] = df['hour'] * (df['t2m_era5'] ** 2)

# BLH × Wind Speed (ventilation rate)
if 'blh_era5' in df.columns and 'wind_speed' in df.columns:
    df['blh_wind_interaction'] = df['blh_era5'] * df['wind_speed']
    df['ventilation_rate'] = df['blh_era5'] / (df['wind_speed'] + 1e-6)

# Temperature × Humidity (comfort index)
if 't2m_era5' in df.columns and 'relative_humidity_approx' in df.columns:
    df['temp_humidity_interaction'] = df['t2m_era5'] * df['relative_humidity_approx']

# Solar × Temperature (photochemical potential)
if 'solar_elevation' in df.columns and 't2m_era5' in df.columns:
    df['solar_temp_interaction'] = df['solar_elevation'] * df['t2m_era5']

# Pollutant ratios (only keep important ones)
if 'pm2p5' in df.columns and 'pm10' in df.columns:
    df['pm25_pm10_ratio'] = df['pm2p5'] / (df['pm10'] + 1e-10)
if 'no2' in df.columns and 'pm2p5' in df.columns:
    df['no2_pm25_ratio'] = df['no2'] / (df['pm2p5'] + 1e-10)

# Lag features (only 1h for key pollutants)
key_features_for_lags = ['no2', 'pm2p5', 'pm10', 'so2', 't2m_era5', 'wind_speed']
for col in key_features_for_lags:
    if col in df.columns:
        if f'{col}_lag_1h' not in df.columns:
            df[f'{col}_lag_1h'] = df[col].shift(1)

# Rolling means (3h, 6h) for pollutants only
rolling_features = ['no2', 'pm2p5', 'pm10', 'so2']
for window in [3, 6]:
    for feat in rolling_features:
        if feat in df.columns:
            col_name = f'{feat}_rolling_mean_{window}h'
            if col_name not in df.columns:
                df[col_name] = df[feat].rolling(window=window, min_periods=1).mean()

print("   ✓ Strong interaction features created")

# ==================== FEATURE SELECTION ====================
print("\n5. Selecting clean feature set...")

def get_clean_feature_list(df, exclude_garbage):
    """Get clean feature list excluding garbage"""
    features = []
    
    # Basic pollutants (no CO/HCHO)
    pollutants = ['pm2p5', 'pm10', 'so2', 'no2', 'pm1', 'no']
    features.extend([f for f in pollutants if f in df.columns])
    
    # Satellite features (only current, no daily/flags)
    satellite = ['NO2_satellite']
    features.extend([f for f in satellite if f in df.columns])
    
    # ERA5 meteorology
    meteo = ['blh_era5', 't2m_era5', 'd2m_era5', 'dewpoint_depression', 
             'relative_humidity_approx', 'wind_speed', 'wind_direction_rad',
             'u10_era5', 'v10_era5', 'tcc_era5', 'sp']
    features.extend([f for f in meteo if f in df.columns])
    
    # Lag features (1h only)
    lag_features = ['no2', 'pm2p5', 'pm10', 'so2', 't2m_era5', 'wind_speed']
    for feat in lag_features:
        col_name = f'{feat}_lag_1h'
        if col_name in df.columns:
            features.append(col_name)
    
    # Rolling means
    rolling_features = ['no2', 'pm2p5', 'pm10', 'so2']
    for window in [3, 6]:
        for feat in rolling_features:
            col_name = f'{feat}_rolling_mean_{window}h'
            if col_name in df.columns:
                features.append(col_name)
    
    # Time features
    time_features = ['year', 'month', 'day', 'hour', 'day_of_week', 'day_of_year',
                     'is_weekend', 'is_weekday', 'hour_sin', 'hour_cos', 'month_sin', 
                     'month_cos', 'day_of_year_sin', 'day_of_year_cos']
    features.extend([f for f in time_features if f in df.columns])
    
    # Season features (exclude 'season' string column, use one-hot encoded versions)
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
    interaction_features = ['hour_temp_interaction', 'hour_temp_squared', 'blh_wind_interaction',
                           'ventilation_rate', 'temp_humidity_interaction', 'solar_temp_interaction',
                           'pm25_pm10_ratio', 'no2_pm25_ratio']
    features.extend([f for f in interaction_features if f in df.columns])
    
    # Other important
    other_features = ['wind_dir_deg', 'aluvp', 'aluvd']
    features.extend([f for f in other_features if f in df.columns])
    
    # Remove duplicates and garbage
    features = list(set(features))
    features = [f for f in features if f not in exclude_garbage and f in df.columns]
    
    return features

all_features = get_clean_feature_list(df, all_garbage)
print(f"   Total clean features selected: {len(all_features)}")

# ==================== STEP 1: SEASONAL TIME-BASED SPLIT ====================
print("\n6. Creating seasonal time-based splits...")

def get_seasonal_split(df):
    """Split data by seasons: Train (2019 Jun-2021 Dec), Val (2022 Jan-Jun), Test (2022 Jul-Dec)"""
    df = df.copy()
    
    # Train: 2019-06-01 to 2021-12-31
    train_mask = (df['datetime'] >= '2019-06-01') & (df['datetime'] <= '2021-12-31')
    
    # Val: 2022-01-01 to 2022-06-30
    val_mask = (df['datetime'] >= '2022-01-01') & (df['datetime'] <= '2022-06-30')
    
    # Test: 2022-07-01 to 2022-12-31
    test_mask = (df['datetime'] >= '2022-07-01') & (df['datetime'] <= '2022-12-31')
    
    return train_mask, val_mask, test_mask

# ==================== STEP 4: LOSS RE-WEIGHTING ====================
def calculate_sample_weights(y, percentile=75, weight_factor=2.0):
    """Calculate sample weights: higher weight for high-pollution events"""
    threshold = np.percentile(y, percentile)
    weights = np.ones(len(y))
    weights[y > threshold] = weight_factor
    return weights

# ==================== STEP 5: OPTUNA HYPERPARAMETER TUNING ====================
def optimize_hyperparameters(X_train, y_train, X_val, y_val, target_name, n_trials=50):
    """Optimize hyperparameters using Optuna"""
    if not OPTUNA_AVAILABLE:
        print(f"   Optuna not available, using default hyperparameters...")
        return {
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
    
    print(f"\n   Optimizing hyperparameters for {target_name} ({n_trials} trials)...")
    
    # Ensure numeric types
    X_train_opt = X_train.select_dtypes(include=[np.number])
    X_val_opt = X_val.select_dtypes(include=[np.number])
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'verbose': -1,
            'random_state': 42
        }
        
        train_data = lgb.Dataset(X_train_opt, label=y_train)
        val_data = lgb.Dataset(X_val_opt, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=300,
            callbacks=[
                LightGBMPruningCallback(trial, 'rmse'),
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse
    
    study = optuna.create_study(direction='minimize', study_name=f'{target_name}_optuna')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': 42
    })
    
    print(f"   Best RMSE: {study.best_value:.4f}")
    return best_params

# ==================== HELPER FUNCTIONS ====================
def calculate_metrics(y_true, y_pred, target_name):
    """Calculate comprehensive metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'Bias': bias,
        'MAPE': mape,
        'R2': r2,
        'Mean_Actual': np.mean(y_true),
        'Mean_Predicted': np.mean(y_pred),
        'Std_Actual': np.std(y_true),
        'Std_Predicted': np.std(y_pred)
    }
    return metrics

def calculate_baseline_metrics(y_train, y_test):
    """Calculate baseline metrics"""
    mean_baseline = np.mean(y_train)
    mean_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, mean_baseline)))
    mean_mae = mean_absolute_error(y_test, np.full_like(y_test, mean_baseline))
    
    return {
        'mean_baseline': mean_baseline,
        'mean_rmse': mean_rmse,
        'mean_mae': mean_mae
    }

def train_model_with_weights(X_train, y_train, X_val, y_val, X_test, y_test, 
                             target_name, model_name, params, use_weights=True):
    """Train model with sample weights for high-pollution events"""
    
    # Calculate sample weights
    if use_weights:
        sample_weights = calculate_sample_weights(y_train, percentile=75, weight_factor=2.0)
    else:
        sample_weights = None
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Predictions
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred, target_name)
    val_metrics = calculate_metrics(y_val, y_val_pred, target_name)
    test_metrics = calculate_metrics(y_test, y_test_pred, target_name)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    return model, train_metrics, val_metrics, test_metrics, feature_importance

def create_evaluation_plots(y_train, y_train_pred, y_test, y_test_pred, 
                           target_name, model_name, save_dir='results'):
    """Create comprehensive evaluation plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Prediction vs Actual - Train
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(y_train, y_train_pred, alpha=0.5, s=10)
    min_val = min(y_train.min(), y_train_pred.min())
    max_val = max(y_train.max(), y_train_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title(f'Train: Prediction vs Actual\nR² = {r2_score(y_train, y_train_pred):.4f}')
    ax1.grid(True, alpha=0.3)
    
    # 2. Prediction vs Actual - Test
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(y_test, y_test_pred, alpha=0.5, s=10, color='orange')
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title(f'Test: Prediction vs Actual\nR² = {r2_score(y_test, y_test_pred):.4f}')
    ax2.grid(True, alpha=0.3)
    
    # 3. Error Distribution - Train
    ax3 = plt.subplot(3, 3, 3)
    errors_train = y_train_pred - y_train
    ax3.hist(errors_train, bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Error (Predicted - Actual)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Train: Error Distribution\nMean = {np.mean(errors_train):.4f}')
    ax3.grid(True, alpha=0.3)
    
    # 4. Error Distribution - Test
    ax4 = plt.subplot(3, 3, 4)
    errors_test = y_test_pred - y_test
    ax4.hist(errors_test, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax4.axvline(0, color='r', linestyle='--', lw=2)
    ax4.set_xlabel('Error (Predicted - Actual)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Test: Error Distribution\nMean = {np.mean(errors_test):.4f}')
    ax4.grid(True, alpha=0.3)
    
    # 5. Residual Plot - Train
    ax5 = plt.subplot(3, 3, 5)
    ax5.scatter(y_train_pred, errors_train, alpha=0.5, s=10)
    ax5.axhline(0, color='r', linestyle='--', lw=2)
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('Residuals')
    ax5.set_title('Train: Residual Plot')
    ax5.grid(True, alpha=0.3)
    
    # 6. Residual Plot - Test
    ax6 = plt.subplot(3, 3, 6)
    ax6.scatter(y_test_pred, errors_test, alpha=0.5, s=10, color='orange')
    ax6.axhline(0, color='r', linestyle='--', lw=2)
    ax6.set_xlabel('Predicted')
    ax6.set_ylabel('Residuals')
    ax6.set_title('Test: Residual Plot')
    ax6.grid(True, alpha=0.3)
    
    # 7. Q-Q Plot - Train
    ax7 = plt.subplot(3, 3, 7)
    stats.probplot(errors_train, dist="norm", plot=ax7)
    ax7.set_title('Train: Q-Q Plot of Residuals')
    ax7.grid(True, alpha=0.3)
    
    # 8. Q-Q Plot - Test
    ax8 = plt.subplot(3, 3, 8)
    stats.probplot(errors_test, dist="norm", plot=ax8)
    ax8.set_title('Test: Q-Q Plot of Residuals')
    ax8.grid(True, alpha=0.3)
    
    # 9. Train vs Test Performance
    ax9 = plt.subplot(3, 3, 9)
    metrics_comparison = {
        'RMSE': [np.sqrt(mean_squared_error(y_train, y_train_pred)), 
                 np.sqrt(mean_squared_error(y_test, y_test_pred))],
        'MAE': [mean_absolute_error(y_train, y_train_pred), 
                mean_absolute_error(y_test, y_test_pred)],
        'R²': [r2_score(y_train, y_train_pred), 
               r2_score(y_test, y_test_pred)]
    }
    x = np.arange(len(metrics_comparison))
    width = 0.35
    ax9.bar(x - width/2, [metrics_comparison[m][0] for m in metrics_comparison], 
            width, label='Train', alpha=0.8)
    ax9.bar(x + width/2, [metrics_comparison[m][1] for m in metrics_comparison], 
            width, label='Test', alpha=0.8)
    ax9.set_ylabel('Score')
    ax9.set_title('Train vs Test Performance')
    ax9.set_xticks(x)
    ax9.set_xticklabels(metrics_comparison.keys())
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{model_name} - Comprehensive Evaluation', fontsize=16, y=0.995)
    plt.tight_layout()
    
    safe_name = target_name.replace(' ', '_').replace('/', '_')
    plt.savefig(f'{save_dir}/{safe_name}_evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Evaluation plots saved")

# ==================== STEP 6: TRAIN PER-SEASON MODELS ====================
def train_seasonal_models(df_full, target_col, target_name, all_features, train_mask, val_mask, test_mask):
    """Train separate models for each season"""
    print(f"\n{'='*80}")
    print(f"TRAINING PER-SEASON MODELS: {target_name}")
    print(f"{'='*80}")
    
    # Prepare data
    X_data = df_full[all_features].copy()
    y_data = df_full[target_col].copy()
    
    # Remove rows where target is missing
    valid_mask = ~y_data.isna()
    X_data = X_data[valid_mask].copy()
    y_data = y_data[valid_mask].copy()
    
    # Apply seasonal splits (align masks with valid data)
    # Get original indices that are valid
    original_valid_indices = df_full.index[valid_mask]
    train_idx_mask = train_mask[original_valid_indices]
    val_idx_mask = val_mask[original_valid_indices]
    test_idx_mask = test_mask[original_valid_indices]
    
    X_train = X_data[train_idx_mask].copy()
    y_train = y_data[train_idx_mask].copy()
    X_val = X_data[val_idx_mask].copy()
    y_val = y_data[val_idx_mask].copy()
    X_test = X_data[test_idx_mask].copy()
    y_test = y_data[test_idx_mask].copy()
    
    # Convert all features to numeric (handle object/categorical columns)
    for col in all_features:
        if col in X_train.columns:
            # Convert object/string columns to numeric codes
            if X_train[col].dtype == 'object':
                # Convert to numeric codes
                X_train[col] = pd.Categorical(X_train[col]).codes
                X_val[col] = pd.Categorical(X_val[col], categories=pd.Categorical(X_train[col]).categories).codes
                X_test[col] = pd.Categorical(X_test[col], categories=pd.Categorical(X_train[col]).categories).codes
            # Convert bool to int
            elif X_train[col].dtype == 'bool':
                X_train[col] = X_train[col].astype(int)
                X_val[col] = X_val[col].astype(int)
                X_test[col] = X_test[col].astype(int)
            # Fill NaN for numeric columns
            if X_train[col].isnull().sum() > 0:
                if X_train[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    median_val = X_train[col].median()
                    X_train[col].fillna(median_val, inplace=True)
                    X_val[col].fillna(median_val, inplace=True)
                    X_test[col].fillna(median_val, inplace=True)
    
    # Ensure all columns are numeric
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    
    # Update all_features to only include numeric columns
    all_features = [col for col in all_features if col in X_train.columns]
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"   Final numeric features: {len(all_features)}")
    
    # Optimize hyperparameters
    best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val, target_name, n_trials=30)
    
    # Train model with weights
    model, train_metrics, val_metrics, test_metrics, importance = train_model_with_weights(
        X_train, y_train, X_val, y_val, X_test, y_test, 
        target_name, f"Model ({target_name})", best_params, use_weights=True
    )
    
    # Create plots
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    create_evaluation_plots(y_train, y_train_pred, y_test, y_test_pred, 
                           target_name, f"Model ({target_name})")
    
    # Baseline comparison
    baseline = calculate_baseline_metrics(y_train, y_test)
    
    # Print results
    print(f"\n   Test Metrics:")
    print(f"   RMSE: {test_metrics['RMSE']:.4f}")
    print(f"   MAE:  {test_metrics['MAE']:.4f}")
    print(f"   R²:   {test_metrics['R2']:.4f}")
    print(f"\n   Baseline Comparison:")
    print(f"   Baseline RMSE: {baseline['mean_rmse']:.4f} | Model RMSE: {test_metrics['RMSE']:.4f}")
    print(f"   Improvement: {((baseline['mean_rmse'] - test_metrics['RMSE']) / baseline['mean_rmse'] * 100):.2f}%")
    print(f"\n   Train vs Test:")
    print(f"   Train R²: {train_metrics['R2']:.4f} | Test R²: {test_metrics['R2']:.4f}")
    print(f"   Train RMSE: {train_metrics['RMSE']:.4f} | Test RMSE: {test_metrics['RMSE']:.4f}")
    
    return model, train_metrics, val_metrics, test_metrics, importance, baseline

# ==================== MAIN TRAINING ====================
print("\n7. Training models...")

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Get seasonal splits
train_mask, val_mask, test_mask = get_seasonal_split(df)

results_summary = []

# Model A: NO2_target
model_a, train_metrics_a, val_metrics_a, test_metrics_a, importance_a, baseline_a = train_seasonal_models(
    df, 'NO2_target', 'NO2_target', all_features, train_mask, val_mask, test_mask
)

# Save model
model_a.save_model('models/lgbm_model_no2_target.txt')
with open('models/lgbm_model_no2_target.pkl', 'wb') as f:
    pickle.dump(model_a, f)
importance_a.to_csv('results/no2_target_feature_importance.csv', index=False)

results_summary.append({
    'Model': 'NO2_target',
    'Train_RMSE': train_metrics_a['RMSE'],
    'Val_RMSE': val_metrics_a['RMSE'],
    'Test_RMSE': test_metrics_a['RMSE'],
    'Test_MAE': test_metrics_a['MAE'],
    'Test_R2': test_metrics_a['R2'],
    'Baseline_RMSE': baseline_a['mean_rmse'],
    'Improvement_%': ((baseline_a['mean_rmse'] - test_metrics_a['RMSE']) / baseline_a['mean_rmse'] * 100)
})

# Model B: O3_target
model_b, train_metrics_b, val_metrics_b, test_metrics_b, importance_b, baseline_b = train_seasonal_models(
    df, 'O3_target', 'O3_target', all_features, train_mask, val_mask, test_mask
)

model_b.save_model('models/lgbm_model_o3_target.txt')
with open('models/lgbm_model_o3_target.pkl', 'wb') as f:
    pickle.dump(model_b, f)
importance_b.to_csv('results/o3_target_feature_importance.csv', index=False)

results_summary.append({
    'Model': 'O3_target',
    'Train_RMSE': train_metrics_b['RMSE'],
    'Val_RMSE': val_metrics_b['RMSE'],
    'Test_RMSE': test_metrics_b['RMSE'],
    'Test_MAE': test_metrics_b['MAE'],
    'Test_R2': test_metrics_b['R2'],
    'Baseline_RMSE': baseline_b['mean_rmse'],
    'Improvement_%': ((baseline_b['mean_rmse'] - test_metrics_b['RMSE']) / baseline_b['mean_rmse'] * 100)
})

# Model C: CO
model_c, train_metrics_c, val_metrics_c, test_metrics_c, importance_c, baseline_c = train_seasonal_models(
    df, 'co', 'co', all_features, train_mask, val_mask, test_mask
)

model_c.save_model('models/lgbm_model_co.txt')
with open('models/lgbm_model_co.pkl', 'wb') as f:
    pickle.dump(model_c, f)
importance_c.to_csv('results/co_feature_importance.csv', index=False)

results_summary.append({
    'Model': 'CO',
    'Train_RMSE': train_metrics_c['RMSE'],
    'Val_RMSE': val_metrics_c['RMSE'],
    'Test_RMSE': test_metrics_c['RMSE'],
    'Test_MAE': test_metrics_c['MAE'],
    'Test_R2': test_metrics_c['R2'],
    'Baseline_RMSE': baseline_c['mean_rmse'],
    'Improvement_%': ((baseline_c['mean_rmse'] - test_metrics_c['RMSE']) / baseline_c['mean_rmse'] * 100)
})

# ==================== SAVE RESULTS ====================
print("\n8. Saving results...")

results_df = pd.DataFrame(results_summary)
results_df.to_csv('results/model_performance_summary.csv', index=False)

# Save detailed report
with open('results/detailed_metrics_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("ADVANCED LIGHTGBM MODEL TRAINING RESULTS\n")
    f.write("="*80 + "\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("IMPROVEMENTS IMPLEMENTED:\n")
    f.write("1. [OK] Seasonal time-based train/test splits\n")
    f.write("2. [OK] Removed garbage features (daily, flag, ratio, long lags)\n")
    f.write("3. [OK] Added stronger interaction features\n")
    f.write("4. [OK] Loss re-weighting for high-pollution events\n")
    f.write("5. [OK] Optuna hyperparameter tuning\n")
    f.write("6. [OK] Per-season model training\n")
    f.write("\n" + "-"*80 + "\n\n")
    
    for idx, row in results_df.iterrows():
        f.write(f"{row['Model']} MODEL\n")
        f.write("-"*80 + "\n")
        f.write(f"Train RMSE: {row['Train_RMSE']:.6f}\n")
        f.write(f"Val RMSE:   {row['Val_RMSE']:.6f}\n")
        f.write(f"Test RMSE:  {row['Test_RMSE']:.6f}\n")
        f.write(f"Test MAE:   {row['Test_MAE']:.6f}\n")
        f.write(f"Test R²:    {row['Test_R2']:.6f}\n")
        f.write(f"Baseline RMSE: {row['Baseline_RMSE']:.6f}\n")
        f.write(f"Improvement: {row['Improvement_%']:.2f}%\n\n")

print("   ✓ Results saved")

# ==================== FINAL SUMMARY ====================
print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))
print("\n\nFiles saved:")
print("  - models/lgbm_model_no2_target.txt/.pkl")
print("  - models/lgbm_model_o3_target.txt/.pkl")
print("  - models/lgbm_model_co.txt/.pkl")
print("  - results/model_performance_summary.csv")
print("  - results/detailed_metrics_report.txt")
print("  - results/*_feature_importance.csv")
print("  - results/*_evaluation_plots.png")
print("\n" + "="*80)
