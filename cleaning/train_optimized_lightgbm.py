"""
Optimized Single LightGBM Model Training
- One strong LightGBM model per pollutant (NO2, O3, CO)
- Clean feature set with proper selection
- Proper time-based train/val/test split
- Good lag and meteorological features
- Hyperparameter tuning (Optuna)
- Photochemical features for O3
- Feature pruning
- Target smoothing
- Peak-weighted loss
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Optuna for hyperparameter tuning
try:
    import optuna
    from optuna.integration import LightGBMPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Using default hyperparameters.")

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
print("OPTIMIZED SINGLE LIGHTGBM MODEL TRAINING")
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
df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

# Season features
def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5, 6]:
        return 'summer'
    elif month in [7, 8, 9]:
        return 'monsoon'
    else:
        return 'post_monsoon'

df['is_winter'] = (df['month'].isin([12, 1, 2])).astype(int)
df['is_summer'] = (df['month'].isin([3, 4, 5, 6])).astype(int)
df['is_monsoon'] = (df['month'].isin([7, 8, 9])).astype(int)
df['is_post_monsoon'] = (df['month'].isin([10, 11])).astype(int)

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

# Critical interaction features
if 'hour' in df.columns and 't2m_era5' in df.columns:
    df['hour_temp_interaction'] = df['hour'] * df['t2m_era5']
if 'blh_era5' in df.columns and 'wind_speed' in df.columns:
    df['blh_wind_interaction'] = df['blh_era5'] * df['wind_speed']
    df['ventilation_rate'] = df['blh_era5'] / (df['wind_speed'] + 1e-6)
if 't2m_era5' in df.columns and 'relative_humidity_approx' in df.columns:
    df['temp_humidity_interaction'] = df['t2m_era5'] * df['relative_humidity_approx']

# PHOTOCHEMICAL FEATURES FOR O3 (Critical)
print("   Creating photochemical features for O3...")
if 'solar_elevation' in df.columns:
    df['solar_elevation_abs'] = np.abs(df['solar_elevation'])
    if 'SZA_deg' in df.columns:
        df['photolysis_rate_approx'] = np.maximum(0, np.cos(np.radians(df['SZA_deg'])))
    else:
        df['photolysis_rate_approx'] = np.maximum(0, np.sin(np.radians(df['solar_elevation'])))
    if 't2m_era5' in df.columns:
        df['temp_sunlight_interaction'] = df['t2m_era5'] * df['photolysis_rate_approx']
        df['temp_solar_elevation'] = df['t2m_era5'] * df['solar_elevation_abs']
    df['weekend_traffic_reduction'] = df['is_weekend'] * (1 - df['is_rush_hour'])

# Lag features (1h, 3h for key pollutants)
key_features_for_lags = ['no2', 'pm2p5', 'pm10', 'so2', 't2m_era5', 'wind_speed']
for col in key_features_for_lags:
    if col in df.columns:
        for lag in [1, 3]:
            if f'{col}_lag_{lag}h' not in df.columns:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)

# Rolling means (3h, 6h, 12h)
rolling_features = ['no2', 'pm2p5', 'pm10', 'so2']
for window in [3, 6, 12]:
    for feat in rolling_features:
        if feat in df.columns:
            col_name = f'{feat}_rolling_mean_{window}h'
            if col_name not in df.columns:
                df[col_name] = df[feat].rolling(window=window, min_periods=1).mean()

print("   ✓ Features created")

# ==================== FEATURE SELECTION ====================
print("\n3. Selecting optimal feature set...")

def get_feature_list(df, target_name='NO2_target'):
    """Get clean, optimal feature list"""
    features = []
    
    # Basic pollutants (no CO/HCHO)
    pollutants = ['pm2p5', 'pm10', 'so2', 'no2', 'pm1', 'no']
    features.extend([f for f in pollutants if f in df.columns])
    
    # Satellite (only current NO2, no daily/flags)
    if 'NO2_satellite' in df.columns:
        features.append('NO2_satellite')
    
    # ERA5 meteorology (critical drivers)
    meteo = ['blh_era5', 't2m_era5', 'd2m_era5', 'dewpoint_depression', 
             'relative_humidity_approx', 'wind_speed', 'wind_direction_rad',
             'u10_era5', 'v10_era5', 'tcc_era5', 'sp']
    features.extend([f for f in meteo if f in df.columns])
    
    # Lag features (1h, 3h)
    for lag in [1, 3]:
        for feat in ['no2', 'pm2p5', 'pm10', 'so2', 't2m_era5', 'wind_speed']:
            col_name = f'{feat}_lag_{lag}h'
            if col_name in df.columns:
                features.append(col_name)
    
    # Rolling means (3h, 6h, 12h)
    for window in [3, 6, 12]:
        for feat in ['no2', 'pm2p5', 'pm10', 'so2']:
            col_name = f'{feat}_rolling_mean_{window}h'
            if col_name in df.columns:
                features.append(col_name)
    
    # Time features
    time_features = ['year', 'month', 'day', 'hour', 'day_of_week', 'day_of_year',
                     'is_weekend', 'is_weekday', 'hour_sin', 'hour_cos', 
                     'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos']
    features.extend([f for f in time_features if f in df.columns])
    
    # Season features
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

# ==================== TARGET SMOOTHING ====================
def smooth_target(y, window=3):
    """Apply rolling mean smoothing to target (optional)"""
    return pd.Series(y).rolling(window=window, min_periods=1, center=True).mean().values

# ==================== PEAK-WEIGHTED LOSS ====================
def calculate_sample_weights(y, percentile=75, weight_factor=2.0):
    """Calculate sample weights: higher weight for high-pollution events"""
    threshold = np.percentile(y, percentile)
    weights = np.ones(len(y))
    weights[y > threshold] = weight_factor
    return weights

# ==================== HYPERPARAMETER TUNING ====================
def optimize_hyperparameters(X_train, y_train, X_val, y_val, target_name, n_trials=50):
    """Optimize hyperparameters using Optuna"""
    if not OPTUNA_AVAILABLE:
        print(f"   Using default hyperparameters (Optuna not available)...")
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
    
    print(f"   Optimizing hyperparameters ({n_trials} trials)...")
    
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
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
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

# ==================== FEATURE PRUNING ====================
def prune_features(X_train, y_train, X_val, y_val, k=50):
    """Select top k features using f_regression"""
    selector = SelectKBest(score_func=f_regression, k=min(k, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    return X_train_selected, X_val_selected, selected_features, selector

# ==================== DATA PREPARATION ====================
def prepare_data(df, target_col, features, train_mask, val_mask, test_mask):
    """Prepare data with proper type conversion"""
    # Filter to valid rows
    valid_mask = ~df[target_col].isna()
    
    # Combine masks
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
    
    # Update features list
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
def train_optimized_model(df, target_col, target_name, train_mask, val_mask, test_mask):
    """Train optimized single LightGBM model"""
    print(f"\n{'='*80}")
    print(f"TRAINING OPTIMIZED MODEL: {target_name}")
    print(f"{'='*80}")
    
    # Get features
    features = get_feature_list(df, target_name)
    print(f"   Initial features: {len(features)}")
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_data(
        df, target_col, features, train_mask, val_mask, test_mask
    )
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Optional: Target smoothing (can be disabled)
    # y_train = smooth_target(y_train, window=3)
    
    # Feature pruning (select top k features)
    print(f"   Pruning features (selecting top 50)...")
    X_train_pruned, X_val_pruned, selected_features, selector = prune_features(
        X_train, y_train, X_val, y_val, k=50
    )
    X_test_pruned = selector.transform(X_test)
    
    # Convert back to DataFrame for LightGBM
    X_train_final = pd.DataFrame(X_train_pruned, columns=selected_features, index=X_train.index)
    X_val_final = pd.DataFrame(X_val_pruned, columns=selected_features, index=X_val.index)
    X_test_final = pd.DataFrame(X_test_pruned, columns=selected_features, index=X_test.index)
    
    print(f"   Selected features: {len(selected_features)}")
    
    # Optimize hyperparameters
    best_params = optimize_hyperparameters(
        X_train_final, y_train, X_val_final, y_val, target_name, n_trials=30
    )
    
    # Calculate sample weights (peak-weighted loss)
    sample_weights = calculate_sample_weights(y_train, percentile=75, weight_factor=2.0)
    
    # Train model
    print(f"   Training model with peak-weighted loss...")
    train_data = lgb.Dataset(X_train_final, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val_final, label=y_val, reference=train_data)
    
    model = lgb.train(
        best_params,
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
    y_train_pred = model.predict(X_train_final, num_iteration=model.best_iteration)
    y_val_pred = model.predict(X_val_final, num_iteration=model.best_iteration)
    y_test_pred = model.predict(X_test_final, num_iteration=model.best_iteration)
    
    # Calculate metrics
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
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    # Baseline
    baseline_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean())))
    
    # Print results
    print(f"\n   Results:")
    print(f"   Train RMSE: {train_metrics['RMSE']:.4f}, R²: {train_metrics['R2']:.4f}")
    print(f"   Val RMSE:   {val_metrics['RMSE']:.4f}, R²: {val_metrics['R2']:.4f}")
    print(f"   Test RMSE:  {test_metrics['RMSE']:.4f}, R²: {test_metrics['R2']:.4f}")
    print(f"   Baseline RMSE: {baseline_rmse:.4f}")
    print(f"   Improvement: {((baseline_rmse - test_metrics['RMSE']) / baseline_rmse * 100):.2f}%")
    
    return model, train_metrics, val_metrics, test_metrics, feature_importance, baseline_rmse, selector

# ==================== MAIN TRAINING ====================
print("\n4. Training optimized models...")

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Proper time-based splits
train_mask = (df['datetime'] >= '2019-06-01') & (df['datetime'] <= '2021-12-31')
val_mask = (df['datetime'] >= '2022-01-01') & (df['datetime'] <= '2022-06-30')
test_mask = (df['datetime'] >= '2022-07-01') & (df['datetime'] <= '2022-12-31')

results_summary = []

# Train models
for target_name, target_col in [('NO2_target', 'NO2_target'), ('O3_target', 'O3_target'), ('CO', 'co')]:
    model, train_metrics, val_metrics, test_metrics, importance, baseline_rmse, selector = train_optimized_model(
        df, target_col, target_name, train_mask, val_mask, test_mask
    )
    
    # Save model
    model.save_model(f'models/optimized_lgbm_{target_name}.txt')
    with open(f'models/optimized_lgbm_{target_name}.pkl', 'wb') as f:
        pickle.dump((model, selector), f)  # Save model + feature selector
    
    # Save feature importance
    importance.to_csv(f'results/optimized_{target_name}_feature_importance.csv', index=False)
    
    results_summary.append({
        'Model': target_name,
        'Train_RMSE': train_metrics['RMSE'],
        'Train_R2': train_metrics['R2'],
        'Val_RMSE': val_metrics['RMSE'],
        'Val_R2': val_metrics['R2'],
        'Test_RMSE': test_metrics['RMSE'],
        'Test_MAE': test_metrics['MAE'],
        'Test_R2': test_metrics['R2'],
        'Baseline_RMSE': baseline_rmse,
        'Improvement_%': ((baseline_rmse - test_metrics['RMSE']) / baseline_rmse * 100)
    })

# ==================== SAVE RESULTS ====================
print("\n5. Saving results...")

results_df = pd.DataFrame(results_summary)
results_df.to_csv('results/optimized_model_performance_summary.csv', index=False)

with open('results/optimized_metrics_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("OPTIMIZED SINGLE LIGHTGBM MODEL TRAINING RESULTS\n")
    f.write("="*80 + "\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("FEATURES:\n")
    f.write("- Clean feature set with proper selection\n")
    f.write("- Proper time-based train/val/test split\n")
    f.write("- Good lag features (1h, 3h)\n")
    f.write("- Good meteorological drivers\n")
    f.write("- Hyperparameter tuning (Optuna)\n")
    f.write("- Photochemical features for O3\n")
    f.write("- Feature pruning (top 50 features)\n")
    f.write("- Peak-weighted loss\n")
    f.write("\n" + "-"*80 + "\n\n")
    
    for idx, row in results_df.iterrows():
        f.write(f"{row['Model']} MODEL\n")
        f.write("-"*80 + "\n")
        f.write(f"Train RMSE: {row['Train_RMSE']:.6f}, R²: {row['Train_R2']:.6f}\n")
        f.write(f"Val RMSE:   {row['Val_RMSE']:.6f}, R²: {row['Val_R2']:.6f}\n")
        f.write(f"Test RMSE:  {row['Test_RMSE']:.6f}, MAE: {row['Test_MAE']:.6f}, R²: {row['Test_R2']:.6f}\n")
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
print("  - models/optimized_lgbm_*.txt/.pkl")
print("  - results/optimized_model_performance_summary.csv")
print("  - results/optimized_metrics_report.txt")
print("  - results/optimized_*_feature_importance.csv")
print("\n" + "="*80)

