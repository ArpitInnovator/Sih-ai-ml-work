"""
Anti-Overfitting LightGBM Model Training
Fixes severe overfitting caused by distribution shift between train/test periods
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
import warnings
import pickle
import os
from datetime import datetime

# Optuna for hyperparameter tuning
try:
    import optuna
    from optuna.integration import LightGBMPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings('ignore')

print("="*80)
print("ANTI-OVERFITTING LIGHTGBM MODEL TRAINING")
print("="*80)

# ==================== LOAD DATA ====================
print("\n1. Loading and analyzing data...")
df = pd.read_csv('master_site1_final_cleaned.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# Check distribution shift
train_mask = (df['datetime'] >= '2019-06-01') & (df['datetime'] <= '2021-12-31')
test_mask = (df['datetime'] >= '2022-07-01') & (df['datetime'] <= '2022-12-31')

print("\n   Distribution Analysis:")
for target in ['NO2_target', 'O3_target']:
    if target in df.columns:
        train_mean = df[train_mask][target].mean()
        train_std = df[train_mask][target].std()
        test_mean = df[test_mask][target].mean()
        test_std = df[test_mask][target].std()
        shift_pct = ((test_mean - train_mean) / train_mean) * 100
        print(f"   {target}:")
        print(f"     Train: Mean={train_mean:.2f}, Std={train_std:.2f}")
        print(f"     Test:  Mean={test_mean:.2f}, Std={test_std:.2f}")
        print(f"     Shift: {shift_pct:.1f}% (mean), {((test_std-train_std)/train_std*100):.1f}% (std)")

# ==================== FEATURE ENGINEERING ====================
print("\n2. Creating features...")

# Time features
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Season
df['is_winter'] = (df['month'].isin([12, 1, 2])).astype(int)
df['is_summer'] = (df['month'].isin([3, 4, 5, 6])).astype(int)
df['is_monsoon'] = (df['month'].isin([7, 8, 9])).astype(int)

# Derived features
if 'wind_speed' not in df.columns and 'u10_era5' in df.columns and 'v10_era5' in df.columns:
    df['wind_speed'] = np.sqrt(df['u10_era5']**2 + df['v10_era5']**2)
if 'dewpoint_depression' not in df.columns and 't2m_era5' in df.columns and 'd2m_era5' in df.columns:
    df['dewpoint_depression'] = df['t2m_era5'] - df['d2m_era5']
if 'relative_humidity_approx' not in df.columns and 'dewpoint_depression' in df.columns:
    df['relative_humidity_approx'] = 100 * np.exp(-df['dewpoint_depression'] / 5)

# Interactions
if 'blh_era5' in df.columns and 'wind_speed' in df.columns:
    df['ventilation_rate'] = df['blh_era5'] / (df['wind_speed'] + 1e-6)

# O3 photochemical features
if 'solar_elevation' in df.columns and 't2m_era5' in df.columns:
    df['temp_solar_interaction'] = df['t2m_era5'] * np.abs(df['solar_elevation'])

# Lags (1h only, conservative)
for col in ['no2', 'pm2p5', 'pm10', 't2m_era5']:
    if col in df.columns:
        df[f'{col}_lag_1h'] = df[col].shift(1)

# Rolling means (3h, 6h)
for window in [3, 6]:
    for feat in ['no2', 'pm2p5', 'pm10']:
        if feat in df.columns:
            df[f'{feat}_rolling_mean_{window}h'] = df[feat].rolling(window=window, min_periods=1).mean()

print("   ✓ Features created")

# ==================== FEATURE SELECTION ====================
def get_features(df, target_name):
    """Conservative feature set"""
    features = []
    
    # Core pollutants
    features.extend([f for f in ['pm2p5', 'pm10', 'so2', 'no2'] if f in df.columns])
    
    # Meteorology (essential only)
    meteo = ['blh_era5', 't2m_era5', 'd2m_era5', 'wind_speed', 'u10_era5', 'v10_era5', 
             'relative_humidity_approx', 'tcc_era5', 'sp']
    features.extend([f for f in meteo if f in df.columns])
    
    # Lags (1h only)
    for feat in ['no2', 'pm2p5', 'pm10', 't2m_era5']:
        if f'{feat}_lag_1h' in df.columns:
            features.append(f'{feat}_lag_1h')
    
    # Rolling (3h, 6h)
    for window in [3, 6]:
        for feat in ['no2', 'pm2p5', 'pm10']:
            if f'{feat}_rolling_mean_{window}h' in df.columns:
                features.append(f'{feat}_rolling_mean_{window}h')
    
    # Time
    time_features = ['month', 'hour', 'day_of_week', 'is_weekend', 
                    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                    'is_winter', 'is_summer', 'is_monsoon']
    features.extend([f for f in time_features if f in df.columns])
    
    # Solar
    if 'solar_elevation' in df.columns:
        features.append('solar_elevation')
    
    # Interactions (conservative)
    if 'ventilation_rate' in df.columns:
        features.append('ventilation_rate')
    if target_name == 'O3_target' and 'temp_solar_interaction' in df.columns:
        features.append('temp_solar_interaction')
    
    return [f for f in features if f in df.columns]

# ==================== ANTI-OVERFITTING HYPERPARAMETERS ====================
def get_conservative_params():
    """Very conservative hyperparameters to prevent overfitting"""
    return {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15,  # Reduced from 31
        'max_depth': 5,    # Reduced from 7
        'learning_rate': 0.03,  # Reduced from 0.05
        'feature_fraction': 0.7,  # Reduced from 0.9
        'bagging_fraction': 0.7,  # Reduced from 0.8
        'bagging_freq': 5,
        'min_data_in_leaf': 50,  # Increased from 20
        'lambda_l1': 1.0,  # Increased from 0.1
        'lambda_l2': 1.0,  # Increased from 0.1
        'min_gain_to_split': 0.1,  # Added
        'max_bin': 255,
        'verbose': -1,
        'random_state': 42
    }

# ==================== TARGET NORMALIZATION ====================
def normalize_target(y_train, y_val, y_test):
    """Normalize targets to handle distribution shift"""
    scaler = RobustScaler()
    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_val_scaled = scaler.transform(y_val.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1)).ravel()
    return y_train_scaled, y_val_scaled, y_test_scaled, scaler

# ==================== PREPARE DATA ====================
def prepare_data(df, target_col, features, train_mask, val_mask, test_mask, normalize_targets=True):
    """Prepare data with target normalization"""
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
    
    # Normalize targets to handle distribution shift
    target_scaler = None
    if normalize_targets:
        y_train, y_val, y_test, target_scaler = normalize_target(y_train, y_val, y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, features, target_scaler

# ==================== TRAIN MODEL ====================
def train_anti_overfit_model(df, target_col, target_name, train_mask, val_mask, test_mask):
    """Train model with anti-overfitting measures"""
    print(f"\n{'='*80}")
    print(f"TRAINING ANTI-OVERFITTING MODEL: {target_name}")
    print(f"{'='*80}")
    
    features = get_features(df, target_name)
    print(f"   Features: {len(features)}")
    
    X_train, y_train, X_val, y_val, X_test, y_test, features, target_scaler = prepare_data(
        df, target_col, features, train_mask, val_mask, test_mask, normalize_targets=True
    )
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Use conservative hyperparameters
    params = get_conservative_params()
    print(f"   Using conservative hyperparameters (anti-overfitting)")
    
    # Train WITHOUT sample weights (they can cause overfitting)
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        num_boost_round=200,  # Reduced from 500
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),  # More aggressive early stopping
            lgb.log_evaluation(period=50)
        ]
    )
    
    # Predictions (on normalized scale)
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Inverse transform if normalized
    if target_scaler is not None:
        y_train = target_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
        y_val = target_scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
        y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_train_pred = target_scaler.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()
        y_val_pred = target_scaler.inverse_transform(y_val_pred.reshape(-1, 1)).ravel()
        y_test_pred = target_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()
    
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
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    # Baseline
    baseline_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean())))
    
    # Overfitting check
    train_test_gap = train_metrics['R2'] - test_metrics['R2']
    overfitting_warning = "⚠️ SEVERE OVERFITTING" if train_test_gap > 0.3 else "⚠️ MODERATE OVERFITTING" if train_test_gap > 0.15 else "✓ Good generalization"
    
    print(f"\n   Results:")
    print(f"   Train RMSE: {train_metrics['RMSE']:.4f}, R²: {train_metrics['R2']:.4f}")
    print(f"   Val RMSE:   {val_metrics['RMSE']:.4f}, R²: {val_metrics['R2']:.4f}")
    print(f"   Test RMSE:  {test_metrics['RMSE']:.4f}, R²: {test_metrics['R2']:.4f}")
    print(f"   Train-Test R² gap: {train_test_gap:.4f} {overfitting_warning}")
    print(f"   Baseline RMSE: {baseline_rmse:.4f}")
    print(f"   Improvement: {((baseline_rmse - test_metrics['RMSE']) / baseline_rmse * 100):.2f}%")
    
    return model, train_metrics, val_metrics, test_metrics, feature_importance, baseline_rmse, target_scaler

# ==================== MAIN ====================
print("\n3. Training anti-overfitting models...")

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

train_mask = (df['datetime'] >= '2019-06-01') & (df['datetime'] <= '2021-12-31')
val_mask = (df['datetime'] >= '2022-01-01') & (df['datetime'] <= '2022-06-30')
test_mask = (df['datetime'] >= '2022-07-01') & (df['datetime'] <= '2022-12-31')

results_summary = []

for target_name, target_col in [('NO2_target', 'NO2_target'), ('O3_target', 'O3_target'), ('CO', 'co')]:
    model, train_metrics, val_metrics, test_metrics, importance, baseline_rmse, target_scaler = train_anti_overfit_model(
        df, target_col, target_name, train_mask, val_mask, test_mask
    )
    
    # Save model + scaler
    model.save_model(f'models/anti_overfit_lgbm_{target_name}.txt')
    with open(f'models/anti_overfit_lgbm_{target_name}.pkl', 'wb') as f:
        pickle.dump((model, target_scaler), f)
    
    importance.to_csv(f'results/anti_overfit_{target_name}_feature_importance.csv', index=False)
    
    results_summary.append({
        'Model': target_name,
        'Train_RMSE': train_metrics['RMSE'],
        'Train_R2': train_metrics['R2'],
        'Val_RMSE': val_metrics['RMSE'],
        'Val_R2': val_metrics['R2'],
        'Test_RMSE': test_metrics['RMSE'],
        'Test_MAE': test_metrics['MAE'],
        'Test_R2': test_metrics['R2'],
        'Train_Test_R2_Gap': train_metrics['R2'] - test_metrics['R2'],
        'Baseline_RMSE': baseline_rmse,
        'Improvement_%': ((baseline_rmse - test_metrics['RMSE']) / baseline_rmse * 100)
    })

# Save results
results_df = pd.DataFrame(results_summary)
results_df.to_csv('results/anti_overfit_performance_summary.csv', index=False)

with open('results/anti_overfit_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("ANTI-OVERFITTING MODEL RESULTS\n")
    f.write("="*80 + "\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("ANTI-OVERFITTING MEASURES:\n")
    f.write("- Conservative hyperparameters (reduced complexity)\n")
    f.write("- Target normalization (RobustScaler) to handle distribution shift\n")
    f.write("- No sample weights (prevents overfitting to peaks)\n")
    f.write("- Aggressive early stopping\n")
    f.write("- Reduced number of features\n")
    f.write("- Increased regularization (lambda_l1=1.0, lambda_l2=1.0)\n")
    f.write("\n" + "-"*80 + "\n\n")
    
    for idx, row in results_df.iterrows():
        f.write(f"{row['Model']} MODEL\n")
        f.write("-"*80 + "\n")
        f.write(f"Train RMSE: {row['Train_RMSE']:.6f}, R²: {row['Train_R2']:.6f}\n")
        f.write(f"Val RMSE:   {row['Val_RMSE']:.6f}, R²: {row['Val_R2']:.6f}\n")
        f.write(f"Test RMSE:  {row['Test_RMSE']:.6f}, MAE: {row['Test_MAE']:.6f}, R²: {row['Test_R2']:.6f}\n")
        f.write(f"Train-Test R² Gap: {row['Train_Test_R2_Gap']:.6f}\n")
        f.write(f"Baseline RMSE: {row['Baseline_RMSE']:.6f}\n")
        f.write(f"Improvement: {row['Improvement_%']:.2f}%\n\n")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))
print("\n" + "="*80)

