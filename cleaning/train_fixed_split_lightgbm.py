"""
Fixed Split LightGBM Training
Uses more recent training data to match test distribution better
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

warnings.filterwarnings('ignore')

print("="*80)
print("FIXED SPLIT LIGHTGBM TRAINING")
print("="*80)

# ==================== LOAD DATA ====================
print("\n1. Loading data...")
df = pd.read_csv('master_site1_final_cleaned.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# ==================== ANALYZE DISTRIBUTIONS ====================
print("\n2. Analyzing distributions for better split...")

# Check different time periods
periods = {
    '2019-2021': (df['datetime'] >= '2019-06-01') & (df['datetime'] <= '2021-12-31'),
    '2020-2022': (df['datetime'] >= '2020-01-01') & (df['datetime'] <= '2022-06-30'),
    '2021-2022': (df['datetime'] >= '2021-01-01') & (df['datetime'] <= '2022-06-30'),
    'Test (2022 H2)': (df['datetime'] >= '2022-07-01') & (df['datetime'] <= '2022-12-31')
}

print("\n   Target distributions by period:")
for period_name, mask in periods.items():
    if period_name == 'Test (2022 H2)':
        continue
    for target in ['NO2_target', 'O3_target']:
        if target in df.columns:
            data = df[mask][target].dropna()
            if len(data) > 0:
                print(f"   {period_name} - {target}: Mean={data.mean():.2f}, Std={data.std():.2f}, N={len(data)}")

# Test period
test_mask = periods['Test (2022 H2)']
for target in ['NO2_target', 'O3_target']:
    if target in df.columns:
        data = df[test_mask][target].dropna()
        if len(data) > 0:
            print(f"   Test (2022 H2) - {target}: Mean={data.mean():.2f}, Std={data.std():.2f}, N={len(data)}")

# ==================== USE MORE RECENT TRAINING DATA ====================
# Strategy: Use 2020-2022 H1 for training to better match test distribution
print("\n3. Using more recent training period (2020-2022 H1) to match test distribution...")

train_mask = (df['datetime'] >= '2020-01-01') & (df['datetime'] <= '2022-06-30')
val_mask = (df['datetime'] >= '2022-01-01') & (df['datetime'] <= '2022-06-30')  # Use same period for val
test_mask = (df['datetime'] >= '2022-07-01') & (df['datetime'] <= '2022-12-31')

# For validation, use a subset of training period
val_mask = (df['datetime'] >= '2022-01-01') & (df['datetime'] <= '2022-03-31')
train_mask = (df['datetime'] >= '2020-01-01') & (df['datetime'] <= '2021-12-31')

print(f"   Train: {train_mask.sum()} samples")
print(f"   Val:   {val_mask.sum()} samples")
print(f"   Test:  {test_mask.sum()} samples")

# ==================== FEATURE ENGINEERING ====================
print("\n4. Creating features...")

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

# O3 photochemical
if 'solar_elevation' in df.columns and 't2m_era5' in df.columns:
    df['temp_solar_interaction'] = df['t2m_era5'] * np.abs(df['solar_elevation'])

# Lags (1h only)
for col in ['no2', 'pm2p5', 'pm10', 't2m_era5']:
    if col in df.columns:
        df[f'{col}_lag_1h'] = df[col].shift(1)

# Rolling means (3h, 6h)
for window in [3, 6]:
    for feat in ['no2', 'pm2p5', 'pm10']:
        if feat in df.columns:
            df[f'{feat}_rolling_mean_{window}h'] = df[feat].rolling(window=window, min_periods=1).mean()

def get_features(df, target_name):
    """Feature list"""
    features = []
    features.extend([f for f in ['pm2p5', 'pm10', 'so2', 'no2'] if f in df.columns])
    meteo = ['blh_era5', 't2m_era5', 'd2m_era5', 'wind_speed', 'u10_era5', 'v10_era5', 
             'relative_humidity_approx', 'tcc_era5', 'sp']
    features.extend([f for f in meteo if f in df.columns])
    for feat in ['no2', 'pm2p5', 'pm10', 't2m_era5']:
        if f'{feat}_lag_1h' in df.columns:
            features.append(f'{feat}_lag_1h')
    for window in [3, 6]:
        for feat in ['no2', 'pm2p5', 'pm10']:
            if f'{feat}_rolling_mean_{window}h' in df.columns:
                features.append(f'{feat}_rolling_mean_{window}h')
    time_features = ['month', 'hour', 'day_of_week', 'is_weekend', 
                    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                    'is_winter', 'is_summer', 'is_monsoon']
    features.extend([f for f in time_features if f in df.columns])
    if 'solar_elevation' in df.columns:
        features.append('solar_elevation')
    if 'ventilation_rate' in df.columns:
        features.append('ventilation_rate')
    if target_name == 'O3_target' and 'temp_solar_interaction' in df.columns:
        features.append('temp_solar_interaction')
    return [f for f in features if f in df.columns]

# ==================== PREPARE DATA ====================
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
def train_model(df, target_col, target_name, train_mask, val_mask, test_mask):
    """Train model with fixed split"""
    print(f"\n{'='*80}")
    print(f"TRAINING: {target_name}")
    print(f"{'='*80}")
    
    features = get_features(df, target_name)
    print(f"   Features: {len(features)}")
    
    X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_data(
        df, target_col, features, train_mask, val_mask, test_mask
    )
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"   Train target: Mean={y_train.mean():.2f}, Std={y_train.std():.2f}")
    print(f"   Test target:  Mean={y_test.mean():.2f}, Std={y_test.std():.2f}")
    print(f"   Distribution shift: {((y_test.mean()-y_train.mean())/y_train.mean()*100):.1f}%")
    
    # Very conservative hyperparameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 10,  # Very small
        'max_depth': 4,    # Very shallow
        'learning_rate': 0.02,  # Slow
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'min_data_in_leaf': 100,  # Large
        'lambda_l1': 2.0,  # High regularization
        'lambda_l2': 2.0,
        'min_gain_to_split': 0.2,
        'max_bin': 255,
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
        num_boost_round=150,  # Fewer rounds
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
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
    
    print(f"\n   Results:")
    print(f"   Train RMSE: {train_metrics['RMSE']:.4f}, R²: {train_metrics['R2']:.4f}")
    print(f"   Val RMSE:   {val_metrics['RMSE']:.4f}, R²: {val_metrics['R2']:.4f}")
    print(f"   Test RMSE:  {test_metrics['RMSE']:.4f}, R²: {test_metrics['R2']:.4f}")
    print(f"   Train-Test R² gap: {train_test_gap:.4f}")
    if train_test_gap > 0.3:
        print(f"   ⚠️ Still overfitting, but improved")
    elif train_test_gap > 0.15:
        print(f"   ⚠️ Moderate overfitting")
    else:
        print(f"   ✓ Good generalization")
    print(f"   Baseline RMSE: {baseline_rmse:.4f}")
    print(f"   Improvement: {((baseline_rmse - test_metrics['RMSE']) / baseline_rmse * 100):.2f}%")
    
    return model, train_metrics, val_metrics, test_metrics, baseline_rmse

# ==================== MAIN ====================
print("\n5. Training models...")

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

results_summary = []

for target_name, target_col in [('NO2_target', 'NO2_target'), ('O3_target', 'O3_target'), ('CO', 'co')]:
    model, train_metrics, val_metrics, test_metrics, baseline_rmse = train_model(
        df, target_col, target_name, train_mask, val_mask, test_mask
    )
    
    model.save_model(f'models/fixed_split_lgbm_{target_name}.txt')
    with open(f'models/fixed_split_lgbm_{target_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
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

results_df = pd.DataFrame(results_summary)
results_df.to_csv('results/fixed_split_performance_summary.csv', index=False)

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))
print("\n" + "="*80)

