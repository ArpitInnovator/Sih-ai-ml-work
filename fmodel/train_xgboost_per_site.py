"""
XGBoost Model Training Script - Per Site Models (FULLY OPTIMIZED VERSION)
Trains separate XGBoost models for each site (7 sites total).
Each site model predicts both O3_target and NO2_target.

ALL 6 CRITICAL IMPROVEMENTS APPLIED:
1. ✅ Added lag features (NO2/O3_target_lag1, lag2, lag3, lag6, lag12)
2. ✅ Added rolling window features (NO2_roll3, roll6, roll12, O3_roll3, etc.)
3. ✅ Fixed satellite noise (smooth and drop raw columns)
4. ✅ Improved temporal features (sin/cos encoding for hour and month)
5. ✅ True time-series split (Jan-Oct train, Nov val, Dec test)
6. ✅ Optimized hyperparameters (max_depth=5, etc.)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_DIR = "Data_SIH_2025_with_blh/Data_SIH_2025_with_blh/with_satellite"
MODEL_DIR = "models/xgboost_per_site"
N_SITES = 7

# Features to exclude from training (ONLY targets, keep temporal features)
EXCLUDE_FEATURES = ['O3_target', 'NO2_target', 'site_id']

# XGBoost hyperparameters - OPTIMIZED
XGB_PARAMS = {
    'O3': {
        'objective': 'reg:squarederror',
        'n_estimators': 300,
        'max_depth': 5,  # Optimized from 3
        'learning_rate': 0.1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 5,
        'gamma': 0.3,  # Increased from 0.2
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'random_state': 42,
        'n_jobs': -1
    },
    'NO2': {
        'objective': 'reg:squarederror',
        'n_estimators': 300,
        'max_depth': 5,  # Optimized from 3
        'learning_rate': 0.1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 5,
        'gamma': 0.3,  # Increased from 0.2
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'random_state': 42,
        'n_jobs': -1
    }
}


def load_site_data(data_dir, site_id):
    """Load training data for a specific site."""
    train_file = os.path.join(data_dir, f"site_{site_id}_train_data_with_satellite.csv")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    print(f"Loading site {site_id} data...")
    df = pd.read_csv(train_file)
    
    # Ensure data is sorted by year, month, day, hour (temporal order)
    if all(col in df.columns for col in ['year', 'month', 'day', 'hour']):
        df = df.sort_values(by=['year', 'month', 'day', 'hour']).reset_index(drop=True)
        print(f"  Data sorted by temporal order")
    
    print(f"  Loaded {len(df)} samples from site {site_id}")
    
    return df


def add_lag_features(df):
    """Add lag features for NO2_target and O3_target (STEP 1)."""
    df = df.copy()
    
    lag_periods = [1, 2, 3, 6, 12]
    
    # Add lag features for NO2_target
    if 'NO2_target' in df.columns:
        for lag in lag_periods:
            df[f'NO2_target_lag{lag}'] = df['NO2_target'].shift(lag)
    
    # Add lag features for O3_target
    if 'O3_target' in df.columns:
        for lag in lag_periods:
            df[f'O3_target_lag{lag}'] = df['O3_target'].shift(lag)
    
    print(f"  Added {len(lag_periods) * 2} lag features")
    return df


def add_rolling_features(df):
    """Add rolling window features (STEP 2)."""
    df = df.copy()
    
    # Rolling features for NO2_target
    if 'NO2_target' in df.columns:
        for window in [3, 6, 12]:
            df[f'NO2_roll{window}'] = df['NO2_target'].rolling(window=window, min_periods=1).mean()
    
    # Rolling features for O3_target
    if 'O3_target' in df.columns:
        for window in [3, 6, 12]:
            df[f'O3_roll{window}'] = df['O3_target'].rolling(window=window, min_periods=1).mean()
    
    # Rolling features for temperature (T_forecast)
    if 'T_forecast' in df.columns:
        for window in [3]:
            df[f'temp_roll{window}'] = df['T_forecast'].rolling(window=window, min_periods=1).mean()
    
    # Rolling features for relative humidity (q_forecast as proxy)
    if 'q_forecast' in df.columns:
        for window in [3]:
            df[f'rh_roll{window}'] = df['q_forecast'].rolling(window=window, min_periods=1).mean()
    
    # Rolling features for wind speed
    if 'wind_speed' in df.columns:
        for window in [3]:
            df[f'ws_roll{window}'] = df['wind_speed'].rolling(window=window, min_periods=1).mean()
    
    print(f"  Added rolling window features")
    return df


def smooth_satellite_data(df):
    """Smooth satellite data and drop raw columns (STEP 3)."""
    df = df.copy()
    
    satellite_cols = ['NO2_satellite_filled', 'HCHO_satellite_filled', 'ratio_satellite']
    
    for col in satellite_cols:
        if col in df.columns:
            # Apply rolling mean with window=3, min_periods=1
            df[f'{col}_smooth'] = df[col].rolling(window=3, min_periods=1, center=True).mean()
            # Drop the original raw column
            df = df.drop(col, axis=1)
    
    print(f"  Smoothed and dropped raw satellite columns")
    return df


def add_temporal_features(df):
    """Add improved temporal features with sin/cos encoding (STEP 4)."""
    df = df.copy()
    
    # Create datetime column if year, month, day, hour exist
    if all(col in df.columns for col in ['year', 'month', 'day', 'hour']):
        # Convert to datetime
        df['datetime'] = pd.to_datetime(
            df[['year', 'month', 'day']].astype(int).astype(str).agg('-'.join, axis=1) + 
            ' ' + df['hour'].astype(int).astype(str) + ':00:00',
            errors='coerce'
        )
        
        # Add day of week (0=Monday, 6=Sunday)
        df['dayofweek'] = df['datetime'].dt.dayofweek
        
        # Add season (1=Spring, 2=Summer, 3=Fall, 4=Winter)
        df['season'] = df['datetime'].dt.month % 12 // 3 + 1
        
        # Add is_weekend
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Sin/Cos encoding for hour (STEP 4)
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Sin/Cos encoding for month (STEP 4)
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Drop datetime column (not needed for training)
        df = df.drop('datetime', axis=1)
    
    print(f"  Added temporal features with sin/cos encoding")
    return df


def prepare_features(df):
    """Prepare features and targets for training with all improvements."""
    # Step 1: Add lag features (MUST be done before other operations)
    df = add_lag_features(df)
    
    # Step 2: Add rolling window features
    df = add_rolling_features(df)
    
    # Step 3: Smooth satellite data and drop raw columns
    df = smooth_satellite_data(df)
    
    # Step 4: Add temporal features with sin/cos encoding
    df = add_temporal_features(df)
    
    # Get all feature columns (exclude targets)
    feature_cols = [col for col in df.columns if col not in EXCLUDE_FEATURES]
    
    # Check if targets exist
    if 'O3_target' not in df.columns or 'NO2_target' not in df.columns:
        raise ValueError("Target columns (O3_target, NO2_target) not found in data!")
    
    X = df[feature_cols].copy()
    y_o3 = df['O3_target'].copy()
    y_no2 = df['NO2_target'].copy()
    
    # Handle missing values (lag features will have NaN at the beginning)
    X = X.fillna(X.median())
    
    # Check for infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"  Total features used: {len(feature_cols)}")
    lag_features = [col for col in feature_cols if 'lag' in col or 'roll' in col]
    print(f"  Lag & rolling features: {len(lag_features)}")
    
    return X, y_o3, y_no2, feature_cols


def split_time_series_by_month(df, X, y_o3, y_no2):
    """
    Split time-series data by month (STEP 5).
    Train: Jan-Oct, Val: Nov, Test: Dec
    """
    # Create month mask
    train_mask = df['month'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Jan-Oct
    val_mask = df['month'] == 11  # November
    test_mask = df['month'] == 12  # December
    
    # Split features
    X_train = X[train_mask].copy()
    X_val = X[val_mask].copy()
    X_test = X[test_mask].copy()
    
    # Split targets using SAME masks
    y_o3_train = y_o3[train_mask].copy()
    y_o3_val = y_o3[val_mask].copy()
    y_o3_test = y_o3[test_mask].copy()
    
    y_no2_train = y_no2[train_mask].copy()
    y_no2_val = y_no2[val_mask].copy()
    y_no2_test = y_no2[test_mask].copy()
    
    # If no November/December data, fall back to 80/20 split
    if len(X_val) == 0 or len(X_test) == 0:
        print(f"  ⚠️  No Nov/Dec data found, using 80/20 temporal split")
        n_samples = len(X)
        split_idx = int(0.8 * n_samples)
        X_train = X.iloc[:split_idx].copy()
        X_val = X.iloc[split_idx:].copy()
        X_test = X_val.copy()  # Use val as test if no separate test set
        
        y_o3_train = y_o3.iloc[:split_idx].copy()
        y_o3_val = y_o3.iloc[split_idx:].copy()
        y_o3_test = y_o3_val.copy()
        
        y_no2_train = y_no2.iloc[:split_idx].copy()
        y_no2_val = y_no2.iloc[split_idx:].copy()
        y_no2_test = y_no2_val.copy()
    
    return (X_train, X_val, X_test, 
            y_o3_train, y_o3_val, y_o3_test,
            y_no2_train, y_no2_val, y_no2_test)


def train_model(X_train, y_train, X_val, y_val, target_name, params):
    """Train XGBoost model for a single target with early stopping and track metrics."""
    # Early stopping parameters
    early_stopping_rounds = 30
    min_rounds = 20
    
    # Create base params without n_estimators for incremental training
    base_params = {k: v for k, v in params.items() if k != 'n_estimators'}
    max_rounds = params.get('n_estimators', 300)
    
    # Lists to store metrics at each iteration
    iterations = []
    train_rmse_history = []
    train_r2_history = []
    val_rmse_history = []
    val_r2_history = []
    
    # Initialize model
    model = xgb.XGBRegressor(**base_params, n_estimators=min_rounds)
    
    # Manual early stopping: train incrementally and monitor validation loss
    best_val_rmse = float('inf')
    best_model = None
    rounds_without_improvement = 0
    best_iteration = min_rounds
    
    # Train model first (simplified approach)
    print(f"  Training model and tracking metrics...")
    
    try:
        # Try using callbacks for newer XGBoost versions (2.0+)
        callbacks = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True)]
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            callbacks=callbacks
        )
        best_iteration = getattr(model, 'best_iteration', max_rounds)
    except (AttributeError, TypeError):
        # Fallback: Manual early stopping
        try:
            # Try early_stopping_rounds in fit (XGBoost 1.x)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
            best_iteration = getattr(model, 'best_iteration', max_rounds)
        except TypeError:
            # Manual early stopping implementation
            print(f"  Using manual early stopping (max {max_rounds} rounds, patience {early_stopping_rounds})")
            
            # Train incrementally
            for round_num in range(min_rounds, max_rounds + 1, 10):
                # Update n_estimators
                temp_model = xgb.XGBRegressor(**base_params, n_estimators=round_num)
                temp_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                # Evaluate on validation set
                y_val_pred = temp_model.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                
                # Check for improvement
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_model = temp_model
                    best_iteration = round_num
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 10
                    if rounds_without_improvement >= early_stopping_rounds:
                        print(f"  Early stopping at round {round_num} (best: {best_iteration})")
                        break
            
            # Use best model
            if best_model is not None:
                model = best_model
            else:
                # If no improvement, use the last model
                model = xgb.XGBRegressor(**base_params, n_estimators=best_iteration)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # After training, calculate metrics at different iteration points for plotting
    print(f"  Calculating metrics at each iteration for plotting...")
    actual_iterations = min(best_iteration, max_rounds)
    
    # Sample iterations (every 5 iterations for efficiency, or all if < 100)
    step = 1 if actual_iterations < 100 else 5
    
    for iteration in range(1, actual_iterations + 1, step):
        try:
            # Get predictions at this iteration using iteration_range
            y_train_pred_iter = model.predict(X_train, iteration_range=(0, iteration))
            y_val_pred_iter = model.predict(X_val, iteration_range=(0, iteration))
            
            # Calculate metrics
            train_rmse_iter = np.sqrt(mean_squared_error(y_train, y_train_pred_iter))
            train_r2_iter = r2_score(y_train, y_train_pred_iter)
            val_rmse_iter = np.sqrt(mean_squared_error(y_val, y_val_pred_iter))
            val_r2_iter = r2_score(y_val, y_val_pred_iter)
            
            iterations.append(iteration)
            train_rmse_history.append(train_rmse_iter)
            train_r2_history.append(train_r2_iter)
            val_rmse_history.append(val_rmse_iter)
            val_r2_history.append(val_r2_iter)
        except Exception:
            # If iteration_range doesn't work, skip this iteration
            continue
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_bias = np.mean(y_train_pred - y_train)
    
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    val_bias = np.mean(y_val_pred - y_val)
    
    print(f"  {target_name} - Train R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, Bias: {train_bias:.4f}")
    print(f"  {target_name} - Val R²:   {val_r2:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, Bias: {val_bias:.4f}")
    
    # Check for overfitting
    if train_r2 - val_r2 > 0.15:
        print(f"  ⚠️  WARNING: Potential overfitting detected (Train R² - Val R² = {train_r2 - val_r2:.4f})")
    
    # Store training history
    training_history = {
        'iterations': iterations,
        'train_rmse': train_rmse_history,
        'train_r2': train_r2_history,
        'val_rmse': val_rmse_history,
        'val_r2': val_r2_history
    }
    
    return model, {
        'train_rmse': float(train_rmse),
        'train_mae': float(train_mae),
        'train_bias': float(train_bias),
        'train_r2': float(train_r2),
        'val_rmse': float(val_rmse),
        'val_mae': float(val_mae),
        'val_bias': float(val_bias),
        'val_r2': float(val_r2),
        'best_iteration': best_iteration,
        'training_history': training_history
    }


def get_feature_importance(model, feature_names, top_n=20):
    """Get top N feature importances."""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)


def plot_training_curves(training_history, site_id, target_name, model_dir, timestamp):
    """Plot R² and RMSE curves throughout training iterations."""
    if not training_history or not training_history.get('iterations'):
        print(f"  ⚠️  No training history available for plotting")
        return
    
    iterations = training_history['iterations']
    train_rmse = training_history['train_rmse']
    train_r2 = training_history['train_r2']
    val_rmse = training_history['val_rmse']
    val_r2 = training_history['val_r2']
    
    # Create plots directory
    plots_dir = os.path.join(model_dir, f"site_{site_id}", "training_curves")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 5)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: RMSE over iterations
    ax1.plot(iterations, train_rmse, label='Train RMSE', linewidth=2, color='blue', alpha=0.7)
    ax1.plot(iterations, val_rmse, label='Validation RMSE', linewidth=2, color='red', alpha=0.7)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title(f'{target_name} - RMSE vs Iteration (Site {site_id})', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # Plot 2: R² over iterations
    ax2.plot(iterations, train_r2, label='Train R²', linewidth=2, color='blue', alpha=0.7)
    ax2.plot(iterations, val_r2, label='Validation R²', linewidth=2, color='red', alpha=0.7)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title(f'{target_name} - R² vs Iteration (Site {site_id})', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0, top=1.1)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(plots_dir, f"{target_name}_training_curves_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Training curves saved: {plot_path}")
    
    # Also create combined plot with both metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot RMSE (left y-axis)
    ax2_twin = ax.twinx()
    line1 = ax.plot(iterations, train_r2, label='Train R²', linewidth=2, color='blue', alpha=0.7)
    line2 = ax.plot(iterations, val_r2, label='Val R²', linewidth=2, color='red', alpha=0.7)
    line3 = ax2_twin.plot(iterations, train_rmse, label='Train RMSE', linewidth=2, color='blue', 
                          linestyle='--', alpha=0.7)
    line4 = ax2_twin.plot(iterations, val_rmse, label='Val RMSE', linewidth=2, color='red', 
                          linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('R²', fontsize=12, color='black')
    ax2_twin.set_ylabel('RMSE', fontsize=12, color='black')
    ax.set_title(f'{target_name} - Training Curves (Site {site_id})', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=1.1)
    
    plt.tight_layout()
    
    # Save combined plot
    plot_path_combined = os.path.join(plots_dir, f"{target_name}_combined_curves_{timestamp}.png")
    plt.savefig(plot_path_combined, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Combined curves saved: {plot_path_combined}")


def save_site_models(site_id, o3_model, no2_model, feature_cols, o3_metrics, no2_metrics, 
                     o3_importance, no2_importance, model_dir, timestamp, 
                     o3_training_history=None, no2_training_history=None):
    """Save models and metadata for a site."""
    os.makedirs(model_dir, exist_ok=True)
    site_dir = os.path.join(model_dir, f"site_{site_id}")
    os.makedirs(site_dir, exist_ok=True)
    
    # Save models
    o3_model_path = os.path.join(site_dir, f"O3_model_{timestamp}.pkl")
    no2_model_path = os.path.join(site_dir, f"NO2_model_{timestamp}.pkl")
    
    joblib.dump(o3_model, o3_model_path)
    joblib.dump(no2_model, no2_model_path)
    
    print(f"  Models saved to: {site_dir}")
    
    # Save metadata
    metadata = {
        'site_id': site_id,
        'timestamp': timestamp,
        'n_features': len(feature_cols),
        'features': feature_cols,
        'O3_metrics': o3_metrics,
        'NO2_metrics': no2_metrics,
        'model_type': 'XGBoost',
        'hyperparameters': {
            'O3': XGB_PARAMS['O3'],
            'NO2': XGB_PARAMS['NO2']
        },
        'improvements_applied': [
            'Added lag features (NO2/O3_target_lag1, lag2, lag3, lag6, lag12)',
            'Added rolling window features (NO2_roll3, roll6, roll12, O3_roll3, etc.)',
            'Fixed satellite noise (smooth and drop raw columns)',
            'Improved temporal features (sin/cos encoding for hour and month)',
            'True time-series split (Jan-Oct train, Nov val, Dec test)',
            'Optimized hyperparameters (max_depth=5, etc.)'
        ]
    }
    
    metadata_path = os.path.join(site_dir, f"metadata_{timestamp}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save feature importance
    o3_importance_path = os.path.join(site_dir, f"O3_importance_{timestamp}.csv")
    no2_importance_path = os.path.join(site_dir, f"NO2_importance_{timestamp}.csv")
    o3_importance.to_csv(o3_importance_path, index=False)
    no2_importance.to_csv(no2_importance_path, index=False)
    
    # Save standardized metrics (RMSE, MAE, Bias) in CSV format
    metrics_data = {
        'Site': [site_id, site_id],
        'Target': ['O3', 'NO2'],
        'RMSE': [o3_metrics['val_rmse'], no2_metrics['val_rmse']],
        'MAE': [o3_metrics['val_mae'], no2_metrics['val_mae']],
        'Bias': [o3_metrics['val_bias'], no2_metrics['val_bias']],
        'R²': [o3_metrics['val_r2'], no2_metrics['val_r2']]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv_path = os.path.join(site_dir, f"metrics_{timestamp}.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"  Standardized metrics saved to: {metrics_csv_path}")
    
    # Plot training curves
    if o3_training_history:
        print(f"\n  Plotting O3 training curves...")
        plot_training_curves(o3_training_history, site_id, 'O3', model_dir, timestamp)
    
    if no2_training_history:
        print(f"  Plotting NO2 training curves...")
        plot_training_curves(no2_training_history, site_id, 'NO2', model_dir, timestamp)
    
    return site_dir


def train_site_model(site_id, data_dir, model_dir, timestamp):
    """Train models for a single site."""
    print(f"\n{'='*70}")
    print(f"Training models for Site {site_id}")
    print(f"{'='*70}")
    
    # Load site data
    df = load_site_data(data_dir, site_id)
    
    # Prepare features (includes all 6 improvements)
    X, y_o3, y_no2, feature_cols = prepare_features(df)
    
    # STEP 5: Split time-series data by month (Jan-Oct train, Nov val, Dec test)
    print(f"\n  Splitting data by month: Train (Jan-Oct), Val (Nov), Test (Dec)")
    (X_train, X_val, X_test,
     y_o3_train, y_o3_val, y_o3_test,
     y_no2_train, y_no2_val, y_no2_test) = split_time_series_by_month(df, X, y_o3, y_no2)
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  ✅ Using SAME split for O3 and NO2 targets")
    
    # Train O3 model
    print(f"\n  Training O3 model...")
    o3_model, o3_metrics = train_model(
        X_train, y_o3_train, X_val, y_o3_val, 'O3', XGB_PARAMS['O3']
    )
    o3_training_history = o3_metrics.get('training_history', None)
    
    # Train NO2 model
    print(f"\n  Training NO2 model...")
    no2_model, no2_metrics = train_model(
        X_train, y_no2_train, X_val, y_no2_val, 'NO2', XGB_PARAMS['NO2']
    )
    no2_training_history = no2_metrics.get('training_history', None)
    
    # Feature importance
    o3_importance = get_feature_importance(o3_model, feature_cols, top_n=20)
    no2_importance = get_feature_importance(no2_model, feature_cols, top_n=20)
    
    # Save models
    site_dir = save_site_models(
        site_id, o3_model, no2_model, feature_cols,
        o3_metrics, no2_metrics, o3_importance, no2_importance,
        model_dir, timestamp,
        o3_training_history=o3_training_history,
        no2_training_history=no2_training_history
    )
    
    return {
        'site_id': site_id,
        'o3_metrics': o3_metrics,
        'no2_metrics': no2_metrics,
        'n_samples': len(df),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test)
    }


def main():
    """Main training function."""
    print("="*70)
    print("XGBoost Model Training - Per Site Models (O3 and NO2)")
    print("FULLY OPTIMIZED VERSION - All 6 Critical Improvements Applied")
    print("="*70)
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Create timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Train models for each site
    all_results = []
    
    for site_id in range(1, N_SITES + 1):
        try:
            result = train_site_model(site_id, DATA_DIR, MODEL_DIR, timestamp)
            all_results.append(result)
        except Exception as e:
            print(f"\nError training site {site_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print("Training Summary - All Sites")
    print(f"{'='*70}")
    print(f"Timestamp: {timestamp}")
    print(f"\n{'Site':<6} {'Target':<8} {'RMSE':<12} {'MAE':<12} {'Bias':<12} {'R²':<10} {'Overfit?':<10}")
    print("-" * 70)
    
    # Prepare data for comprehensive metrics CSV
    all_metrics_data = []
    
    for result in all_results:
        site_id = result['site_id']
        o3_metrics = result['o3_metrics']
        no2_metrics = result['no2_metrics']
        
        # Check overfitting
        o3_overfit = "Yes" if o3_metrics['train_r2'] - o3_metrics['val_r2'] > 0.15 else "No"
        no2_overfit = "Yes" if no2_metrics['train_r2'] - no2_metrics['val_r2'] > 0.15 else "No"
        
        # O3 metrics
        print(f"Site {site_id:<3} {'O3':<8} {o3_metrics['val_rmse']:<12.4f} {o3_metrics['val_mae']:<12.4f} {o3_metrics['val_bias']:<12.4f} {o3_metrics['val_r2']:<10.4f} {o3_overfit:<10}")
        all_metrics_data.append({
            'Site': site_id,
            'Target': 'O3',
            'RMSE': o3_metrics['val_rmse'],
            'MAE': o3_metrics['val_mae'],
            'Bias': o3_metrics['val_bias'],
            'R²': o3_metrics['val_r2'],
            'Train_RMSE': o3_metrics['train_rmse'],
            'Train_MAE': o3_metrics['train_mae'],
            'Train_Bias': o3_metrics['train_bias'],
            'Train_R²': o3_metrics['train_r2'],
            'Overfitting': o3_overfit,
            'Best_Iteration': o3_metrics.get('best_iteration', 'N/A')
        })
        
        # NO2 metrics
        print(f"Site {site_id:<3} {'NO2':<8} {no2_metrics['val_rmse']:<12.4f} {no2_metrics['val_mae']:<12.4f} {no2_metrics['val_bias']:<12.4f} {no2_metrics['val_r2']:<10.4f} {no2_overfit:<10}")
        all_metrics_data.append({
            'Site': site_id,
            'Target': 'NO2',
            'RMSE': no2_metrics['val_rmse'],
            'MAE': no2_metrics['val_mae'],
            'Bias': no2_metrics['val_bias'],
            'R²': no2_metrics['val_r2'],
            'Train_RMSE': no2_metrics['train_rmse'],
            'Train_MAE': no2_metrics['train_mae'],
            'Train_Bias': no2_metrics['train_bias'],
            'Train_R²': no2_metrics['train_r2'],
            'Overfitting': no2_overfit,
            'Best_Iteration': no2_metrics.get('best_iteration', 'N/A')
        })
    
    # Save comprehensive metrics CSV
    metrics_df = pd.DataFrame(all_metrics_data)
    metrics_csv_path = os.path.join(MODEL_DIR, f"all_sites_metrics_{timestamp}.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nComprehensive metrics saved to: {metrics_csv_path}")
    
    # Save overall summary JSON
    summary_path = os.path.join(MODEL_DIR, f"training_summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'n_sites': len(all_results),
            'results': all_results,
            'improvements_applied': [
                'Added lag features (NO2/O3_target_lag1, lag2, lag3, lag6, lag12)',
                'Added rolling window features (NO2_roll3, roll6, roll12, O3_roll3, etc.)',
                'Fixed satellite noise (smooth and drop raw columns)',
                'Improved temporal features (sin/cos encoding for hour and month)',
                'True time-series split (Jan-Oct train, Nov val, Dec test)',
                'Optimized hyperparameters (max_depth=5, etc.)'
            ]
        }, f, indent=2)
    
    print(f"Summary JSON saved to: {summary_path}")
    print(f"All models saved in: {MODEL_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
