"""
Complete Training, Evaluation, and Prediction Pipeline for SIH PS-10
Trains XGBoost models for O3 and NO2 prediction
"""

import pandas as pd
import numpy as np
try:
    import xgboost as xgb
except ImportError:
    print("Error: xgboost not installed. Please run: pip install xgboost")
    import sys
    sys.exit(1)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys
from pathlib import Path
import warnings
from datetime import datetime
import pickle
import joblib
warnings.filterwarnings('ignore')

# Global variable to hold output file handle
output_file_handle = None
original_stdout = sys.stdout

class TeeOutput:
    """Class to write to both console and file"""
    def __init__(self, file_handle):
        self.file = file_handle
        self.console = original_stdout
    
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.file.flush()
    
    def flush(self):
        self.console.flush()
        self.file.flush()

def setup_output_file(site_num, data_dir='Data_SIH_2025'):
    """Setup output file for logging results"""
    global output_file_handle
    
    output_dir = os.path.join(data_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'site_{site_num}_results.txt')
    
    output_file_handle = open(output_file, 'w', encoding='utf-8')
    sys.stdout = TeeOutput(output_file_handle)
    
    # Write header
    print("="*70)
    print(f"SIH PS-10 - Training and Evaluation Results")
    print(f"Site: {site_num}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print()
    
    return output_file

def close_output_file():
    """Close output file and restore stdout"""
    global output_file_handle, original_stdout
    
    if output_file_handle:
        sys.stdout = original_stdout
        output_file_handle.close()
        output_file_handle = None

def calculate_ria(y_true, y_pred):
    """
    Calculate R² Index of Agreement (RIA / d-statistic)
    RIA = 1 - (Σ(y_true - y_pred)²) / (Σ(|y_pred - mean(y_true)| + |y_true - mean(y_true)|)²)
    Ranges from 0 to 1, where 1 is perfect agreement
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true))) ** 2)
    
    if denominator == 0:
        return 0.0
    
    ria = 1 - (numerator / denominator)
    return ria

def calculate_bias(y_true, y_pred):
    """
    Calculate Bias (mean of residuals: predicted - actual)
    Bias = mean(y_pred - y_true)
    Positive bias means model overpredicts, negative means underpredicts
    """
    return np.mean(y_pred - y_true)

def prepare_features(df, is_training=True):
    """Prepare features for model training/prediction"""
    
    # Columns to exclude from features
    exclude_cols = ['O3_target', 'NO2_target', 'datetime']
    
    if not is_training:
        # For unseen data, we don't have targets, but we still exclude datetime
        exclude_cols = ['datetime']
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Select features
    X = df[feature_cols].copy()
    
    # Fill NaN values (for lag features at the beginning of time series)
    X = X.fillna(0)
    
    return X, feature_cols

def train_models(train_file, test_file, site_num):
    """Train XGBoost models for O3 and NO2"""
    
    print(f"\n{'='*70}")
    print(f"Training Models for Site {site_num}")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    print(f"Train set: {len(train_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    
    # Prepare features
    print("\nPreparing features...")
    X_train, feature_cols = prepare_features(train_df, is_training=True)
    X_test, _ = prepare_features(test_df, is_training=True)
    
    # Prepare targets
    y_train_o3 = train_df['O3_target'].values
    y_train_no2 = train_df['NO2_target'].values
    y_test_o3 = test_df['O3_target'].values
    y_test_no2 = test_df['NO2_target'].values
    
    print(f"Number of features: {len(feature_cols)}")
    
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Train O3 model
    print("\n" + "-"*70)
    print("Training O3 Model...")
    print("-"*70)
    model_o3 = xgb.XGBRegressor(**params)
    model_o3.fit(X_train, y_train_o3,
                 eval_set=[(X_test, y_test_o3)],
                 verbose=False)
    
    # Train NO2 model
    print("\n" + "-"*70)
    print("Training NO2 Model...")
    print("-"*70)
    model_no2 = xgb.XGBRegressor(**params)
    model_no2.fit(X_train, y_train_no2,
                  eval_set=[(X_test, y_test_no2)],
                  verbose=False)
    
    # Predictions
    print("\nMaking predictions...")
    y_pred_o3_train = model_o3.predict(X_train)
    y_pred_o3_test = model_o3.predict(X_test)
    
    y_pred_no2_train = model_no2.predict(X_train)
    y_pred_no2_test = model_no2.predict(X_test)
    
    # Calculate metrics for O3
    print("\n" + "="*70)
    print("O3 TARGET EVALUATION")
    print("="*70)
    
    # Train metrics
    rmse_train_o3 = np.sqrt(mean_squared_error(y_train_o3, y_pred_o3_train))
    mae_train_o3 = mean_absolute_error(y_train_o3, y_pred_o3_train)
    bias_train_o3 = calculate_bias(y_train_o3, y_pred_o3_train)
    r2_train_o3 = r2_score(y_train_o3, y_pred_o3_train)
    ria_train_o3 = calculate_ria(y_train_o3, y_pred_o3_train)
    
    # Test metrics
    rmse_test_o3 = np.sqrt(mean_squared_error(y_test_o3, y_pred_o3_test))
    mae_test_o3 = mean_absolute_error(y_test_o3, y_pred_o3_test)
    bias_test_o3 = calculate_bias(y_test_o3, y_pred_o3_test)
    r2_test_o3 = r2_score(y_test_o3, y_pred_o3_test)
    ria_test_o3 = calculate_ria(y_test_o3, y_pred_o3_test)
    
    # Baseline (O3_forecast)
    baseline_o3_test = test_df['O3_forecast'].values
    rmse_baseline_o3 = np.sqrt(mean_squared_error(y_test_o3, baseline_o3_test))
    mae_baseline_o3 = mean_absolute_error(y_test_o3, baseline_o3_test)
    bias_baseline_o3 = calculate_bias(y_test_o3, baseline_o3_test)
    r2_baseline_o3 = r2_score(y_test_o3, baseline_o3_test)
    ria_baseline_o3 = calculate_ria(y_test_o3, baseline_o3_test)
    
    print(f"\nTRAIN SET:")
    print(f"  RMSE: {rmse_train_o3:.4f}")
    print(f"  MAE:  {mae_train_o3:.4f}")
    print(f"  Bias: {bias_train_o3:.4f}")
    print(f"  R²:   {r2_train_o3:.4f}")
    print(f"  RIA:  {ria_train_o3:.4f}")
    
    print(f"\nTEST SET:")
    print(f"  RMSE: {rmse_test_o3:.4f}")
    print(f"  MAE:  {mae_test_o3:.4f}")
    print(f"  Bias: {bias_test_o3:.4f}")
    print(f"  R²:   {r2_test_o3:.4f}")
    print(f"  RIA:  {ria_test_o3:.4f}")
    
    print(f"\nBASELINE (O3_forecast) on TEST SET:")
    print(f"  RMSE: {rmse_baseline_o3:.4f}")
    print(f"  MAE:  {mae_baseline_o3:.4f}")
    print(f"  Bias: {bias_baseline_o3:.4f}")
    print(f"  R²:   {r2_baseline_o3:.4f}")
    print(f"  RIA:  {ria_baseline_o3:.4f}")
    
    improvement_o3 = ((rmse_baseline_o3 - rmse_test_o3) / rmse_baseline_o3) * 100
    print(f"\n  Improvement: {improvement_o3:.2f}% RMSE reduction")
    
    # Calculate metrics for NO2
    print("\n" + "="*70)
    print("NO2 TARGET EVALUATION")
    print("="*70)
    
    # Train metrics
    rmse_train_no2 = np.sqrt(mean_squared_error(y_train_no2, y_pred_no2_train))
    mae_train_no2 = mean_absolute_error(y_train_no2, y_pred_no2_train)
    bias_train_no2 = calculate_bias(y_train_no2, y_pred_no2_train)
    r2_train_no2 = r2_score(y_train_no2, y_pred_no2_train)
    ria_train_no2 = calculate_ria(y_train_no2, y_pred_no2_train)
    
    # Test metrics
    rmse_test_no2 = np.sqrt(mean_squared_error(y_test_no2, y_pred_no2_test))
    mae_test_no2 = mean_absolute_error(y_test_no2, y_pred_no2_test)
    bias_test_no2 = calculate_bias(y_test_no2, y_pred_no2_test)
    r2_test_no2 = r2_score(y_test_no2, y_pred_no2_test)
    ria_test_no2 = calculate_ria(y_test_no2, y_pred_no2_test)
    
    # Baseline (NO2_forecast)
    baseline_no2_test = test_df['NO2_forecast'].values
    rmse_baseline_no2 = np.sqrt(mean_squared_error(y_test_no2, baseline_no2_test))
    mae_baseline_no2 = mean_absolute_error(y_test_no2, baseline_no2_test)
    bias_baseline_no2 = calculate_bias(y_test_no2, baseline_no2_test)
    r2_baseline_no2 = r2_score(y_test_no2, baseline_no2_test)
    ria_baseline_no2 = calculate_ria(y_test_no2, baseline_no2_test)
    
    print(f"\nTRAIN SET:")
    print(f"  RMSE: {rmse_train_no2:.4f}")
    print(f"  MAE:  {mae_train_no2:.4f}")
    print(f"  Bias: {bias_train_no2:.4f}")
    print(f"  R²:   {r2_train_no2:.4f}")
    print(f"  RIA:  {ria_train_no2:.4f}")
    
    print(f"\nTEST SET:")
    print(f"  RMSE: {rmse_test_no2:.4f}")
    print(f"  MAE:  {mae_test_no2:.4f}")
    print(f"  Bias: {bias_test_no2:.4f}")
    print(f"  R²:   {r2_test_no2:.4f}")
    print(f"  RIA:  {ria_test_no2:.4f}")
    
    print(f"\nBASELINE (NO2_forecast) on TEST SET:")
    print(f"  RMSE: {rmse_baseline_no2:.4f}")
    print(f"  MAE:  {mae_baseline_no2:.4f}")
    print(f"  Bias: {bias_baseline_no2:.4f}")
    print(f"  R²:   {r2_baseline_no2:.4f}")
    print(f"  RIA:  {ria_baseline_no2:.4f}")
    
    improvement_no2 = ((rmse_baseline_no2 - rmse_test_no2) / rmse_baseline_no2) * 100
    print(f"\n  Improvement: {improvement_no2:.2f}% RMSE reduction")
    
    # Per-hour error analysis
    print("\n" + "="*70)
    print("PER-HOUR ERROR ANALYSIS")
    print("="*70)
    
    test_df_with_pred = test_df.copy()
    test_df_with_pred['O3_pred'] = y_pred_o3_test
    test_df_with_pred['NO2_pred'] = y_pred_no2_test
    test_df_with_pred['O3_error'] = np.abs(y_test_o3 - y_pred_o3_test)
    test_df_with_pred['NO2_error'] = np.abs(y_test_no2 - y_pred_no2_test)
    
    hourly_o3 = test_df_with_pred.groupby('hour')['O3_error'].mean()
    hourly_no2 = test_df_with_pred.groupby('hour')['NO2_error'].mean()
    
    print("\nAverage Absolute Error by Hour (O3):")
    for hour in range(24):
        if hour in hourly_o3.index:
            print(f"  Hour {hour:2d}: {hourly_o3[hour]:.4f}")
    
    print("\nAverage Absolute Error by Hour (NO2):")
    for hour in range(24):
        if hour in hourly_no2.index:
            print(f"  Hour {hour:2d}: {hourly_no2[hour]:.4f}")
    
    print(f"\nOverall Average Error - O3: {test_df_with_pred['O3_error'].mean():.4f}")
    print(f"Overall Average Error - NO2: {test_df_with_pred['NO2_error'].mean():.4f}")
    
    # Save results
    results = {
        'site': site_num,
        'O3_train_rmse': rmse_train_o3,
        'O3_train_mae': mae_train_o3,
        'O3_train_bias': bias_train_o3,
        'O3_train_r2': r2_train_o3,
        'O3_train_ria': ria_train_o3,
        'O3_test_rmse': rmse_test_o3,
        'O3_test_mae': mae_test_o3,
        'O3_test_bias': bias_test_o3,
        'O3_test_r2': r2_test_o3,
        'O3_test_ria': ria_test_o3,
        'O3_baseline_rmse': rmse_baseline_o3,
        'O3_baseline_mae': mae_baseline_o3,
        'O3_baseline_bias': bias_baseline_o3,
        'O3_improvement': improvement_o3,
        'NO2_train_rmse': rmse_train_no2,
        'NO2_train_mae': mae_train_no2,
        'NO2_train_bias': bias_train_no2,
        'NO2_train_r2': r2_train_no2,
        'NO2_train_ria': ria_train_no2,
        'NO2_test_rmse': rmse_test_no2,
        'NO2_test_mae': mae_test_no2,
        'NO2_test_bias': bias_test_no2,
        'NO2_test_r2': r2_test_no2,
        'NO2_test_ria': ria_test_no2,
        'NO2_baseline_rmse': rmse_baseline_no2,
        'NO2_baseline_mae': mae_baseline_no2,
        'NO2_baseline_bias': bias_baseline_no2,
        'NO2_improvement': improvement_no2,
    }
    
    return model_o3, model_no2, results, feature_cols

def process_unseen_data(unseen_file, site_num):
    """
    Process unseen input data for prediction
    Note: This needs to handle missing lag features differently since we don't have historical targets
    """
    print(f"\n{'='*70}")
    print(f"Processing Unseen Data for Site {site_num}")
    print(f"{'='*70}\n")
    
    # Load unseen data
    df = pd.read_csv(unseen_file)
    print(f"Loaded {len(df)} rows from unseen data")
    
    # Convert to datetime
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['day'] = df['day'].astype(int)
    df['hour'] = df['hour'].astype(int)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Handle satellite data (same as training)
    satellite_cols = ['NO2_satellite', 'HCHO_satellite', 'ratio_satellite']
    df['date'] = df['datetime'].dt.date
    
    for col in satellite_cols:
        if col in df.columns:
            df[col] = df.groupby('date')[col].transform(lambda x: x.ffill().bfill())
            flag_col = col.replace('_satellite', '_sat_flag')
            df[flag_col] = (~df[col].isnull()).astype(int)
    
    df = df.drop('date', axis=1)
    
    # For unseen data, we can't create lag features from targets (we don't have them)
    # Strategy: Use forecast values as proxies for lag features
    # This is a reasonable approximation since forecasts are available
    df['O3_target_lag1'] = df['O3_forecast'].shift(1).fillna(df['O3_forecast'].iloc[0])
    df['O3_target_lag24'] = df['O3_forecast'].shift(24).fillna(df['O3_forecast'].iloc[0])
    df['O3_target_lag168'] = df['O3_forecast'].shift(168).fillna(df['O3_forecast'].iloc[0])
    df['NO2_target_lag1'] = df['NO2_forecast'].shift(1).fillna(df['NO2_forecast'].iloc[0])
    df['NO2_target_lag24'] = df['NO2_forecast'].shift(24).fillna(df['NO2_forecast'].iloc[0])
    df['NO2_target_lag168'] = df['NO2_forecast'].shift(168).fillna(df['NO2_forecast'].iloc[0])
    
    print("Unseen data processed (using forecast values as lag feature proxies)")
    
    return df

def save_models(model_o3, model_no2, feature_cols, site_num, data_dir='Data_SIH_2025', format='pkl'):
    """
    Save trained models to file
    
    Args:
        model_o3: Trained O3 model
        model_no2: Trained NO2 model
        feature_cols: List of feature column names
        site_num: Site number
        data_dir: Data directory
        format: 'pkl' (pickle), 'joblib', or 'both'
    """
    models_dir = os.path.join(data_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_data = {
        'model_o3': model_o3,
        'model_no2': model_no2,
        'feature_cols': feature_cols,
        'site_num': site_num,
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if format in ['pkl', 'both']:
        # Save as pickle
        pkl_file = os.path.join(models_dir, f'site_{site_num}_models.pkl')
        with open(pkl_file, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n[OK] Models saved as pickle: {pkl_file}")
    
    if format in ['joblib', 'both']:
        # Save as joblib (better for sklearn/xgboost models)
        joblib_file = os.path.join(models_dir, f'site_{site_num}_models.joblib')
        joblib.dump(model_data, joblib_file)
        print(f"[OK] Models saved as joblib: {joblib_file}")
    
    # Also save feature columns separately for easy reference
    feature_file = os.path.join(models_dir, f'site_{site_num}_features.txt')
    with open(feature_file, 'w') as f:
        f.write('\n'.join(feature_cols))
    
    return models_dir

def load_models(site_num, data_dir='Data_SIH_2025', format='joblib'):
    """
    Load trained models from file
    
    Args:
        site_num: Site number
        data_dir: Data directory
        format: 'pkl' or 'joblib' (default: joblib)
    
    Returns:
        dict with 'model_o3', 'model_no2', 'feature_cols', 'site_num', 'saved_at'
    """
    models_dir = os.path.join(data_dir, 'models')
    
    if format == 'pkl':
        model_file = os.path.join(models_dir, f'site_{site_num}_models.pkl')
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
    else:  # joblib
        model_file = os.path.join(models_dir, f'site_{site_num}_models.joblib')
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        model_data = joblib.load(model_file)
    
    return model_data

def predict_unseen_data(unseen_df, model_o3, model_no2, feature_cols, site_num):
    """Predict O3 and NO2 for unseen data"""
    
    print(f"\n{'='*70}")
    print(f"Predicting for Unseen Data - Site {site_num}")
    print(f"{'='*70}\n")
    
    # Prepare features
    X_unseen, _ = prepare_features(unseen_df, is_training=False)
    
    # Ensure all feature columns are present
    for col in feature_cols:
        if col not in X_unseen.columns:
            X_unseen[col] = 0
            print(f"Warning: Added missing feature {col}")
    
    # Reorder columns to match training
    X_unseen = X_unseen[feature_cols]
    
    # Make predictions
    print("Making predictions...")
    predictions_o3 = model_o3.predict(X_unseen)
    predictions_no2 = model_no2.predict(X_unseen)
    
    # Create output dataframe
    output_df = unseen_df[['year', 'month', 'day', 'hour', 'datetime']].copy()
    output_df['O3_target'] = predictions_o3
    output_df['NO2_target'] = predictions_no2
    
    print(f"Generated predictions for {len(output_df)} rows")
    
    return output_df

def train_unified_model(data_dir='Data_SIH_2025'):
    """
    Train a single unified model on all sites combined
    Combines training data from all 7 sites and trains one model for O3 and one for NO2
    """
    print(f"\n{'='*70}")
    print("Training Unified Model on All Sites Combined")
    print(f"{'='*70}\n")
    
    # Setup output file
    result_file = setup_output_file("unified", data_dir)
    
    try:
        # Collect all training and test data from all sites
        all_train_data = []
        all_test_data = []
        
        print("Loading data from all sites...")
        for site_num in range(1, 8):
            train_file = os.path.join(data_dir, 'processed', f'site_{site_num}_train_processed.csv')
            test_file = os.path.join(data_dir, 'processed', f'site_{site_num}_test_processed.csv')
            
            if os.path.exists(train_file) and os.path.exists(test_file):
                train_df = pd.read_csv(train_file)
                test_df = pd.read_csv(test_file)
                
                # Add site identifier as a feature (optional, but can help)
                train_df['site_id'] = site_num
                test_df['site_id'] = site_num
                
                all_train_data.append(train_df)
                all_test_data.append(test_df)
                print(f"  Loaded Site {site_num}: {len(train_df)} train, {len(test_df)} test rows")
            else:
                print(f"  Warning: Site {site_num} files not found, skipping")
        
        if not all_train_data:
            print("Error: No training data found!")
            return None
        
        # Combine all training data
        print(f"\nCombining data from {len(all_train_data)} sites...")
        combined_train = pd.concat(all_train_data, ignore_index=True)
        combined_test = pd.concat(all_test_data, ignore_index=True)
        
        print(f"Combined training set: {len(combined_train)} rows")
        print(f"Combined test set: {len(combined_test)} rows")
        
        # Prepare features
        print("\nPreparing features...")
        X_train, feature_cols = prepare_features(combined_train, is_training=True)
        X_test, _ = prepare_features(combined_test, is_training=True)
        
        # Prepare targets
        y_train_o3 = combined_train['O3_target'].values
        y_train_no2 = combined_train['NO2_target'].values
        y_test_o3 = combined_test['O3_target'].values
        y_test_no2 = combined_test['NO2_target'].values
        
        print(f"Number of features: {len(feature_cols)}")
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train O3 model
        print("\n" + "-"*70)
        print("Training Unified O3 Model...")
        print("-"*70)
        model_o3 = xgb.XGBRegressor(**params)
        model_o3.fit(X_train, y_train_o3,
                     eval_set=[(X_test, y_test_o3)],
                     verbose=False)
        
        # Train NO2 model
        print("\n" + "-"*70)
        print("Training Unified NO2 Model...")
        print("-"*70)
        model_no2 = xgb.XGBRegressor(**params)
        model_no2.fit(X_train, y_train_no2,
                      eval_set=[(X_test, y_test_no2)],
                      verbose=False)
        
        # Make predictions
        print("\nMaking predictions...")
        y_pred_o3_train = model_o3.predict(X_train)
        y_pred_o3_test = model_o3.predict(X_test)
        y_pred_no2_train = model_no2.predict(X_train)
        y_pred_no2_test = model_no2.predict(X_test)
        
        # Calculate metrics for O3
        print("\n" + "="*70)
        print("UNIFIED O3 MODEL EVALUATION")
        print("="*70)
        
        rmse_train_o3 = np.sqrt(mean_squared_error(y_train_o3, y_pred_o3_train))
        mae_train_o3 = mean_absolute_error(y_train_o3, y_pred_o3_train)
        bias_train_o3 = calculate_bias(y_train_o3, y_pred_o3_train)
        r2_train_o3 = r2_score(y_train_o3, y_pred_o3_train)
        ria_train_o3 = calculate_ria(y_train_o3, y_pred_o3_train)
        
        rmse_test_o3 = np.sqrt(mean_squared_error(y_test_o3, y_pred_o3_test))
        mae_test_o3 = mean_absolute_error(y_test_o3, y_pred_o3_test)
        bias_test_o3 = calculate_bias(y_test_o3, y_pred_o3_test)
        r2_test_o3 = r2_score(y_test_o3, y_pred_o3_test)
        ria_test_o3 = calculate_ria(y_test_o3, y_pred_o3_test)
        
        baseline_o3_test = combined_test['O3_forecast'].values
        rmse_baseline_o3 = np.sqrt(mean_squared_error(y_test_o3, baseline_o3_test))
        mae_baseline_o3 = mean_absolute_error(y_test_o3, baseline_o3_test)
        bias_baseline_o3 = calculate_bias(y_test_o3, baseline_o3_test)
        r2_baseline_o3 = r2_score(y_test_o3, baseline_o3_test)
        ria_baseline_o3 = calculate_ria(y_test_o3, baseline_o3_test)
        
        print(f"\nTRAIN SET:")
        print(f"  RMSE: {rmse_train_o3:.4f}")
        print(f"  MAE:  {mae_train_o3:.4f}")
        print(f"  Bias: {bias_train_o3:.4f}")
        print(f"  R²:   {r2_train_o3:.4f}")
        print(f"  RIA:  {ria_train_o3:.4f}")
        
        print(f"\nTEST SET:")
        print(f"  RMSE: {rmse_test_o3:.4f}")
        print(f"  MAE:  {mae_test_o3:.4f}")
        print(f"  Bias: {bias_test_o3:.4f}")
        print(f"  R²:   {r2_test_o3:.4f}")
        print(f"  RIA:  {ria_test_o3:.4f}")
        
        print(f"\nBASELINE (O3_forecast) on TEST SET:")
        print(f"  RMSE: {rmse_baseline_o3:.4f}")
        print(f"  MAE:  {mae_baseline_o3:.4f}")
        print(f"  Bias: {bias_baseline_o3:.4f}")
        print(f"  R²:   {r2_baseline_o3:.4f}")
        print(f"  RIA:  {ria_baseline_o3:.4f}")
        
        improvement_o3 = ((rmse_baseline_o3 - rmse_test_o3) / rmse_baseline_o3) * 100
        print(f"\n  Improvement: {improvement_o3:.2f}% RMSE reduction")
        
        # Calculate metrics for NO2
        print("\n" + "="*70)
        print("UNIFIED NO2 MODEL EVALUATION")
        print("="*70)
        
        rmse_train_no2 = np.sqrt(mean_squared_error(y_train_no2, y_pred_no2_train))
        mae_train_no2 = mean_absolute_error(y_train_no2, y_pred_no2_train)
        bias_train_no2 = calculate_bias(y_train_no2, y_pred_no2_train)
        r2_train_no2 = r2_score(y_train_no2, y_pred_no2_train)
        ria_train_no2 = calculate_ria(y_train_no2, y_pred_no2_train)
        
        rmse_test_no2 = np.sqrt(mean_squared_error(y_test_no2, y_pred_no2_test))
        mae_test_no2 = mean_absolute_error(y_test_no2, y_pred_no2_test)
        bias_test_no2 = calculate_bias(y_test_no2, y_pred_no2_test)
        r2_test_no2 = r2_score(y_test_no2, y_pred_no2_test)
        ria_test_no2 = calculate_ria(y_test_no2, y_pred_no2_test)
        
        baseline_no2_test = combined_test['NO2_forecast'].values
        rmse_baseline_no2 = np.sqrt(mean_squared_error(y_test_no2, baseline_no2_test))
        mae_baseline_no2 = mean_absolute_error(y_test_no2, baseline_no2_test)
        bias_baseline_no2 = calculate_bias(y_test_no2, baseline_no2_test)
        r2_baseline_no2 = r2_score(y_test_no2, baseline_no2_test)
        ria_baseline_no2 = calculate_ria(y_test_no2, baseline_no2_test)
        
        print(f"\nTRAIN SET:")
        print(f"  RMSE: {rmse_train_no2:.4f}")
        print(f"  MAE:  {mae_train_no2:.4f}")
        print(f"  Bias: {bias_train_no2:.4f}")
        print(f"  R²:   {r2_train_no2:.4f}")
        print(f"  RIA:  {ria_train_no2:.4f}")
        
        print(f"\nTEST SET:")
        print(f"  RMSE: {rmse_test_no2:.4f}")
        print(f"  MAE:  {mae_test_no2:.4f}")
        print(f"  Bias: {bias_test_no2:.4f}")
        print(f"  R²:   {r2_test_no2:.4f}")
        print(f"  RIA:  {ria_test_no2:.4f}")
        
        print(f"\nBASELINE (NO2_forecast) on TEST SET:")
        print(f"  RMSE: {rmse_baseline_no2:.4f}")
        print(f"  MAE:  {mae_baseline_no2:.4f}")
        print(f"  Bias: {bias_baseline_no2:.4f}")
        print(f"  R²:   {r2_baseline_no2:.4f}")
        print(f"  RIA:  {ria_baseline_no2:.4f}")
        
        improvement_no2 = ((rmse_baseline_no2 - rmse_test_no2) / rmse_baseline_no2) * 100
        print(f"\n  Improvement: {improvement_no2:.2f}% RMSE reduction")
        
        # Retrain on full data (train + test combined)
        print(f"\n{'='*70}")
        print("Retraining on Full Combined Data")
        print(f"{'='*70}\n")
        
        combined_full = pd.concat([combined_train, combined_test], ignore_index=True)
        X_train_full, _ = prepare_features(combined_full, is_training=True)
        y_train_full_o3 = combined_full['O3_target'].values
        y_train_full_no2 = combined_full['NO2_target'].values
        
        model_o3 = xgb.XGBRegressor(**params)
        model_o3.fit(X_train_full, y_train_full_o3, verbose=False)
        
        model_no2 = xgb.XGBRegressor(**params)
        model_no2.fit(X_train_full, y_train_full_no2, verbose=False)
        
        print("Unified models retrained on full combined data")
        
        # Save unified models
        print("\nSaving unified models...")
        save_models(model_o3, model_no2, feature_cols, "unified", data_dir, format='both')
        
        print(f"\n{'='*70}")
        print(f"[OK] Unified model training complete!")
        print(f"[OK] Results saved to: {result_file}")
        print(f"{'='*70}")
        
        return {
            'model_o3': model_o3,
            'model_no2': model_no2,
            'feature_cols': feature_cols
        }
    finally:
        close_output_file()

def process_site(site_num, data_dir='Data_SIH_2025', retrain_on_full=False):
    """Complete pipeline for one site"""
    
    # Setup output file for logging
    result_file = setup_output_file(site_num, data_dir)
    
    try:
        # File paths
        train_file = os.path.join(data_dir, 'processed', f'site_{site_num}_train_processed.csv')
        test_file = os.path.join(data_dir, 'processed', f'site_{site_num}_test_processed.csv')
        unseen_file = os.path.join(data_dir, f'site_{site_num}_unseen_input_data.csv')
        
        # Check if files exist
        if not os.path.exists(train_file):
            print(f"Error: {train_file} not found!")
            print(f"\n[OK] Results saved to: {result_file}")
            return None
        
        # STEP 1-3: Train and evaluate
        model_o3, model_no2, results, feature_cols = train_models(train_file, test_file, site_num)
        
        # STEP 4: Retrain on full training data (train + test combined)
        if retrain_on_full:
            print(f"\n{'='*70}")
            print(f"Retraining on Full Training Data - Site {site_num}")
            print(f"{'='*70}\n")
            
            # Combine train and test
            train_full = pd.read_csv(train_file)
            test_full = pd.read_csv(test_file)
            train_full = pd.concat([train_full, test_full], ignore_index=True)
            
            X_train_full, _ = prepare_features(train_full, is_training=True)
            y_train_full_o3 = train_full['O3_target'].values
            y_train_full_no2 = train_full['NO2_target'].values
            
            # Retrain models
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
            
            model_o3 = xgb.XGBRegressor(**params)
            model_o3.fit(X_train_full, y_train_full_o3, verbose=False)
            
            model_no2 = xgb.XGBRegressor(**params)
            model_no2.fit(X_train_full, y_train_full_no2, verbose=False)
            
            print("Models retrained on full training data")
            
            # Save trained models
            print("\nSaving trained models...")
            save_models(model_o3, model_no2, feature_cols, site_num, data_dir, format='both')
        
        # STEP 5: Predict for unseen data
        if os.path.exists(unseen_file):
            unseen_df = process_unseen_data(unseen_file, site_num)
            predictions_df = predict_unseen_data(unseen_df, model_o3, model_no2, feature_cols, site_num)
            
            # Save predictions
            output_dir = os.path.join(data_dir, 'predictions')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'site_{site_num}_predictions.csv')
            predictions_df.to_csv(output_file, index=False)
            print(f"\n[OK] Predictions saved to: {output_file}")
        else:
            print(f"\nWarning: Unseen file {unseen_file} not found, skipping predictions")
        
        print(f"\n{'='*70}")
        print(f"[OK] All results saved to: {result_file}")
        print(f"{'='*70}")
        
        return results
    finally:
        # Always close output file and restore stdout
        close_output_file()

def main():
    """Main function to process all sites or a single site"""
    import sys
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        # Check if user wants unified model
        if arg == 'unified' or arg == 'all' or arg == 'combined':
            try:
                train_unified_model()
            except Exception as e:
                close_output_file()
                print(f"\n[ERROR] Error training unified model: {str(e)}\n")
                import traceback
                traceback.print_exc()
        else:
            # Single site training
            site_num = int(arg)
            try:
                process_site(site_num, retrain_on_full=True)
            except Exception as e:
                close_output_file()  # Ensure file is closed on error
                print(f"\n[ERROR] Error processing Site {site_num}: {str(e)}\n")
                import traceback
                traceback.print_exc()
    else:
        # Default: Train unified model on all sites combined
        print("="*70)
        print("No site specified - Training Unified Model on All Sites")
        print("="*70)
        print("\nTo train individual sites, use: python train_and_evaluate.py <site_num>")
        print("To explicitly train unified model: python train_and_evaluate.py unified\n")
        try:
            train_unified_model()
        except Exception as e:
            close_output_file()
            print(f"\n[ERROR] Error training unified model: {str(e)}\n")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

