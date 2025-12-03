import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

print("="*80)
print("FEATURE ENGINEERING AND DATA ANALYSIS")
print("="*80)

# Load the cleaned dataset
print("\n1. Loading dataset...")
df = pd.read_csv('master_site1_final_cleaned.csv')
print(f"   Original shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# Convert datetime column
if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

# ==================== FEATURE ENGINEERING ====================
print("\n2. Performing Feature Engineering...")

# Time-based features
if 'datetime' in df.columns:
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding for time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    print("   ✓ Time-based features created")

# Weather interaction features
if 't2m_era5' in df.columns and 'd2m_era5' in df.columns:
    df['dewpoint_depression'] = df['t2m_era5'] - df['d2m_era5']
    df['relative_humidity_approx'] = 100 * (np.exp((17.27 * df['d2m_era5']) / (237.7 + df['d2m_era5'])) / 
                                           np.exp((17.27 * df['t2m_era5']) / (237.7 + df['t2m_era5'])))
    print("   ✓ Temperature-humidity features created")

# Wind features
if 'u10_era5' in df.columns and 'v10_era5' in df.columns:
    df['wind_magnitude'] = np.sqrt(df['u10_era5']**2 + df['v10_era5']**2)
    df['wind_direction_rad'] = np.arctan2(df['v10_era5'], df['u10_era5'])
    df['wind_u_squared'] = df['u10_era5']**2
    df['wind_v_squared'] = df['v10_era5']**2
    print("   ✓ Wind features created")

# Air quality ratios and interactions
if 'no2' in df.columns and 'pm2p5' in df.columns:
    df['no2_pm25_ratio'] = df['no2'] / (df['pm2p5'] + 1e-10)
    print("   ✓ Air quality ratios created")

if 'pm2p5' in df.columns and 'pm10' in df.columns:
    df['pm25_pm10_ratio'] = df['pm2p5'] / (df['pm10'] + 1e-10)
    print("   ✓ PM ratios created")

# Satellite features
if 'NO2_satellite' in df.columns and 'HCHO_satellite' in df.columns:
    df['satellite_ratio'] = df['NO2_satellite'] / (df['HCHO_satellite'] + 1e-10)
    print("   ✓ Satellite interaction features created")

# AOD features
if 'aod550' in df.columns and 'bcaod550' in df.columns:
    df['aod_ratio'] = df['bcaod550'] / (df['aod550'] + 1e-10)
    print("   ✓ AOD features created")

# Boundary layer features
if 'blh_era5' in df.columns and 'wind_speed' in df.columns:
    df['blh_wind_interaction'] = df['blh_era5'] * df['wind_speed']
    print("   ✓ Boundary layer features created")

# Solar features
if 'solar_elevation' in df.columns:
    df['solar_elevation_squared'] = df['solar_elevation']**2
    df['is_daytime'] = (df['solar_elevation'] > 0).astype(int)
    print("   ✓ Solar features created")

# Pressure features
if 'sp' in df.columns:
    df['pressure_normalized'] = (df['sp'] - df['sp'].mean()) / df['sp'].std()
    print("   ✓ Pressure features created")

# Lag features (if datetime exists)
if 'datetime' in df.columns:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target columns from lag features
    target_cols = ['NO2_target', 'O3_target', 'co', 'hcho']
    numeric_cols = [col for col in numeric_cols if col not in target_cols]
    
    # Create 1-hour and 3-hour lag features for key variables
    key_features = ['no2', 'pm2p5', 'pm10', 'so2', 'co', 't2m_era5', 'wind_speed']
    for col in key_features:
        if col in df.columns:
            df[f'{col}_lag_1h'] = df[col].shift(1)
            df[f'{col}_lag_3h'] = df[col].shift(3)
    print("   ✓ Lag features created")

# Rolling statistics
if 'datetime' in df.columns:
    key_features = ['no2', 'pm2p5', 'NO2_target', 'O3_target']
    for col in key_features:
        if col in df.columns:
            df[f'{col}_rolling_mean_6h'] = df[col].rolling(window=6, min_periods=1).mean()
            df[f'{col}_rolling_std_6h'] = df[col].rolling(window=6, min_periods=1).std()
    print("   ✓ Rolling statistics created")

print(f"\n   Final shape after feature engineering: {df.shape}")
print(f"   Total features: {df.shape[1]}")

# ==================== DATA SUMMARY ====================
print("\n3. Generating Data Summary...")

summary_stats = df.describe()
print(f"\n   Basic Statistics:")
print(f"   - Total rows: {len(df)}")
print(f"   - Total columns: {len(df.columns)}")
print(f"   - Missing values: {df.isnull().sum().sum()}")
print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ==================== CORRELATION MATRIX ====================
print("\n4. Creating Correlation Matrix Heatmap...")

# Select numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Calculate correlation matrix
corr_matrix = df[numeric_cols].corr()

# Create figure with subplots
fig, axes = plt.subplots(1, 1, figsize=(20, 16))

# Create heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, 
            fmt='.2f', vmin=-1, vmax=1, ax=axes)

axes.set_title('Correlation Matrix Heatmap - All Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_matrix_heatmap.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: correlation_matrix_heatmap.png")

# ==================== TARGET CORRELATIONS ====================
print("\n5. Creating Target Variable Correlation Analysis...")

targets = ['NO2_target', 'O3_target', 'co', 'hcho']
targets_present = [t for t in targets if t in df.columns]

if targets_present:
    # Get top correlations for each target
    fig, axes = plt.subplots(len(targets_present), 1, figsize=(14, 5*len(targets_present)))
    if len(targets_present) == 1:
        axes = [axes]
    
    for idx, target in enumerate(targets_present):
        if target in corr_matrix.columns:
            target_corr = corr_matrix[target].drop(target).sort_values(ascending=False)
            
            # Top 20 positive and negative correlations
            top_pos = target_corr.head(15)
            top_neg = target_corr.tail(15)
            
            axes[idx].barh(range(len(top_pos)), top_pos.values, color='green', alpha=0.7, label='Positive')
            axes[idx].barh(range(len(top_pos), len(top_pos) + len(top_neg)), 
                          top_neg.values, color='red', alpha=0.7, label='Negative')
            axes[idx].set_yticks(range(len(top_pos) + len(top_neg)))
            axes[idx].set_yticklabels(list(top_pos.index) + list(top_neg.index))
            axes[idx].set_xlabel('Correlation Coefficient')
            axes[idx].set_title(f'Top Correlations with {target}', fontsize=12, fontweight='bold')
            axes[idx].legend()
            axes[idx].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            axes[idx].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('target_correlations.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: target_correlations.png")

# ==================== DISTRIBUTION ANALYSIS ====================
print("\n6. Creating Distribution Analysis...")

# Distribution of target variables
if targets_present:
    n_targets = len(targets_present)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, target in enumerate(targets_present[:4]):
        if target in df.columns:
            axes[idx].hist(df[target].dropna(), bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {target}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(target)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_targets, 4):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('target_distributions.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: target_distributions.png")

# ==================== MISSING VALUES ANALYSIS ====================
print("\n7. Creating Missing Values Analysis...")

missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
missing_pct = (missing_data / len(df)) * 100

if len(missing_data) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    top_missing = missing_data.head(20)
    ax.barh(range(len(top_missing)), top_missing.values)
    ax.set_yticks(range(len(top_missing)))
    ax.set_yticklabels(top_missing.index)
    ax.set_xlabel('Number of Missing Values')
    ax.set_title('Top 20 Features with Missing Values', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('missing_values_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: missing_values_analysis.png")
else:
    print("   ✓ No missing values found!")

# ==================== FEATURE IMPORTANCE (Correlation-based) ====================
print("\n8. Creating Feature Importance Analysis...")

if targets_present:
    fig, axes = plt.subplots(len(targets_present), 1, figsize=(14, 5*len(targets_present)))
    if len(targets_present) == 1:
        axes = [axes]
    
    for idx, target in enumerate(targets_present):
        if target in corr_matrix.columns:
            # Get absolute correlations
            target_corr_abs = corr_matrix[target].drop(target).abs().sort_values(ascending=False)
            top_features = target_corr_abs.head(25)
            
            axes[idx].barh(range(len(top_features)), top_features.values, color='steelblue', alpha=0.8)
            axes[idx].set_yticks(range(len(top_features)))
            axes[idx].set_yticklabels(top_features.index)
            axes[idx].set_xlabel('Absolute Correlation')
            axes[idx].set_title(f'Top 25 Most Important Features for {target} (by Correlation)', 
                              fontsize=12, fontweight='bold')
            axes[idx].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: feature_importance.png")

# ==================== TIME SERIES ANALYSIS ====================
print("\n9. Creating Time Series Analysis...")

if 'datetime' in df.columns and targets_present:
    fig, axes = plt.subplots(len(targets_present), 1, figsize=(16, 5*len(targets_present)))
    if len(targets_present) == 1:
        axes = [axes]
    
    for idx, target in enumerate(targets_present):
        if target in df.columns:
            # Sample data for visualization (every 10th point for performance)
            sample_df = df[['datetime', target]].iloc[::10].copy()
            axes[idx].plot(sample_df['datetime'], sample_df[target], linewidth=0.5, alpha=0.7)
            axes[idx].set_title(f'Time Series: {target}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Date')
            axes[idx].set_ylabel(target)
            axes[idx].grid(alpha=0.3)
            axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: time_series_analysis.png")

# ==================== STATISTICAL SUMMARY ====================
print("\n10. Generating Statistical Summary Report...")

summary_report = []
summary_report.append("="*80)
summary_report.append("FEATURE ENGINEERING AND ANALYSIS REPORT")
summary_report.append("="*80)
summary_report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
summary_report.append(f"\nDataset Information:")
summary_report.append(f"  - Original shape: {df.shape}")
summary_report.append(f"  - Total features: {df.shape[1]}")
summary_report.append(f"  - Total samples: {df.shape[0]}")
summary_report.append(f"  - Missing values: {df.isnull().sum().sum()}")
summary_report.append(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

summary_report.append(f"\n\nTarget Variables:")
for target in targets_present:
    if target in df.columns:
        summary_report.append(f"  - {target}:")
        summary_report.append(f"    Mean: {df[target].mean():.4f}")
        summary_report.append(f"    Std: {df[target].std():.4f}")
        summary_report.append(f"    Min: {df[target].min():.4f}")
        summary_report.append(f"    Max: {df[target].max():.4f}")
        summary_report.append(f"    Missing: {df[target].isnull().sum()} ({df[target].isnull().sum()/len(df)*100:.2f}%)")

summary_report.append(f"\n\nTop Correlated Features (by absolute correlation):")
if targets_present:
    for target in targets_present[:2]:  # Show for first 2 targets
        if target in corr_matrix.columns:
            target_corr_abs = corr_matrix[target].drop(target).abs().sort_values(ascending=False)
            summary_report.append(f"\n  {target}:")
            for i, (feature, corr) in enumerate(target_corr_abs.head(10).items()):
                actual_corr = corr_matrix[target][feature]
                summary_report.append(f"    {i+1}. {feature}: {actual_corr:.4f}")

summary_report.append(f"\n\nFeature Categories:")
feature_categories = {
    'Time Features': [col for col in df.columns if col in ['year', 'month', 'day', 'hour', 'day_of_week', 
                                                           'hour_sin', 'hour_cos', 'month_sin', 'month_cos']],
    'Weather Features': [col for col in df.columns if any(x in col.lower() for x in ['t2m', 'd2m', 'wind', 'blh', 'tcc', 'sp'])],
    'Air Quality Features': [col for col in df.columns if any(x in col.lower() for x in ['no2', 'pm', 'so2', 'co', 'o3', 'hcho'])],
    'Satellite Features': [col for col in df.columns if 'satellite' in col.lower()],
    'Forecast Features': [col for col in df.columns if 'forecast' in col.lower()],
    'Engineered Features': [col for col in df.columns if any(x in col for x in ['_lag_', '_rolling_', '_ratio', '_interaction', 'dewpoint', 'humidity'])]
}

for category, features in feature_categories.items():
    if features:
        summary_report.append(f"  {category}: {len(features)} features")
        summary_report.append(f"    {', '.join(features[:10])}")
        if len(features) > 10:
            summary_report.append(f"    ... and {len(features) - 10} more")

# Save report
with open('feature_engineering_report.txt', 'w') as f:
    f.write('\n'.join(summary_report))

print("   ✓ Saved: feature_engineering_report.txt")

# ==================== SAVE ENGINEERED DATASET ====================
print("\n11. Saving Engineered Dataset...")
df.to_csv('master_site1_final_engineered.csv', index=False)
print("   ✓ Saved: master_site1_final_engineered.csv")

# ==================== FINAL SUMMARY ====================
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  1. correlation_matrix_heatmap.png - Full correlation matrix")
print("  2. target_correlations.png - Target variable correlations")
print("  3. target_distributions.png - Distribution of target variables")
print("  4. missing_values_analysis.png - Missing values visualization")
print("  5. feature_importance.png - Feature importance by correlation")
print("  6. time_series_analysis.png - Time series plots")
print("  7. feature_engineering_report.txt - Detailed text report")
print("  8. master_site1_final_engineered.csv - Dataset with engineered features")
print("\n" + "="*80)

