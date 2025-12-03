# Feature Engineering and Data Analysis

This script performs comprehensive feature engineering and generates important data visualizations for the cleaned dataset.

## What It Does

### 1. Feature Engineering
- **Time-based features**: Year, month, day, hour, day of week, cyclical encodings
- **Weather interactions**: Dewpoint depression, relative humidity approximation
- **Wind features**: Wind magnitude, direction, squared components
- **Air quality ratios**: NO2/PM2.5, PM2.5/PM10 ratios
- **Satellite interactions**: NO2/HCHO ratios
- **AOD features**: AOD ratios and interactions
- **Boundary layer features**: BLH-wind interactions
- **Solar features**: Solar elevation squared, daytime indicator
- **Pressure features**: Normalized pressure
- **Lag features**: 1-hour and 3-hour lags for key variables
- **Rolling statistics**: 6-hour rolling mean and standard deviation

### 2. Generated Visualizations

1. **correlation_matrix_heatmap.png**
   - Full correlation matrix for all features
   - Shows relationships between all variables
   - Size: 20x16 inches, high resolution

2. **target_correlations.png**
   - Top correlations for each target variable (NO2_target, O3_target, co, hcho)
   - Shows both positive and negative correlations
   - Helps identify most important features

3. **target_distributions.png**
   - Distribution histograms for target variables
   - Shows data distribution patterns

4. **missing_values_analysis.png**
   - Visualization of missing values across features
   - Helps identify data quality issues

5. **feature_importance.png**
   - Top 25 most important features for each target
   - Based on absolute correlation values
   - Useful for feature selection

6. **time_series_analysis.png**
   - Time series plots for target variables
   - Shows temporal patterns in the data

### 3. Generated Reports

1. **feature_engineering_report.txt**
   - Comprehensive text report with:
     - Dataset statistics
     - Target variable summaries
     - Top correlated features
     - Feature categories breakdown

2. **master_site1_final_engineered.csv**
   - Complete dataset with all engineered features
   - Ready for machine learning models

## Usage

```bash
# Install required packages
pip install -r requirements.txt

# Run the analysis
python feature_engineering_analysis.py
```

## Requirements

- pandas >= 1.5.0
- numpy >= 1.23.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0

## Output Files

All generated files will be saved in the current directory:
- `correlation_matrix_heatmap.png`
- `target_correlations.png`
- `target_distributions.png`
- `missing_values_analysis.png`
- `feature_importance.png`
- `time_series_analysis.png`
- `feature_engineering_report.txt`
- `master_site1_final_engineered.csv`

## Notes

- The script automatically handles missing values in calculations
- Large datasets may take a few minutes to process
- All visualizations are saved in high resolution (300 DPI)
- The correlation matrix shows only the lower triangle to avoid redundancy

