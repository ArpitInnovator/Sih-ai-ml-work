import pandas as pd

print("Adding forecast columns to cleaned dataset...")

# Load datasets
df_orig = pd.read_csv('master_site1_final.csv')
df_clean = pd.read_csv('master_site1_final_cleaned.csv')

# Forecast columns to add
forecast_cols = ['NO2_forecast', 'O3_forecast', 'T_forecast', 
                 'q_forecast', 'u_forecast', 'v_forecast', 'w_forecast']

print(f"\nOriginal cleaned shape: {df_clean.shape}")

# Add forecast columns
for col in forecast_cols:
    if col in df_orig.columns:
        df_clean[col] = df_orig[col]
        print(f"  ✓ Added: {col}")
    else:
        print(f"  ✗ NOT FOUND: {col}")

# Save updated cleaned dataset
df_clean.to_csv('master_site1_final_cleaned.csv', index=False)

print(f"\nUpdated cleaned dataset: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
print(f"\nTotal columns now: {len(df_clean.columns)}")
print("\nAdded forecast columns:")
for col in forecast_cols:
    if col in df_clean.columns:
        print(f"  - {col}")


