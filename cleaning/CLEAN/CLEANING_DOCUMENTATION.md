# Dataset Cleaning Documentation

**Date:** 2025-12-03 23:53:49

**Original Dataset:** master_site1_final.csv
**Cleaned Dataset:** master_site1_final_cleaned.csv

**Original Shape:** 27408 rows × 148 columns
**Cleaned Shape:** 27408 rows × 44 columns

**Columns Removed:** 104
**Columns Kept:** 44

**Note:** The following forecast columns were explicitly kept as requested:
- NO2_forecast
- O3_forecast
- T_forecast
- q_forecast
- u_forecast
- v_forecast
- w_forecast

---


## ✅ FEATURES KEPT

### Total: 44 features

### Forecast Variables (7 features) - KEPT as requested
- `NO2_forecast` - KEPT - Forecast variable (explicitly requested)
- `O3_forecast` - KEPT - Forecast variable (explicitly requested)
- `T_forecast` - KEPT - Forecast variable (explicitly requested)
- `q_forecast` - KEPT - Forecast variable (explicitly requested)
- `u_forecast` - KEPT - Forecast variable (explicitly requested)
- `v_forecast` - KEPT - Forecast variable (explicitly requested)
- `w_forecast` - KEPT - Forecast variable (explicitly requested)


### Targets (4 features)
- `NO2_target` - KEPT - Target variable
- `O3_target` - KEPT - Target variable
- `co` - KEPT - Target variable
- `hcho` - KEPT - Target variable

### Meteorological (10 features)
- `blh_era5` - KEPT - Meteorological feature (ERA5)
- `d2m_era5` - KEPT - Meteorological feature (ERA5)
- `solar_elevation` - KEPT - Meteorological feature
- `sp` - KEPT - Meteorological feature
- `t2m_era5` - KEPT - Meteorological feature (ERA5)
- `tcc_era5` - KEPT - Meteorological feature (ERA5)
- `u10_era5` - KEPT - Meteorological feature (ERA5)
- `v10_era5` - KEPT - Meteorological feature (ERA5)
- `wind_dir_deg` - KEPT - Meteorological feature
- `wind_speed` - KEPT - Meteorological feature

### Satellite Observations (11 features)
- `HCHO_satellite` - KEPT - Satellite current observation
- `HCHO_satellite_daily` - KEPT - Satellite current observation
- `HCHO_satellite_flag` - KEPT - Satellite current observation
- `NO2_satellite` - KEPT - Satellite current observation
- `NO2_satellite_daily` - KEPT - Satellite current observation
- `NO2_satellite_flag` - KEPT - Satellite current observation
- `NO2_satellite_lag_1h` - KEPT - Satellite observation (1h lag - acceptable)
- `NO2_satellite_lag_3h` - KEPT - Satellite observation (3h lag - acceptable)
- `ratio_satellite` - KEPT - Satellite current observation
- `ratio_satellite_daily` - KEPT - Satellite current observation
- `ratio_satellite_flag` - KEPT - Satellite current observation

### Local Pollutants (8 features)
- `aluvd` - KEPT - Local pollution feature
- `aluvp` - KEPT - Local pollution feature
- `aod550` - KEPT - Local pollution feature
- `bcaod550` - KEPT - Local pollution feature
- `pm1` - KEPT - Local pollution feature
- `pm10` - KEPT - Local pollution feature
- `pm2p5` - KEPT - Local pollution feature
- `so2` - KEPT - Local pollution feature

### Time Features (1 features)
- `datetime` - KEPT - Time feature (for temporal feature generation)

### Other (3 features)
- `SZA_deg` - KEPT - Valid feature with <50% missing
- `no` - KEPT - Local pollutant measurement
- `no2` - KEPT - Local pollutant measurement

---


## ❌ FEATURES REMOVED

### Total: 105 features


### Daily satellite lag field (5 features)
- `NO2_satellite_flag_lag_12h`
- `NO2_satellite_flag_lag_1h`
- `NO2_satellite_flag_lag_24h`
- `NO2_satellite_flag_lag_3h`
- `NO2_satellite_flag_lag_6h`

### Daily satellite lag field (temporally misaligned) (5 features)
- `NO2_satellite_daily_lag_12h`
- `NO2_satellite_daily_lag_1h`
- `NO2_satellite_daily_lag_24h`
- `NO2_satellite_daily_lag_3h`
- `NO2_satellite_daily_lag_6h`

### Derived/transformed duplicate (trig transformation) (5 features)
- `cosSZA`
- `cos_hour`
- `hour_cos`
- `hour_sin`
- `sin_hour`

### Future leakage (forecast variable) (3 features)
- `go3`
- `gtco3`
- `tcco`

### Future leakage (forecast/prediction variable) (17 features)
- `NO2_forecast`
- `NO2_forecast_lag_12h`
- `NO2_forecast_lag_1h`
- `NO2_forecast_lag_24h`
- `NO2_forecast_lag_3h`
- `NO2_forecast_lag_6h`
- `O3_forecast`
- `O3_forecast_lag_12h`
- `O3_forecast_lag_1h`
- `O3_forecast_lag_24h`
- `O3_forecast_lag_3h`
- `O3_forecast_lag_6h`
- `T_forecast`
- `q_forecast`
- `u_forecast`
- `v_forecast`
- `w_forecast`

### Global forecast field (4 features)
- `tc_no`
- `tchcho`
- `tcno2`
- `tcso2`

### Global forecast field lag (15 features)
- `go3_lag_12h`
- `go3_lag_1h`
- `go3_lag_24h`
- `go3_lag_3h`
- `go3_lag_6h`
- `gtco3_lag_12h`
- `gtco3_lag_1h`
- `gtco3_lag_24h`
- `gtco3_lag_3h`
- `gtco3_lag_6h`
- `tcno2_lag_12h`
- `tcno2_lag_1h`
- `tcno2_lag_24h`
- `tcno2_lag_3h`
- `tcno2_lag_6h`

### Identifier/metadata column (7 features)
- `day_cams`
- `file_month`
- `file_year`
- `hour_cams`
- `month_cams`
- `trainable`
- `year_cams`

### Lag > 6h (temporal leakage risk) (14 features)
- `NO2_satellite_lag_12h`
- `NO2_satellite_lag_24h`
- `NO2_target_lag1_lag_12h`
- `NO2_target_lag1_lag_24h`
- `NO2_target_lag24_lag_12h`
- `NO2_target_lag24_lag_24h`
- `NO2_target_lag_12h`
- `NO2_target_lag_24h`
- `O3_target_lag1_lag_12h`
- `O3_target_lag1_lag_24h`
- `O3_target_lag_12h`
- `O3_target_lag_24h`
- `no2_lag_12h`
- `no2_lag_24h`

### Nested lag pattern (9 features)
- `NO2_target_lag1_lag_1h`
- `NO2_target_lag1_lag_3h`
- `NO2_target_lag1_lag_6h`
- `NO2_target_lag24_lag_1h`
- `NO2_target_lag24_lag_3h`
- `NO2_target_lag24_lag_6h`
- `O3_target_lag1_lag_1h`
- `O3_target_lag1_lag_3h`
- `O3_target_lag1_lag_6h`

### Old time feature (will regenerate) (8 features)
- `day`
- `day_of_week`
- `day_of_year`
- `dayofweek`
- `hour`
- `is_weekend`
- `month`
- `year`

### Target lag variable (potential leakage) (9 features)
- `NO2_target_lag1`
- `NO2_target_lag24`
- `NO2_target_lag_1h`
- `NO2_target_lag_3h`
- `NO2_target_lag_6h`
- `O3_target_lag1`
- `O3_target_lag_1h`
- `O3_target_lag_3h`
- `O3_target_lag_6h`

### Unclassified lag/forecast variable (4 features)
- `NO2_satellite_lag_6h`
- `no2_lag_1h`
- `no2_lag_3h`
- `no2_lag_6h`

---


## 📊 Summary Statistics

### Missing Data in Kept Features
- `aluvd`: 8.6% missing
- `aluvp`: 8.6% missing
- `no`: 8.6% missing
- `so2`: 8.6% missing
- `sp`: 8.6% missing

---


## 🔍 Cleaning Rules Applied

1. **Targets Kept:** NO2_target, O3_target, CO, HCHO (surface measurement)

2. **Removed Future Leakage:** All forecast, prediction, and lead variables

3. **Removed Derived Features:** Trigonometric transformations (sin_hour, cos_hour, etc.)

4. **Removed Sparse Features:** Columns with >80% missing data

5. **Removed Daily Lag Fields:** Satellite daily products with lag times

6. **Removed Identifiers:** File metadata and constant columns

7. **Kept Meteorological Features:** Temperature, humidity, wind, pressure, etc.

8. **Kept Satellite Observations:** Current observations and short lags (≤6h)

9. **Kept Local Pollutants:** PM2.5, PM10, SO2, AOD, etc.

10. **Removed Global Forecasts:** go3, gtco3, tcco, and similar fields

11. **Time Features:** Kept datetime only (will regenerate cyclical features)