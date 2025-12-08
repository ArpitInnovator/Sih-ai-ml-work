# TCN-GNN Model Architecture Summary

## Complete Model Structure

### 1. Inputs (3 Groups)

#### A. Hourly Sequence Features (Time-Series, Last 24-48 Hours)
- **Shape**: `[batch_size, seq_length, hourly_feature_dim]`
- **Features** (15 total):
  - `NO2_forecast`, `O3_forecast`, `T_forecast`
  - `wind_speed`, `wind_dir_deg`
  - `blh_forecast`
  - `sin_hour`, `cos_hour`
  - `NO2_forecast_per_blh`, `O3_forecast_per_blh`
  - `blh_delta_1h`, `blh_rel_change`
  - `cosSZA`, `solar_elevation`, `sunset_flag`

#### B. Daily Satellite Features (Once Per Day)
- **Shape**: `[batch_size, 3]`
- **Features**:
  - `NO2_satellite_filled`
  - `HCHO_satellite_filled`
  - `ratio_satellite`
- **Note**: These are NOT time-series; passed as flat vectors (not repeated hourly)

#### C. Static Site Features
- **Shape**: `[batch_size, 2]`
- **Features**:
  - `latitude`
  - `longitude`

---

### 2. TCN: Temporal Encoder

**Purpose**: Processes ONLY the hourly sequence input to learn:
- Diurnal patterns
- Photochemistry cycles
- Lag effects
- Forecast corrections

**Structure**:
```
Input: [batch, seq_len, hourly_feature_dim]
  ↓
Initial Conv1D (kernel=3, padding=1) → ReLU → BatchNorm → Dropout
  ↓
Dilated Conv1D (dilation=1) → ReLU → BatchNorm → Dropout
  ↓
Dilated Conv1D (dilation=2) → ReLU → BatchNorm → Dropout
  ↓
Dilated Conv1D (dilation=4) → ReLU → BatchNorm → Dropout
  ↓
GlobalAveragePooling
  ↓
Output: [batch, temporal_hidden_dim] (default: 128)
```

**Key Features**:
- Causal convolutions (no future information leakage)
- Dilated convolutions capture long-range dependencies
- Global average pooling summarizes temporal patterns

---

### 3. Feature Combination Layer

**Purpose**: Combine all information into a single vector per site

**Structure**:
```
Inputs:
  - Temporal embedding: [batch, 128]
  - Daily satellite: [batch, 3]
  - Static features: [batch, 2]
  ↓
Concatenate → [batch, 133]
  ↓
Dense(256) → ReLU → Dropout
  ↓
Dense(256) → ReLU
  ↓
Output: [batch, 256] → reduced to [batch, 128] for GNN input
```

**Output**: Combined node embedding representing each site

---

### 4. GNN Layer (GAT - Graph Attention Network)

**Purpose**: Learn spatial relationships between sites

**Graph Structure**:
- Nodes: Each sample in the batch (representing a site at a time)
- Edges: Fully connected within batch (GAT learns attention weights)
- Edge weights: Learned by attention mechanism

**Structure**:
```
Input: [batch, 128]
  ↓
GAT Layer 1 (4 heads) → ELU → Dropout
  ↓
[batch, 512] (128 * 4 heads)
  ↓
GAT Layer 2 (1 head) → [batch, 128]
  ↓
Output: [batch, 128] (spatially enriched node embedding)
```

**What GNN Learns**:
- Regional transport patterns
- Chemical coupling between sites
- Spatial smoothing
- Satellite footprint influence

---

### 5. MLP Prediction Head

**Purpose**: Map GNN output to final predictions

**Structure**:
```
Input: [batch, 128]
  ↓
Dense(128) → ReLU → Dropout
  ↓
Dense(64) → ReLU → Dropout
  ↓
Dense(2)
  ↓
Output: [batch, 2] → [NO₂_pred, O₃_pred]
```

---

### 6. Training Pipeline

#### Loss Function
- **MAE (Mean Absolute Error)**: `Loss = MAE(NO₂_pred, NO₂_true) + MAE(O₃_pred, O₃_true)`
- **Alternative**: Huber Loss (robust to outliers)

#### Optimizer
- **AdamW** with learning rate = 1e-3
- Weight decay = 1e-5
- Gradient clipping (max_norm = 1.0)

#### Training Method
- **Blocked Time-Series Split**:
  - Train on older dates → Validate on newer dates
  - Prevents data leakage
  - Maintains temporal order

#### Training Schedule
- Maximum epochs: 50-100
- Early stopping on validation MAE (patience = 10)
- Learning rate reduction on plateau (factor = 0.5, patience = 5)

---

### 7. Data Flow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT DATA                                │
├─────────────────────────────────────────────────────────────┤
│ Hourly Data: [batch, 24, 15]                                 │
│ Daily Satellite: [batch, 3]                                 │
│ Static Features: [batch, 2]                                 │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              TCN ENCODER                                    │
│  Learns temporal patterns (diurnal, photochemistry)       │
│  Output: [batch, 128]                                       │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│         FEATURE COMBINATION                                  │
│  Combines: temporal + daily satellite + static             │
│  Output: [batch, 128]                                       │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              GNN LAYER (GAT)                                │
│  Learns spatial + chemical relationships                    │
│  Output: [batch, 128]                                       │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              MLP PREDICTION HEAD                            │
│  Maps embedding to predictions                             │
│  Output: [batch, 2] → [NO₂_pred, O₃_pred]                  │
└─────────────────────────────────────────────────────────────┘
                        ↓
                    LOSS
              (MAE or Huber)
```

---

### 8. Why This Structure Works

1. **TCN**: 
   - Handles hourly pollution & meteorology changes
   - Captures temporal dependencies (diurnal cycles, lag effects)
   - Causal convolutions prevent future information leakage

2. **GNN**: 
   - Models spatial interactions between sites
   - Learns regional chemistry and transport patterns
   - Attention mechanism focuses on important spatial connections

3. **Satellite Data**: 
   - Enhances chemical regime understanding
   - Provides daily-scale chemical context
   - Helps model understand photochemical processes

4. **MLP**: 
   - Performs final mapping to predictions
   - Learns non-linear relationships
   - Outputs both NO₂ and O₃ simultaneously

---

### 9. Model Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `temporal_hidden_dim` | 128 | TCN output dimension |
| `gnn_hidden_dim` | 128 | GNN hidden dimension |
| `mlp_hidden_dim1` | 128 | First MLP layer size |
| `mlp_hidden_dim2` | 64 | Second MLP layer size |
| `num_gnn_heads` | 4 | GAT attention heads |
| `dropout` | 0.1 | Dropout rate |
| `seq_length` | 24 | Sequence window (hours) |
| `learning_rate` | 1e-3 | AdamW learning rate |
| `batch_size` | 32 | Training batch size |

---

### 10. Key Implementation Details

- **Data Preprocessing**: StandardScaler for all feature groups
- **Sequence Creation**: Handles cross-day sequences properly
- **Graph Construction**: Fully connected within batch (GAT learns weights)
- **Causal Convolutions**: TCN uses causal padding
- **Early Stopping**: Based on validation loss
- **Model Checkpointing**: Saves best model based on validation performance

---

## File Structure

```
fmodel/
├── tcn_gnn_model.py          # Model architecture
├── data_loader.py             # Data loading and preprocessing
├── train_tcn_gnn.py           # Training script
├── run_training_example.py    # Example usage
├── requirements_tcn_gnn.txt   # Dependencies
├── README_TCN_GNN.md          # Usage documentation
└── ARCHITECTURE_SUMMARY.md    # This file
```

---

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements_tcn_gnn.txt
   ```

2. Run training:
   ```bash
   python train_tcn_gnn.py --data_dir "Data_SIH_2025_with_blh/Data_SIH_2025_with_blh/with_satellite"
   ```

3. Or use the example script:
   ```bash
   python run_training_example.py
   ```

---

## Model Outputs

After training, the model saves:
- `best_model.pt`: Model weights and configuration
- `training_history.json`: Training metrics
- `training_curves.png`: Visualization plots
- `args.json`: Training arguments
