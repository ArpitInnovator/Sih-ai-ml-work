"""
Training Script for TCN-GNN Model

Implements:
- Blocked time-series split
- MAE/Huber loss
- AdamW optimizer
- Early stopping
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from tcn_gnn_model import TCNGNNModel
from data_loader import create_data_loaders


class HuberLoss(nn.Module):
    """Huber loss for robust training"""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        error = pred - target
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        return torch.mean(0.5 * quadratic ** 2 + self.delta * linear)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        hourly_seq = batch['hourly_seq'].to(device)
        daily_satellite = batch['daily_satellite'].to(device)
        static = batch['static'].to(device)
        target = batch['target'].to(device)
        edge_index = batch['edge_index'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(hourly_seq, daily_satellite, static, edge_index)
        
        # Compute loss
        loss = criterion(predictions, target)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            hourly_seq = batch['hourly_seq'].to(device)
            daily_satellite = batch['daily_satellite'].to(device)
            static = batch['static'].to(device)
            target = batch['target'].to(device)
            edge_index = batch['edge_index'].to(device)
            
            # Forward pass
            predictions = model(hourly_seq, daily_satellite, static, edge_index)
            
            # Compute loss
            loss = criterion(predictions, target)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    
    # Compute metrics
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    mae_no2 = np.mean(np.abs(all_predictions[:, 0] - all_targets[:, 0]))
    mae_o3 = np.mean(np.abs(all_predictions[:, 1] - all_targets[:, 1]))
    
    return avg_loss, mae_no2, mae_o3


def train(
    data_dir: Path,
    site_ids: list,
    seq_length: int = 24,
    prediction_horizon: int = 1,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    loss_type: str = 'mae',
    patience: int = 10,
    save_dir: Path = None
):
    """
    Train TCN-GNN model
    
    Args:
        data_dir: Directory containing CSV files
        site_ids: List of site IDs to train on
        seq_length: Length of sequence window
        prediction_horizon: Hours ahead to predict
        batch_size: Batch size
        learning_rate: Learning rate for AdamW
        num_epochs: Maximum number of epochs
        loss_type: 'mae' or 'huber'
        patience: Early stopping patience
        save_dir: Directory to save model and results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, scaler_h, scaler_d, scaler_s = create_data_loaders(
        data_dir=data_dir,
        site_ids=site_ids,
        seq_length=seq_length,
        prediction_horizon=prediction_horizon,
        batch_size=batch_size,
        use_time_split=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get feature dimensions from first batch
    sample_batch = next(iter(train_loader))
    hourly_feature_dim = sample_batch['hourly_seq'].shape[-1]
    daily_satellite_dim = sample_batch['daily_satellite'].shape[-1]
    static_feature_dim = sample_batch['static'].shape[-1]
    
    print(f"Hourly feature dim: {hourly_feature_dim}")
    print(f"Daily satellite dim: {daily_satellite_dim}")
    print(f"Static feature dim: {static_feature_dim}")
    
    # Create model
    model = TCNGNNModel(
        hourly_feature_dim=hourly_feature_dim,
        daily_satellite_dim=daily_satellite_dim,
        static_feature_dim=static_feature_dim,
        temporal_hidden_dim=128,
        gnn_hidden_dim=128,
        mlp_hidden_dim1=128,
        mlp_hidden_dim2=64,
        num_gnn_heads=4,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    if loss_type == 'mae':
        criterion = nn.L1Loss()
    elif loss_type == 'huber':
        criterion = HuberLoss(delta=1.0)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae_no2': [],
        'val_mae_o3': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_mae_no2, val_mae_o3 = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae_no2'].append(val_mae_no2)
        history['val_mae_o3'].append(val_mae_o3)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val MAE NO2: {val_mae_no2:.4f}")
        print(f"Val MAE O3: {val_mae_o3:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print("✓ New best model!")
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model weights")
    
    # Save model and results
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'hourly_feature_dim': hourly_feature_dim,
                'daily_satellite_dim': daily_satellite_dim,
                'static_feature_dim': static_feature_dim,
            },
            'scaler_hourly': scaler_h,
            'scaler_daily': scaler_d,
            'scaler_static': scaler_s,
        }, save_dir / 'best_model.pt')
        
        # Save history
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot training curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['val_mae_no2'], label='NO2 MAE')
        plt.plot(history['val_mae_o3'], label='O3 MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.title('Validation MAE')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=150)
        plt.close()
        
        print(f"\nModel and results saved to {save_dir}")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_mae_no2, test_mae_o3 = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE NO2: {test_mae_no2:.4f}")
    print(f"Test MAE O3: {test_mae_o3:.4f}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train TCN-GNN Model')
    parser.add_argument('--data_dir', type=str, 
                       default='Data_SIH_2025_with_blh/Data_SIH_2025_with_blh/with_satellite',
                       help='Directory containing CSV files')
    parser.add_argument('--site_ids', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7],
                       help='Site IDs to train on')
    parser.add_argument('--seq_length', type=int, default=24,
                       help='Length of sequence window (hours)')
    parser.add_argument('--prediction_horizon', type=int, default=1,
                       help='Hours ahead to predict')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--loss_type', type=str, default='mae', choices=['mae', 'huber'],
                       help='Loss function type')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--save_dir', type=str, default='models/tcn_gnn',
                       help='Directory to save model and results')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    
    # Create timestamped save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = save_dir / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(save_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Train model
    model, history = train(
        data_dir=data_dir,
        site_ids=args.site_ids,
        seq_length=args.seq_length,
        prediction_horizon=args.prediction_horizon,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        loss_type=args.loss_type,
        patience=args.patience,
        save_dir=save_dir
    )
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
