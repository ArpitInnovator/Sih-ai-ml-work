"""
TCN-GNN Model Architecture for Air Quality Prediction

This module implements a Temporal Convolutional Network (TCN) combined with
Graph Neural Network (GNN) for predicting NO2 and O3 concentrations.

Architecture:
1. TCN: Processes hourly time-series sequences
2. Feature Combination: Combines temporal embeddings with daily satellite and static features
3. GNN: Learns spatial relationships between sites
4. MLP: Final prediction head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np


class DilatedConv1d(nn.Module):
    """Dilated 1D Convolutional layer for TCN with causal padding"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.padding = padding
        
    def forward(self, x):
        x = self.conv(x)
        # Remove right padding to maintain sequence length (causal convolution)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network Encoder
    
    Processes hourly sequence features (last 24-48 hours) to extract
    temporal patterns (diurnal cycles, photochemistry, lag effects).
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initial Conv1D layer
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Dilated convolutional layers
        self.dilated_convs = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.dilated_convs.append(
                DilatedConv1d(hidden_dim, hidden_dim, kernel_size=3, 
                             dilation=dilation, dropout=dropout)
            )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim] or [batch_size, input_dim, seq_len]
        Returns:
            embedding: [batch_size, hidden_dim]
        """
        # Ensure input is [batch, channels, seq_len]
        if len(x.shape) == 3 and x.shape[1] != self.input_dim:
            x = x.transpose(1, 2)  # [batch, seq_len, features] -> [batch, features, seq_len]
        
        x = self.initial_conv(x)
        
        # Apply dilated convolutions
        for conv in self.dilated_convs:
            x = conv(x)
        
        # Global average pooling
        x = self.global_pool(x)  # [batch, hidden_dim, 1]
        x = x.squeeze(-1)  # [batch, hidden_dim]
        
        return x


class FeatureCombiner(nn.Module):
    """
    Combines temporal embedding with daily satellite and static features
    """
    def __init__(self, temporal_dim, daily_satellite_dim, static_dim, output_dim=256):
        super().__init__()
        combined_dim = temporal_dim + daily_satellite_dim + static_dim
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, temporal_emb, daily_satellite, static_features):
        """
        Args:
            temporal_emb: [batch, temporal_dim]
            daily_satellite: [batch, daily_satellite_dim]
            static_features: [batch, static_dim]
        Returns:
            combined: [batch, output_dim]
        """
        combined = torch.cat([temporal_emb, daily_satellite, static_features], dim=1)
        return self.fc(combined)


class GNNLayer(nn.Module):
    """
    Graph Neural Network Layer using Graph Attention Network (GAT)
    
    Learns spatial relationships between sites based on:
    - Physical distance (K nearest neighbors)
    - Pollutant correlation
    """
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.gat1 = GATConv(
            input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True
        )
        self.gat2 = GATConv(
            hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, concat=False
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: [num_nodes, input_dim] node features
            edge_index: [2, num_edges] edge connectivity
            batch: [num_nodes] batch assignment for each node
        Returns:
            out: [num_nodes, hidden_dim] updated node features
        """
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        return x


class MLPPredictionHead(nn.Module):
    """
    Multi-Layer Perceptron for final predictions
    """
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, output_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim2, output_dim)
        )
        
    def forward(self, x):
        return self.fc(x)


class TCNGNNModel(nn.Module):
    """
    Complete TCN-GNN Model for Air Quality Prediction
    
    Architecture:
    1. TCN Encoder: Processes hourly sequences → temporal embedding
    2. Feature Combiner: Combines temporal + daily satellite + static features
    3. GNN Layer: Learns spatial relationships between sites
    4. MLP Head: Predicts NO2 and O3 concentrations
    """
    def __init__(
        self,
        hourly_feature_dim,
        daily_satellite_dim=3,  # NO2_satellite, HCHO_satellite, ratio_satellite
        static_feature_dim=2,   # latitude, longitude
        temporal_hidden_dim=128,
        gnn_hidden_dim=128,
        mlp_hidden_dim1=128,
        mlp_hidden_dim2=64,
        num_gnn_heads=4,
        dropout=0.1
    ):
        super().__init__()
        
        # 1. TCN Encoder for hourly sequences
        self.tcn_encoder = TCNEncoder(
            input_dim=hourly_feature_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=3,
            dropout=dropout
        )
        
        # 2. Feature Combiner
        self.feature_combiner = FeatureCombiner(
            temporal_dim=temporal_hidden_dim,
            daily_satellite_dim=daily_satellite_dim,
            static_dim=static_feature_dim,
            output_dim=gnn_hidden_dim
        )
        
        # 3. GNN Layer
        self.gnn = GNNLayer(
            input_dim=gnn_hidden_dim,
            hidden_dim=gnn_hidden_dim,
            num_heads=num_gnn_heads,
            dropout=dropout
        )
        
        # 4. MLP Prediction Head
        self.mlp_head = MLPPredictionHead(
            input_dim=gnn_hidden_dim,
            hidden_dim1=mlp_hidden_dim1,
            hidden_dim2=mlp_hidden_dim2,
            output_dim=2  # NO2_pred, O3_pred
        )
        
    def forward(self, hourly_seq, daily_satellite, static_features, edge_index, batch=None):
        """
        Forward pass through the complete model
        
        Args:
            hourly_seq: [batch_size, seq_len, hourly_feature_dim] hourly sequence features
            daily_satellite: [batch_size, daily_satellite_dim] daily satellite features
            static_features: [batch_size, static_feature_dim] static site features
            edge_index: [2, num_edges] graph edge connectivity
            batch: [num_nodes] batch assignment for graph nodes
            
        Returns:
            predictions: [batch_size, 2] predictions [NO2_pred, O3_pred]
        """
        # 1. TCN Encoder
        temporal_emb = self.tcn_encoder(hourly_seq)  # [batch, temporal_hidden_dim]
        
        # 2. Feature Combination
        node_features = self.feature_combiner(
            temporal_emb, daily_satellite, static_features
        )  # [batch, gnn_hidden_dim]
        
        # 3. GNN Layer
        gnn_output = self.gnn(node_features, edge_index, batch)  # [batch, gnn_hidden_dim]
        
        # 4. MLP Prediction Head
        predictions = self.mlp_head(gnn_output)  # [batch, 2]
        
        return predictions


def create_graph_edges(site_coords, k_neighbors=3):
    """
    Create graph edges based on K-nearest neighbors using physical distance
    
    Args:
        site_coords: List of (lat, lon) tuples for each site
        k_neighbors: Number of nearest neighbors to connect
        
    Returns:
        edge_index: [2, num_edges] tensor of edge connections
    """
    from scipy.spatial.distance import cdist
    
    n_sites = len(site_coords)
    coords_array = np.array(site_coords)
    
    # Compute pairwise distances
    distances = cdist(coords_array, coords_array, metric='euclidean')
    
    # For each site, find k nearest neighbors (excluding self)
    edges = []
    for i in range(n_sites):
        # Get k+1 nearest (including self), then exclude self
        nearest = np.argsort(distances[i])[:k_neighbors + 1]
        nearest = nearest[nearest != i]  # Remove self
        
        for j in nearest:
            edges.append([i, j])
            edges.append([j, i])  # Make undirected
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index
