"""Graph Convolutional Network for molecular property prediction."""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool


class GNN(nn.Module):
    """
    Graph Neural Network for molecular property prediction.

    Architecture:
    - Input projection layer
    - Multiple GCN/GAT layers with residual connections
    - Global mean + max pooling for graph-level representation
    - MLP head for multi-task prediction

    Args:
        num_node_features: Number of input node features (default: 140)
        hidden_channels: Hidden layer size
        num_layers: Number of graph convolution layers
        num_tasks: Number of prediction tasks
        dropout: Dropout rate
        conv_type: Type of convolution ('gcn' or 'gat')
    """

    def __init__(
        self,
        num_node_features=141,
        hidden_channels=256,
        num_layers=4,
        num_tasks=12,
        dropout=0.2,
        conv_type='gcn'
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Initial projection
        self.input_proj = nn.Linear(num_node_features, hidden_channels)

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            if conv_type == 'gcn':
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            else:  # GAT
                self.convs.append(GATConv(hidden_channels, hidden_channels // 4, heads=4))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output MLP (*2 for mean+max pooling concatenation)
        self.output = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_tasks)
        )

    def forward(self, x, edge_index, batch):
        # Initial projection
        x = self.input_proj(x)
        x = torch.relu(x)

        # Message passing layers with residual connections
        for i in range(self.num_layers):
            x_res = x
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
            x = x + x_res  # Residual connection

        # Global pooling - combine mean and max for richer representation
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Output prediction
        return self.output(x)
