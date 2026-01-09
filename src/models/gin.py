"""Graph Isomorphism Network (GIN) for molecular property prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
from typing import Optional


class GINBlock(nn.Module):
    """
    A single GIN convolution block with batch norm and activation.

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        eps: Initial epsilon value for GIN
        train_eps: Whether to learn epsilon
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        eps: float = 0.0,
        train_eps: bool = True,
    ):
        super().__init__()

        # MLP for GIN aggregation
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

        self.conv = GINConv(self.mlp, eps=eps, train_eps=train_eps)
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x, edge_index)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x


class GIN(nn.Module):
    """
    Graph Isomorphism Network for molecular property prediction.

    GIN is provably as powerful as the Weisfeiler-Lehman (WL) graph
    isomorphism test. It uses sum aggregation and MLPs for updates.

    Reference: Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019

    Architecture:
    - Input projection layer
    - Multiple GIN layers with batch normalization
    - Jumping knowledge (optional): concatenate all layer outputs
    - Global pooling (sum/mean/max)
    - MLP head for multi-task prediction

    Args:
        num_node_features: Number of input node features
        hidden_channels: Hidden layer size
        num_layers: Number of GIN layers
        num_tasks: Number of prediction tasks
        dropout: Dropout rate
        pooling: Global pooling type ('sum', 'mean', 'max', 'attention')
        jumping_knowledge: Whether to use jumping knowledge
        train_eps: Whether to learn epsilon in GIN
    """

    def __init__(
        self,
        num_node_features: int = 141,
        hidden_channels: int = 256,
        num_layers: int = 5,
        num_tasks: int = 12,
        dropout: float = 0.3,
        pooling: str = "sum",
        jumping_knowledge: bool = True,
        train_eps: bool = True,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.pooling = pooling

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(num_node_features, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )

        # GIN layers
        self.gin_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gin_layers.append(
                GINBlock(hidden_channels, hidden_channels, train_eps=train_eps)
            )

        # Jumping knowledge linear layers (one per layer)
        if jumping_knowledge:
            self.jk_linears = nn.ModuleList([
                nn.Linear(hidden_channels, hidden_channels)
                for _ in range(num_layers)
            ])

        # Calculate output dimension
        if jumping_knowledge:
            pool_dim = hidden_channels * num_layers
        else:
            pool_dim = hidden_channels

        # Attention pooling (optional)
        if pooling == "attention":
            self.pool_attention = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.Tanh(),
                nn.Linear(hidden_channels, 1),
            )

        # Output MLP
        self.output = nn.Sequential(
            nn.Linear(pool_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_tasks),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (N, num_node_features)
            edge_index: Edge indices (2, E)
            batch: Batch assignment (N,)

        Returns:
            Predictions (batch_size, num_tasks)
        """
        # Initial projection
        x = self.input_proj(x)

        # Store outputs of each layer for jumping knowledge
        layer_outputs = []

        # GIN message passing
        for i, gin_layer in enumerate(self.gin_layers):
            x = gin_layer(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.jumping_knowledge:
                layer_outputs.append(self.jk_linears[i](x))

        # Global pooling
        if self.jumping_knowledge:
            # Concatenate all layer outputs after pooling
            pooled_outputs = []
            for layer_out in layer_outputs:
                pooled = self._global_pool(layer_out, batch)
                pooled_outputs.append(pooled)
            x = torch.cat(pooled_outputs, dim=1)
        else:
            x = self._global_pool(x, batch)

        # Output prediction
        return self.output(x)

    def _global_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Apply global pooling."""
        if self.pooling == "sum":
            return global_add_pool(x, batch)
        elif self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "max":
            return global_max_pool(x, batch)
        elif self.pooling == "attention":
            # Attention-based pooling
            attn_weights = self.pool_attention(x)
            attn_weights = F.softmax(attn_weights, dim=0)
            # Weighted sum per graph
            return global_add_pool(x * attn_weights, batch)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get molecular embeddings before the output layer.

        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment

        Returns:
            Molecular embeddings
        """
        # Initial projection
        x = self.input_proj(x)

        # GIN message passing
        layer_outputs = []
        for i, gin_layer in enumerate(self.gin_layers):
            x = gin_layer(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.jumping_knowledge:
                layer_outputs.append(self.jk_linears[i](x))

        # Global pooling
        if self.jumping_knowledge:
            pooled_outputs = []
            for layer_out in layer_outputs:
                pooled = self._global_pool(layer_out, batch)
                pooled_outputs.append(pooled)
            x = torch.cat(pooled_outputs, dim=1)
        else:
            x = self._global_pool(x, batch)

        return x


class GINEdge(nn.Module):
    """
    GIN variant that incorporates edge features.

    This extends GIN to handle bond features in molecular graphs.

    Args:
        num_node_features: Number of input node features
        num_edge_features: Number of edge features
        hidden_channels: Hidden layer size
        num_layers: Number of GIN layers
        num_tasks: Number of prediction tasks
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_node_features: int = 141,
        num_edge_features: int = 12,
        hidden_channels: int = 256,
        num_layers: int = 5,
        num_tasks: int = 12,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Input projections
        self.node_proj = nn.Linear(num_node_features, hidden_channels)
        self.edge_proj = nn.Linear(num_edge_features, hidden_channels)

        # GIN layers with edge features
        self.gin_layers = nn.ModuleList()
        self.edge_linears = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            # MLP for node update
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.gin_layers.append(GINConv(mlp, train_eps=True))
            self.edge_linears.append(nn.Linear(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output MLP
        self.output = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_tasks),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with edge features.

        Args:
            x: Node features (N, num_node_features)
            edge_index: Edge indices (2, E)
            edge_attr: Edge features (E, num_edge_features)
            batch: Batch assignment (N,)

        Returns:
            Predictions (batch_size, num_tasks)
        """
        # Project inputs
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)

        # Message passing with edge features
        for i in range(self.num_layers):
            # Add edge information to source nodes
            row, col = edge_index
            edge_features = self.edge_linears[i](edge_attr)

            # Aggregate edge features to nodes
            x_edge = torch.zeros_like(x)
            x_edge.index_add_(0, row, edge_features)

            # GIN convolution
            x = self.gin_layers[i](x + x_edge, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling (mean + max)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        return self.output(x)
