"""AttentiveFP model for molecular property prediction."""

import torch.nn as nn
from torch_geometric.nn.models import AttentiveFP


class AttentiveFPPredictor(nn.Module):
    """
    AttentiveFP wrapper for multi-task molecular property prediction.

    AttentiveFP uses graph attention with edge features for state-of-the-art
    performance on molecular property prediction tasks.

    Args:
        in_channels: Number of input node features
        hidden_channels: Hidden layer size
        out_channels: Number of output tasks
        edge_dim: Number of edge features
        num_layers: Number of AttentiveFP layers
        num_timesteps: Number of attention timesteps
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=256,
        out_channels=12,
        edge_dim=12,
        num_layers=3,
        num_timesteps=3,
        dropout=0.2
    ):
        super().__init__()

        self.attentive_fp = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,  # Output embedding
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout
        )

        # Multi-task prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # Get graph-level embedding from AttentiveFP
        embedding = self.attentive_fp(x, edge_index, edge_attr, batch)
        # Predict toxicity
        return self.predictor(embedding)
