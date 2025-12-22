"""MLP-based molecular property predictor using fingerprints."""

import torch.nn as nn


class MolecularPropertyPredictor(nn.Module):
    """
    Neural network for predicting molecular properties from fingerprints.

    Architecture: Fingerprint -> MLP with batch norm -> Multi-task output

    Args:
        input_size: Size of input fingerprint (default: 2048 for Morgan fingerprint)
        hidden_sizes: List of hidden layer sizes
        num_tasks: Number of prediction tasks (default: 12 for Tox21)
        dropout: Dropout rate
    """

    def __init__(self, input_size=2048, hidden_sizes=[1024, 512, 256], num_tasks=12, dropout=0.3):
        super().__init__()

        layers = []
        prev_size = input_size

        # Build hidden layers with batch norm and dropout
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        self.encoder = nn.Sequential(*layers)

        # Multi-task prediction head (one output per toxicity endpoint)
        self.predictor = nn.Linear(hidden_sizes[-1], num_tasks)

    def forward(self, x):
        hidden = self.encoder(x)
        return self.predictor(hidden)
