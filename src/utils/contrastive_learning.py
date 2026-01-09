"""Contrastive learning utilities for molecular representation learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive learning."""

    temperature: float = 0.07
    projection_dim: int = 128
    hidden_dim: int = 256
    loss_type: str = "infonce"  # "infonce", "triplet", "ntxent"
    hard_negative_mining: bool = False
    num_negatives: int = -1  # -1 means all negatives


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss.

    Maximizes agreement between positive pairs and minimizes
    agreement with negative pairs.

    Reference: Oord et al., "Representation Learning with Contrastive Predictive Coding"

    Args:
        temperature: Temperature scaling parameter
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            anchor: Anchor embeddings [N, D]
            positive: Positive embeddings [N, D]
            negatives: Optional explicit negatives [N, K, D]

        Returns:
            InfoNCE loss
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)

        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature

        if negatives is not None:
            # Explicit negatives
            negatives = F.normalize(negatives, dim=-1)
            neg_sim = torch.bmm(anchor.unsqueeze(1), negatives.transpose(1, 2))
            neg_sim = neg_sim.squeeze(1) / self.temperature
            logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        else:
            # In-batch negatives
            sim_matrix = torch.mm(anchor, positive.t()) / self.temperature
            # Mask out self-similarity
            mask = torch.eye(anchor.size(0), device=anchor.device).bool()
            logits = sim_matrix

        # Labels: positive is always at position 0
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

        if negatives is not None:
            loss = F.cross_entropy(logits, labels)
        else:
            # For in-batch negatives, diagonal is positive
            loss = F.cross_entropy(logits, torch.arange(anchor.size(0), device=anchor.device))

        return loss


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    Used in SimCLR for self-supervised learning.

    Reference: Chen et al., "A Simple Framework for Contrastive Learning"

    Args:
        temperature: Temperature parameter
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss for paired views.

        Args:
            z_i: First view embeddings [N, D]
            z_j: Second view embeddings [N, D]

        Returns:
            NT-Xent loss
        """
        batch_size = z_i.size(0)

        # Normalize
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        # Concatenate representations
        representations = torch.cat([z_i, z_j], dim=0)

        # Similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        ) / self.temperature

        # Create labels (positive pairs)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ], dim=0).to(z_i.device)

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z_i.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))

        # Cross entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


class TripletLoss(nn.Module):
    """
    Triplet margin loss for contrastive learning.

    Args:
        margin: Margin between positive and negative distances
        mining: Mining strategy ('none', 'hard', 'semi-hard')
    """

    def __init__(self, margin: float = 1.0, mining: str = "none"):
        super().__init__()
        self.margin = margin
        self.mining = mining
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings [N, D]
            positive: Positive embeddings [N, D]
            negative: Negative embeddings [N, D]

        Returns:
            Triplet loss
        """
        if self.mining == "hard":
            anchor, positive, negative = self._hard_mining(anchor, positive, negative)
        elif self.mining == "semi-hard":
            anchor, positive, negative = self._semi_hard_mining(anchor, positive, negative)

        return self.triplet_loss(anchor, positive, negative)

    def _hard_mining(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select hardest negatives."""
        with torch.no_grad():
            distances = torch.cdist(anchor, negative)
            hard_idx = distances.argmin(dim=1)

        return anchor, positive, negative[hard_idx]

    def _semi_hard_mining(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select semi-hard negatives."""
        with torch.no_grad():
            pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
            neg_dist = torch.cdist(anchor, negative) ** 2

            # Semi-hard: negatives that are farther than positives but within margin
            mask = (neg_dist > pos_dist.unsqueeze(1)) & (neg_dist < pos_dist.unsqueeze(1) + self.margin)

            # Fallback to random if no semi-hard negatives
            semi_hard_idx = []
            for i in range(anchor.size(0)):
                valid = mask[i].nonzero().squeeze(-1)
                if valid.numel() > 0:
                    idx = valid[torch.randint(valid.numel(), (1,))]
                else:
                    idx = torch.randint(negative.size(0), (1,), device=anchor.device)
                semi_hard_idx.append(idx)

            semi_hard_idx = torch.cat(semi_hard_idx)

        return anchor, positive, negative[semi_hard_idx]


class MolecularAugmentation:
    """
    Molecular augmentation strategies for contrastive learning.

    Generates augmented views of molecular graphs.
    """

    def __init__(
        self,
        node_drop_rate: float = 0.1,
        edge_drop_rate: float = 0.1,
        feature_mask_rate: float = 0.1,
        subgraph_rate: float = 0.2,
    ):
        self.node_drop_rate = node_drop_rate
        self.edge_drop_rate = edge_drop_rate
        self.feature_mask_rate = feature_mask_rate
        self.subgraph_rate = subgraph_rate

    def node_dropping(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly drop nodes from the graph."""
        num_nodes = x.size(0)
        keep_mask = torch.rand(num_nodes, device=x.device) > self.node_drop_rate

        # Ensure at least one node per graph
        for b in batch.unique():
            graph_mask = batch == b
            if not keep_mask[graph_mask].any():
                # Keep random node
                graph_indices = graph_mask.nonzero().squeeze(-1)
                keep_idx = graph_indices[torch.randint(graph_indices.numel(), (1,))]
                keep_mask[keep_idx] = True

        # Filter nodes
        new_x = x[keep_mask]
        new_batch = batch[keep_mask]

        # Remap edges
        node_mapping = torch.zeros(num_nodes, dtype=torch.long, device=x.device) - 1
        node_mapping[keep_mask] = torch.arange(keep_mask.sum(), device=x.device)

        edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
        new_edge_index = node_mapping[edge_index[:, edge_mask]]

        return new_x, new_edge_index, new_batch

    def edge_perturbation(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Randomly drop and add edges."""
        num_edges = edge_index.size(1)

        # Drop edges
        keep_mask = torch.rand(num_edges, device=edge_index.device) > self.edge_drop_rate
        new_edge_index = edge_index[:, keep_mask]

        # Add random edges
        num_add = int(num_edges * self.edge_drop_rate)
        if num_add > 0:
            new_src = torch.randint(num_nodes, (num_add,), device=edge_index.device)
            new_dst = torch.randint(num_nodes, (num_add,), device=edge_index.device)
            added_edges = torch.stack([new_src, new_dst])
            new_edge_index = torch.cat([new_edge_index, added_edges], dim=1)

        return new_edge_index

    def feature_masking(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Randomly mask node features."""
        mask = torch.rand(x.shape, device=x.device) > self.feature_mask_rate
        return x * mask

    def subgraph_sampling(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract random subgraphs using random walks."""
        num_nodes = x.size(0)
        keep_mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)

        for b in batch.unique():
            graph_mask = batch == b
            graph_nodes = graph_mask.nonzero().squeeze(-1)
            num_keep = max(1, int(graph_nodes.numel() * (1 - self.subgraph_rate)))

            # Random walk starting from random node
            start = graph_nodes[torch.randint(graph_nodes.numel(), (1,))]
            walked = {start.item()}

            current = start
            for _ in range(num_keep - 1):
                # Get neighbors
                neighbors = edge_index[1, edge_index[0] == current]
                neighbors = neighbors[torch.isin(neighbors, graph_nodes)]

                if neighbors.numel() > 0:
                    current = neighbors[torch.randint(neighbors.numel(), (1,))]
                    walked.add(current.item())
                else:
                    # Random jump
                    remaining = graph_nodes[~torch.tensor(list(walked), device=x.device).unsqueeze(0).eq(graph_nodes.unsqueeze(1)).any(1)]
                    if remaining.numel() > 0:
                        current = remaining[torch.randint(remaining.numel(), (1,))]
                        walked.add(current.item())

            for node in walked:
                keep_mask[node] = True

        return self.node_dropping.__wrapped__(self, x, edge_index, batch)

    def augment(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Apply random augmentation."""
        aug_type = np.random.choice(["node_drop", "edge_perturb", "feature_mask"])

        if aug_type == "node_drop":
            x, edge_index, batch = self.node_dropping(x, edge_index, batch)
            if edge_attr is not None:
                edge_attr = None  # Edge attributes no longer valid
        elif aug_type == "edge_perturb":
            edge_index = self.edge_perturbation(edge_index, x.size(0))
            if edge_attr is not None:
                edge_attr = None
        else:
            x = self.feature_masking(x)

        return x, edge_index, batch, edge_attr


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.

    Maps representations to contrastive embedding space.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
        num_layers: Number of MLP layers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.projector = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class MomentumEncoder(nn.Module):
    """
    Momentum encoder for MoCo-style contrastive learning.

    Maintains a slowly-updating copy of the encoder.

    Args:
        encoder: Base encoder network
        momentum: Momentum coefficient for updates
    """

    def __init__(self, encoder: nn.Module, momentum: float = 0.999):
        super().__init__()
        self.encoder = encoder
        self.momentum = momentum

        # Create momentum encoder
        self.momentum_encoder = self._create_momentum_encoder()

    def _create_momentum_encoder(self) -> nn.Module:
        """Create a copy of the encoder for momentum updates."""
        import copy
        momentum_encoder = copy.deepcopy(self.encoder)
        for param in momentum_encoder.parameters():
            param.requires_grad = False
        return momentum_encoder

    @torch.no_grad()
    def update_momentum_encoder(self) -> None:
        """Update momentum encoder with exponential moving average."""
        for param, momentum_param in zip(
            self.encoder.parameters(), self.momentum_encoder.parameters()
        ):
            momentum_param.data = (
                self.momentum * momentum_param.data + (1 - self.momentum) * param.data
            )

    def forward(self, x, use_momentum: bool = False):
        if use_momentum:
            return self.momentum_encoder(x)
        return self.encoder(x)


class ContrastiveLearner:
    """
    Contrastive learning trainer for molecular representations.

    Args:
        encoder: Molecular encoder network
        projector: Projection head
        config: Contrastive learning configuration
        device: Device for training
    """

    def __init__(
        self,
        encoder: nn.Module,
        projector: Optional[nn.Module] = None,
        config: Optional[ContrastiveConfig] = None,
        device: str = "cuda",
    ):
        self.encoder = encoder.to(device)
        self.config = config or ContrastiveConfig()
        self.device = device

        if projector is None:
            # Auto-detect encoder output dimension
            projector = ProjectionHead(
                input_dim=256,  # Default
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.projection_dim,
            )
        self.projector = projector.to(device)

        self.augmenter = MolecularAugmentation()

        # Loss function
        if self.config.loss_type == "infonce":
            self.criterion = InfoNCELoss(self.config.temperature)
        elif self.config.loss_type == "ntxent":
            self.criterion = NTXentLoss(self.config.temperature)
        elif self.config.loss_type == "triplet":
            self.criterion = TripletLoss()
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Get projected embeddings."""
        h = self.encoder(x, edge_index, edge_attr, batch)
        z = self.projector(h)
        return z

    def train_step(
        self,
        batch_data,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """
        Perform one contrastive training step.

        Args:
            batch_data: Graph batch from DataLoader
            optimizer: Optimizer

        Returns:
            Dict with loss values
        """
        self.encoder.train()
        self.projector.train()

        x = batch_data.x.to(self.device)
        edge_index = batch_data.edge_index.to(self.device)
        edge_attr = getattr(batch_data, "edge_attr", None)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)
        batch = batch_data.batch.to(self.device)

        # Generate two augmented views
        x1, edge_index1, batch1, edge_attr1 = self.augmenter.augment(
            x.clone(), edge_index.clone(), batch.clone(),
            edge_attr.clone() if edge_attr is not None else None
        )
        x2, edge_index2, batch2, edge_attr2 = self.augmenter.augment(
            x.clone(), edge_index.clone(), batch.clone(),
            edge_attr.clone() if edge_attr is not None else None
        )

        # Get embeddings
        z1 = self.get_embeddings(x1, edge_index1, edge_attr1, batch1)
        z2 = self.get_embeddings(x2, edge_index2, edge_attr2, batch2)

        # Compute loss
        optimizer.zero_grad()

        if self.config.loss_type in ["infonce", "ntxent"]:
            loss = self.criterion(z1, z2)
        else:
            # For triplet loss, use in-batch negatives
            neg_idx = torch.randperm(z1.size(0))
            loss = self.criterion(z1, z2, z2[neg_idx])

        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss using labels.

    Pulls together samples with same label, pushes apart different labels.

    Reference: Khosla et al., "Supervised Contrastive Learning"

    Args:
        temperature: Temperature parameter
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            features: Embeddings [N, D]
            labels: Labels [N, T] for multi-label or [N] for single-label

        Returns:
            Loss value
        """
        features = F.normalize(features, dim=-1)
        batch_size = features.size(0)

        # Create mask for positive pairs (same label)
        if len(labels.shape) > 1:
            # Multi-label: positive if any label matches
            labels_match = torch.mm(labels.float(), labels.float().t())
            mask = (labels_match > 0).float()
        else:
            mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Remove self-similarity
        mask = mask - torch.eye(batch_size, device=features.device)

        # Similarity matrix
        similarity = torch.mm(features, features.t()) / self.temperature

        # Log-sum-exp trick for numerical stability
        sim_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_max.detach()

        # Compute log probability
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(
            exp_sim.sum(dim=1, keepdim=True) - exp_sim.diag().unsqueeze(1)
        )

        # Mean of log-likelihood over positive pairs
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.clamp(mask_sum, min=1)  # Avoid division by zero

        mean_log_prob = (mask * log_prob).sum(dim=1) / mask_sum

        loss = -mean_log_prob.mean()

        return loss


def create_contrastive_pretrainer(
    encoder: nn.Module,
    config: Optional[ContrastiveConfig] = None,
    use_momentum: bool = False,
    device: str = "cuda",
) -> ContrastiveLearner:
    """
    Create a contrastive learning pretrainer.

    Args:
        encoder: Molecular encoder
        config: Configuration
        use_momentum: Whether to use MoCo-style momentum
        device: Device for training

    Returns:
        ContrastiveLearner instance
    """
    config = config or ContrastiveConfig()

    if use_momentum:
        encoder = MomentumEncoder(encoder)

    projector = ProjectionHead(
        input_dim=256,
        hidden_dim=config.hidden_dim,
        output_dim=config.projection_dim,
    )

    return ContrastiveLearner(encoder, projector, config, device)
