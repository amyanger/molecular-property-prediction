"""Multi-task learning utilities for molecular property prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """Configuration for a single task."""

    name: str
    weight: float = 1.0
    loss_type: str = "bce"  # "bce", "mse", "focal"
    num_classes: int = 1
    is_classification: bool = True


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task learning."""

    tasks: List[TaskConfig] = field(default_factory=list)
    weighting_strategy: str = "fixed"  # "fixed", "uncertainty", "gradnorm", "pcgrad"
    shared_layers: int = 3
    task_specific_layers: int = 1


class TaskSpecificHead(nn.Module):
    """
    Task-specific prediction head.

    Separate prediction head for each task allowing task-specific
    representations.

    Args:
        input_dim: Input feature dimension
        hidden_dims: Hidden layer dimensions
        output_dim: Output dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        output_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128]

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MultiTaskModel(nn.Module):
    """
    Multi-task model with shared backbone and task-specific heads.

    Args:
        backbone: Shared feature extractor
        num_tasks: Number of tasks
        feature_dim: Dimension of backbone output
        head_hidden_dims: Hidden dimensions for task heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_tasks: int = 12,
        feature_dim: int = 256,
        head_hidden_dims: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.backbone = backbone
        self.num_tasks = num_tasks

        if head_hidden_dims is None:
            head_hidden_dims = [128]

        # Create task-specific heads
        self.task_heads = nn.ModuleList([
            TaskSpecificHead(
                input_dim=feature_dim,
                hidden_dims=head_hidden_dims,
                output_dim=1,
                dropout=dropout,
            )
            for _ in range(num_tasks)
        ])

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through backbone and all task heads.

        Returns:
            Predictions for all tasks [B, num_tasks]
        """
        # Get shared features
        features = self.backbone(*args, **kwargs)

        # Apply task-specific heads
        outputs = []
        for head in self.task_heads:
            outputs.append(head(features))

        return torch.cat(outputs, dim=-1)

    def forward_task(self, task_id: int, *args, **kwargs) -> torch.Tensor:
        """Forward pass for a single task."""
        features = self.backbone(*args, **kwargs)
        return self.task_heads[task_id](features)


class CrossStitchUnit(nn.Module):
    """
    Cross-stitch unit for multi-task learning.

    Learns linear combinations of features from different tasks.

    Reference: Misra et al., "Cross-stitch Networks for Multi-task Learning"

    Args:
        num_tasks: Number of tasks
        feature_dim: Feature dimension
    """

    def __init__(self, num_tasks: int, feature_dim: int):
        super().__init__()
        self.num_tasks = num_tasks

        # Cross-stitch weights (initialized to identity)
        self.weights = nn.Parameter(torch.eye(num_tasks))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply cross-stitch transformation.

        Args:
            features: List of features from each task [T x [B, D]]

        Returns:
            Cross-stitched features [T x [B, D]]
        """
        # Stack features: [B, T, D]
        stacked = torch.stack(features, dim=1)
        batch_size, _, feat_dim = stacked.shape

        # Apply cross-stitch: [T, T] x [B, T, D] -> [B, T, D]
        weights = F.softmax(self.weights, dim=1)
        stitched = torch.einsum("ij,bje->bie", weights, stacked)

        # Split back to list
        return [stitched[:, i] for i in range(self.num_tasks)]


class GradNormWeighting(nn.Module):
    """
    GradNorm adaptive task weighting.

    Automatically balances task weights based on training dynamics.

    Reference: Chen et al., "GradNorm: Gradient Normalization for Adaptive
    Loss Balancing in Deep Multitask Networks"

    Args:
        num_tasks: Number of tasks
        alpha: Asymmetry parameter
    """

    def __init__(self, num_tasks: int, alpha: float = 1.5):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha

        # Learnable task weights
        self.log_weights = nn.Parameter(torch.zeros(num_tasks))

        # Track initial losses
        self.initial_losses = None
        self.training_ratios = None

    def get_weights(self) -> torch.Tensor:
        """Get normalized task weights."""
        weights = torch.exp(self.log_weights)
        return weights / weights.sum() * self.num_tasks

    def forward(
        self,
        losses: torch.Tensor,
        shared_params: List[nn.Parameter],
    ) -> torch.Tensor:
        """
        Compute weighted loss with GradNorm update.

        Args:
            losses: Individual task losses [T]
            shared_params: Shared layer parameters

        Returns:
            Weighted total loss
        """
        weights = self.get_weights()

        # Store initial losses
        if self.initial_losses is None:
            self.initial_losses = losses.detach().clone()

        # Compute training ratios
        current_ratios = losses.detach() / self.initial_losses.clamp(min=1e-8)
        avg_ratio = current_ratios.mean()
        target_ratios = (current_ratios / avg_ratio) ** self.alpha

        # Weighted loss
        weighted_loss = (weights * losses).sum()

        # GradNorm loss for weight update (computed separately)
        if shared_params and self.training:
            grad_norms = []
            for i, loss in enumerate(losses):
                grads = torch.autograd.grad(
                    loss, shared_params, retain_graph=True, allow_unused=True
                )
                grad_norm = sum(g.norm() for g in grads if g is not None)
                grad_norms.append(weights[i] * grad_norm)

            avg_grad_norm = sum(grad_norms) / len(grad_norms)

            # Target gradient norms
            target_grad_norms = avg_grad_norm * target_ratios

            # GradNorm loss
            gradnorm_loss = sum(
                (gn - tgn.detach()).abs()
                for gn, tgn in zip(grad_norms, target_grad_norms)
            )

            return weighted_loss, gradnorm_loss

        return weighted_loss, torch.tensor(0.0)


class UncertaintyWeighting(nn.Module):
    """
    Uncertainty-based task weighting.

    Learns task weights as homoscedastic uncertainty.

    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to
    Weigh Losses for Scene Geometry and Semantics"

    Args:
        num_tasks: Number of tasks
    """

    def __init__(self, num_tasks: int):
        super().__init__()

        # Log variance parameters
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty-weighted loss.

        Args:
            losses: Individual task losses [T]

        Returns:
            Weighted total loss
        """
        # Weight = 1 / (2 * sigma^2) = 0.5 * exp(-log_var)
        precision = 0.5 * torch.exp(-self.log_vars)

        # Weighted loss + regularization term
        weighted_loss = (precision * losses + 0.5 * self.log_vars).sum()

        return weighted_loss


class PCGrad:
    """
    Projecting Conflicting Gradients for multi-task learning.

    Projects gradients to reduce task interference.

    Reference: Yu et al., "Gradient Surgery for Multi-Task Learning"
    """

    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks

    def backward(
        self,
        losses: List[torch.Tensor],
        shared_params: List[nn.Parameter],
    ) -> None:
        """
        Apply PCGrad backward pass.

        Args:
            losses: List of task losses
            shared_params: Shared parameters to update
        """
        # Compute per-task gradients
        task_grads = []
        for loss in losses:
            grads = torch.autograd.grad(
                loss, shared_params, retain_graph=True, allow_unused=True
            )
            flat_grad = torch.cat([
                g.view(-1) if g is not None else torch.zeros(p.numel(), device=loss.device)
                for g, p in zip(grads, shared_params)
            ])
            task_grads.append(flat_grad)

        # Project conflicting gradients
        projected_grads = []
        for i, g_i in enumerate(task_grads):
            g_proj = g_i.clone()
            for j, g_j in enumerate(task_grads):
                if i != j:
                    # Check for conflict
                    dot = torch.dot(g_proj, g_j)
                    if dot < 0:
                        # Project out conflicting component
                        g_proj = g_proj - (dot / (g_j.norm() ** 2 + 1e-8)) * g_j
            projected_grads.append(g_proj)

        # Average projected gradients
        avg_grad = torch.stack(projected_grads).mean(dim=0)

        # Apply gradients
        offset = 0
        for param in shared_params:
            param_size = param.numel()
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            param.grad.add_(avg_grad[offset:offset + param_size].view_as(param))
            offset += param_size


class TaskBalancedBatchSampler:
    """
    Balanced batch sampler for multi-task learning.

    Ensures each batch has balanced representation of valid labels
    across tasks.

    Args:
        labels: Task labels [N, T]
        batch_size: Batch size
        drop_last: Whether to drop last incomplete batch
    """

    def __init__(
        self,
        labels: torch.Tensor,
        batch_size: int,
        drop_last: bool = False,
    ):
        self.labels = labels
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_samples = len(labels)
        self.num_tasks = labels.shape[1]

        # Find valid indices per task
        self.task_indices = []
        for t in range(self.num_tasks):
            valid = (~torch.isnan(labels[:, t])).nonzero().squeeze(-1)
            self.task_indices.append(valid)

    def __iter__(self):
        # Round-robin sampling from tasks
        task_iters = [
            iter(idx[torch.randperm(len(idx))].tolist())
            for idx in self.task_indices
        ]

        batch = []
        task_idx = 0

        while True:
            try:
                sample_idx = next(task_iters[task_idx])
                batch.append(sample_idx)
                task_idx = (task_idx + 1) % self.num_tasks

                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []

            except StopIteration:
                # Restart exhausted iterator
                valid = self.task_indices[task_idx]
                task_iters[task_idx] = iter(valid[torch.randperm(len(valid))].tolist())

                if all(len(idx) == 0 for idx in self.task_indices):
                    break

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        return self.num_samples // self.batch_size


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss with flexible weighting.

    Args:
        num_tasks: Number of tasks
        task_weights: Optional fixed task weights
        weighting_strategy: Weighting strategy
        focal_gamma: Gamma for focal loss
    """

    def __init__(
        self,
        num_tasks: int,
        task_weights: Optional[List[float]] = None,
        weighting_strategy: str = "fixed",
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.weighting_strategy = weighting_strategy
        self.focal_gamma = focal_gamma

        if task_weights is None:
            task_weights = [1.0] * num_tasks
        self.register_buffer("task_weights", torch.tensor(task_weights))

        if weighting_strategy == "uncertainty":
            self.uncertainty_weighting = UncertaintyWeighting(num_tasks)
        elif weighting_strategy == "gradnorm":
            self.gradnorm_weighting = GradNormWeighting(num_tasks)

    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            predictions: Model predictions [B, T]
            labels: Ground truth labels [B, T]
            mask: Valid label mask [B, T]

        Returns:
            Dict with total loss and per-task losses
        """
        if mask is None:
            mask = ~torch.isnan(labels)
            labels = torch.nan_to_num(labels, 0.0)

        # Compute per-task losses
        task_losses = []

        for t in range(self.num_tasks):
            task_pred = predictions[:, t]
            task_label = labels[:, t]
            task_mask = mask[:, t].float()

            # BCE loss
            bce = F.binary_cross_entropy_with_logits(
                task_pred, task_label, reduction="none"
            )

            # Apply focal weighting
            if self.focal_gamma > 0:
                probs = torch.sigmoid(task_pred)
                pt = task_label * probs + (1 - task_label) * (1 - probs)
                focal_weight = (1 - pt) ** self.focal_gamma
                bce = focal_weight * bce

            # Masked mean
            loss = (bce * task_mask).sum() / task_mask.sum().clamp(min=1)
            task_losses.append(loss)

        task_losses = torch.stack(task_losses)

        # Apply weighting strategy
        if self.weighting_strategy == "fixed":
            total_loss = (self.task_weights * task_losses).mean()

        elif self.weighting_strategy == "uncertainty":
            total_loss = self.uncertainty_weighting(task_losses)

        elif self.weighting_strategy == "gradnorm":
            total_loss, _ = self.gradnorm_weighting(task_losses, [])

        else:
            total_loss = task_losses.mean()

        return {
            "total_loss": total_loss,
            "task_losses": task_losses,
        }


class TaskAttention(nn.Module):
    """
    Task attention for adaptive feature aggregation.

    Learns task-specific attention over shared features.

    Args:
        feature_dim: Feature dimension
        num_tasks: Number of tasks
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        feature_dim: int,
        num_tasks: int,
        num_heads: int = 4,
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        # Task embeddings
        self.task_embeddings = nn.Embedding(num_tasks, feature_dim)

        # Attention projections
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply task attention.

        Args:
            features: Shared features [B, D]

        Returns:
            Task-specific features [B, T, D]
        """
        batch_size = features.size(0)

        # Get task queries
        task_ids = torch.arange(self.num_tasks, device=features.device)
        task_emb = self.task_embeddings(task_ids)  # [T, D]

        # Expand features for all tasks
        features = features.unsqueeze(1)  # [B, 1, D]

        # Compute attention
        q = self.q_proj(task_emb).unsqueeze(0).expand(batch_size, -1, -1)  # [B, T, D]
        k = self.k_proj(features)  # [B, 1, D]
        v = self.v_proj(features)  # [B, 1, D]

        # Multi-head attention
        q = q.view(batch_size, self.num_tasks, self.num_heads, self.head_dim)
        k = k.view(batch_size, 1, self.num_heads, self.head_dim)
        v = v.view(batch_size, 1, self.num_heads, self.head_dim)

        attn = torch.einsum("bthd,bshd->bths", q, k) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bths,bshd->bthd", attn, v)
        out = out.view(batch_size, self.num_tasks, -1)

        return self.out_proj(out) + task_emb.unsqueeze(0)


def analyze_task_relationships(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_tasks: int,
    device: str = "cuda",
) -> np.ndarray:
    """
    Analyze task relationships via gradient similarity.

    Args:
        model: Multi-task model
        dataloader: Data loader
        num_tasks: Number of tasks
        device: Device for computation

    Returns:
        Task similarity matrix [T, T]
    """
    model.to(device)
    model.eval()

    task_gradients = [[] for _ in range(num_tasks)]

    for batch in dataloader:
        x, y = batch[0].to(device), batch[1].to(device)
        mask = ~torch.isnan(y)

        for t in range(num_tasks):
            task_mask = mask[:, t]
            if task_mask.sum() == 0:
                continue

            model.zero_grad()
            outputs = model(x)
            loss = F.binary_cross_entropy_with_logits(
                outputs[task_mask, t], y[task_mask, t]
            )
            loss.backward()

            # Flatten gradients
            grad = torch.cat([
                p.grad.view(-1) for p in model.parameters() if p.grad is not None
            ])
            task_gradients[t].append(grad.cpu())

        if len(task_gradients[0]) >= 10:
            break

    # Compute similarity matrix
    similarity = np.zeros((num_tasks, num_tasks))

    for i in range(num_tasks):
        for j in range(num_tasks):
            if task_gradients[i] and task_gradients[j]:
                grad_i = torch.stack(task_gradients[i]).mean(0)
                grad_j = torch.stack(task_gradients[j]).mean(0)
                cos_sim = F.cosine_similarity(grad_i.unsqueeze(0), grad_j.unsqueeze(0))
                similarity[i, j] = cos_sim.item()

    return similarity
