"""Custom loss functions for molecular property prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

    The focal loss down-weights easy examples and focuses on hard negatives:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter, higher = more focus on hard examples (default: 2.0)
        reduction: Reduction method ('none', 'mean', 'sum')
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predicted logits (N, num_tasks) or (N,)
            targets: Ground truth labels (N, num_tasks) or (N,)
            mask: Optional mask for valid samples (N, num_tasks) or (N,)

        Returns:
            Focal loss value
        """
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)

        # Binary cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction="none"
        )

        # Compute p_t
        p_t = p * targets + (1 - p) * (1 - targets)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Compute focal loss
        focal_loss = alpha_t * focal_weight * ce_loss

        # Apply mask if provided
        if mask is not None:
            focal_loss = focal_loss * mask

        # Apply reduction
        if self.reduction == "mean":
            if mask is not None:
                return focal_loss.sum() / mask.sum().clamp(min=1)
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss with per-class weights.

    Useful for handling class imbalance by assigning higher weights
    to the minority class.

    Args:
        pos_weight: Weight for positive class (default: None, auto-computed)
        reduction: Reduction method ('none', 'mean', 'sum')
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute weighted BCE loss."""
        loss = F.binary_cross_entropy_with_logits(
            inputs,
            targets.float(),
            pos_weight=self.pos_weight,
            reduction="none",
        )

        if mask is not None:
            loss = loss * mask

        if self.reduction == "mean":
            if mask is not None:
                return loss.sum() / mask.sum().clamp(min=1)
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification with class imbalance.

    Reference: Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification"

    Uses different gamma values for positive and negative samples.

    Args:
        gamma_neg: Focusing parameter for negative samples
        gamma_pos: Focusing parameter for positive samples
        clip: Probability margin for clipping negative samples
        reduction: Reduction method
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute asymmetric loss."""
        # Probability
        p = torch.sigmoid(inputs)

        # Positive and negative terms
        xs_pos = p
        xs_neg = 1 - p

        # Asymmetric clipping
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss = -one_sided_w * (los_pos + los_neg)
        else:
            loss = -(los_pos + los_neg)

        if mask is not None:
            loss = loss * mask

        if self.reduction == "mean":
            if mask is not None:
                return loss.sum() / mask.sum().clamp(min=1)
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingBCELoss(nn.Module):
    """
    Binary Cross-Entropy with Label Smoothing.

    Softens hard labels to prevent overconfidence:
    smoothed_label = label * (1 - smoothing) + 0.5 * smoothing

    Args:
        smoothing: Label smoothing factor (0.0 = no smoothing)
        reduction: Reduction method
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute label-smoothed BCE loss."""
        # Apply label smoothing
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        loss = F.binary_cross_entropy_with_logits(
            inputs, targets_smooth.float(), reduction="none"
        )

        if mask is not None:
            loss = loss * mask

        if self.reduction == "mean":
            if mask is not None:
                return loss.sum() / mask.sum().clamp(min=1)
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with learnable task weights.

    Implements uncertainty-based task weighting from:
    Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses"

    Args:
        num_tasks: Number of tasks
        base_loss: Base loss function to use
        learnable_weights: Whether to learn task weights
    """

    def __init__(
        self,
        num_tasks: int,
        base_loss: Optional[nn.Module] = None,
        learnable_weights: bool = True,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.base_loss = base_loss or nn.BCEWithLogitsLoss(reduction="none")
        self.learnable_weights = learnable_weights

        if learnable_weights:
            # Log variance parameters (initialized to 0 = weight of 1)
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        else:
            self.register_buffer("log_vars", torch.zeros(num_tasks))

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-task loss with learned weights.

        Returns:
            Tuple of (total_loss, per_task_losses)
        """
        # Compute per-task losses
        losses = []
        for i in range(self.num_tasks):
            task_inputs = inputs[:, i]
            task_targets = targets[:, i]

            if mask is not None:
                task_mask = mask[:, i]
                task_loss = self.base_loss(task_inputs, task_targets.float())
                task_loss = (task_loss * task_mask).sum() / task_mask.sum().clamp(min=1)
            else:
                task_loss = self.base_loss(task_inputs, task_targets.float()).mean()

            losses.append(task_loss)

        losses = torch.stack(losses)

        # Apply uncertainty weighting
        # precision = exp(-log_var), loss = precision * loss + log_var
        precision = torch.exp(-self.log_vars)
        weighted_losses = precision * losses + self.log_vars

        total_loss = weighted_losses.sum()

        return total_loss, losses

    def get_task_weights(self) -> torch.Tensor:
        """Get current task weights (precision = exp(-log_var))."""
        return torch.exp(-self.log_vars).detach()


class CombinedLoss(nn.Module):
    """
    Combine multiple loss functions with configurable weights.

    Args:
        losses: List of (loss_fn, weight) tuples
    """

    def __init__(self, losses: list[tuple[nn.Module, float]]):
        super().__init__()
        self.losses = nn.ModuleList([loss for loss, _ in losses])
        self.weights = [weight for _, weight in losses]

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute combined loss."""
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(inputs, targets, mask)
        return total_loss


def compute_class_weights(targets: np.ndarray, mask: Optional[np.ndarray] = None) -> torch.Tensor:
    """
    Compute positive class weights based on class frequencies.

    Args:
        targets: Target labels (n_samples, n_tasks)
        mask: Optional mask for valid samples

    Returns:
        Positive class weights (n_tasks,)
    """
    n_tasks = targets.shape[1]
    pos_weights = []

    for i in range(n_tasks):
        if mask is not None:
            task_mask = mask[:, i].astype(bool)
            task_targets = targets[task_mask, i]
        else:
            task_targets = targets[:, i]
            task_targets = task_targets[task_targets >= 0]  # Remove -1 labels

        if len(task_targets) == 0:
            pos_weights.append(1.0)
            continue

        n_pos = (task_targets == 1).sum()
        n_neg = (task_targets == 0).sum()

        if n_pos == 0:
            pos_weights.append(1.0)
        else:
            # Weight = n_neg / n_pos
            pos_weights.append(n_neg / n_pos)

    return torch.tensor(pos_weights, dtype=torch.float32)


def get_loss_function(
    name: str,
    num_tasks: int = 12,
    pos_weight: Optional[torch.Tensor] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to get loss by name.

    Args:
        name: Loss function name
        num_tasks: Number of tasks (for multi-task loss)
        pos_weight: Positive class weights
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function module
    """
    losses = {
        "bce": lambda: nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        "focal": lambda: FocalLoss(**kwargs),
        "weighted_bce": lambda: WeightedBCELoss(pos_weight=pos_weight, **kwargs),
        "asymmetric": lambda: AsymmetricLoss(**kwargs),
        "label_smoothing": lambda: LabelSmoothingBCELoss(**kwargs),
        "multitask": lambda: MultiTaskLoss(num_tasks=num_tasks, **kwargs),
    }

    if name not in losses:
        raise ValueError(f"Unknown loss function: {name}. Available: {list(losses.keys())}")

    return losses[name]()
