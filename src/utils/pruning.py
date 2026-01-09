"""Model pruning utilities for efficient inference."""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Optional, Union, Callable
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class PruningResult:
    """Container for pruning results."""

    original_params: int
    remaining_params: int
    pruned_params: int
    sparsity: float
    layer_sparsities: dict


class ModelPruner:
    """
    Prune neural network models for efficient inference.

    Supports various pruning methods including magnitude-based,
    structured, and gradual pruning.

    Args:
        model: PyTorch model to prune
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.original_state = None
        self._save_original_state()

    def _save_original_state(self) -> None:
        """Save original model state for restoration."""
        self.original_state = {
            name: param.clone() for name, param in self.model.named_parameters()
        }

    def restore_original(self) -> None:
        """Restore model to original unpruned state."""
        if self.original_state is None:
            logger.warning("No original state saved")
            return

        for name, param in self.model.named_parameters():
            if name in self.original_state:
                param.data.copy_(self.original_state[name])

        logger.info("Model restored to original state")

    def prune_magnitude(
        self,
        amount: float = 0.3,
        prune_type: str = "unstructured",
    ) -> PruningResult:
        """
        Prune weights with lowest magnitude.

        Args:
            amount: Fraction of weights to prune (0.0 to 1.0)
            prune_type: "unstructured" or "structured"

        Returns:
            PruningResult
        """
        parameters_to_prune = []

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                parameters_to_prune.append((module, "weight"))

        if prune_type == "unstructured":
            # Global unstructured pruning
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )
        else:
            # Per-layer structured pruning
            for module, name in parameters_to_prune:
                if isinstance(module, nn.Linear):
                    prune.ln_structured(module, name, amount=amount, n=2, dim=0)
                elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                    prune.ln_structured(module, name, amount=amount, n=2, dim=0)

        return self._get_pruning_stats()

    def prune_random(self, amount: float = 0.3) -> PruningResult:
        """
        Randomly prune weights.

        Args:
            amount: Fraction of weights to prune

        Returns:
            PruningResult
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                prune.random_unstructured(module, name="weight", amount=amount)

        return self._get_pruning_stats()

    def prune_gradual(
        self,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.5,
        current_step: int = 0,
        total_steps: int = 100,
    ) -> PruningResult:
        """
        Apply gradual magnitude pruning (cubic schedule).

        Args:
            initial_sparsity: Starting sparsity
            final_sparsity: Target final sparsity
            current_step: Current training step
            total_steps: Total pruning steps

        Returns:
            PruningResult
        """
        # Cubic sparsity schedule
        progress = min(current_step / total_steps, 1.0)
        current_sparsity = final_sparsity + (initial_sparsity - final_sparsity) * \
                          (1 - progress) ** 3

        return self.prune_magnitude(amount=current_sparsity)

    def prune_attention_heads(
        self,
        head_importance: dict[str, np.ndarray],
        num_heads_to_prune: int,
    ) -> PruningResult:
        """
        Prune attention heads based on importance scores.

        Args:
            head_importance: Dict mapping layer names to head importance scores
            num_heads_to_prune: Number of heads to prune per layer

        Returns:
            PruningResult
        """
        for layer_name, importance in head_importance.items():
            module = dict(self.model.named_modules()).get(layer_name)
            if module is None:
                continue

            # Find heads to prune (lowest importance)
            heads_to_prune = np.argsort(importance)[:num_heads_to_prune]

            # Apply custom pruning mask
            if hasattr(module, "weight"):
                mask = torch.ones_like(module.weight)
                head_dim = module.weight.shape[0] // len(importance)

                for head in heads_to_prune:
                    start = head * head_dim
                    end = (head + 1) * head_dim
                    mask[start:end] = 0

                prune.custom_from_mask(module, name="weight", mask=mask)

        return self._get_pruning_stats()

    def _get_pruning_stats(self) -> PruningResult:
        """Get statistics about current pruning state."""
        total_params = 0
        nonzero_params = 0
        layer_sparsities = {}

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if hasattr(module, "weight"):
                    weight = module.weight
                    if hasattr(module, "weight_mask"):
                        weight = weight * module.weight_mask

                    layer_total = weight.numel()
                    layer_nonzero = (weight != 0).sum().item()
                    total_params += layer_total
                    nonzero_params += layer_nonzero

                    layer_sparsities[name] = 1.0 - layer_nonzero / layer_total

        pruned_params = total_params - nonzero_params
        sparsity = pruned_params / total_params if total_params > 0 else 0.0

        return PruningResult(
            original_params=total_params,
            remaining_params=nonzero_params,
            pruned_params=pruned_params,
            sparsity=sparsity,
            layer_sparsities=layer_sparsities,
        )

    def make_permanent(self) -> None:
        """Make pruning permanent (remove masks and zero out weights)."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if hasattr(module, "weight_orig"):
                    prune.remove(module, "weight")

        logger.info("Pruning made permanent")

    def get_layer_importance(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: Callable,
        device: str = "cuda",
    ) -> dict[str, float]:
        """
        Compute layer importance using Fisher information.

        Args:
            dataloader: Data for computing importance
            criterion: Loss function
            device: Device for computation

        Returns:
            Dict mapping layer names to importance scores
        """
        self.model.to(device)
        self.model.eval()

        # Collect gradients
        gradient_norms = {}

        for name, param in self.model.named_parameters():
            if "weight" in name and param.requires_grad:
                gradient_norms[name] = 0.0

        num_batches = 0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in batch[:-1])
                targets = batch[-1].to(device)
            else:
                continue

            self.model.zero_grad()
            outputs = self.model(*inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for name, param in self.model.named_parameters():
                if name in gradient_norms and param.grad is not None:
                    gradient_norms[name] += param.grad.norm().item()

            num_batches += 1

        # Average gradient norms
        for name in gradient_norms:
            gradient_norms[name] /= num_batches

        return gradient_norms


def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        only_trainable: Whether to count only trainable parameters

    Returns:
        Number of parameters
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: nn.Module) -> int:
    """
    Count non-zero parameters (after pruning).

    Args:
        model: PyTorch model

    Returns:
        Number of non-zero parameters
    """
    nonzero = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            nonzero += (param != 0).sum().item()
        else:
            nonzero += param.numel()
    return nonzero


def compute_model_sparsity(model: nn.Module) -> float:
    """
    Compute overall model sparsity.

    Args:
        model: PyTorch model

    Returns:
        Sparsity ratio (0 to 1)
    """
    total = count_parameters(model, only_trainable=False)
    nonzero = count_nonzero_parameters(model)
    return 1.0 - nonzero / total if total > 0 else 0.0


def prune_for_inference(
    model: nn.Module,
    sparsity: float = 0.3,
) -> tuple[nn.Module, PruningResult]:
    """
    Convenience function to prune model for inference.

    Args:
        model: Model to prune
        sparsity: Target sparsity

    Returns:
        Tuple of (pruned_model, pruning_result)
    """
    pruner = ModelPruner(model)
    result = pruner.prune_magnitude(amount=sparsity)
    pruner.make_permanent()

    logger.info(f"Model pruned to {result.sparsity:.1%} sparsity")
    logger.info(f"Parameters: {result.original_params:,} -> {result.remaining_params:,}")

    return model, result
