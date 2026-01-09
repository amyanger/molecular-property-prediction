"""Advanced early stopping utilities for training."""

import numpy as np
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StoppingReason(Enum):
    """Reason for stopping training."""

    NOT_STOPPED = "not_stopped"
    PATIENCE_EXCEEDED = "patience_exceeded"
    TARGET_REACHED = "target_reached"
    MIN_DELTA_NOT_MET = "min_delta_not_met"
    MAX_EPOCHS_REACHED = "max_epochs_reached"
    DIVERGING = "diverging"
    MANUAL = "manual"


@dataclass
class EarlyStoppingState:
    """State of early stopping."""

    best_value: float = float("-inf")
    best_epoch: int = 0
    counter: int = 0
    stopped: bool = False
    reason: StoppingReason = StoppingReason.NOT_STOPPED
    history: list[float] = field(default_factory=list)


class EarlyStopping:
    """
    Advanced early stopping with multiple strategies.

    Monitors a metric and stops training when no improvement
    is seen for a specified number of epochs.

    Args:
        patience: Number of epochs without improvement before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for minimizing, 'max' for maximizing
        baseline: Baseline value to compare against
        restore_best: Whether to restore best weights when stopping
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
        baseline: Optional[float] = None,
        restore_best: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.restore_best = restore_best

        self.state = EarlyStoppingState()
        self.best_weights = None

        # Set comparison function
        if mode == "min":
            self.is_better = lambda new, best: new < best - min_delta
            self.state.best_value = float("inf")
        else:
            self.is_better = lambda new, best: new > best + min_delta
            self.state.best_value = float("-inf")

        if baseline is not None:
            self.state.best_value = baseline

    def step(
        self,
        value: float,
        epoch: int,
        model: Optional[Any] = None,
    ) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value
            epoch: Current epoch number
            model: Optional model to save best weights

        Returns:
            True if training should stop
        """
        self.state.history.append(value)

        if self.is_better(value, self.state.best_value):
            self.state.best_value = value
            self.state.best_epoch = epoch
            self.state.counter = 0

            # Save best weights
            if model is not None and self.restore_best:
                import torch
                self.best_weights = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
        else:
            self.state.counter += 1

        if self.state.counter >= self.patience:
            self.state.stopped = True
            self.state.reason = StoppingReason.PATIENCE_EXCEEDED
            logger.info(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best value: {self.state.best_value:.4f} at epoch {self.state.best_epoch}"
            )
            return True

        return False

    def restore_best_weights(self, model: Any) -> None:
        """Restore model to best weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(f"Restored best weights from epoch {self.state.best_epoch}")

    def reset(self) -> None:
        """Reset early stopping state."""
        if self.mode == "min":
            self.state.best_value = float("inf")
        else:
            self.state.best_value = float("-inf")

        if self.baseline is not None:
            self.state.best_value = self.baseline

        self.state.best_epoch = 0
        self.state.counter = 0
        self.state.stopped = False
        self.state.reason = StoppingReason.NOT_STOPPED
        self.state.history = []
        self.best_weights = None


class ReduceLROnPlateau:
    """
    Reduce learning rate when metric plateaus.

    Similar to PyTorch's ReduceLROnPlateau but with more features.

    Args:
        optimizer: Wrapped optimizer
        mode: 'min' or 'max'
        factor: Factor to reduce LR by
        patience: Number of epochs to wait before reducing
        min_lr: Minimum learning rate
        threshold: Threshold for measuring improvement
        cooldown: Epochs to wait after reducing before resuming
    """

    def __init__(
        self,
        optimizer,
        mode: str = "max",
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-7,
        threshold: float = 1e-4,
        cooldown: int = 0,
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.cooldown = cooldown

        self.best = float("inf") if mode == "min" else float("-inf")
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.num_reductions = 0

        self.is_better = (
            lambda new, best: new < best - threshold
            if mode == "min"
            else lambda new, best: new > best + threshold
        )

    def step(self, value: float) -> bool:
        """
        Update scheduler and potentially reduce LR.

        Args:
            value: Current metric value

        Returns:
            True if LR was reduced
        """
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False

        if self.is_better(value, self.best):
            self.best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            return True

        return False

    def _reduce_lr(self) -> None:
        """Reduce learning rate."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group["lr"]
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group["lr"] = new_lr

        self.num_reductions += 1
        logger.info(f"Reducing learning rate to {new_lr:.2e}")


class DivergenceDetector:
    """
    Detect training divergence.

    Monitors loss and stops training if loss increases significantly
    or becomes NaN/Inf.

    Args:
        threshold: Threshold for loss increase detection
        window_size: Window size for smoothing
        consecutive_increases: Number of consecutive increases to trigger
    """

    def __init__(
        self,
        threshold: float = 2.0,
        window_size: int = 10,
        consecutive_increases: int = 5,
    ):
        self.threshold = threshold
        self.window_size = window_size
        self.consecutive_increases = consecutive_increases

        self.losses = []
        self.increase_counter = 0
        self.baseline_loss = None

    def step(self, loss: float) -> bool:
        """
        Check for divergence.

        Args:
            loss: Current loss value

        Returns:
            True if divergence detected
        """
        # Check for NaN/Inf
        if np.isnan(loss) or np.isinf(loss):
            logger.warning("NaN/Inf loss detected - training diverged")
            return True

        self.losses.append(loss)

        # Set baseline from first window
        if len(self.losses) == self.window_size:
            self.baseline_loss = np.mean(self.losses)

        if self.baseline_loss is None:
            return False

        # Check for significant increase
        if loss > self.baseline_loss * self.threshold:
            self.increase_counter += 1
        else:
            self.increase_counter = 0

        if self.increase_counter >= self.consecutive_increases:
            logger.warning("Training appears to be diverging")
            return True

        return False

    def reset(self) -> None:
        """Reset detector state."""
        self.losses = []
        self.increase_counter = 0
        self.baseline_loss = None


class TargetMetricStopper:
    """
    Stop training when target metric is reached.

    Args:
        target: Target metric value
        mode: 'min' or 'max'
        patience: Epochs to wait after reaching target
    """

    def __init__(
        self,
        target: float,
        mode: str = "max",
        patience: int = 5,
    ):
        self.target = target
        self.mode = mode
        self.patience = patience

        self.target_reached = False
        self.epochs_since_target = 0

    def step(self, value: float) -> bool:
        """
        Check if target is reached.

        Args:
            value: Current metric value

        Returns:
            True if should stop
        """
        if self.mode == "max":
            reached = value >= self.target
        else:
            reached = value <= self.target

        if reached:
            if not self.target_reached:
                self.target_reached = True
                logger.info(f"Target metric {self.target} reached!")

            self.epochs_since_target += 1

            if self.epochs_since_target >= self.patience:
                logger.info("Stopping - target maintained for patience period")
                return True

        return False


class CombinedEarlyStopping:
    """
    Combine multiple early stopping criteria.

    Args:
        stoppers: List of early stopping objects
        mode: 'any' (stop if any triggers) or 'all' (stop if all trigger)
    """

    def __init__(
        self,
        stoppers: list,
        mode: str = "any",
    ):
        self.stoppers = stoppers
        self.mode = mode

    def step(self, **kwargs) -> bool:
        """
        Check all stopping criteria.

        Args:
            **kwargs: Arguments to pass to each stopper

        Returns:
            True if should stop
        """
        results = [stopper.step(**kwargs) for stopper in self.stoppers]

        if self.mode == "any":
            return any(results)
        else:
            return all(results)

    def reset(self) -> None:
        """Reset all stoppers."""
        for stopper in self.stoppers:
            if hasattr(stopper, "reset"):
                stopper.reset()
