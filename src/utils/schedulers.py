"""Learning rate scheduler utilities for training molecular property prediction models."""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    _LRScheduler,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
    ExponentialLR,
    ReduceLROnPlateau,
    OneCycleLR,
    LinearLR,
    SequentialLR,
)
import math
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.

    The learning rate starts from warmup_start_lr and linearly increases
    to base_lr over warmup_epochs, then follows cosine annealing.

    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of epochs
        warmup_start_lr: Starting learning rate for warmup
        min_lr: Minimum learning rate
        last_epoch: The index of last epoch
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_start_lr: float = 1e-7,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class WarmupLinearScheduler(_LRScheduler):
    """
    Linear decay scheduler with warmup.

    Learning rate linearly increases during warmup, then linearly decreases.

    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of epochs
        warmup_start_lr: Starting learning rate for warmup
        end_lr: Final learning rate
        last_epoch: The index of last epoch
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_start_lr: float = 1e-7,
        end_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            return [
                base_lr + (self.end_lr - base_lr) * progress
                for base_lr in self.base_lrs
            ]


class CyclicCosineScheduler(_LRScheduler):
    """
    Cyclic cosine annealing with warm restarts and cycle length multiplier.

    Similar to SGDR but with customizable cycle growth.

    Args:
        optimizer: Wrapped optimizer
        cycle_length: Length of first cycle
        cycle_mult: Cycle length multiplier
        warmup_epochs: Warmup epochs per cycle
        min_lr: Minimum learning rate
        max_cycles: Maximum number of cycles (None for unlimited)
        last_epoch: The index of last epoch
    """

    def __init__(
        self,
        optimizer: Optimizer,
        cycle_length: int = 10,
        cycle_mult: float = 2.0,
        warmup_epochs: int = 1,
        min_lr: float = 1e-7,
        max_cycles: Optional[int] = None,
        last_epoch: int = -1,
    ):
        self.cycle_length = cycle_length
        self.cycle_mult = cycle_mult
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.max_cycles = max_cycles
        self.current_cycle = 0
        self.cycle_epoch = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Determine current cycle and position within cycle
        epoch = self.last_epoch
        cycle_start = 0
        current_length = self.cycle_length

        while epoch >= cycle_start + current_length:
            cycle_start += current_length
            current_length = int(current_length * self.cycle_mult)
            self.current_cycle += 1

            if self.max_cycles and self.current_cycle >= self.max_cycles:
                # Stay at minimum LR after max cycles
                return [self.min_lr for _ in self.base_lrs]

        self.cycle_epoch = epoch - cycle_start

        if self.cycle_epoch < self.warmup_epochs:
            # Warmup within cycle
            alpha = self.cycle_epoch / max(1, self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine decay within cycle
            progress = (self.cycle_epoch - self.warmup_epochs) / max(
                1, current_length - self.warmup_epochs
            )
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class PolynomialLRScheduler(_LRScheduler):
    """
    Polynomial learning rate decay.

    lr = base_lr * (1 - epoch/total_epochs)^power

    Args:
        optimizer: Wrapped optimizer
        total_epochs: Total number of epochs
        power: Polynomial power (1.0 = linear decay)
        min_lr: Minimum learning rate
        last_epoch: The index of last epoch
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_epochs: int,
        power: float = 1.0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.total_epochs = total_epochs
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.total_epochs:
            return [self.min_lr for _ in self.base_lrs]

        decay_factor = (1 - self.last_epoch / self.total_epochs) ** self.power
        return [
            max(self.min_lr, base_lr * decay_factor)
            for base_lr in self.base_lrs
        ]


class GradualWarmupScheduler(_LRScheduler):
    """
    Gradually warmup learning rate then hand off to another scheduler.

    Args:
        optimizer: Wrapped optimizer
        multiplier: Target learning rate multiplier
        warmup_epochs: Number of warmup epochs
        after_scheduler: Scheduler to use after warmup
    """

    def __init__(
        self,
        optimizer: Optimizer,
        multiplier: float,
        warmup_epochs: int,
        after_scheduler: Optional[_LRScheduler] = None,
    ):
        self.multiplier = multiplier
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [
            base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.warmup_epochs + 1.0)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            self.after_scheduler.step()
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            super().step(epoch)


class LRSchedulerFactory:
    """Factory class for creating learning rate schedulers."""

    @staticmethod
    def create(
        name: str,
        optimizer: Optimizer,
        total_epochs: int,
        **kwargs,
    ) -> Union[_LRScheduler, ReduceLROnPlateau]:
        """
        Create a learning rate scheduler by name.

        Args:
            name: Scheduler name
            optimizer: Wrapped optimizer
            total_epochs: Total training epochs
            **kwargs: Additional scheduler arguments

        Returns:
            Learning rate scheduler
        """
        schedulers = {
            "cosine": lambda: CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
                eta_min=kwargs.get("min_lr", 1e-7),
            ),
            "cosine_warmup": lambda: WarmupCosineScheduler(
                optimizer,
                warmup_epochs=kwargs.get("warmup_epochs", 5),
                total_epochs=total_epochs,
                warmup_start_lr=kwargs.get("warmup_start_lr", 1e-7),
                min_lr=kwargs.get("min_lr", 1e-7),
            ),
            "cosine_warm_restarts": lambda: CosineAnnealingWarmRestarts(
                optimizer,
                T_0=kwargs.get("T_0", 10),
                T_mult=kwargs.get("T_mult", 2),
                eta_min=kwargs.get("min_lr", 1e-7),
            ),
            "cyclic_cosine": lambda: CyclicCosineScheduler(
                optimizer,
                cycle_length=kwargs.get("cycle_length", 10),
                cycle_mult=kwargs.get("cycle_mult", 2.0),
                warmup_epochs=kwargs.get("warmup_epochs", 1),
                min_lr=kwargs.get("min_lr", 1e-7),
            ),
            "linear_warmup": lambda: WarmupLinearScheduler(
                optimizer,
                warmup_epochs=kwargs.get("warmup_epochs", 5),
                total_epochs=total_epochs,
                warmup_start_lr=kwargs.get("warmup_start_lr", 1e-7),
                end_lr=kwargs.get("end_lr", 0.0),
            ),
            "step": lambda: StepLR(
                optimizer,
                step_size=kwargs.get("step_size", 10),
                gamma=kwargs.get("gamma", 0.1),
            ),
            "exponential": lambda: ExponentialLR(
                optimizer,
                gamma=kwargs.get("gamma", 0.95),
            ),
            "polynomial": lambda: PolynomialLRScheduler(
                optimizer,
                total_epochs=total_epochs,
                power=kwargs.get("power", 1.0),
                min_lr=kwargs.get("min_lr", 0.0),
            ),
            "reduce_on_plateau": lambda: ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get("mode", "max"),
                factor=kwargs.get("factor", 0.5),
                patience=kwargs.get("patience", 5),
                min_lr=kwargs.get("min_lr", 1e-7),
                verbose=kwargs.get("verbose", True),
            ),
            "one_cycle": lambda: OneCycleLR(
                optimizer,
                max_lr=kwargs.get("max_lr", optimizer.param_groups[0]["lr"] * 10),
                total_steps=total_epochs * kwargs.get("steps_per_epoch", 100),
                pct_start=kwargs.get("pct_start", 0.3),
                anneal_strategy=kwargs.get("anneal_strategy", "cos"),
            ),
            "constant": lambda: LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=1.0,
                total_iters=total_epochs,
            ),
        }

        if name not in schedulers:
            available = list(schedulers.keys())
            raise ValueError(f"Unknown scheduler: {name}. Available: {available}")

        scheduler = schedulers[name]()
        logger.info(f"Created {name} scheduler")
        return scheduler


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    total_epochs: int,
    **kwargs,
) -> Union[_LRScheduler, ReduceLROnPlateau]:
    """
    Convenience function to get a scheduler by name.

    Args:
        name: Scheduler name
        optimizer: Wrapped optimizer
        total_epochs: Total training epochs
        **kwargs: Additional scheduler arguments

    Returns:
        Learning rate scheduler
    """
    return LRSchedulerFactory.create(name, optimizer, total_epochs, **kwargs)


def create_warmup_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    main_scheduler: str = "cosine",
    **kwargs,
) -> _LRScheduler:
    """
    Create a scheduler with warmup followed by another scheduler.

    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        main_scheduler: Main scheduler to use after warmup
        **kwargs: Additional scheduler arguments

    Returns:
        Sequential scheduler with warmup
    """
    # Warmup scheduler
    warmup = LinearLR(
        optimizer,
        start_factor=kwargs.get("warmup_start_factor", 0.01),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    # Main scheduler
    remaining_epochs = total_epochs - warmup_epochs
    main = get_scheduler(
        main_scheduler,
        optimizer,
        remaining_epochs,
        **kwargs,
    )

    # Combine
    return SequentialLR(
        optimizer,
        schedulers=[warmup, main],
        milestones=[warmup_epochs],
    )
