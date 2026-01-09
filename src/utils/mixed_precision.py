"""Mixed precision training utilities for faster training and reduced memory."""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Callable, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""

    enabled: bool = True
    dtype: torch.dtype = torch.float16
    loss_scale: str = "dynamic"  # "dynamic" or float value
    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    max_scale: float = 2**24


class MixedPrecisionTrainer:
    """
    Mixed precision training wrapper for PyTorch models.

    Enables FP16/BF16 training with automatic loss scaling
    for faster training and reduced GPU memory usage.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        config: MixedPrecisionConfig
        device: Device to use
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[MixedPrecisionConfig] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or MixedPrecisionConfig()
        self.device = device

        # Check if mixed precision is supported
        self.enabled = self.config.enabled and torch.cuda.is_available()

        if self.enabled:
            # Initialize gradient scaler
            self.scaler = GradScaler(
                init_scale=self.config.init_scale,
                growth_factor=self.config.growth_factor,
                backoff_factor=self.config.backoff_factor,
                growth_interval=self.config.growth_interval,
                enabled=True,
            )
            logger.info(f"Mixed precision training enabled with {self.config.dtype}")
        else:
            self.scaler = None
            logger.info("Mixed precision training disabled")

        # Statistics
        self.overflow_count = 0
        self.step_count = 0

    def forward_backward(
        self,
        inputs: tuple,
        targets: torch.Tensor,
        loss_fn: Callable,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform forward and backward pass with mixed precision.

        Args:
            inputs: Model inputs (as tuple)
            targets: Target tensor
            loss_fn: Loss function
            mask: Optional mask tensor

        Returns:
            Tuple of (loss, predictions)
        """
        self.step_count += 1

        if self.enabled:
            with autocast(dtype=self.config.dtype):
                predictions = self.model(*inputs)
                if mask is not None:
                    loss = loss_fn(predictions, targets, mask)
                else:
                    loss = loss_fn(predictions, targets)

            # Scale loss and backward
            self.scaler.scale(loss).backward()
        else:
            predictions = self.model(*inputs)
            if mask is not None:
                loss = loss_fn(predictions, targets, mask)
            else:
                loss = loss_fn(predictions, targets)
            loss.backward()

        return loss, predictions

    def step(self, max_grad_norm: Optional[float] = None) -> bool:
        """
        Perform optimizer step with gradient scaling.

        Args:
            max_grad_norm: Maximum gradient norm for clipping

        Returns:
            True if step was successful (no overflow)
        """
        if self.enabled:
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_grad_norm,
                )

            # Check for overflow
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            new_scale = self.scaler.get_scale()

            if new_scale < old_scale:
                self.overflow_count += 1
                return False
            return True
        else:
            # Standard training
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_grad_norm,
                )
            self.optimizer.step()
            return True

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad()

    def get_scale(self) -> float:
        """Get current loss scale."""
        if self.enabled and self.scaler:
            return self.scaler.get_scale()
        return 1.0

    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        state = {
            "optimizer": self.optimizer.state_dict(),
            "overflow_count": self.overflow_count,
            "step_count": self.step_count,
        }
        if self.enabled and self.scaler:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint."""
        self.optimizer.load_state_dict(state["optimizer"])
        self.overflow_count = state.get("overflow_count", 0)
        self.step_count = state.get("step_count", 0)
        if self.enabled and self.scaler and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            "enabled": self.enabled,
            "dtype": str(self.config.dtype) if self.enabled else "float32",
            "current_scale": self.get_scale(),
            "overflow_count": self.overflow_count,
            "step_count": self.step_count,
            "overflow_rate": self.overflow_count / max(1, self.step_count),
        }


def get_optimal_dtype(device: torch.device) -> torch.dtype:
    """
    Get optimal dtype for mixed precision on device.

    Args:
        device: PyTorch device

    Returns:
        Recommended dtype (float16 or bfloat16)
    """
    if device.type != "cuda":
        return torch.float32

    # Check for bfloat16 support (Ampere+)
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def convert_to_channels_last(model: nn.Module) -> nn.Module:
    """
    Convert model to channels_last memory format for better performance.

    Args:
        model: PyTorch model

    Returns:
        Model in channels_last format
    """
    return model.to(memory_format=torch.channels_last)


class AutocastWrapper(nn.Module):
    """
    Wrapper that automatically applies autocast during forward.

    Args:
        model: Model to wrap
        dtype: Autocast dtype
    """

    def __init__(self, model: nn.Module, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, *args, **kwargs):
        with autocast(dtype=self.dtype):
            return self.model(*args, **kwargs)


def benchmark_precision(
    model: nn.Module,
    inputs: tuple,
    num_iterations: int = 100,
    warmup: int = 10,
) -> dict:
    """
    Benchmark different precision modes.

    Args:
        model: Model to benchmark
        inputs: Model inputs
        num_iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        Dictionary with benchmark results
    """
    import time

    model.eval()
    results = {}

    # FP32 baseline
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(*inputs)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(num_iterations):
            _ = model(*inputs)
        torch.cuda.synchronize()
        results["fp32_time"] = (time.time() - start) / num_iterations

    # FP16
    with torch.no_grad():
        with autocast(dtype=torch.float16):
            # Warmup
            for _ in range(warmup):
                _ = model(*inputs)
            torch.cuda.synchronize()

            start = time.time()
            for _ in range(num_iterations):
                _ = model(*inputs)
            torch.cuda.synchronize()
            results["fp16_time"] = (time.time() - start) / num_iterations

    # BF16 if supported
    if torch.cuda.is_bf16_supported():
        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                # Warmup
                for _ in range(warmup):
                    _ = model(*inputs)
                torch.cuda.synchronize()

                start = time.time()
                for _ in range(num_iterations):
                    _ = model(*inputs)
                torch.cuda.synchronize()
                results["bf16_time"] = (time.time() - start) / num_iterations

    # Calculate speedups
    results["fp16_speedup"] = results["fp32_time"] / results["fp16_time"]
    if "bf16_time" in results:
        results["bf16_speedup"] = results["fp32_time"] / results["bf16_time"]

    return results
