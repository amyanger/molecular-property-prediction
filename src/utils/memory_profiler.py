"""Memory profiling utilities for training monitoring."""

import torch
import torch.nn as nn
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import gc
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage."""

    timestamp: str
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    peak_reserved_mb: float
    cpu_memory_mb: float
    label: str = ""


@dataclass
class LayerMemory:
    """Memory usage for a single layer."""

    name: str
    params_mb: float
    activations_mb: float
    gradients_mb: float
    total_mb: float


@dataclass
class MemoryProfile:
    """Complete memory profile."""

    total_params_mb: float
    total_activations_mb: float
    total_gradients_mb: float
    total_model_mb: float
    peak_memory_mb: float
    layer_breakdown: list[LayerMemory] = field(default_factory=list)
    snapshots: list[MemorySnapshot] = field(default_factory=list)


class GPUMemoryProfiler:
    """
    Profile GPU memory usage during training.

    Tracks memory allocation, peaks, and per-layer usage
    to identify memory bottlenecks.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.snapshots: list[MemorySnapshot] = []
        self.is_tracking = False

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, GPU profiling disabled")

    def reset_stats(self) -> None:
        """Reset CUDA memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        """
        Take a memory snapshot.

        Args:
            label: Label for this snapshot

        Returns:
            MemorySnapshot
        """
        if not torch.cuda.is_available():
            return MemorySnapshot(
                timestamp=datetime.now().isoformat(),
                allocated_mb=0,
                reserved_mb=0,
                max_allocated_mb=0,
                peak_reserved_mb=0,
                cpu_memory_mb=self._get_cpu_memory(),
                label=label,
            )

        snapshot = MemorySnapshot(
            timestamp=datetime.now().isoformat(),
            allocated_mb=torch.cuda.memory_allocated() / 1024**2,
            reserved_mb=torch.cuda.memory_reserved() / 1024**2,
            max_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2,
            peak_reserved_mb=torch.cuda.max_memory_reserved() / 1024**2,
            cpu_memory_mb=self._get_cpu_memory(),
            label=label,
        )

        self.snapshots.append(snapshot)
        return snapshot

    def _get_cpu_memory(self) -> float:
        """Get current CPU memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024**2
        except ImportError:
            return 0.0

    def profile_model(
        self,
        model: nn.Module,
        sample_input: tuple,
        backward: bool = True,
    ) -> MemoryProfile:
        """
        Profile memory usage for a model.

        Args:
            model: Model to profile
            sample_input: Sample input for forward pass
            backward: Whether to include backward pass

        Returns:
            MemoryProfile
        """
        model.to(self.device)
        model.train()
        self.reset_stats()

        layer_memories = []
        total_params_mb = 0
        total_activations_mb = 0
        total_gradients_mb = 0

        # Profile parameters
        for name, param in model.named_parameters():
            param_mb = param.numel() * param.element_size() / 1024**2
            total_params_mb += param_mb

        # Move inputs to device
        inputs = tuple(
            x.to(self.device) if isinstance(x, torch.Tensor) else x
            for x in sample_input
        )

        # Take baseline snapshot
        self.take_snapshot("baseline")

        # Forward pass
        gc.collect()
        torch.cuda.empty_cache()
        before_forward = self.take_snapshot("before_forward")

        outputs = model(*inputs)

        after_forward = self.take_snapshot("after_forward")
        total_activations_mb = after_forward.allocated_mb - before_forward.allocated_mb

        # Backward pass
        if backward and outputs.requires_grad:
            loss = outputs.sum()

            before_backward = self.take_snapshot("before_backward")
            loss.backward()
            after_backward = self.take_snapshot("after_backward")

            total_gradients_mb = after_backward.allocated_mb - before_backward.allocated_mb

        # Profile per-layer memory
        for name, module in model.named_modules():
            if list(module.children()):  # Skip container modules
                continue

            param_mb = sum(
                p.numel() * p.element_size() / 1024**2
                for p in module.parameters()
            )

            grad_mb = sum(
                p.grad.numel() * p.grad.element_size() / 1024**2
                for p in module.parameters()
                if p.grad is not None
            )

            if param_mb > 0:
                layer_memories.append(LayerMemory(
                    name=name,
                    params_mb=param_mb,
                    activations_mb=0,  # Hard to measure per-layer
                    gradients_mb=grad_mb,
                    total_mb=param_mb + grad_mb,
                ))

        # Sort by memory usage
        layer_memories.sort(key=lambda x: x.total_mb, reverse=True)

        return MemoryProfile(
            total_params_mb=total_params_mb,
            total_activations_mb=total_activations_mb,
            total_gradients_mb=total_gradients_mb,
            total_model_mb=total_params_mb + total_activations_mb + total_gradients_mb,
            peak_memory_mb=self.snapshots[-1].max_allocated_mb if self.snapshots else 0,
            layer_breakdown=layer_memories,
            snapshots=self.snapshots.copy(),
        )

    def profile_batch_sizes(
        self,
        model: nn.Module,
        input_generator: Callable[[int], tuple],
        batch_sizes: list[int],
    ) -> dict[int, float]:
        """
        Profile memory usage for different batch sizes.

        Args:
            model: Model to profile
            input_generator: Function that generates inputs for batch size
            batch_sizes: List of batch sizes to test

        Returns:
            Dict mapping batch size to peak memory (MB)
        """
        results = {}

        for batch_size in batch_sizes:
            self.reset_stats()
            gc.collect()
            torch.cuda.empty_cache()

            try:
                sample_input = input_generator(batch_size)
                profile = self.profile_model(model, sample_input)
                results[batch_size] = profile.peak_memory_mb
                logger.info(f"Batch size {batch_size}: {profile.peak_memory_mb:.1f} MB")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"Batch size {batch_size}: OOM")
                    results[batch_size] = float("inf")
                else:
                    raise

        return results

    def find_max_batch_size(
        self,
        model: nn.Module,
        input_generator: Callable[[int], tuple],
        start_batch: int = 1,
        max_batch: int = 1024,
        memory_limit_mb: Optional[float] = None,
    ) -> int:
        """
        Find maximum batch size that fits in memory.

        Args:
            model: Model to profile
            input_generator: Function that generates inputs for batch size
            start_batch: Starting batch size
            max_batch: Maximum batch size to try
            memory_limit_mb: Memory limit (None = use total GPU memory)

        Returns:
            Maximum batch size
        """
        if memory_limit_mb is None and torch.cuda.is_available():
            memory_limit_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2

        max_working = start_batch
        current = start_batch

        while current <= max_batch:
            self.reset_stats()
            gc.collect()
            torch.cuda.empty_cache()

            try:
                sample_input = input_generator(current)
                profile = self.profile_model(model, sample_input)

                if profile.peak_memory_mb < memory_limit_mb * 0.9:  # 90% threshold
                    max_working = current
                    current *= 2
                else:
                    break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                raise

        # Binary search for exact max
        low, high = max_working, min(max_working * 2, max_batch)

        while low < high - 1:
            mid = (low + high) // 2
            self.reset_stats()
            gc.collect()
            torch.cuda.empty_cache()

            try:
                sample_input = input_generator(mid)
                profile = self.profile_model(model, sample_input)

                if profile.peak_memory_mb < memory_limit_mb * 0.9:
                    low = mid
                else:
                    high = mid
            except RuntimeError:
                high = mid

        return low


class MemoryTracker:
    """
    Track memory usage during training.

    Records memory at regular intervals for later analysis.
    """

    def __init__(self, interval: int = 10):
        """
        Args:
            interval: Steps between memory recordings
        """
        self.interval = interval
        self.profiler = GPUMemoryProfiler()
        self.step_count = 0
        self.history: list[dict] = []

    def step(self, metrics: Optional[dict] = None) -> None:
        """
        Record memory at current step.

        Args:
            metrics: Optional metrics to record alongside memory
        """
        self.step_count += 1

        if self.step_count % self.interval == 0:
            snapshot = self.profiler.take_snapshot(f"step_{self.step_count}")

            record = {
                "step": self.step_count,
                "allocated_mb": snapshot.allocated_mb,
                "reserved_mb": snapshot.reserved_mb,
                "max_allocated_mb": snapshot.max_allocated_mb,
                "cpu_memory_mb": snapshot.cpu_memory_mb,
            }

            if metrics:
                record.update(metrics)

            self.history.append(record)

    def get_summary(self) -> dict:
        """Get memory usage summary."""
        if not self.history:
            return {}

        allocated = [h["allocated_mb"] for h in self.history]
        reserved = [h["reserved_mb"] for h in self.history]

        return {
            "mean_allocated_mb": sum(allocated) / len(allocated),
            "max_allocated_mb": max(allocated),
            "mean_reserved_mb": sum(reserved) / len(reserved),
            "max_reserved_mb": max(reserved),
            "num_recordings": len(self.history),
        }


def estimate_model_memory(model: nn.Module) -> dict:
    """
    Estimate model memory requirements.

    Args:
        model: PyTorch model

    Returns:
        Dict with memory estimates
    """
    params_memory = 0
    buffers_memory = 0

    for param in model.parameters():
        params_memory += param.numel() * param.element_size()

    for buffer in model.buffers():
        buffers_memory += buffer.numel() * buffer.element_size()

    # Estimate gradient memory (same as params for most cases)
    gradient_memory = params_memory

    return {
        "params_mb": params_memory / 1024**2,
        "buffers_mb": buffers_memory / 1024**2,
        "gradients_mb": gradient_memory / 1024**2,
        "total_mb": (params_memory + buffers_memory + gradient_memory) / 1024**2,
    }
