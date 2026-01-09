"""Model checkpointing utilities for saving and loading training state."""

import json
from pathlib import Path
from typing import Optional, Any
from datetime import datetime
import torch


class CheckpointManager:
    """
    Manage model checkpoints during training.

    Features:
    - Save best model based on validation metric
    - Save periodic checkpoints
    - Resume training from checkpoint
    - Keep only N most recent checkpoints

    Usage:
        checkpoint_mgr = CheckpointManager(
            checkpoint_dir="models/checkpoints",
            model_name="gnn",
            keep_n_checkpoints=3
        )

        # During training
        if checkpoint_mgr.is_best(val_auc):
            checkpoint_mgr.save_best(model, optimizer, epoch, val_auc)

        # Resume training
        start_epoch = checkpoint_mgr.load_latest(model, optimizer)
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        model_name: str = "model",
        keep_n_checkpoints: int = 3,
        metric_name: str = "val_auc",
        higher_is_better: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.keep_n_checkpoints = keep_n_checkpoints
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better

        self.best_metric = float('-inf') if higher_is_better else float('inf')
        self.checkpoints = []

        self._load_checkpoint_history()

    def _load_checkpoint_history(self):
        """Load existing checkpoint history."""
        history_file = self.checkpoint_dir / f"{self.model_name}_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                data = json.load(f)
                self.best_metric = data.get('best_metric', self.best_metric)
                self.checkpoints = data.get('checkpoints', [])

    def _save_checkpoint_history(self):
        """Save checkpoint history."""
        history_file = self.checkpoint_dir / f"{self.model_name}_history.json"
        with open(history_file, 'w') as f:
            json.dump({
                'best_metric': self.best_metric,
                'checkpoints': self.checkpoints,
                'model_name': self.model_name,
            }, f, indent=2)

    def is_best(self, metric: float) -> bool:
        """Check if current metric is the best so far."""
        if self.higher_is_better:
            return metric > self.best_metric
        return metric < self.best_metric

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric: float,
        scheduler: Optional[Any] = None,
        extra_data: Optional[dict] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metric: Current metric value
            scheduler: Optional learning rate scheduler
            extra_data: Optional extra data to save
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_name}_epoch{epoch:04d}_{timestamp}.pt"
        filepath = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            self.metric_name: metric,
            'timestamp': timestamp,
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if extra_data:
            checkpoint.update(extra_data)

        torch.save(checkpoint, filepath)

        # Update checkpoints list
        self.checkpoints.append({
            'filename': filename,
            'epoch': epoch,
            'metric': metric,
            'timestamp': timestamp,
        })

        # Update best metric
        if is_best:
            self.best_metric = metric
            # Also save as best model
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
            torch.save(checkpoint, best_path)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        self._save_checkpoint_history()

        return str(filepath)

    def save_best(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric: float,
        **kwargs,
    ) -> str:
        """Save checkpoint if it's the best so far."""
        is_best = self.is_best(metric)
        return self.save_checkpoint(model, optimizer, epoch, metric, is_best=is_best, **kwargs)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the N most recent."""
        if len(self.checkpoints) <= self.keep_n_checkpoints:
            return

        # Sort by timestamp (newest first)
        self.checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)

        # Remove old checkpoints
        to_remove = self.checkpoints[self.keep_n_checkpoints:]
        self.checkpoints = self.checkpoints[:self.keep_n_checkpoints]

        for ckpt in to_remove:
            filepath = self.checkpoint_dir / ckpt['filename']
            if filepath.exists():
                filepath.unlink()

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str | Path] = None,
        device: torch.device = torch.device('cpu'),
    ) -> dict:
        """
        Load a checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            checkpoint_path: Path to checkpoint (default: best checkpoint)
            device: Device to load to

        Returns:
            Dictionary with checkpoint data
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pt"

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device('cpu'),
    ) -> dict:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        return self.load_checkpoint(model, optimizer, device=device, checkpoint_path=best_path)

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device('cpu'),
    ) -> int:
        """
        Load the latest checkpoint and return the epoch to resume from.

        Returns:
            Epoch number to resume training from
        """
        if not self.checkpoints:
            return 0

        # Get most recent checkpoint
        latest = max(self.checkpoints, key=lambda x: x['timestamp'])
        checkpoint_path = self.checkpoint_dir / latest['filename']

        checkpoint = self.load_checkpoint(model, optimizer, device=device, checkpoint_path=checkpoint_path)

        return checkpoint.get('epoch', 0) + 1

    def get_best_metric(self) -> float:
        """Get the best metric value achieved."""
        return self.best_metric

    def list_checkpoints(self) -> list[dict]:
        """List all available checkpoints."""
        return sorted(self.checkpoints, key=lambda x: x['epoch'])


def save_model_for_inference(
    model: torch.nn.Module,
    filepath: str | Path,
    model_config: Optional[dict] = None,
    metadata: Optional[dict] = None,
):
    """
    Save model for inference (without optimizer state).

    Args:
        model: Model to save
        filepath: Output path
        model_config: Model configuration for reconstruction
        metadata: Optional metadata
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }

    if model_config:
        save_data['model_config'] = model_config

    if metadata:
        save_data['metadata'] = metadata

    torch.save(save_data, filepath)


def load_model_for_inference(
    model: torch.nn.Module,
    filepath: str | Path,
    device: torch.device = torch.device('cpu'),
) -> dict:
    """
    Load model for inference.

    Args:
        model: Model instance to load weights into
        filepath: Path to saved model
        device: Device to load to

    Returns:
        Dictionary with metadata and config
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return {
        'config': checkpoint.get('model_config'),
        'metadata': checkpoint.get('metadata'),
        'timestamp': checkpoint.get('timestamp'),
    }
