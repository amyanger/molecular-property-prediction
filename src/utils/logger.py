"""Logging utility for molecular property prediction."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "molprop",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional path to log file
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance

    Usage:
        logger = setup_logger("training", level=logging.DEBUG)
        logger.info("Starting training...")
        logger.debug("Batch size: 64")
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Default format
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "molprop") -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If no handlers, set up with defaults
    if not logger.handlers:
        return setup_logger(name)

    return logger


class TrainingLogger:
    """
    Specialized logger for training progress.

    Provides convenient methods for logging training metrics.

    Usage:
        logger = TrainingLogger("gnn_training")
        logger.log_epoch(1, train_loss=0.5, val_auc=0.82)
        logger.log_best_model(epoch=5, val_auc=0.85)
    """

    def __init__(self, name: str = "training", log_file: Optional[Path] = None):
        self.logger = setup_logger(name, log_file=log_file)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_auc: float,
        lr: Optional[float] = None,
    ) -> None:
        """
        Log training progress for an epoch.

        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
            train_loss: Training loss
            val_auc: Validation AUC
            lr: Current learning rate (optional)
        """
        msg = f"Epoch {epoch:3d}/{total_epochs} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}"
        if lr is not None:
            msg += f" | LR: {lr:.2e}"
        self.logger.info(msg)

    def log_best_model(self, epoch: int, val_auc: float, path: Optional[str] = None) -> None:
        """Log when a new best model is saved."""
        msg = f"New best model at epoch {epoch} (Val AUC: {val_auc:.4f})"
        if path:
            msg += f" -> Saved to {path}"
        self.logger.info(msg)

    def log_early_stopping(self, epoch: int, patience: int) -> None:
        """Log early stopping event."""
        self.logger.info(f"Early stopping triggered at epoch {epoch} (patience: {patience})")

    def log_metrics(self, metrics: dict, prefix: str = "") -> None:
        """
        Log a dictionary of metrics.

        Args:
            metrics: Dictionary of metric names to values
            prefix: Optional prefix for the log message
        """
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        msg = f"{prefix} {metrics_str}" if prefix else metrics_str
        self.logger.info(msg)

    def log_model_summary(self, model_name: str, num_params: int, device: str) -> None:
        """Log model summary information."""
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Parameters: {num_params:,}")
        self.logger.info(f"Device: {device}")

    def log_data_summary(self, train_size: int, val_size: int, test_size: int) -> None:
        """Log dataset split summary."""
        self.logger.info(f"Data splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    def separator(self, char: str = "=", length: int = 60) -> None:
        """Print a separator line."""
        self.logger.info(char * length)


# Global logger instance
_default_logger: Optional[logging.Logger] = None


def get_default_logger() -> logging.Logger:
    """Get or create the default global logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger("molprop")
    return _default_logger
