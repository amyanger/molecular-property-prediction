"""Model evaluation metrics utilities."""

import numpy as np
from typing import Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
)


def compute_auc_roc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Area Under ROC Curve.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        mask: Optional mask for valid samples

    Returns:
        AUC-ROC score
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(np.unique(y_true)) < 2:
        return 0.5

    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5


def compute_auc_pr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Area Under Precision-Recall Curve.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        mask: Optional mask for valid samples

    Returns:
        AUC-PR score
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(np.unique(y_true)) < 2:
        return 0.0

    try:
        return average_precision_score(y_true, y_pred)
    except ValueError:
        return 0.0


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        mask: Optional mask for valid samples

    Returns:
        Dictionary with all metrics
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    y_pred_binary = (y_pred >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
        'auc_roc': compute_auc_roc(y_true, y_pred),
        'auc_pr': compute_auc_pr(y_true, y_pred),
    }

    try:
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred_binary)
    except ValueError:
        metrics['mcc'] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


def compute_multitask_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_names: Optional[list[str]] = None,
    threshold: float = 0.5,
) -> dict:
    """
    Compute metrics for multi-task prediction.

    Args:
        y_true: Ground truth labels (n_samples, n_tasks)
        y_pred: Predicted probabilities (n_samples, n_tasks)
        task_names: Optional list of task names
        threshold: Classification threshold

    Returns:
        Dictionary with per-task and aggregate metrics
    """
    n_tasks = y_true.shape[1]

    if task_names is None:
        task_names = [f"task_{i}" for i in range(n_tasks)]

    results = {
        'per_task': {},
        'aggregate': {},
    }

    all_aucs = []
    all_auc_prs = []

    for i, task in enumerate(task_names):
        # Create mask for valid labels (not -1)
        mask = y_true[:, i] != -1

        if mask.sum() < 10:
            continue

        task_y_true = y_true[mask, i]
        task_y_pred = y_pred[mask, i]

        task_metrics = compute_classification_metrics(
            task_y_true, task_y_pred, threshold
        )
        results['per_task'][task] = task_metrics

        all_aucs.append(task_metrics['auc_roc'])
        all_auc_prs.append(task_metrics['auc_pr'])

    # Aggregate metrics
    results['aggregate'] = {
        'mean_auc_roc': np.mean(all_aucs) if all_aucs else 0.0,
        'std_auc_roc': np.std(all_aucs) if all_aucs else 0.0,
        'mean_auc_pr': np.mean(all_auc_prs) if all_auc_prs else 0.0,
        'std_auc_pr': np.std(all_auc_prs) if all_auc_prs else 0.0,
        'min_auc_roc': min(all_aucs) if all_aucs else 0.0,
        'max_auc_roc': max(all_aucs) if all_aucs else 0.0,
        'n_tasks_evaluated': len(all_aucs),
    }

    return results


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = 'f1',
    n_thresholds: int = 100,
) -> tuple[float, float]:
    """
    Find optimal classification threshold.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        metric: Metric to optimize ('f1', 'accuracy', 'mcc')
        n_thresholds: Number of thresholds to try

    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_threshold = 0.5
    best_score = 0.0

    for t in thresholds:
        y_pred_binary = (y_pred >= t).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred_binary, zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred_binary)
        elif metric == 'mcc':
            try:
                score = matthews_corrcoef(y_true, y_pred_binary)
            except ValueError:
                score = 0.0
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score


def compute_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration metrics (ECE, MCE).

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        Dictionary with calibration metrics
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = y_pred[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()

            bin_error = abs(avg_accuracy - avg_confidence)
            ece += prop_in_bin * bin_error
            mce = max(mce, bin_error)

            bin_data.append({
                'bin': f"({bin_lower:.1f}, {bin_upper:.1f}]",
                'count': int(in_bin.sum()),
                'avg_confidence': float(avg_confidence),
                'avg_accuracy': float(avg_accuracy),
                'error': float(bin_error),
            })

    return {
        'expected_calibration_error': float(ece),
        'max_calibration_error': float(mce),
        'bin_data': bin_data,
    }


class MetricsTracker:
    """Track metrics during training."""

    def __init__(self, task_names: Optional[list[str]] = None):
        self.task_names = task_names
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'epoch': [],
        }
        self.best_val_auc = 0.0
        self.best_epoch = 0

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        val_auc: Optional[float] = None,
    ) -> bool:
        """
        Update metrics and return True if new best.

        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            val_auc: Validation AUC

        Returns:
            True if val_auc is new best
        """
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_auc'].append(val_auc)

        is_best = False
        if val_auc is not None and val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            self.best_epoch = epoch
            is_best = True

        return is_best

    def get_summary(self) -> dict:
        """Get training summary."""
        return {
            'best_val_auc': self.best_val_auc,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'total_epochs': len(self.history['epoch']),
            'history': self.history,
        }
