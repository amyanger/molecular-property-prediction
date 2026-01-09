"""Model calibration utilities for reliable uncertainty estimates."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass, field
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
import logging

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics."""

    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float
    reliability_diagram: dict = field(default_factory=dict)


@dataclass
class CalibratedPredictions:
    """Calibrated prediction results."""

    original_probs: np.ndarray
    calibrated_probs: np.ndarray
    method: str
    calibration_params: dict = field(default_factory=dict)


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for model calibration.

    Learns a single temperature parameter to scale logits before softmax.

    Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017

    Args:
        initial_temp: Initial temperature value
    """

    def __init__(self, initial_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        Learn optimal temperature on validation data.

        Args:
            logits: Model logits (before sigmoid)
            labels: Ground truth labels
            lr: Learning rate for optimization
            max_iter: Maximum iterations

        Returns:
            Optimal temperature value
        """
        self.temperature.data = torch.ones(1) * 1.5

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            scaled_logits = self(logits)
            loss = criterion(scaled_logits, labels.float())
            loss.backward()
            return loss

        optimizer.step(closure)

        optimal_temp = self.temperature.item()
        logger.info(f"Optimal temperature: {optimal_temp:.4f}")
        return optimal_temp


class PlattScaling:
    """
    Platt scaling for probability calibration.

    Fits a sigmoid function to map scores to calibrated probabilities:
    P(y=1|x) = 1 / (1 + exp(A*f(x) + B))

    Args:
        prior_correction: Whether to apply prior probability correction
    """

    def __init__(self, prior_correction: bool = True):
        self.prior_correction = prior_correction
        self.a = None
        self.b = None

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Fit Platt scaling parameters.

        Args:
            scores: Model scores/probabilities
            labels: Ground truth labels
        """
        # Initialize parameters
        prior1 = labels.sum()
        prior0 = len(labels) - prior1

        if self.prior_correction:
            hi_target = (prior1 + 1) / (prior1 + 2)
            lo_target = 1 / (prior0 + 2)
        else:
            hi_target = 1.0
            lo_target = 0.0

        # Target probabilities
        targets = np.where(labels == 1, hi_target, lo_target)

        # Optimization
        def objective(params):
            a, b = params
            probs = 1 / (1 + np.exp(a * scores + b))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return -np.sum(targets * np.log(probs) + (1 - targets) * np.log(1 - probs))

        result = minimize(objective, [0.0, 0.0], method="BFGS")
        self.a, self.b = result.x

        logger.info(f"Platt scaling params: A={self.a:.4f}, B={self.b:.4f}")

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to scores."""
        if self.a is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return 1 / (1 + np.exp(self.a * scores + self.b))


class IsotonicCalibration:
    """
    Isotonic regression for probability calibration.

    Non-parametric approach that finds a monotonic function
    to map scores to calibrated probabilities.
    """

    def __init__(self):
        self.isotonic = IsotonicRegression(out_of_bounds="clip")

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Fit isotonic regression.

        Args:
            scores: Model scores/probabilities
            labels: Ground truth labels
        """
        self.isotonic.fit(scores, labels)
        logger.info("Isotonic calibration fitted")

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration."""
        return self.isotonic.transform(scores)


class BetaCalibration:
    """
    Beta calibration for probability calibration.

    Fits a beta distribution-based transformation.

    Reference: Kull et al., "Beta calibration: a well-founded and easily
    implemented improvement on logistic calibration", AISTATS 2017
    """

    def __init__(self):
        self.a = None
        self.b = None
        self.c = None

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Fit beta calibration parameters.

        Args:
            scores: Model scores/probabilities (0-1)
            labels: Ground truth labels
        """
        # Clip scores to avoid log(0)
        scores = np.clip(scores, 1e-7, 1 - 1e-7)

        # Log-odds transformation
        log_odds = np.log(scores / (1 - scores))

        def objective(params):
            a, b, c = params
            # Beta calibration formula
            calibrated_log_odds = a * np.log(scores) + b * np.log(1 - scores) + c
            calibrated = 1 / (1 + np.exp(-calibrated_log_odds))
            calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)

            # Negative log-likelihood
            return -np.sum(
                labels * np.log(calibrated) + (1 - labels) * np.log(1 - calibrated)
            )

        result = minimize(objective, [1.0, -1.0, 0.0], method="BFGS")
        self.a, self.b, self.c = result.x

        logger.info(f"Beta calibration params: a={self.a:.4f}, b={self.b:.4f}, c={self.c:.4f}")

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply beta calibration."""
        if self.a is None:
            raise ValueError("Model not fitted. Call fit() first.")

        scores = np.clip(scores, 1e-7, 1 - 1e-7)
        calibrated_log_odds = self.a * np.log(scores) + self.b * np.log(1 - scores) + self.c
        return 1 / (1 + np.exp(-calibrated_log_odds))


def compute_calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """
    Compute calibration metrics.

    Args:
        probabilities: Predicted probabilities
        labels: Ground truth labels
        n_bins: Number of bins for ECE/MCE

    Returns:
        CalibrationMetrics
    """
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    mce = 0.0
    reliability_data = {"bins": [], "accuracy": [], "confidence": [], "count": []}

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find samples in this bin
        in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = probabilities[in_bin].mean()
            avg_accuracy = labels[in_bin].mean()

            bin_error = abs(avg_accuracy - avg_confidence)
            ece += prop_in_bin * bin_error
            mce = max(mce, bin_error)

            reliability_data["bins"].append(f"({bin_lower:.1f}, {bin_upper:.1f}]")
            reliability_data["accuracy"].append(float(avg_accuracy))
            reliability_data["confidence"].append(float(avg_confidence))
            reliability_data["count"].append(int(in_bin.sum()))

    # Brier score
    brier_score = np.mean((probabilities - labels) ** 2)

    return CalibrationMetrics(
        ece=float(ece),
        mce=float(mce),
        brier_score=float(brier_score),
        reliability_diagram=reliability_data,
    )


def calibrate_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    method: str = "temperature",
    val_split: float = 0.5,
) -> CalibratedPredictions:
    """
    Calibrate predictions using specified method.

    Args:
        probabilities: Original predicted probabilities
        labels: Ground truth labels
        method: Calibration method ('temperature', 'platt', 'isotonic', 'beta')
        val_split: Fraction of data for fitting calibration

    Returns:
        CalibratedPredictions
    """
    # Split data
    n = len(probabilities)
    indices = np.random.permutation(n)
    n_fit = int(n * val_split)

    fit_probs = probabilities[indices[:n_fit]]
    fit_labels = labels[indices[:n_fit]]
    transform_probs = probabilities

    if method == "temperature":
        # Temperature scaling works on logits
        fit_logits = np.log(fit_probs / (1 - fit_probs + 1e-7))
        ts = TemperatureScaling()
        temp = ts.calibrate(
            torch.tensor(fit_logits),
            torch.tensor(fit_labels)
        )
        all_logits = np.log(transform_probs / (1 - transform_probs + 1e-7))
        calibrated = 1 / (1 + np.exp(-all_logits / temp))
        params = {"temperature": temp}

    elif method == "platt":
        ps = PlattScaling()
        ps.fit(fit_probs, fit_labels)
        calibrated = ps.transform(transform_probs)
        params = {"a": ps.a, "b": ps.b}

    elif method == "isotonic":
        iso = IsotonicCalibration()
        iso.fit(fit_probs, fit_labels)
        calibrated = iso.transform(transform_probs)
        params = {}

    elif method == "beta":
        bc = BetaCalibration()
        bc.fit(fit_probs, fit_labels)
        calibrated = bc.transform(transform_probs)
        params = {"a": bc.a, "b": bc.b, "c": bc.c}

    else:
        raise ValueError(f"Unknown method: {method}")

    return CalibratedPredictions(
        original_probs=probabilities,
        calibrated_probs=calibrated,
        method=method,
        calibration_params=params,
    )


def compare_calibration_methods(
    probabilities: np.ndarray,
    labels: np.ndarray,
    methods: Optional[list[str]] = None,
) -> dict[str, CalibrationMetrics]:
    """
    Compare different calibration methods.

    Args:
        probabilities: Predicted probabilities
        labels: Ground truth labels
        methods: List of methods to compare

    Returns:
        Dict mapping method names to CalibrationMetrics
    """
    if methods is None:
        methods = ["none", "temperature", "platt", "isotonic", "beta"]

    results = {}

    for method in methods:
        if method == "none":
            metrics = compute_calibration_metrics(probabilities, labels)
        else:
            calibrated = calibrate_predictions(probabilities, labels, method)
            metrics = compute_calibration_metrics(calibrated.calibrated_probs, labels)

        results[method] = metrics
        logger.info(f"{method}: ECE={metrics.ece:.4f}, MCE={metrics.mce:.4f}, "
                    f"Brier={metrics.brier_score:.4f}")

    return results
