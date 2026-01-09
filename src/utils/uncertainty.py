"""Uncertainty quantification utilities for molecular property prediction."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyResult:
    """Container for prediction with uncertainty estimates."""

    predictions: np.ndarray  # Mean predictions (N, num_tasks)
    epistemic_uncertainty: np.ndarray  # Model uncertainty (N, num_tasks)
    aleatoric_uncertainty: Optional[np.ndarray] = None  # Data uncertainty
    total_uncertainty: Optional[np.ndarray] = None  # Combined uncertainty
    prediction_std: Optional[np.ndarray] = None  # Standard deviation
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (lower, upper)


def enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers during inference for MC Dropout."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def disable_dropout(model: nn.Module) -> None:
    """Disable dropout layers (normal inference mode)."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()


class MCDropout:
    """
    Monte Carlo Dropout for uncertainty estimation.

    Performs multiple forward passes with dropout enabled to estimate
    prediction uncertainty (epistemic uncertainty).

    Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016

    Args:
        model: Neural network model with dropout layers
        n_samples: Number of MC samples
        apply_sigmoid: Whether to apply sigmoid to outputs
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 30,
        apply_sigmoid: bool = True,
    ):
        self.model = model
        self.n_samples = n_samples
        self.apply_sigmoid = apply_sigmoid

    @torch.no_grad()
    def predict(
        self,
        *inputs,
        confidence_level: float = 0.95,
    ) -> UncertaintyResult:
        """
        Make predictions with uncertainty estimates.

        Args:
            *inputs: Model inputs (e.g., x, edge_index, batch)
            confidence_level: Confidence level for intervals (default: 95%)

        Returns:
            UncertaintyResult with predictions and uncertainties
        """
        self.model.eval()
        enable_dropout(self.model)

        # Collect MC samples
        samples = []
        for _ in range(self.n_samples):
            output = self.model(*inputs)
            if self.apply_sigmoid:
                output = torch.sigmoid(output)
            samples.append(output.cpu().numpy())

        samples = np.stack(samples, axis=0)  # (n_samples, N, num_tasks)

        # Compute statistics
        mean_pred = samples.mean(axis=0)
        std_pred = samples.std(axis=0)

        # Epistemic uncertainty (model uncertainty) = variance of predictions
        epistemic = std_pred ** 2

        # Confidence intervals
        alpha = 1 - confidence_level
        lower = np.percentile(samples, alpha / 2 * 100, axis=0)
        upper = np.percentile(samples, (1 - alpha / 2) * 100, axis=0)

        disable_dropout(self.model)

        return UncertaintyResult(
            predictions=mean_pred,
            epistemic_uncertainty=epistemic,
            prediction_std=std_pred,
            confidence_intervals=(lower, upper),
        )


class DeepEnsemble:
    """
    Deep Ensemble for uncertainty estimation.

    Combines predictions from multiple independently trained models.

    Reference: Lakshminarayanan et al., "Simple and Scalable Predictive
    Uncertainty Estimation using Deep Ensembles", NeurIPS 2017

    Args:
        models: List of trained models
        apply_sigmoid: Whether to apply sigmoid to outputs
    """

    def __init__(
        self,
        models: list[nn.Module],
        apply_sigmoid: bool = True,
    ):
        self.models = models
        self.apply_sigmoid = apply_sigmoid
        self.n_models = len(models)

    @torch.no_grad()
    def predict(
        self,
        *inputs,
        confidence_level: float = 0.95,
    ) -> UncertaintyResult:
        """
        Make predictions with uncertainty estimates.

        Args:
            *inputs: Model inputs
            confidence_level: Confidence level for intervals

        Returns:
            UncertaintyResult with predictions and uncertainties
        """
        # Collect predictions from all models
        predictions = []
        for model in self.models:
            model.eval()
            output = model(*inputs)
            if self.apply_sigmoid:
                output = torch.sigmoid(output)
            predictions.append(output.cpu().numpy())

        predictions = np.stack(predictions, axis=0)  # (n_models, N, num_tasks)

        # Statistics
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        epistemic = std_pred ** 2

        # Confidence intervals
        alpha = 1 - confidence_level
        lower = np.percentile(predictions, alpha / 2 * 100, axis=0)
        upper = np.percentile(predictions, (1 - alpha / 2) * 100, axis=0)

        return UncertaintyResult(
            predictions=mean_pred,
            epistemic_uncertainty=epistemic,
            prediction_std=std_pred,
            confidence_intervals=(lower, upper),
        )


class EvidentialUncertainty(nn.Module):
    """
    Evidential deep learning for uncertainty estimation.

    Predicts parameters of a Dirichlet distribution for classification
    uncertainty estimation.

    Reference: Sensoy et al., "Evidential Deep Learning to Quantify
    Classification Uncertainty", NeurIPS 2018

    Args:
        in_features: Input feature dimension
        num_classes: Number of classes (2 for binary)
    """

    def __init__(self, in_features: int, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.evidence_layer = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features

        Returns:
            Tuple of (probabilities, uncertainty, evidence)
        """
        # Get evidence (positive values)
        evidence = torch.softplus(self.evidence_layer(x))

        # Dirichlet parameters
        alpha = evidence + 1

        # Dirichlet strength
        S = alpha.sum(dim=-1, keepdim=True)

        # Expected probabilities
        prob = alpha / S

        # Uncertainty (inverse of total evidence)
        uncertainty = self.num_classes / S

        return prob, uncertainty.squeeze(-1), evidence


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for calibrated uncertainty.

    Learns a temperature parameter to calibrate model predictions.

    Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017

    Args:
        temperature: Initial temperature value
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([temperature]))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
    ) -> float:
        """
        Learn optimal temperature on validation data.

        Args:
            logits: Model logits
            labels: True labels
            max_iter: Maximum optimization iterations

        Returns:
            Optimal temperature value
        """
        from torch.optim import LBFGS

        self.temperature.data = torch.tensor([1.5])

        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = self(logits)
            loss = nn.functional.binary_cross_entropy_with_logits(
                scaled_logits, labels.float()
            )
            loss.backward()
            return loss

        optimizer.step(closure)

        return self.temperature.item()


class BayesianLayer(nn.Module):
    """
    Bayesian linear layer with weight uncertainty.

    Uses variational inference with reparameterization trick.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        prior_sigma: Prior standard deviation
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.full((out_features, in_features), -5.0))

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_log_var = nn.Parameter(torch.full((out_features,), -5.0))

        # Prior
        self.prior_sigma = prior_sigma

        # Initialize weights
        nn.init.xavier_normal_(self.weight_mu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sampled weights."""
        if self.training:
            # Sample weights using reparameterization trick
            weight_std = torch.exp(0.5 * self.weight_log_var)
            weight_eps = torch.randn_like(weight_std)
            weight = self.weight_mu + weight_std * weight_eps

            bias_std = torch.exp(0.5 * self.bias_log_var)
            bias_eps = torch.randn_like(bias_std)
            bias = self.bias_mu + bias_std * bias_eps
        else:
            # Use mean weights during inference
            weight = self.weight_mu
            bias = self.bias_mu

        return nn.functional.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence from prior."""
        # KL(q||p) for Gaussian
        weight_var = torch.exp(self.weight_log_var)
        bias_var = torch.exp(self.bias_log_var)

        kl = 0.5 * (
            (self.weight_mu ** 2 + weight_var) / self.prior_sigma ** 2
            - 1
            - self.weight_log_var
            + np.log(self.prior_sigma ** 2)
        ).sum()

        kl += 0.5 * (
            (self.bias_mu ** 2 + bias_var) / self.prior_sigma ** 2
            - 1
            - self.bias_log_var
            + np.log(self.prior_sigma ** 2)
        ).sum()

        return kl


def compute_uncertainty_metrics(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute metrics for uncertainty quality evaluation.

    Args:
        predictions: Model predictions
        uncertainties: Uncertainty estimates
        targets: True labels
        n_bins: Number of bins for calibration

    Returns:
        Dictionary with uncertainty quality metrics
    """
    # Flatten if multi-task
    if predictions.ndim > 1:
        predictions = predictions.flatten()
        uncertainties = uncertainties.flatten()
        targets = targets.flatten()

    # Remove invalid samples
    valid_mask = targets >= 0
    predictions = predictions[valid_mask]
    uncertainties = uncertainties[valid_mask]
    targets = targets[valid_mask]

    # Compute prediction errors
    errors = np.abs(predictions - targets)

    # Spearman correlation between uncertainty and error
    from scipy.stats import spearmanr
    correlation, p_value = spearmanr(uncertainties, errors)

    # Area under the sparsification curve (AUSC)
    sorted_indices = np.argsort(uncertainties)[::-1]  # High to low uncertainty
    cumulative_errors = np.cumsum(errors[sorted_indices])
    ausc = np.trapz(cumulative_errors) / len(predictions)

    # Oracle AUSC (sorted by actual errors)
    oracle_indices = np.argsort(errors)[::-1]
    oracle_cumulative = np.cumsum(errors[oracle_indices])
    oracle_ausc = np.trapz(oracle_cumulative) / len(predictions)

    # Random AUSC
    random_ausc = errors.sum() * len(predictions) / 2

    # Normalized AUSC
    if oracle_ausc != random_ausc:
        normalized_ausc = (random_ausc - ausc) / (random_ausc - oracle_ausc)
    else:
        normalized_ausc = 0.0

    return {
        "uncertainty_error_correlation": float(correlation),
        "correlation_p_value": float(p_value),
        "ausc": float(ausc),
        "oracle_ausc": float(oracle_ausc),
        "normalized_ausc": float(normalized_ausc),
        "mean_uncertainty": float(uncertainties.mean()),
        "mean_error": float(errors.mean()),
    }
