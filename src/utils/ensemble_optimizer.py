"""Bayesian optimization for ensemble weight tuning."""

import numpy as np
from typing import Optional, Callable, Tuple
from dataclasses import dataclass, field
from scipy.optimize import minimize
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnsembleWeightResult:
    """Result of ensemble weight optimization."""

    optimal_weights: dict[str, float]
    optimal_metric: float
    history: list[dict] = field(default_factory=list)
    n_iterations: int = 0


class BayesianEnsembleOptimizer:
    """
    Bayesian optimization for ensemble weight tuning.

    Uses Gaussian Process surrogate model to efficiently
    search the weight space.

    Args:
        model_names: List of model names in ensemble
        metric_fn: Function that takes weights dict and returns metric
        n_iterations: Number of optimization iterations
        acquisition: Acquisition function ('ei', 'ucb', 'pi')
    """

    def __init__(
        self,
        model_names: list[str],
        metric_fn: Callable[[dict[str, float]], float],
        n_iterations: int = 50,
        acquisition: str = "ei",
    ):
        self.model_names = model_names
        self.metric_fn = metric_fn
        self.n_iterations = n_iterations
        self.acquisition = acquisition

        self.n_models = len(model_names)
        self.X_observed = []
        self.y_observed = []

    def _weights_to_simplex(self, weights: np.ndarray) -> np.ndarray:
        """Convert unconstrained weights to simplex (sum to 1)."""
        exp_weights = np.exp(weights)
        return exp_weights / exp_weights.sum()

    def _simplex_to_dict(self, simplex: np.ndarray) -> dict[str, float]:
        """Convert simplex array to named dict."""
        return {name: float(w) for name, w in zip(self.model_names, simplex)}

    def _sample_prior(self, n_samples: int) -> np.ndarray:
        """Sample from prior (uniform on simplex)."""
        samples = np.random.dirichlet(np.ones(self.n_models), size=n_samples)
        return samples

    def _expected_improvement(
        self,
        x: np.ndarray,
        gp_mean: Callable,
        gp_std: Callable,
        y_best: float,
        xi: float = 0.01,
    ) -> float:
        """Compute expected improvement acquisition."""
        mean = gp_mean(x)
        std = gp_std(x)

        if std == 0:
            return 0.0

        z = (mean - y_best - xi) / std
        ei = (mean - y_best - xi) * norm.cdf(z) + std * norm.pdf(z)
        return ei

    def _upper_confidence_bound(
        self,
        x: np.ndarray,
        gp_mean: Callable,
        gp_std: Callable,
        kappa: float = 2.0,
    ) -> float:
        """Compute UCB acquisition."""
        return gp_mean(x) + kappa * gp_std(x)

    def optimize(
        self,
        initial_weights: Optional[dict[str, float]] = None,
        n_initial: int = 5,
    ) -> EnsembleWeightResult:
        """
        Optimize ensemble weights.

        Args:
            initial_weights: Starting weights (None = uniform)
            n_initial: Number of random initial evaluations

        Returns:
            EnsembleWeightResult
        """
        history = []

        # Initial random samples
        if n_initial > 0:
            initial_samples = self._sample_prior(n_initial)
            for sample in initial_samples:
                weights_dict = self._simplex_to_dict(sample)
                metric = self.metric_fn(weights_dict)
                self.X_observed.append(sample)
                self.y_observed.append(metric)
                history.append({
                    "iteration": len(history),
                    "weights": weights_dict,
                    "metric": metric,
                })
                logger.info(f"Initial sample: metric={metric:.4f}")

        # Main optimization loop
        for i in range(self.n_iterations - n_initial):
            # Fit Gaussian Process surrogate
            gp_mean, gp_std = self._fit_gp()

            # Find next point to evaluate
            y_best = max(self.y_observed)
            next_weights = self._acquire_next(gp_mean, gp_std, y_best)

            # Evaluate
            weights_dict = self._simplex_to_dict(next_weights)
            metric = self.metric_fn(weights_dict)

            self.X_observed.append(next_weights)
            self.y_observed.append(metric)

            history.append({
                "iteration": len(history),
                "weights": weights_dict,
                "metric": metric,
            })

            if metric > y_best:
                logger.info(f"New best: metric={metric:.4f}, weights={weights_dict}")
            else:
                logger.debug(f"Iteration {i}: metric={metric:.4f}")

        # Find best
        best_idx = np.argmax(self.y_observed)
        optimal_weights = self._simplex_to_dict(self.X_observed[best_idx])
        optimal_metric = self.y_observed[best_idx]

        return EnsembleWeightResult(
            optimal_weights=optimal_weights,
            optimal_metric=optimal_metric,
            history=history,
            n_iterations=len(history),
        )

    def _fit_gp(self) -> Tuple[Callable, Callable]:
        """Fit Gaussian Process to observations."""
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)

        # Simple GP with RBF kernel
        def rbf_kernel(x1, x2, length_scale=0.1):
            dist = np.sum((x1 - x2) ** 2)
            return np.exp(-dist / (2 * length_scale ** 2))

        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = rbf_kernel(X[i], X[j])

        K += 1e-6 * np.eye(n)  # Regularization

        K_inv = np.linalg.inv(K)

        def gp_mean(x):
            k_star = np.array([rbf_kernel(x, xi) for xi in X])
            return k_star @ K_inv @ y

        def gp_std(x):
            k_star = np.array([rbf_kernel(x, xi) for xi in X])
            k_star_star = rbf_kernel(x, x)
            var = k_star_star - k_star @ K_inv @ k_star
            return np.sqrt(max(var, 1e-6))

        return gp_mean, gp_std

    def _acquire_next(
        self,
        gp_mean: Callable,
        gp_std: Callable,
        y_best: float,
    ) -> np.ndarray:
        """Find next point to evaluate using acquisition function."""
        best_acquisition = -np.inf
        best_x = None

        # Random search on simplex
        candidates = self._sample_prior(1000)

        for x in candidates:
            if self.acquisition == "ei":
                acq = self._expected_improvement(x, gp_mean, gp_std, y_best)
            elif self.acquisition == "ucb":
                acq = self._upper_confidence_bound(x, gp_mean, gp_std)
            else:
                acq = self._expected_improvement(x, gp_mean, gp_std, y_best)

            if acq > best_acquisition:
                best_acquisition = acq
                best_x = x

        return best_x


class GridSearchEnsembleOptimizer:
    """
    Grid search optimization for ensemble weights.

    Simple but effective for small number of models.

    Args:
        model_names: List of model names
        metric_fn: Function that takes weights and returns metric
        grid_resolution: Number of points per dimension
    """

    def __init__(
        self,
        model_names: list[str],
        metric_fn: Callable[[dict[str, float]], float],
        grid_resolution: int = 10,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ):
        self.model_names = model_names
        self.metric_fn = metric_fn
        self.grid_resolution = grid_resolution
        self.min_weight = min_weight
        self.max_weight = max_weight

    def optimize(self) -> EnsembleWeightResult:
        """
        Run grid search optimization.

        Returns:
            EnsembleWeightResult
        """
        from itertools import product

        # Generate grid
        weights_range = np.linspace(self.min_weight, self.max_weight, self.grid_resolution)

        best_weights = None
        best_metric = -np.inf
        history = []

        # Generate all combinations
        for combo in product(weights_range, repeat=len(self.model_names)):
            weights = np.array(combo)

            # Normalize to sum to 1
            if weights.sum() == 0:
                continue
            weights = weights / weights.sum()

            weights_dict = {
                name: float(w) for name, w in zip(self.model_names, weights)
            }

            metric = self.metric_fn(weights_dict)
            history.append({"weights": weights_dict, "metric": metric})

            if metric > best_metric:
                best_metric = metric
                best_weights = weights_dict

        logger.info(f"Grid search complete. Best metric: {best_metric:.4f}")
        logger.info(f"Optimal weights: {best_weights}")

        return EnsembleWeightResult(
            optimal_weights=best_weights,
            optimal_metric=best_metric,
            history=history,
            n_iterations=len(history),
        )


def optimize_ensemble_weights(
    predictions: dict[str, np.ndarray],
    targets: np.ndarray,
    method: str = "bayesian",
    n_iterations: int = 50,
    metric: str = "auc_roc",
) -> EnsembleWeightResult:
    """
    Convenience function to optimize ensemble weights.

    Args:
        predictions: Dict mapping model names to predictions
        targets: Ground truth labels
        method: Optimization method ('bayesian', 'grid')
        n_iterations: Number of iterations for Bayesian opt
        metric: Metric to optimize ('auc_roc', 'f1', 'accuracy')

    Returns:
        EnsembleWeightResult
    """
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    model_names = list(predictions.keys())

    def compute_metric(weights: dict[str, float]) -> float:
        # Weighted average of predictions
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        for name, weight in weights.items():
            ensemble_pred += weight * predictions[name]

        # Compute metric
        if targets.ndim == 1:
            valid_mask = targets >= 0
            if metric == "auc_roc":
                return roc_auc_score(targets[valid_mask], ensemble_pred[valid_mask])
            elif metric == "f1":
                return f1_score(targets[valid_mask], (ensemble_pred[valid_mask] > 0.5).astype(int))
            else:
                return accuracy_score(targets[valid_mask], (ensemble_pred[valid_mask] > 0.5).astype(int))
        else:
            # Multi-task: average across tasks
            aucs = []
            for i in range(targets.shape[1]):
                valid_mask = targets[:, i] >= 0
                if valid_mask.sum() > 10:
                    auc = roc_auc_score(
                        targets[valid_mask, i],
                        ensemble_pred[valid_mask, i]
                    )
                    aucs.append(auc)
            return np.mean(aucs)

    if method == "bayesian":
        optimizer = BayesianEnsembleOptimizer(
            model_names, compute_metric, n_iterations=n_iterations
        )
    else:
        optimizer = GridSearchEnsembleOptimizer(model_names, compute_metric)

    return optimizer.optimize()
