"""Data drift detection utilities for monitoring model performance."""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Container for drift detection results."""

    has_drift: bool
    drift_score: float
    p_value: float
    drift_type: str  # "covariate", "label", "prediction"
    details: dict = field(default_factory=dict)


class DataDriftDetector:
    """
    Detect data drift in molecular property prediction.

    Monitors for distribution shifts in input features (covariate drift)
    and model predictions (concept drift).

    Args:
        reference_data: Reference (training) data for comparison
        n_components: Number of PCA components for dimensionality reduction
        significance_level: Significance level for statistical tests
    """

    def __init__(
        self,
        reference_data: Optional[np.ndarray] = None,
        n_components: int = 50,
        significance_level: float = 0.05,
    ):
        self.reference_data = reference_data
        self.n_components = n_components
        self.significance_level = significance_level
        self.pca = None
        self.reference_transformed = None

        if reference_data is not None:
            self.fit_reference(reference_data)

    def fit_reference(self, reference_data: np.ndarray) -> None:
        """
        Fit reference distribution from training data.

        Args:
            reference_data: Reference feature matrix (N, D)
        """
        self.reference_data = reference_data

        # Fit PCA for dimensionality reduction
        n_components = min(self.n_components, reference_data.shape[0], reference_data.shape[1])
        self.pca = PCA(n_components=n_components)
        self.reference_transformed = self.pca.fit_transform(reference_data)

        # Store reference statistics
        self.reference_mean = self.reference_transformed.mean(axis=0)
        self.reference_std = self.reference_transformed.std(axis=0)
        self.reference_cov = np.cov(self.reference_transformed.T)

        logger.info(f"Reference fitted with {reference_data.shape[0]} samples, "
                    f"{n_components} components")

    def detect_covariate_drift(
        self,
        new_data: np.ndarray,
        method: str = "ks",
    ) -> DriftReport:
        """
        Detect covariate (input feature) drift.

        Args:
            new_data: New feature matrix to compare
            method: Statistical test ('ks', 'mmd', 'psi')

        Returns:
            DriftReport
        """
        if self.reference_transformed is None:
            raise ValueError("Reference data not fitted. Call fit_reference first.")

        # Transform new data
        new_transformed = self.pca.transform(new_data)

        if method == "ks":
            return self._ks_test_drift(new_transformed)
        elif method == "mmd":
            return self._mmd_drift(new_transformed)
        elif method == "psi":
            return self._psi_drift(new_transformed)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _ks_test_drift(self, new_data: np.ndarray) -> DriftReport:
        """Kolmogorov-Smirnov test for drift detection."""
        p_values = []
        statistics = []

        for i in range(new_data.shape[1]):
            statistic, p_value = stats.ks_2samp(
                self.reference_transformed[:, i],
                new_data[:, i]
            )
            p_values.append(p_value)
            statistics.append(statistic)

        # Bonferroni correction
        min_p_value = min(p_values) * len(p_values)
        max_statistic = max(statistics)

        has_drift = min_p_value < self.significance_level

        return DriftReport(
            has_drift=has_drift,
            drift_score=max_statistic,
            p_value=min_p_value,
            drift_type="covariate",
            details={
                "method": "ks_test",
                "per_component_p_values": p_values,
                "per_component_statistics": statistics,
                "num_drifted_components": sum(
                    p < self.significance_level / len(p_values) for p in p_values
                ),
            }
        )

    def _mmd_drift(self, new_data: np.ndarray) -> DriftReport:
        """Maximum Mean Discrepancy test for drift detection."""
        # Compute MMD with Gaussian kernel
        sigma = self._median_heuristic(
            np.vstack([self.reference_transformed, new_data])
        )

        mmd = self._compute_mmd(self.reference_transformed, new_data, sigma)

        # Permutation test for p-value
        n_permutations = 100
        combined = np.vstack([self.reference_transformed, new_data])
        n_ref = len(self.reference_transformed)
        null_mmds = []

        for _ in range(n_permutations):
            perm = np.random.permutation(len(combined))
            ref_perm = combined[perm[:n_ref]]
            new_perm = combined[perm[n_ref:]]
            null_mmds.append(self._compute_mmd(ref_perm, new_perm, sigma))

        p_value = np.mean(np.array(null_mmds) >= mmd)
        has_drift = p_value < self.significance_level

        return DriftReport(
            has_drift=has_drift,
            drift_score=mmd,
            p_value=p_value,
            drift_type="covariate",
            details={
                "method": "mmd",
                "kernel_sigma": sigma,
                "null_mmd_mean": np.mean(null_mmds),
            }
        )

    def _compute_mmd(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sigma: float,
    ) -> float:
        """Compute MMD between two samples."""
        xx = self._rbf_kernel(x, x, sigma)
        yy = self._rbf_kernel(y, y, sigma)
        xy = self._rbf_kernel(x, y, sigma)

        return np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)

    def _rbf_kernel(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """RBF (Gaussian) kernel."""
        from scipy.spatial.distance import cdist
        dists = cdist(x, y, metric='sqeuclidean')
        return np.exp(-dists / (2 * sigma ** 2))

    def _median_heuristic(self, data: np.ndarray) -> float:
        """Compute median heuristic for kernel bandwidth."""
        from scipy.spatial.distance import pdist
        dists = pdist(data, metric='euclidean')
        return np.median(dists)

    def _psi_drift(self, new_data: np.ndarray) -> DriftReport:
        """Population Stability Index for drift detection."""
        psi_values = []

        for i in range(new_data.shape[1]):
            psi = self._compute_psi(
                self.reference_transformed[:, i],
                new_data[:, i]
            )
            psi_values.append(psi)

        mean_psi = np.mean(psi_values)
        max_psi = np.max(psi_values)

        # PSI thresholds: < 0.1 = no drift, 0.1-0.25 = moderate, > 0.25 = significant
        has_drift = max_psi > 0.25

        return DriftReport(
            has_drift=has_drift,
            drift_score=max_psi,
            p_value=0.0,  # PSI doesn't provide p-value
            drift_type="covariate",
            details={
                "method": "psi",
                "mean_psi": mean_psi,
                "per_component_psi": psi_values,
                "num_significant_drift": sum(psi > 0.25 for psi in psi_values),
                "num_moderate_drift": sum(0.1 <= psi <= 0.25 for psi in psi_values),
            }
        )

    def _compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute PSI for a single feature."""
        # Create bins from expected distribution
        bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        # Compute proportions
        expected_counts = np.histogram(expected, bins=bins)[0]
        actual_counts = np.histogram(actual, bins=bins)[0]

        expected_props = expected_counts / len(expected)
        actual_props = actual_counts / len(actual)

        # Avoid division by zero
        expected_props = np.clip(expected_props, 1e-6, None)
        actual_props = np.clip(actual_props, 1e-6, None)

        # PSI formula
        psi = np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))

        return psi


class PredictionDriftDetector:
    """
    Detect drift in model predictions.

    Monitors for changes in prediction distribution that may indicate
    concept drift or model degradation.

    Args:
        significance_level: Significance level for statistical tests
    """

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.reference_predictions = None

    def fit_reference(self, predictions: np.ndarray) -> None:
        """Fit reference from initial predictions."""
        self.reference_predictions = predictions
        self.reference_mean = predictions.mean(axis=0)
        self.reference_std = predictions.std(axis=0)

    def detect_drift(self, new_predictions: np.ndarray) -> DriftReport:
        """
        Detect drift in predictions.

        Args:
            new_predictions: New prediction matrix

        Returns:
            DriftReport
        """
        if self.reference_predictions is None:
            raise ValueError("Reference not fitted")

        # Per-task KS tests
        n_tasks = new_predictions.shape[1] if new_predictions.ndim > 1 else 1

        if n_tasks == 1:
            ref = self.reference_predictions.flatten()
            new = new_predictions.flatten()
            statistic, p_value = stats.ks_2samp(ref, new)
            has_drift = p_value < self.significance_level

            return DriftReport(
                has_drift=has_drift,
                drift_score=statistic,
                p_value=p_value,
                drift_type="prediction",
                details={"method": "ks_test"},
            )

        p_values = []
        statistics = []

        for i in range(n_tasks):
            stat, p_val = stats.ks_2samp(
                self.reference_predictions[:, i],
                new_predictions[:, i]
            )
            p_values.append(p_val)
            statistics.append(stat)

        min_p = min(p_values) * n_tasks  # Bonferroni
        has_drift = min_p < self.significance_level

        return DriftReport(
            has_drift=has_drift,
            drift_score=max(statistics),
            p_value=min_p,
            drift_type="prediction",
            details={
                "method": "ks_test",
                "per_task_p_values": p_values,
                "per_task_statistics": statistics,
                "drifted_tasks": [i for i, p in enumerate(p_values)
                                 if p < self.significance_level / n_tasks],
            }
        )


class DriftMonitor:
    """
    Continuous drift monitoring for production systems.

    Args:
        covariate_detector: Detector for feature drift
        prediction_detector: Detector for prediction drift
        window_size: Number of samples for detection window
    """

    def __init__(
        self,
        covariate_detector: Optional[DataDriftDetector] = None,
        prediction_detector: Optional[PredictionDriftDetector] = None,
        window_size: int = 1000,
    ):
        self.covariate_detector = covariate_detector or DataDriftDetector()
        self.prediction_detector = prediction_detector or PredictionDriftDetector()
        self.window_size = window_size

        self.feature_buffer = []
        self.prediction_buffer = []
        self.alerts = []

    def update(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
    ) -> Optional[DriftReport]:
        """
        Update monitor with new samples.

        Args:
            features: New feature matrix
            predictions: New predictions

        Returns:
            DriftReport if drift detected, None otherwise
        """
        self.feature_buffer.append(features)
        self.prediction_buffer.append(predictions)

        # Check if we have enough samples
        total_features = sum(f.shape[0] for f in self.feature_buffer)

        if total_features < self.window_size:
            return None

        # Combine buffers
        all_features = np.vstack(self.feature_buffer)[-self.window_size:]
        all_predictions = np.vstack(self.prediction_buffer)[-self.window_size:]

        # Check for drift
        covariate_report = self.covariate_detector.detect_covariate_drift(all_features)
        prediction_report = self.prediction_detector.detect_drift(all_predictions)

        # Clear old buffer data
        if total_features > self.window_size * 2:
            self.feature_buffer = [all_features]
            self.prediction_buffer = [all_predictions]

        # Return most significant drift
        if covariate_report.has_drift or prediction_report.has_drift:
            report = covariate_report if covariate_report.drift_score > prediction_report.drift_score \
                     else prediction_report
            self.alerts.append(report)
            return report

        return None

    def get_alerts(self) -> list[DriftReport]:
        """Get all drift alerts."""
        return self.alerts
