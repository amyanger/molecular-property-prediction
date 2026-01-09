"""SHAP-based explainability utilities for molecular property prediction."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Callable, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class SHAPExplanation:
    """Container for SHAP explanation results."""

    smiles: str
    task_idx: int
    shap_values: np.ndarray
    base_value: float
    prediction: float
    feature_names: Optional[list[str]] = None
    top_features: list[tuple] = field(default_factory=list)


class MolecularSHAP:
    """
    SHAP explanations for molecular property prediction models.

    Provides feature importance explanations for fingerprint-based
    and descriptor-based molecular property predictions.

    Args:
        model: PyTorch model
        background_data: Background dataset for SHAP (samples x features)
        feature_names: Optional names for features
    """

    def __init__(
        self,
        model: nn.Module,
        background_data: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ):
        self.model = model
        self.background = background_data
        self.feature_names = feature_names
        self.device = next(model.parameters()).device

        # Initialize SHAP explainer
        self._init_explainer()

    def _init_explainer(self) -> None:
        """Initialize SHAP explainer."""
        try:
            import shap

            # Wrap model for SHAP
            def model_predict(x):
                self.model.eval()
                with torch.no_grad():
                    x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
                    output = self.model(x_tensor)
                    return torch.sigmoid(output).cpu().numpy()

            # Use KernelSHAP for model-agnostic explanations
            self.explainer = shap.KernelExplainer(
                model_predict,
                self.background[:100],  # Use subset for efficiency
            )

            logger.info("SHAP explainer initialized")

        except ImportError:
            logger.warning("SHAP not installed. Install with: pip install shap")
            self.explainer = None

    def explain(
        self,
        sample: np.ndarray,
        smiles: str = "",
        task_idx: int = 0,
        n_samples: int = 100,
    ) -> Optional[SHAPExplanation]:
        """
        Generate SHAP explanation for a sample.

        Args:
            sample: Input features (1, n_features)
            smiles: SMILES string for the molecule
            task_idx: Which task to explain
            n_samples: Number of samples for SHAP estimation

        Returns:
            SHAPExplanation or None if SHAP not available
        """
        if self.explainer is None:
            logger.warning("SHAP explainer not available")
            return None

        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        # Get SHAP values
        shap_values = self.explainer.shap_values(sample, nsamples=n_samples)

        # Handle multi-task output
        if isinstance(shap_values, list):
            task_shap = shap_values[task_idx]
        else:
            task_shap = shap_values[:, :, task_idx] if shap_values.ndim == 3 else shap_values

        task_shap = task_shap.flatten()

        # Get prediction
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(sample, dtype=torch.float32).to(self.device)
            output = self.model(x_tensor)
            prediction = torch.sigmoid(output[0, task_idx]).item()

        # Find top features
        feature_importance = np.abs(task_shap)
        top_indices = np.argsort(feature_importance)[::-1][:10]

        top_features = []
        for idx in top_indices:
            name = self.feature_names[idx] if self.feature_names else f"feature_{idx}"
            top_features.append((name, task_shap[idx], float(feature_importance[idx])))

        return SHAPExplanation(
            smiles=smiles,
            task_idx=task_idx,
            shap_values=task_shap,
            base_value=float(self.explainer.expected_value[task_idx])
            if isinstance(self.explainer.expected_value, (list, np.ndarray))
            else float(self.explainer.expected_value),
            prediction=prediction,
            feature_names=self.feature_names,
            top_features=top_features,
        )

    def explain_batch(
        self,
        samples: np.ndarray,
        smiles_list: list[str],
        task_idx: int = 0,
        n_samples: int = 50,
    ) -> list[SHAPExplanation]:
        """
        Generate SHAP explanations for a batch of samples.

        Args:
            samples: Input features (n_samples, n_features)
            smiles_list: List of SMILES strings
            task_idx: Which task to explain
            n_samples: Number of samples for SHAP estimation

        Returns:
            List of SHAPExplanation objects
        """
        explanations = []
        for i, (sample, smiles) in enumerate(zip(samples, smiles_list)):
            explanation = self.explain(sample, smiles, task_idx, n_samples)
            if explanation:
                explanations.append(explanation)

        return explanations


class GradientExplainer:
    """
    Gradient-based explanations for neural networks.

    Faster alternative to SHAP for deep learning models.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: Optional[list[str]] = None,
    ):
        self.model = model
        self.feature_names = feature_names
        self.device = next(model.parameters()).device

    def explain(
        self,
        sample: np.ndarray,
        task_idx: int = 0,
    ) -> np.ndarray:
        """
        Compute input gradients for explanation.

        Args:
            sample: Input features
            task_idx: Task to explain

        Returns:
            Gradient values
        """
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        self.model.eval()

        x_tensor = torch.tensor(sample, dtype=torch.float32, requires_grad=True)
        x_tensor = x_tensor.to(self.device)

        output = self.model(x_tensor)
        output[0, task_idx].backward()

        gradients = x_tensor.grad.cpu().numpy().flatten()
        return gradients

    def integrated_gradients(
        self,
        sample: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        task_idx: int = 0,
        n_steps: int = 50,
    ) -> np.ndarray:
        """
        Compute integrated gradients for explanation.

        Args:
            sample: Input features
            baseline: Baseline input (default: zeros)
            task_idx: Task to explain
            n_steps: Number of interpolation steps

        Returns:
            Integrated gradient values
        """
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        if baseline is None:
            baseline = np.zeros_like(sample)

        # Interpolate between baseline and sample
        alphas = np.linspace(0, 1, n_steps)
        gradients = []

        for alpha in alphas:
            interpolated = baseline + alpha * (sample - baseline)
            grad = self.explain(interpolated, task_idx)
            gradients.append(grad)

        # Average gradients and scale by input difference
        avg_gradients = np.mean(gradients, axis=0)
        integrated_grads = (sample.flatten() - baseline.flatten()) * avg_gradients

        return integrated_grads


def summarize_shap_explanations(
    explanations: list[SHAPExplanation],
) -> dict:
    """
    Summarize SHAP explanations across multiple samples.

    Args:
        explanations: List of SHAPExplanation objects

    Returns:
        Summary dictionary
    """
    if not explanations:
        return {}

    n_features = len(explanations[0].shap_values)
    feature_names = explanations[0].feature_names

    # Aggregate SHAP values
    all_shap = np.array([e.shap_values for e in explanations])

    # Mean absolute SHAP values (global importance)
    mean_abs_shap = np.mean(np.abs(all_shap), axis=0)

    # Sort by importance
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    top_features = []
    for idx in sorted_idx[:20]:
        name = feature_names[idx] if feature_names else f"feature_{idx}"
        top_features.append({
            "name": name,
            "mean_abs_shap": float(mean_abs_shap[idx]),
            "mean_shap": float(np.mean(all_shap[:, idx])),
            "std_shap": float(np.std(all_shap[:, idx])),
        })

    return {
        "n_samples": len(explanations),
        "n_features": n_features,
        "top_features": top_features,
        "mean_prediction": float(np.mean([e.prediction for e in explanations])),
    }


def get_fingerprint_bit_explanations(
    shap_values: np.ndarray,
    smiles: str,
    radius: int = 2,
) -> list[dict]:
    """
    Map fingerprint bit SHAP values to molecular substructures.

    Args:
        shap_values: SHAP values for fingerprint bits
        smiles: SMILES string
        radius: Morgan fingerprint radius

    Returns:
        List of substructure explanations
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    # Get bit info
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=len(shap_values), bitInfo=bit_info
    )

    explanations = []
    for bit_idx, shap_val in enumerate(shap_values):
        if abs(shap_val) < 0.001:
            continue

        if bit_idx in bit_info:
            # Get atoms involved in this bit
            atom_envs = bit_info[bit_idx]
            explanations.append({
                "bit_index": int(bit_idx),
                "shap_value": float(shap_val),
                "atom_environments": [
                    {"center_atom": env[0], "radius": env[1]}
                    for env in atom_envs
                ],
            })

    # Sort by absolute SHAP value
    explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return explanations[:20]  # Return top 20
