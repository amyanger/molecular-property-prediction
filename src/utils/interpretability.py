"""Model interpretability utilities for understanding predictions."""

import numpy as np
import torch
from typing import Optional
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def compute_fingerprint_importance(
    model: torch.nn.Module,
    smiles: str,
    task_idx: int = 0,
    device: torch.device = torch.device('cpu'),
) -> np.ndarray:
    """
    Compute importance of each fingerprint bit using gradient-based attribution.

    Args:
        model: Trained MLP model
        smiles: SMILES string
        task_idx: Task index for multi-task models
        device: Torch device

    Returns:
        Array of importance scores for each fingerprint bit
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048)

    # Generate fingerprint
    fp_gen = GetMorganGenerator(radius=2, fpSize=2048)
    fp = fp_gen.GetFingerprintAsNumPy(mol)

    # Convert to tensor with gradient
    x = torch.tensor(fp, dtype=torch.float, requires_grad=True).unsqueeze(0).to(device)

    model.eval()
    output = model(x)

    # Get gradient for specified task
    if output.dim() > 1:
        target = output[0, task_idx]
    else:
        target = output[0]

    target.backward()

    # Importance = input * gradient (integrated gradients approximation)
    importance = (x.grad[0] * x[0]).cpu().detach().numpy()

    return importance


def get_important_substructures(
    smiles: str,
    importance: np.ndarray,
    top_k: int = 10,
) -> list[dict]:
    """
    Identify important substructures based on fingerprint importance.

    Args:
        smiles: SMILES string
        importance: Fingerprint importance scores
        top_k: Number of top bits to analyze

    Returns:
        List of dictionaries with bit info and matched atoms
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    # Get top important bits
    top_bits = np.argsort(np.abs(importance))[-top_k:][::-1]

    # Generate bit info
    fp_gen = GetMorganGenerator(radius=2, fpSize=2048)
    fp = fp_gen.GetFingerprint(mol)

    results = []
    bit_info = {}
    fp_gen.GetFingerprint(mol, additionalOutput=bit_info)

    for bit in top_bits:
        result = {
            'bit_idx': int(bit),
            'importance': float(importance[bit]),
            'is_positive': importance[bit] > 0,
        }

        # Try to get atom mapping for this bit
        if bit in bit_info:
            result['atom_indices'] = list(bit_info[bit])

        results.append(result)

    return results


def compute_attention_weights(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None,
) -> Optional[np.ndarray]:
    """
    Extract attention weights from AttentiveFP model.

    Args:
        model: AttentiveFP model
        x: Node features
        edge_index: Edge indices
        edge_attr: Edge features
        batch: Batch assignment

    Returns:
        Attention weights or None if not available
    """
    model.eval()

    # Try to extract attention weights
    # Note: This depends on the specific AttentiveFP implementation
    try:
        with torch.no_grad():
            # Forward pass through attentive_fp
            if hasattr(model, 'attentive_fp'):
                afp = model.attentive_fp

                # Get attention from the model's internal state
                # This is implementation-specific
                if hasattr(afp, 'atom_attentions'):
                    return afp.atom_attentions.cpu().numpy()

    except Exception:
        pass

    return None


def compute_integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    n_steps: int = 50,
    task_idx: int = 0,
    device: torch.device = torch.device('cpu'),
) -> np.ndarray:
    """
    Compute integrated gradients attribution.

    Args:
        model: Neural network model
        x: Input tensor
        baseline: Baseline input (default: zeros)
        n_steps: Number of interpolation steps
        task_idx: Task index for multi-task models
        device: Torch device

    Returns:
        Attribution scores
    """
    if baseline is None:
        baseline = torch.zeros_like(x)

    x = x.to(device)
    baseline = baseline.to(device)

    model.eval()

    # Interpolate between baseline and input
    scaled_inputs = [
        baseline + (float(i) / n_steps) * (x - baseline)
        for i in range(n_steps + 1)
    ]

    gradients = []

    for scaled_input in scaled_inputs:
        scaled_input = scaled_input.clone().requires_grad_(True)
        output = model(scaled_input)

        if output.dim() > 1:
            target = output[0, task_idx]
        else:
            target = output[0]

        model.zero_grad()
        target.backward()

        gradients.append(scaled_input.grad.clone())

    # Average gradients and multiply by (input - baseline)
    avg_gradients = torch.stack(gradients).mean(dim=0)
    integrated_gradients = (x - baseline) * avg_gradients

    return integrated_gradients.cpu().detach().numpy()


def compute_perturbation_importance(
    model: torch.nn.Module,
    x: torch.Tensor,
    task_idx: int = 0,
    n_perturbations: int = 100,
    device: torch.device = torch.device('cpu'),
) -> np.ndarray:
    """
    Compute feature importance using random perturbation.

    Args:
        model: Neural network model
        x: Input tensor
        task_idx: Task index
        n_perturbations: Number of perturbations per feature
        device: Torch device

    Returns:
        Importance scores for each feature
    """
    x = x.to(device)
    model.eval()

    with torch.no_grad():
        baseline_output = torch.sigmoid(model(x))
        if baseline_output.dim() > 1:
            baseline_pred = baseline_output[0, task_idx].item()
        else:
            baseline_pred = baseline_output[0].item()

    n_features = x.shape[-1]
    importance = np.zeros(n_features)

    for i in range(n_features):
        diffs = []

        for _ in range(n_perturbations):
            x_perturbed = x.clone()
            # Flip the bit (for fingerprints) or add noise (for continuous)
            if x[0, i] in [0, 1]:
                x_perturbed[0, i] = 1 - x_perturbed[0, i]
            else:
                x_perturbed[0, i] += torch.randn(1).to(device) * 0.1

            with torch.no_grad():
                perturbed_output = torch.sigmoid(model(x_perturbed))
                if perturbed_output.dim() > 1:
                    perturbed_pred = perturbed_output[0, task_idx].item()
                else:
                    perturbed_pred = perturbed_output[0].item()

            diffs.append(abs(baseline_pred - perturbed_pred))

        importance[i] = np.mean(diffs)

    return importance


class PredictionExplainer:
    """
    Explain model predictions with multiple methods.

    Usage:
        explainer = PredictionExplainer(model, device)
        explanation = explainer.explain(smiles, task_idx=0)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device = torch.device('cpu'),
    ):
        self.model = model
        self.device = device
        self.model.eval()

    def explain(
        self,
        smiles: str,
        task_idx: int = 0,
        method: str = 'gradient',
    ) -> dict:
        """
        Generate explanation for a prediction.

        Args:
            smiles: SMILES string
            task_idx: Task index
            method: Attribution method ('gradient', 'integrated', 'perturbation')

        Returns:
            Dictionary with explanation data
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'error': 'Invalid SMILES'}

        # Generate fingerprint
        fp_gen = GetMorganGenerator(radius=2, fpSize=2048)
        fp = fp_gen.GetFingerprintAsNumPy(mol)
        x = torch.tensor(fp, dtype=torch.float).unsqueeze(0)

        # Get prediction
        with torch.no_grad():
            output = self.model(x.to(self.device))
            pred = torch.sigmoid(output).cpu().numpy()[0]

        # Compute attribution
        if method == 'gradient':
            importance = compute_fingerprint_importance(
                self.model, smiles, task_idx, self.device
            )
        elif method == 'integrated':
            importance = compute_integrated_gradients(
                self.model, x, task_idx=task_idx, device=self.device
            )[0]
        elif method == 'perturbation':
            importance = compute_perturbation_importance(
                self.model, x, task_idx=task_idx, device=self.device
            )
        else:
            importance = np.zeros(2048)

        # Get important substructures
        substructures = get_important_substructures(smiles, importance)

        return {
            'smiles': smiles,
            'prediction': float(pred[task_idx]) if pred.ndim > 0 else float(pred),
            'task_idx': task_idx,
            'method': method,
            'importance': importance.tolist(),
            'top_substructures': substructures,
            'importance_summary': {
                'max': float(np.max(importance)),
                'min': float(np.min(importance)),
                'mean': float(np.mean(importance)),
                'std': float(np.std(importance)),
                'n_positive': int((importance > 0).sum()),
                'n_negative': int((importance < 0).sum()),
            }
        }
