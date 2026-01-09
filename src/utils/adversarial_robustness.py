"""Adversarial robustness testing and training utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Union
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdversarialResult:
    """Results from adversarial attack."""

    original_predictions: np.ndarray
    adversarial_predictions: np.ndarray
    perturbations: np.ndarray
    success_rate: float
    mean_perturbation_norm: float
    robustness_score: float


@dataclass
class RobustnessMetrics:
    """Comprehensive robustness metrics."""

    clean_accuracy: float
    adversarial_accuracy: float
    accuracy_drop: float
    certified_radius: Optional[float] = None
    lipschitz_estimate: Optional[float] = None
    attack_results: dict = field(default_factory=dict)


class FGSM:
    """
    Fast Gradient Sign Method adversarial attack.

    Creates adversarial examples by adding perturbations in the
    direction of the gradient of the loss.

    Reference: Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015

    Args:
        model: Target model
        epsilon: Maximum perturbation magnitude
        targeted: Whether to perform targeted attack
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        targeted: bool = False,
    ):
        self.model = model
        self.epsilon = epsilon
        self.targeted = targeted

    def attack(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate adversarial examples.

        Args:
            inputs: Original inputs
            labels: True labels
            target_labels: Target labels for targeted attack

        Returns:
            Adversarial examples
        """
        inputs = inputs.clone().detach().requires_grad_(True)
        self.model.eval()

        outputs = self.model(inputs)

        if self.targeted and target_labels is not None:
            loss = F.binary_cross_entropy_with_logits(outputs, target_labels.float())
        else:
            loss = F.binary_cross_entropy_with_logits(outputs, labels.float())

        self.model.zero_grad()
        loss.backward()

        # Get sign of gradient
        grad_sign = inputs.grad.sign()

        # Generate adversarial examples
        if self.targeted:
            adversarial = inputs - self.epsilon * grad_sign
        else:
            adversarial = inputs + self.epsilon * grad_sign

        # Clip to valid range
        adversarial = torch.clamp(adversarial, 0, 1)

        return adversarial.detach()


class PGD:
    """
    Projected Gradient Descent adversarial attack.

    Iterative attack that takes multiple small steps and projects
    back to epsilon-ball.

    Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018

    Args:
        model: Target model
        epsilon: Maximum perturbation magnitude
        alpha: Step size
        num_steps: Number of attack iterations
        random_start: Whether to use random initialization
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        num_steps: int = 10,
        random_start: bool = True,
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start

    def attack(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate adversarial examples using PGD.

        Args:
            inputs: Original inputs
            labels: True labels
            mask: Optional mask for valid labels

        Returns:
            Adversarial examples
        """
        self.model.eval()
        original = inputs.clone().detach()

        # Random initialization
        if self.random_start:
            adversarial = inputs + torch.empty_like(inputs).uniform_(
                -self.epsilon, self.epsilon
            )
            adversarial = torch.clamp(adversarial, 0, 1)
        else:
            adversarial = inputs.clone()

        for _ in range(self.num_steps):
            adversarial.requires_grad_(True)

            outputs = self.model(adversarial)
            loss = F.binary_cross_entropy_with_logits(outputs, labels.float())

            if mask is not None:
                loss = (loss * mask).sum() / mask.sum().clamp(min=1)

            self.model.zero_grad()
            loss.backward()

            # PGD step
            grad = adversarial.grad
            adversarial = adversarial.detach() + self.alpha * grad.sign()

            # Project back to epsilon ball
            perturbation = adversarial - original
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
            adversarial = original + perturbation

            # Clip to valid range
            adversarial = torch.clamp(adversarial, 0, 1)

        return adversarial.detach()


class CarliniWagner:
    """
    Carlini & Wagner L2 attack.

    Optimization-based attack that finds minimal perturbations.

    Reference: Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks", S&P 2017

    Args:
        model: Target model
        c: Trade-off constant
        kappa: Confidence parameter
        num_steps: Optimization steps
        learning_rate: Optimizer learning rate
    """

    def __init__(
        self,
        model: nn.Module,
        c: float = 1.0,
        kappa: float = 0.0,
        num_steps: int = 100,
        learning_rate: float = 0.01,
    ):
        self.model = model
        self.c = c
        self.kappa = kappa
        self.num_steps = num_steps
        self.learning_rate = learning_rate

    def attack(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate adversarial examples using C&W attack.

        Args:
            inputs: Original inputs
            labels: True labels
            target_labels: Target labels for targeted attack

        Returns:
            Adversarial examples
        """
        self.model.eval()

        # Initialize perturbation in tanh space
        w = torch.atanh(2 * inputs.clone() - 1)
        w.requires_grad_(True)

        optimizer = torch.optim.Adam([w], lr=self.learning_rate)

        best_adversarial = inputs.clone()
        best_dist = float("inf")

        for _ in range(self.num_steps):
            # Map back to input space
            adversarial = 0.5 * (torch.tanh(w) + 1)

            # L2 distance
            dist = torch.sum((adversarial - inputs) ** 2)

            # Get predictions
            outputs = self.model(adversarial)

            # C&W loss
            if target_labels is not None:
                # Targeted: increase target class
                target_logits = torch.sum(outputs * target_labels.float(), dim=-1)
                other_logits = torch.max(
                    outputs * (1 - target_labels.float()) - 1e4 * target_labels.float(),
                    dim=-1
                )[0]
                cw_loss = torch.clamp(other_logits - target_logits + self.kappa, min=0)
            else:
                # Untargeted: decrease correct class
                correct_logits = torch.sum(outputs * labels.float(), dim=-1)
                other_logits = torch.max(
                    outputs * (1 - labels.float()) - 1e4 * labels.float(),
                    dim=-1
                )[0]
                cw_loss = torch.clamp(correct_logits - other_logits + self.kappa, min=0)

            loss = dist + self.c * cw_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track best adversarial
            if dist.item() < best_dist:
                pred = torch.sigmoid(outputs) > 0.5
                if target_labels is not None:
                    success = (pred == target_labels).all(dim=-1).any()
                else:
                    success = (pred != labels).any(dim=-1).any()

                if success:
                    best_adversarial = adversarial.detach().clone()
                    best_dist = dist.item()

        return best_adversarial


class AdversarialTrainer:
    """
    Adversarial training for robust models.

    Trains models with adversarial examples to improve robustness.

    Args:
        model: Model to train
        attack: Attack method for generating adversarial examples
        mix_ratio: Ratio of adversarial vs clean examples
        device: Device for training
    """

    def __init__(
        self,
        model: nn.Module,
        attack: Optional[Union[FGSM, PGD]] = None,
        mix_ratio: float = 0.5,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.attack = attack or PGD(model, epsilon=0.1, num_steps=7)
        self.mix_ratio = mix_ratio
        self.device = device

    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> dict:
        """
        Perform one adversarial training step.

        Args:
            inputs: Clean inputs
            labels: Ground truth labels
            optimizer: Optimizer
            criterion: Loss function

        Returns:
            Dict with training metrics
        """
        self.model.train()
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Generate adversarial examples
        self.model.eval()
        adv_inputs = self.attack.attack(inputs, labels)
        self.model.train()

        # Mix clean and adversarial
        batch_size = inputs.size(0)
        n_adv = int(batch_size * self.mix_ratio)

        mixed_inputs = torch.cat([inputs[:batch_size - n_adv], adv_inputs[:n_adv]], dim=0)
        mixed_labels = labels

        # Forward pass
        optimizer.zero_grad()
        outputs = self.model(mixed_inputs)
        loss = criterion(outputs, mixed_labels.float())

        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.mean()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute separate losses
        with torch.no_grad():
            clean_outputs = self.model(inputs)
            adv_outputs = self.model(adv_inputs)

            clean_loss = criterion(clean_outputs, labels.float()).mean()
            adv_loss = criterion(adv_outputs, labels.float()).mean()

        return {
            "total_loss": loss.item(),
            "clean_loss": clean_loss.item(),
            "adversarial_loss": adv_loss.item(),
        }


class RobustnessEvaluator:
    """
    Comprehensive robustness evaluation.

    Evaluates model robustness against various attacks and metrics.

    Args:
        model: Model to evaluate
        device: Device for evaluation
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.device = device

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        attacks: Optional[list] = None,
        epsilon_values: Optional[list[float]] = None,
    ) -> RobustnessMetrics:
        """
        Comprehensive robustness evaluation.

        Args:
            dataloader: Test data loader
            attacks: List of attack methods to evaluate
            epsilon_values: Perturbation budgets to test

        Returns:
            RobustnessMetrics with all results
        """
        if attacks is None:
            attacks = ["fgsm", "pgd"]
        if epsilon_values is None:
            epsilon_values = [0.01, 0.05, 0.1, 0.2]

        self.model.eval()

        # Compute clean accuracy
        all_preds = []
        all_labels = []

        for batch in dataloader:
            if len(batch) == 2:
                inputs, labels = batch
            else:
                inputs = batch[0]
                labels = batch[-1]

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                preds = torch.sigmoid(outputs) > 0.5

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        clean_accuracy = (all_preds == all_labels).float().mean().item()

        # Evaluate attacks
        attack_results = {}

        for attack_name in attacks:
            for epsilon in epsilon_values:
                key = f"{attack_name}_eps{epsilon}"

                if attack_name == "fgsm":
                    attacker = FGSM(self.model, epsilon=epsilon)
                elif attack_name == "pgd":
                    attacker = PGD(self.model, epsilon=epsilon)
                else:
                    continue

                adv_preds = []

                for batch in dataloader:
                    if len(batch) == 2:
                        inputs, labels = batch
                    else:
                        inputs = batch[0]
                        labels = batch[-1]

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    adv_inputs = attacker.attack(inputs, labels)

                    with torch.no_grad():
                        adv_outputs = self.model(adv_inputs)
                        preds = torch.sigmoid(adv_outputs) > 0.5

                    adv_preds.append(preds.cpu())

                adv_preds = torch.cat(adv_preds, dim=0)
                adv_accuracy = (adv_preds == all_labels).float().mean().item()

                attack_results[key] = {
                    "adversarial_accuracy": adv_accuracy,
                    "accuracy_drop": clean_accuracy - adv_accuracy,
                }

        # Find worst-case adversarial accuracy
        worst_adv_acc = min(r["adversarial_accuracy"] for r in attack_results.values())

        return RobustnessMetrics(
            clean_accuracy=clean_accuracy,
            adversarial_accuracy=worst_adv_acc,
            accuracy_drop=clean_accuracy - worst_adv_acc,
            attack_results=attack_results,
        )

    def compute_local_lipschitz(
        self,
        inputs: torch.Tensor,
        num_samples: int = 100,
        epsilon: float = 0.01,
    ) -> float:
        """
        Estimate local Lipschitz constant around inputs.

        Args:
            inputs: Input samples
            num_samples: Number of perturbations to sample
            epsilon: Perturbation radius

        Returns:
            Estimated Lipschitz constant
        """
        inputs = inputs.to(self.device)
        self.model.eval()

        with torch.no_grad():
            original_outputs = self.model(inputs)

        max_ratio = 0.0

        for _ in range(num_samples):
            # Random perturbation
            perturbation = torch.empty_like(inputs).uniform_(-epsilon, epsilon)
            perturbed = torch.clamp(inputs + perturbation, 0, 1)

            with torch.no_grad():
                perturbed_outputs = self.model(perturbed)

            # Compute ratio
            output_diff = torch.norm(
                original_outputs - perturbed_outputs, dim=-1
            )
            input_diff = torch.norm(
                perturbation.view(inputs.size(0), -1), dim=-1
            )

            ratio = (output_diff / input_diff.clamp(min=1e-8)).max()
            max_ratio = max(max_ratio, ratio.item())

        return max_ratio


class InputGradientRegularization(nn.Module):
    """
    Input gradient regularization for improved robustness.

    Penalizes large input gradients to smooth decision boundaries.

    Args:
        lambda_reg: Regularization strength
    """

    def __init__(self, lambda_reg: float = 0.1):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        base_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss with input gradient regularization.

        Args:
            model: Model being trained
            inputs: Input batch
            labels: Ground truth labels
            base_loss: Base task loss

        Returns:
            Regularized loss
        """
        inputs.requires_grad_(True)
        outputs = model(inputs)
        loss = F.binary_cross_entropy_with_logits(outputs, labels.float())

        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.mean()

        # Compute input gradients
        grads = torch.autograd.grad(
            loss, inputs, create_graph=True, retain_graph=True
        )[0]

        # Gradient penalty
        grad_norm = grads.view(grads.size(0), -1).norm(2, dim=1)
        grad_penalty = grad_norm.mean()

        return base_loss + self.lambda_reg * grad_penalty


class JacobianRegularization(nn.Module):
    """
    Jacobian regularization for robustness.

    Penalizes large Jacobian norms to improve Lipschitz continuity.

    Args:
        lambda_reg: Regularization strength
        num_proj: Number of random projections for estimation
    """

    def __init__(self, lambda_reg: float = 0.01, num_proj: int = 1):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.num_proj = num_proj

    def forward(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Jacobian regularization term.

        Args:
            model: Model (unused, for interface compatibility)
            inputs: Input batch
            outputs: Model outputs

        Returns:
            Jacobian regularization loss
        """
        batch_size = inputs.size(0)
        input_dim = inputs.view(batch_size, -1).size(1)
        output_dim = outputs.size(-1)

        jacobian_norm = 0.0

        for _ in range(self.num_proj):
            # Random projection vector
            v = torch.randn(batch_size, output_dim, device=inputs.device)
            v = v / v.norm(dim=1, keepdim=True)

            # Vector-Jacobian product
            vjp = torch.autograd.grad(
                outputs, inputs,
                grad_outputs=v,
                create_graph=True,
                retain_graph=True,
            )[0]

            vjp_flat = vjp.view(batch_size, -1)
            jacobian_norm += vjp_flat.norm(2, dim=1).mean()

        jacobian_norm /= self.num_proj

        return self.lambda_reg * jacobian_norm


def certified_radius(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    sigma: float = 0.1,
    n_samples: int = 1000,
    alpha: float = 0.001,
) -> np.ndarray:
    """
    Compute certified robustness radius using randomized smoothing.

    Reference: Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing", ICML 2019

    Args:
        model: Model to certify
        inputs: Input samples
        labels: True labels
        sigma: Noise standard deviation
        n_samples: Number of samples for estimation
        alpha: Failure probability

    Returns:
        Certified radius for each input
    """
    model.eval()
    device = next(model.parameters()).device

    batch_size = inputs.size(0)
    radii = np.zeros(batch_size)

    with torch.no_grad():
        for i in range(batch_size):
            x = inputs[i:i+1].to(device)

            # Sample noisy predictions
            correct_count = 0
            for _ in range(n_samples):
                noise = torch.randn_like(x) * sigma
                noisy_x = x + noise
                outputs = model(noisy_x)
                pred = torch.sigmoid(outputs) > 0.5

                if (pred == labels[i:i+1].to(device)).all():
                    correct_count += 1

            # Estimate probability
            p = correct_count / n_samples

            if p > 0.5:
                from scipy.stats import norm
                radius = sigma * norm.ppf(p)
                radii[i] = max(0, radius)

    return radii
