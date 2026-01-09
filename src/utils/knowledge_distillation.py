"""Knowledge distillation utilities for creating lightweight student models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Iterator
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""

    temperature: float = 3.0
    alpha: float = 0.5  # Weight for distillation loss vs hard label loss
    soft_target_loss: str = "kl_div"  # "kl_div" or "mse"
    feature_distillation: bool = False
    intermediate_layers: list = field(default_factory=list)


@dataclass
class DistillationResult:
    """Results from distillation training."""

    student_loss: float
    distillation_loss: float
    hard_label_loss: float
    total_loss: float
    teacher_accuracy: float
    student_accuracy: float
    compression_ratio: float


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss combining soft and hard targets.

    Computes: L = α * L_distill(soft_targets) + (1-α) * L_hard(true_labels)

    Args:
        temperature: Softening temperature for teacher outputs
        alpha: Weight for distillation loss (0-1)
        soft_target_loss: Type of soft target loss ('kl_div' or 'mse')
    """

    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.5,
        soft_target_loss: str = "kl_div",
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.soft_target_loss = soft_target_loss

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute distillation loss.

        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits (detached)
            labels: Optional hard labels for supervised component
            mask: Optional mask for missing labels

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Soft targets from teacher
        soft_teacher = torch.sigmoid(teacher_logits / self.temperature)
        soft_student = torch.sigmoid(student_logits / self.temperature)

        # Distillation loss
        if self.soft_target_loss == "kl_div":
            # KL divergence for binary targets
            distill_loss = F.binary_cross_entropy(
                soft_student, soft_teacher.detach(), reduction="none"
            )
        else:
            # MSE loss
            distill_loss = F.mse_loss(
                soft_student, soft_teacher.detach(), reduction="none"
            )

        # Apply mask if provided
        if mask is not None:
            distill_loss = distill_loss * mask
            distill_loss = distill_loss.sum() / mask.sum().clamp(min=1)
        else:
            distill_loss = distill_loss.mean()

        # Scale by temperature squared (as in original paper)
        distill_loss = distill_loss * (self.temperature ** 2)

        # Hard label loss
        if labels is not None:
            hard_loss = F.binary_cross_entropy_with_logits(
                student_logits, labels.float(), reduction="none"
            )
            if mask is not None:
                hard_loss = hard_loss * mask
                hard_loss = hard_loss.sum() / mask.sum().clamp(min=1)
            else:
                hard_loss = hard_loss.mean()

            total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        else:
            hard_loss = torch.tensor(0.0)
            total_loss = distill_loss

        return total_loss, {
            "distillation_loss": distill_loss.item(),
            "hard_label_loss": hard_loss.item() if isinstance(hard_loss, torch.Tensor) else hard_loss,
            "total_loss": total_loss.item(),
        }


class FeatureDistillationLoss(nn.Module):
    """
    Feature-based distillation loss for intermediate layer matching.

    Matches intermediate representations between teacher and student.

    Args:
        student_dims: List of student intermediate layer dimensions
        teacher_dims: List of teacher intermediate layer dimensions
    """

    def __init__(
        self,
        student_dims: list[int],
        teacher_dims: list[int],
    ):
        super().__init__()

        # Create projection layers if dimensions don't match
        self.projectors = nn.ModuleList()
        for s_dim, t_dim in zip(student_dims, teacher_dims):
            if s_dim != t_dim:
                self.projectors.append(nn.Linear(s_dim, t_dim))
            else:
                self.projectors.append(nn.Identity())

    def forward(
        self,
        student_features: list[torch.Tensor],
        teacher_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute feature distillation loss.

        Args:
            student_features: List of student intermediate features
            teacher_features: List of teacher intermediate features

        Returns:
            Feature matching loss
        """
        total_loss = 0.0

        for proj, s_feat, t_feat in zip(
            self.projectors, student_features, teacher_features
        ):
            projected = proj(s_feat)
            loss = F.mse_loss(projected, t_feat.detach())
            total_loss += loss

        return total_loss / len(self.projectors)


class AttentionTransferLoss(nn.Module):
    """
    Attention transfer for distillation.

    Transfers attention maps from teacher to student.

    Reference: Zagoruyko & Komodakis, "Paying More Attention to Attention", ICLR 2017
    """

    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def forward(
        self,
        student_attention: torch.Tensor,
        teacher_attention: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention transfer loss.

        Args:
            student_attention: Student attention maps [B, H, N, N]
            teacher_attention: Teacher attention maps [B, H, N, N]

        Returns:
            Attention transfer loss
        """
        # Normalize attention maps
        s_norm = F.normalize(student_attention.pow(self.p).mean(1).view(
            student_attention.size(0), -1
        ), dim=1)
        t_norm = F.normalize(teacher_attention.pow(self.p).mean(1).view(
            teacher_attention.size(0), -1
        ), dim=1)

        return (s_norm - t_norm).pow(2).mean()


class KnowledgeDistiller:
    """
    Train student models using knowledge distillation from teacher.

    Supports various distillation strategies including response-based,
    feature-based, and relation-based distillation.

    Args:
        teacher: Teacher model (should be in eval mode)
        student: Student model to train
        config: Distillation configuration
        device: Device for training
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[DistillationConfig] = None,
        device: str = "cuda",
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.config = config or DistillationConfig()
        self.device = device

        # Set teacher to eval mode
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.distill_loss = DistillationLoss(
            temperature=self.config.temperature,
            alpha=self.config.alpha,
            soft_target_loss=self.config.soft_target_loss,
        )

        self._compute_compression_ratio()

    def _compute_compression_ratio(self) -> None:
        """Compute model compression ratio."""
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        self.compression_ratio = teacher_params / student_params
        logger.info(f"Compression ratio: {self.compression_ratio:.2f}x")
        logger.info(f"Teacher params: {teacher_params:,}, Student params: {student_params:,}")

    def distill_step(
        self,
        inputs: tuple,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Perform one distillation step.

        Args:
            inputs: Model inputs (varies by model type)
            labels: Optional ground truth labels
            mask: Optional mask for missing labels

        Returns:
            Dict containing loss components
        """
        # Get teacher predictions
        with torch.no_grad():
            if isinstance(inputs, torch.Tensor):
                teacher_logits = self.teacher(inputs)
            else:
                teacher_logits = self.teacher(*inputs)

        # Get student predictions
        self.student.train()
        if isinstance(inputs, torch.Tensor):
            student_logits = self.student(inputs)
        else:
            student_logits = self.student(*inputs)

        # Compute distillation loss
        loss, components = self.distill_loss(
            student_logits, teacher_logits, labels, mask
        )

        return {"loss": loss, **components}

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> dict:
        """
        Train student for one epoch using distillation.

        Args:
            dataloader: Training data loader
            optimizer: Optimizer for student model
            scheduler: Optional learning rate scheduler

        Returns:
            Dict containing epoch metrics
        """
        self.student.train()
        epoch_losses = {
            "total_loss": 0.0,
            "distillation_loss": 0.0,
            "hard_label_loss": 0.0,
        }
        num_batches = 0

        for batch in dataloader:
            # Handle different batch formats
            if len(batch) == 2:
                inputs, labels = batch
                mask = None
            elif len(batch) == 3:
                inputs, labels, mask = batch
            else:
                inputs = batch[:-1]
                labels = batch[-1]
                mask = None

            # Move to device
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            else:
                inputs = tuple(
                    x.to(self.device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                )
            labels = labels.to(self.device) if labels is not None else None
            mask = mask.to(self.device) if mask is not None else None

            # Distillation step
            optimizer.zero_grad()
            result = self.distill_step(inputs, labels, mask)
            result["loss"].backward()
            optimizer.step()

            # Accumulate losses
            for key in epoch_losses:
                if key in result:
                    epoch_losses[key] += result[key]
            num_batches += 1

        if scheduler is not None:
            scheduler.step()

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> dict:
        """
        Evaluate student and compare with teacher.

        Args:
            dataloader: Evaluation data loader

        Returns:
            Dict containing evaluation metrics
        """
        self.student.eval()
        self.teacher.eval()

        teacher_preds = []
        student_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    inputs, labels = batch
                else:
                    inputs = batch[:-1]
                    labels = batch[-1]

                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                else:
                    inputs = tuple(
                        x.to(self.device) if isinstance(x, torch.Tensor) else x
                        for x in inputs
                    )
                labels = labels.to(self.device)

                # Get predictions
                if isinstance(inputs, torch.Tensor):
                    t_logits = self.teacher(inputs)
                    s_logits = self.student(inputs)
                else:
                    t_logits = self.teacher(*inputs)
                    s_logits = self.student(*inputs)

                teacher_preds.append(torch.sigmoid(t_logits).cpu())
                student_preds.append(torch.sigmoid(s_logits).cpu())
                all_labels.append(labels.cpu())

        teacher_preds = torch.cat(teacher_preds, dim=0).numpy()
        student_preds = torch.cat(student_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # Compute metrics
        from sklearn.metrics import roc_auc_score

        # Handle multi-task predictions
        if len(teacher_preds.shape) > 1 and teacher_preds.shape[1] > 1:
            valid_aucs_teacher = []
            valid_aucs_student = []
            for i in range(teacher_preds.shape[1]):
                mask = ~np.isnan(all_labels[:, i])
                if mask.sum() > 0 and len(np.unique(all_labels[mask, i])) > 1:
                    valid_aucs_teacher.append(
                        roc_auc_score(all_labels[mask, i], teacher_preds[mask, i])
                    )
                    valid_aucs_student.append(
                        roc_auc_score(all_labels[mask, i], student_preds[mask, i])
                    )
            teacher_auc = np.mean(valid_aucs_teacher) if valid_aucs_teacher else 0.0
            student_auc = np.mean(valid_aucs_student) if valid_aucs_student else 0.0
        else:
            teacher_auc = roc_auc_score(all_labels, teacher_preds)
            student_auc = roc_auc_score(all_labels, student_preds)

        # Agreement between teacher and student
        teacher_binary = (teacher_preds > 0.5).astype(int)
        student_binary = (student_preds > 0.5).astype(int)
        agreement = (teacher_binary == student_binary).mean()

        return {
            "teacher_auc": teacher_auc,
            "student_auc": student_auc,
            "auc_retention": student_auc / teacher_auc if teacher_auc > 0 else 0.0,
            "agreement": agreement,
            "compression_ratio": self.compression_ratio,
        }


class ProgressiveDistillation:
    """
    Progressive knowledge distillation with multiple intermediate models.

    Creates a chain of increasingly smaller models, each distilling
    from the previous one.

    Args:
        teacher: Original teacher model
        student_factory: Callable that creates student models of given size
        sizes: List of model sizes for progressive distillation
        device: Device for training
    """

    def __init__(
        self,
        teacher: nn.Module,
        student_factory: Callable[[int], nn.Module],
        sizes: list[int],
        device: str = "cuda",
    ):
        self.teacher = teacher.to(device)
        self.student_factory = student_factory
        self.sizes = sizes
        self.device = device
        self.intermediate_models = []

    def distill_chain(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs_per_stage: int = 10,
        lr: float = 1e-3,
    ) -> list[nn.Module]:
        """
        Perform progressive distillation through model chain.

        Args:
            dataloader: Training data loader
            epochs_per_stage: Number of epochs per distillation stage
            lr: Learning rate

        Returns:
            List of trained intermediate models
        """
        current_teacher = self.teacher
        self.intermediate_models = []

        for i, size in enumerate(self.sizes):
            logger.info(f"Stage {i+1}/{len(self.sizes)}: Distilling to size {size}")

            # Create student
            student = self.student_factory(size).to(self.device)

            # Create distiller
            distiller = KnowledgeDistiller(
                current_teacher, student, device=self.device
            )

            # Train
            optimizer = torch.optim.Adam(student.parameters(), lr=lr)
            for epoch in range(epochs_per_stage):
                metrics = distiller.train_epoch(dataloader, optimizer)
                if (epoch + 1) % 5 == 0:
                    logger.info(f"  Epoch {epoch+1}: Loss = {metrics['total_loss']:.4f}")

            # Evaluate
            eval_metrics = distiller.evaluate(dataloader)
            logger.info(f"  Student AUC: {eval_metrics['student_auc']:.4f}, "
                       f"Retention: {eval_metrics['auc_retention']:.2%}")

            self.intermediate_models.append(student)
            current_teacher = student

        return self.intermediate_models


def create_student_model(
    teacher: nn.Module,
    compression_factor: float = 2.0,
    hidden_dims: Optional[list[int]] = None,
) -> nn.Module:
    """
    Create a compressed student model based on teacher architecture.

    Args:
        teacher: Teacher model
        compression_factor: Factor to reduce hidden dimensions
        hidden_dims: Optional explicit hidden dimensions

    Returns:
        Smaller student model
    """
    # Analyze teacher architecture
    teacher_dims = []
    for name, module in teacher.named_modules():
        if isinstance(module, nn.Linear):
            teacher_dims.append((module.in_features, module.out_features))

    if not teacher_dims:
        raise ValueError("Could not analyze teacher architecture")

    # Determine student dimensions
    if hidden_dims is None:
        hidden_dims = []
        for in_f, out_f in teacher_dims[:-1]:  # Keep output dimension same
            hidden_dims.append(int(out_f / compression_factor))
        hidden_dims.append(teacher_dims[-1][1])  # Keep output dim

    # Create student model
    layers = []
    input_dim = teacher_dims[0][0]

    for i, hidden_dim in enumerate(hidden_dims[:-1]):
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        input_dim = hidden_dim

    layers.append(nn.Linear(input_dim, hidden_dims[-1]))

    student = nn.Sequential(*layers)

    logger.info(f"Created student with dims: {hidden_dims}")

    return student


def distill_ensemble_to_single(
    ensemble_models: list[nn.Module],
    ensemble_weights: list[float],
    student: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epochs: int = 20,
    device: str = "cuda",
    lr: float = 1e-3,
) -> nn.Module:
    """
    Distill an ensemble of models into a single student model.

    Args:
        ensemble_models: List of teacher models
        ensemble_weights: Weights for each model in ensemble
        student: Student model to train
        dataloader: Training data
        epochs: Number of training epochs
        device: Device for training
        lr: Learning rate

    Returns:
        Trained student model
    """
    # Move models to device
    for model in ensemble_models:
        model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    student = student.to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    criterion = nn.MSELoss()

    weights = torch.tensor(ensemble_weights, device=device)
    weights = weights / weights.sum()

    for epoch in range(epochs):
        student.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            if len(batch) == 2:
                inputs, _ = batch
            else:
                inputs = batch[0]

            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device)

            # Get ensemble predictions
            with torch.no_grad():
                ensemble_preds = []
                for model in ensemble_models:
                    if isinstance(inputs, torch.Tensor):
                        pred = torch.sigmoid(model(inputs))
                    else:
                        pred = torch.sigmoid(model(*[x.to(device) for x in inputs]))
                    ensemble_preds.append(pred)

                # Weighted average
                ensemble_preds = torch.stack(ensemble_preds, dim=0)
                teacher_pred = (ensemble_preds * weights.view(-1, 1, 1)).sum(dim=0)

            # Student prediction
            optimizer.zero_grad()
            if isinstance(inputs, torch.Tensor):
                student_pred = torch.sigmoid(student(inputs))
            else:
                student_pred = torch.sigmoid(student(*[x.to(device) for x in inputs]))

            loss = criterion(student_pred, teacher_pred)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

    return student
