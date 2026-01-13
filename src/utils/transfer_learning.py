"""Transfer learning utilities for molecular property prediction."""

import torch
import torch.nn as nn
from typing import Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy
import logging

logger = logging.getLogger(__name__)


class FreezeStrategy(Enum):
    """Layer freezing strategies for transfer learning."""
    NONE = "none"  # No freezing
    ALL_BUT_LAST = "all_but_last"  # Freeze all but output layer
    GRADUAL_UNFREEZE = "gradual_unfreeze"  # Gradually unfreeze layers
    DISCRIMINATIVE_LR = "discriminative_lr"  # Different LR per layer
    FEATURE_EXTRACTOR = "feature_extractor"  # Freeze feature extraction layers


@dataclass
class TransferConfig:
    """Configuration for transfer learning."""

    freeze_strategy: FreezeStrategy = FreezeStrategy.ALL_BUT_LAST
    freeze_epochs: int = 5  # Epochs to keep layers frozen
    unfreeze_per_epoch: int = 1  # Layers to unfreeze per epoch
    lr_decay_factor: float = 0.9  # LR decay for lower layers
    reinit_head: bool = True  # Whether to reinitialize output layer
    source_tasks: int = 0  # Number of source tasks (0 = auto-detect)
    target_tasks: int = 12  # Number of target tasks


@dataclass
class TransferResult:
    """Results from transfer learning."""

    source_performance: dict = field(default_factory=dict)
    target_performance: dict = field(default_factory=dict)
    transfer_improvement: float = 0.0
    frozen_layers: list = field(default_factory=list)
    training_history: list = field(default_factory=list)


class LayerFreezer:
    """
    Utility for freezing and unfreezing model layers.

    Args:
        model: Model to freeze/unfreeze layers
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.frozen_layers = set()

    def freeze_all(self) -> list[str]:
        """Freeze all parameters."""
        frozen = []
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            frozen.append(name)
            self.frozen_layers.add(name)
        logger.info(f"Frozen {len(frozen)} layers")
        return frozen

    def unfreeze_all(self) -> list[str]:
        """Unfreeze all parameters."""
        unfrozen = []
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            unfrozen.append(name)
            self.frozen_layers.discard(name)
        logger.info(f"Unfrozen {len(unfrozen)} layers")
        return unfrozen

    def freeze_layers(self, layer_names: list[str]) -> list[str]:
        """
        Freeze specific layers by name pattern.

        Args:
            layer_names: List of layer name patterns to freeze

        Returns:
            List of frozen layer names
        """
        frozen = []
        for name, param in self.model.named_parameters():
            for pattern in layer_names:
                if pattern in name:
                    param.requires_grad = False
                    frozen.append(name)
                    self.frozen_layers.add(name)
                    break
        logger.info(f"Frozen {len(frozen)} layers matching patterns")
        return frozen

    def unfreeze_layers(self, layer_names: list[str]) -> list[str]:
        """
        Unfreeze specific layers by name pattern.

        Args:
            layer_names: List of layer name patterns to unfreeze

        Returns:
            List of unfrozen layer names
        """
        unfrozen = []
        for name, param in self.model.named_parameters():
            for pattern in layer_names:
                if pattern in name:
                    param.requires_grad = True
                    unfrozen.append(name)
                    self.frozen_layers.discard(name)
                    break
        logger.info(f"Unfrozen {len(unfrozen)} layers matching patterns")
        return unfrozen

    def freeze_except(self, layer_names: list[str]) -> list[str]:
        """
        Freeze all layers except those matching patterns.

        Args:
            layer_names: Patterns for layers to keep trainable

        Returns:
            List of frozen layer names
        """
        frozen = []
        for name, param in self.model.named_parameters():
            should_freeze = True
            for pattern in layer_names:
                if pattern in name:
                    should_freeze = False
                    break

            if should_freeze:
                param.requires_grad = False
                frozen.append(name)
                self.frozen_layers.add(name)
            else:
                param.requires_grad = True

        logger.info(f"Frozen {len(frozen)} layers, kept {len(layer_names)} patterns trainable")
        return frozen

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_frozen_params(self) -> list[str]:
        """Get list of frozen parameter names."""
        return list(self.frozen_layers)


class TransferLearner:
    """
    Transfer learning manager for molecular models.

    Handles loading pretrained models, adapting architectures,
    and fine-tuning strategies.

    Args:
        model: Target model architecture
        config: Transfer learning configuration
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TransferConfig] = None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.config = config or TransferConfig()
        self.device = device
        self.freezer = LayerFreezer(model)
        self.current_epoch = 0

    def load_pretrained(
        self,
        checkpoint_path: str,
        strict: bool = False,
        exclude_layers: Optional[list[str]] = None,
    ) -> dict:
        """
        Load pretrained weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to require exact architecture match
            exclude_layers: Layer patterns to exclude from loading

        Returns:
            Dict with loading info
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Filter excluded layers
        if exclude_layers:
            filtered_state = {}
            excluded = []
            for name, param in state_dict.items():
                exclude = False
                for pattern in exclude_layers:
                    if pattern in name:
                        exclude = True
                        excluded.append(name)
                        break
                if not exclude:
                    filtered_state[name] = param
            state_dict = filtered_state
            logger.info(f"Excluded {len(excluded)} layers from loading")

        # Handle dimension mismatches
        model_state = self.model.state_dict()
        compatible_state = {}
        mismatched = []

        for name, param in state_dict.items():
            if name in model_state:
                if param.shape == model_state[name].shape:
                    compatible_state[name] = param
                else:
                    mismatched.append((name, param.shape, model_state[name].shape))
            else:
                logger.debug(f"Skipping {name}: not in model")

        # Load compatible weights
        missing, unexpected = self.model.load_state_dict(
            compatible_state, strict=False
        )

        result = {
            "loaded": len(compatible_state),
            "missing": len(missing),
            "unexpected": len(unexpected),
            "mismatched": mismatched,
        }

        logger.info(f"Loaded {result['loaded']} layers, "
                   f"{result['missing']} missing, "
                   f"{len(mismatched)} dimension mismatches")

        return result

    def adapt_output_layer(
        self,
        num_tasks: int,
        layer_name: str = "output",
        reinit: bool = True,
    ) -> None:
        """
        Adapt output layer for new task count.

        Args:
            num_tasks: Number of tasks in target domain
            layer_name: Name pattern for output layer
            reinit: Whether to reinitialize the layer
        """
        # Find and replace output layer
        for name, module in self.model.named_modules():
            if layer_name in name and isinstance(module, nn.Linear):
                in_features = module.in_features
                new_layer = nn.Linear(in_features, num_tasks)

                if reinit:
                    nn.init.xavier_uniform_(new_layer.weight)
                    nn.init.zeros_(new_layer.bias)

                # Replace in model
                parts = name.split(".")
                parent = self.model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_layer.to(self.device))

                logger.info(f"Adapted {name}: {module.out_features} -> {num_tasks} outputs")
                return

        logger.warning(f"Could not find output layer matching '{layer_name}'")

    def apply_freeze_strategy(self) -> None:
        """Apply configured freezing strategy."""
        strategy = self.config.freeze_strategy

        if strategy == FreezeStrategy.NONE:
            self.freezer.unfreeze_all()

        elif strategy == FreezeStrategy.ALL_BUT_LAST:
            self.freezer.freeze_all()
            self.freezer.unfreeze_layers(["output", "head", "classifier", "fc_out"])

        elif strategy == FreezeStrategy.FEATURE_EXTRACTOR:
            # Freeze everything except fully connected layers
            self.freezer.freeze_layers(["conv", "gnn", "gcn", "attention", "embedding"])

        elif strategy == FreezeStrategy.GRADUAL_UNFREEZE:
            self.freezer.freeze_all()
            self.freezer.unfreeze_layers(["output", "head", "classifier"])

        logger.info(f"Applied freeze strategy: {strategy.value}")

    def step_gradual_unfreeze(self) -> int:
        """
        Unfreeze next layer(s) for gradual unfreezing.

        Returns:
            Number of layers unfrozen
        """
        if self.config.freeze_strategy != FreezeStrategy.GRADUAL_UNFREEZE:
            return 0

        self.current_epoch += 1

        if self.current_epoch <= self.config.freeze_epochs:
            return 0

        # Get frozen layers in reverse order (unfreeze from top)
        frozen_layers = list(self.freezer.frozen_layers)
        if not frozen_layers:
            return 0

        # Group by layer number/depth
        layer_groups = {}
        for name in frozen_layers:
            # Extract layer number from name (e.g., "layer.2.weight" -> 2)
            parts = name.split(".")
            for part in parts:
                if part.isdigit():
                    layer_num = int(part)
                    if layer_num not in layer_groups:
                        layer_groups[layer_num] = []
                    layer_groups[layer_num].append(name)
                    break

        if not layer_groups:
            # No numbered layers, unfreeze all
            self.freezer.unfreeze_all()
            return len(frozen_layers)

        # Unfreeze highest numbered layers first
        sorted_layers = sorted(layer_groups.keys(), reverse=True)
        unfrozen = 0

        for _ in range(self.config.unfreeze_per_epoch):
            if sorted_layers:
                layer_num = sorted_layers.pop(0)
                for name in layer_groups[layer_num]:
                    self.freezer.unfreeze_layers([name])
                    unfrozen += 1

        logger.info(f"Unfroze {unfrozen} parameters at epoch {self.current_epoch}")
        return unfrozen

    def get_discriminative_lr_params(
        self,
        base_lr: float,
    ) -> list[dict]:
        """
        Get parameter groups with discriminative learning rates.

        Lower layers get smaller learning rates.

        Args:
            base_lr: Base learning rate for top layers

        Returns:
            List of parameter groups for optimizer
        """
        # Group parameters by depth
        param_groups = []
        layer_depths = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Estimate depth from name
            depth = name.count(".")
            if depth not in layer_depths:
                layer_depths[depth] = []
            layer_depths[depth].append((name, param))

        # Assign LRs by depth
        max_depth = max(layer_depths.keys()) if layer_depths else 0

        for depth in sorted(layer_depths.keys()):
            # Lower depth = lower LR
            lr_multiplier = self.config.lr_decay_factor ** (max_depth - depth)
            lr = base_lr * lr_multiplier

            params = [p for _, p in layer_depths[depth]]
            param_groups.append({
                "params": params,
                "lr": lr,
            })

            logger.debug(f"Depth {depth}: {len(params)} params, lr={lr:.2e}")

        return param_groups

    def fine_tune_step(
        self,
        batch: tuple,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> dict:
        """
        Perform one fine-tuning step.

        Args:
            batch: Input batch
            optimizer: Optimizer
            criterion: Loss function

        Returns:
            Dict with step metrics
        """
        self.model.train()

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
        labels = labels.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # Forward pass
        optimizer.zero_grad()
        if isinstance(inputs, torch.Tensor):
            outputs = self.model(inputs)
        else:
            outputs = self.model(*inputs)

        # Compute loss
        if mask is not None:
            loss = criterion(outputs, labels.float())
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = criterion(outputs, labels.float()).mean()

        # Backward pass
        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}


class DomainAdapter(nn.Module):
    """
    Domain adaptation layer for transfer learning.

    Adds learnable domain adaptation between source and target.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dim = hidden_dim or input_dim

        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

        # Initialize close to identity
        nn.init.zeros_(self.adapter[-2].weight)
        nn.init.zeros_(self.adapter[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply domain adaptation with residual connection."""
        return x + self.adapter(x)


class FeatureExtractorWrapper(nn.Module):
    """
    Wrapper to extract intermediate features from a model.

    Args:
        model: Base model
        layer_names: Names of layers to extract features from
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: list[str],
    ):
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        self.features = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks for feature extraction."""
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook

        for name, module in self.model.named_modules():
            if name in self.layer_names:
                module.register_forward_hook(get_hook(name))

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, dict]:
        """Forward pass returning output and intermediate features."""
        self.features = {}
        output = self.model(*args, **kwargs)
        return output, self.features.copy()


def transfer_weights(
    source_model: nn.Module,
    target_model: nn.Module,
    layer_mapping: Optional[dict[str, str]] = None,
) -> dict:
    """
    Transfer weights between models with different architectures.

    Args:
        source_model: Model to transfer from
        target_model: Model to transfer to
        layer_mapping: Optional mapping of source->target layer names

    Returns:
        Dict with transfer statistics
    """
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()

    transferred = 0
    skipped = 0

    for target_name, target_param in target_state.items():
        # Find corresponding source layer
        source_name = target_name
        if layer_mapping and target_name in layer_mapping:
            source_name = layer_mapping[target_name]

        if source_name in source_state:
            source_param = source_state[source_name]

            if source_param.shape == target_param.shape:
                target_state[target_name] = source_param
                transferred += 1
            else:
                # Try partial transfer for dimension mismatch
                if len(source_param.shape) == len(target_param.shape):
                    # Transfer what fits
                    slices = tuple(
                        slice(0, min(s, t))
                        for s, t in zip(source_param.shape, target_param.shape)
                    )
                    target_state[target_name][slices] = source_param[slices]
                    transferred += 1
                else:
                    skipped += 1
        else:
            skipped += 1

    target_model.load_state_dict(target_state)

    logger.info(f"Transferred {transferred} layers, skipped {skipped}")

    return {
        "transferred": transferred,
        "skipped": skipped,
        "total_target": len(target_state),
    }


def create_task_specific_head(
    input_dim: int,
    num_tasks: int,
    hidden_dims: Optional[list[int]] = None,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Create a task-specific prediction head.

    Args:
        input_dim: Input feature dimension
        num_tasks: Number of output tasks
        hidden_dims: Optional hidden layer dimensions
        dropout: Dropout rate

    Returns:
        Task-specific head module
    """
    if hidden_dims is None:
        hidden_dims = [256]

    layers = []
    current_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(current_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        current_dim = hidden_dim

    layers.append(nn.Linear(current_dim, num_tasks))

    return nn.Sequential(*layers)


def get_pretrained_embeddings(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_name: str = "fc",
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract pretrained embeddings from a model.

    Args:
        model: Pretrained model
        dataloader: Data loader
        layer_name: Layer to extract embeddings from
        device: Device for computation

    Returns:
        Tuple of (embeddings, labels)
    """
    model = model.to(device)
    model.eval()

    # Register hook
    embeddings = []
    labels_list = []

    def hook(module, input, output):
        embeddings.append(output.detach().cpu())

    # Find and hook layer
    for name, module in model.named_modules():
        if layer_name in name:
            handle = module.register_forward_hook(hook)
            break
    else:
        raise ValueError(f"Layer {layer_name} not found")

    # Extract embeddings
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                inputs, labels = batch
            else:
                inputs = batch[0]
                labels = batch[-1]

            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(device)
                model(inputs)
            else:
                inputs = tuple(
                    x.to(device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                )
                model(*inputs)

            labels_list.append(labels)

    handle.remove()

    all_embeddings = torch.cat(embeddings, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    return all_embeddings, all_labels
