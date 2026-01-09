"""Federated learning utilities for distributed molecular model training."""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np
import copy
import logging

logger = logging.getLogger(__name__)


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""

    num_rounds: int = 100
    clients_per_round: float = 1.0  # Fraction of clients per round
    local_epochs: int = 5
    local_batch_size: int = 32
    learning_rate: float = 0.01
    aggregation_strategy: str = "fedavg"  # "fedavg", "fedprox", "scaffold"
    differential_privacy: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0


@dataclass
class ClientUpdate:
    """Update from a federated client."""

    client_id: str
    model_state: Dict[str, torch.Tensor]
    num_samples: int
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class FederatedRound:
    """Results from a federated round."""

    round_num: int
    participating_clients: List[str]
    aggregated_metrics: Dict[str, float]
    client_metrics: Dict[str, Dict[str, float]]


class FederatedClient:
    """
    Client for federated learning.

    Performs local training on private data.

    Args:
        client_id: Unique client identifier
        model: Local model copy
        train_loader: Local training data
        val_loader: Optional validation data
        config: Federated learning configuration
        device: Device for training
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        config: Optional[FederatedConfig] = None,
        device: str = "cuda",
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or FederatedConfig()
        self.device = device

        self.num_samples = len(train_loader.dataset)

    def receive_model(self, global_state: Dict[str, torch.Tensor]) -> None:
        """Receive global model from server."""
        self.model.load_state_dict(global_state)

    def local_train(
        self,
        criterion: Optional[nn.Module] = None,
        global_model: Optional[nn.Module] = None,
    ) -> ClientUpdate:
        """
        Perform local training.

        Args:
            criterion: Loss function
            global_model: Global model (for FedProx)

        Returns:
            ClientUpdate with new model state
        """
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=0.9,
        )

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.config.local_epochs):
            for batch in self.train_loader:
                if len(batch) == 2:
                    inputs, labels = batch
                else:
                    inputs, labels = batch[0], batch[1]

                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                else:
                    inputs = tuple(
                        x.to(self.device) if isinstance(x, torch.Tensor) else x
                        for x in inputs
                    )
                labels = labels.to(self.device)

                optimizer.zero_grad()

                if isinstance(inputs, torch.Tensor):
                    outputs = self.model(inputs)
                else:
                    outputs = self.model(*inputs)

                loss = criterion(outputs, labels.float())
                if loss.dim() > 0:
                    loss = loss.mean()

                # FedProx proximal term
                if (
                    self.config.aggregation_strategy == "fedprox"
                    and global_model is not None
                ):
                    prox_term = 0.0
                    for p, g in zip(
                        self.model.parameters(), global_model.parameters()
                    ):
                        prox_term += ((p - g.to(self.device)) ** 2).sum()
                    loss = loss + 0.01 * prox_term

                loss.backward()

                # Gradient clipping for differential privacy
                if self.config.differential_privacy:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.dp_max_grad_norm,
                    )

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        # Evaluate if validation data available
        metrics = {"train_loss": avg_loss}
        if self.val_loader is not None:
            val_metrics = self._evaluate()
            metrics.update(val_metrics)

        return ClientUpdate(
            client_id=self.client_id,
            model_state=copy.deepcopy(self.model.state_dict()),
            num_samples=self.num_samples,
            metrics=metrics,
        )

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation data."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        criterion = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            for batch in self.val_loader:
                if len(batch) == 2:
                    inputs, labels = batch
                else:
                    inputs, labels = batch[0], batch[1]

                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                else:
                    inputs = tuple(
                        x.to(self.device) if isinstance(x, torch.Tensor) else x
                        for x in inputs
                    )
                labels = labels.to(self.device)

                if isinstance(inputs, torch.Tensor):
                    outputs = self.model(inputs)
                else:
                    outputs = self.model(*inputs)

                loss = criterion(outputs, labels.float()).mean()
                total_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        accuracy = (all_preds == all_labels).float().mean().item()

        return {
            "val_loss": total_loss / len(self.val_loader),
            "val_accuracy": accuracy,
        }


class FederatedServer:
    """
    Server for federated learning.

    Coordinates clients and aggregates updates.

    Args:
        global_model: Initial global model
        config: Federated learning configuration
        device: Device for aggregation
    """

    def __init__(
        self,
        global_model: nn.Module,
        config: Optional[FederatedConfig] = None,
        device: str = "cuda",
    ):
        self.global_model = global_model.to(device)
        self.config = config or FederatedConfig()
        self.device = device
        self.round_history = []

        # For SCAFFOLD
        self.server_control_variate = None

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """Get current global model state."""
        return copy.deepcopy(self.global_model.state_dict())

    def select_clients(
        self,
        available_clients: List[str],
    ) -> List[str]:
        """Select clients for current round."""
        num_clients = int(
            len(available_clients) * self.config.clients_per_round
        )
        num_clients = max(1, num_clients)
        return np.random.choice(
            available_clients, num_clients, replace=False
        ).tolist()

    def aggregate(
        self,
        client_updates: List[ClientUpdate],
    ) -> FederatedRound:
        """
        Aggregate client updates.

        Args:
            client_updates: Updates from participating clients

        Returns:
            FederatedRound with aggregation results
        """
        if self.config.aggregation_strategy == "fedavg":
            self._fedavg_aggregate(client_updates)
        elif self.config.aggregation_strategy == "fedprox":
            self._fedavg_aggregate(client_updates)  # Same aggregation as FedAvg
        elif self.config.aggregation_strategy == "scaffold":
            self._scaffold_aggregate(client_updates)
        else:
            self._fedavg_aggregate(client_updates)

        # Compute aggregated metrics
        total_samples = sum(u.num_samples for u in client_updates)
        aggregated_metrics = {}

        for key in client_updates[0].metrics.keys():
            weighted_sum = sum(
                u.metrics.get(key, 0) * u.num_samples
                for u in client_updates
            )
            aggregated_metrics[key] = weighted_sum / total_samples

        client_metrics = {
            u.client_id: u.metrics for u in client_updates
        }

        round_result = FederatedRound(
            round_num=len(self.round_history) + 1,
            participating_clients=[u.client_id for u in client_updates],
            aggregated_metrics=aggregated_metrics,
            client_metrics=client_metrics,
        )

        self.round_history.append(round_result)

        return round_result

    def _fedavg_aggregate(
        self,
        client_updates: List[ClientUpdate],
    ) -> None:
        """FedAvg aggregation."""
        total_samples = sum(u.num_samples for u in client_updates)

        # Weighted average of model parameters
        global_state = self.global_model.state_dict()

        for key in global_state.keys():
            weighted_sum = sum(
                u.model_state[key].float() * u.num_samples
                for u in client_updates
            )
            global_state[key] = (weighted_sum / total_samples).to(
                global_state[key].dtype
            )

        self.global_model.load_state_dict(global_state)

    def _scaffold_aggregate(
        self,
        client_updates: List[ClientUpdate],
    ) -> None:
        """SCAFFOLD aggregation with control variates."""
        # Simplified SCAFFOLD implementation
        # Full implementation would track client control variates
        self._fedavg_aggregate(client_updates)


class DifferentialPrivacy:
    """
    Differential privacy utilities for federated learning.

    Args:
        epsilon: Privacy budget
        delta: Failure probability
        max_grad_norm: Maximum gradient norm
        noise_multiplier: Noise scale (computed from epsilon if None)
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

        if noise_multiplier is None:
            # Approximate noise multiplier from privacy budget
            self.noise_multiplier = self._compute_noise_multiplier()
        else:
            self.noise_multiplier = noise_multiplier

    def _compute_noise_multiplier(self) -> float:
        """Compute noise multiplier from privacy parameters."""
        # Simplified computation
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients to max norm."""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)

        return total_norm

    def add_noise(self, model: nn.Module, batch_size: int) -> None:
        """Add Gaussian noise to gradients."""
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * (
                    self.noise_multiplier * self.max_grad_norm / batch_size
                )
                param.grad.data.add_(noise)

    def apply_dp_sgd(
        self,
        model: nn.Module,
        batch_size: int,
    ) -> float:
        """Apply DP-SGD: clip gradients and add noise."""
        grad_norm = self.clip_gradients(model)
        self.add_noise(model, batch_size)
        return grad_norm


class SecureAggregation:
    """
    Secure aggregation for federated learning.

    Provides privacy by aggregating encrypted updates.

    Note: This is a simplified simulation. Real secure aggregation
    requires cryptographic protocols.

    Args:
        num_clients: Number of clients
        threshold: Minimum clients for aggregation
    """

    def __init__(
        self,
        num_clients: int,
        threshold: int,
    ):
        self.num_clients = num_clients
        self.threshold = threshold

        # Simulated secret shares
        self.shares = {}

    def create_shares(
        self,
        client_id: str,
        values: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Create secret shares for values."""
        # Simulated: add random mask that sums to zero
        mask = {
            key: torch.randn_like(tensor)
            for key, tensor in values.items()
        }
        self.shares[client_id] = mask

        masked = {
            key: tensor + mask[key]
            for key, tensor in values.items()
        }

        return masked

    def aggregate_shares(
        self,
        all_masked: List[Dict[str, torch.Tensor]],
        client_ids: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Aggregate shares and remove masks."""
        if len(all_masked) < self.threshold:
            raise ValueError(
                f"Need at least {self.threshold} clients for secure aggregation"
            )

        # Sum all masked values
        result = {}
        for key in all_masked[0].keys():
            result[key] = sum(m[key] for m in all_masked)

        # Subtract sum of masks
        for client_id in client_ids:
            if client_id in self.shares:
                for key in result.keys():
                    result[key] -= self.shares[client_id][key]

        return result


class FederatedLearner:
    """
    Complete federated learning system.

    Manages server and clients for distributed training.

    Args:
        model_fn: Function to create model instances
        config: Federated learning configuration
        device: Device for training
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        config: Optional[FederatedConfig] = None,
        device: str = "cuda",
    ):
        self.model_fn = model_fn
        self.config = config or FederatedConfig()
        self.device = device

        # Initialize server
        global_model = model_fn()
        self.server = FederatedServer(global_model, config, device)

        # Client registry
        self.clients: Dict[str, FederatedClient] = {}

    def add_client(
        self,
        client_id: str,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> None:
        """Add a client to the federation."""
        model = self.model_fn()
        model.load_state_dict(self.server.get_global_state())

        client = FederatedClient(
            client_id=client_id,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            device=self.device,
        )

        self.clients[client_id] = client
        logger.info(f"Added client {client_id} with {client.num_samples} samples")

    def train_round(
        self,
        criterion: Optional[nn.Module] = None,
    ) -> FederatedRound:
        """Execute one federated learning round."""
        # Select participating clients
        available = list(self.clients.keys())
        selected = self.server.select_clients(available)

        logger.info(f"Round with {len(selected)} clients: {selected}")

        # Distribute global model
        global_state = self.server.get_global_state()
        for client_id in selected:
            self.clients[client_id].receive_model(global_state)

        # Local training
        updates = []
        for client_id in selected:
            update = self.clients[client_id].local_train(
                criterion=criterion,
                global_model=self.server.global_model,
            )
            updates.append(update)
            logger.debug(
                f"Client {client_id}: loss={update.metrics.get('train_loss', 0):.4f}"
            )

        # Aggregate
        round_result = self.server.aggregate(updates)

        logger.info(
            f"Round {round_result.round_num}: "
            f"loss={round_result.aggregated_metrics.get('train_loss', 0):.4f}"
        )

        return round_result

    def train(
        self,
        num_rounds: Optional[int] = None,
        criterion: Optional[nn.Module] = None,
    ) -> List[FederatedRound]:
        """Run full federated training."""
        num_rounds = num_rounds or self.config.num_rounds
        results = []

        for round_num in range(num_rounds):
            round_result = self.train_round(criterion)
            results.append(round_result)

            if (round_num + 1) % 10 == 0:
                logger.info(f"Completed round {round_num + 1}/{num_rounds}")

        return results

    def get_global_model(self) -> nn.Module:
        """Get the trained global model."""
        return self.server.global_model


def create_federated_dataloaders(
    dataset: torch.utils.data.Dataset,
    num_clients: int,
    batch_size: int = 32,
    iid: bool = True,
    alpha: float = 0.5,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Split dataset for federated learning.

    Args:
        dataset: Full dataset
        num_clients: Number of clients
        batch_size: Batch size per client
        iid: Whether to use IID split
        alpha: Dirichlet concentration for non-IID

    Returns:
        Dict mapping client IDs to DataLoaders
    """
    num_samples = len(dataset)

    if iid:
        # IID: random equal split
        indices = np.random.permutation(num_samples)
        split_size = num_samples // num_clients
        client_indices = [
            indices[i * split_size:(i + 1) * split_size]
            for i in range(num_clients)
        ]
    else:
        # Non-IID: Dirichlet distribution
        # Assuming labels are accessible
        labels = np.array([dataset[i][1] for i in range(num_samples)])

        if len(labels.shape) > 1:
            labels = labels[:, 0]  # Use first task

        num_classes = len(np.unique(labels[~np.isnan(labels)]))
        label_indices = [
            np.where(labels == c)[0] for c in range(num_classes)
        ]

        client_indices = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            class_indices = label_indices[c]
            np.random.shuffle(class_indices)

            # Dirichlet distribution for this class
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (proportions * len(class_indices)).astype(int)

            # Assign to clients
            start = 0
            for i, prop in enumerate(proportions):
                end = min(start + prop, len(class_indices))
                client_indices[i].extend(class_indices[start:end])
                start = end

    # Create DataLoaders
    dataloaders = {}
    for i, indices in enumerate(client_indices):
        client_id = f"client_{i}"
        subset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
        )
        dataloaders[client_id] = loader

    return dataloaders
