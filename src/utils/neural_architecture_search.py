"""Neural Architecture Search utilities for molecular models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import random
import copy
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchSpace:
    """Define the neural architecture search space."""

    # Layer options
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    num_layers: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])
    activation_fns: List[str] = field(default_factory=lambda: ["relu", "gelu", "swish", "tanh"])
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4])

    # GNN-specific options
    gnn_types: List[str] = field(default_factory=lambda: ["gcn", "gat", "gin", "sage"])
    num_heads: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    pooling_types: List[str] = field(default_factory=lambda: ["mean", "max", "sum", "attention"])

    # Normalization options
    norm_types: List[str] = field(default_factory=lambda: ["batch", "layer", "none"])

    # Skip connection options
    skip_connections: List[bool] = field(default_factory=lambda: [True, False])


@dataclass
class Architecture:
    """Representation of a neural architecture."""

    hidden_dims: List[int]
    activation: str
    dropout: float
    gnn_type: str = "gcn"
    num_heads: int = 4
    pooling: str = "mean"
    norm_type: str = "batch"
    skip_connections: bool = True

    def to_dict(self) -> dict:
        return {
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
            "dropout": self.dropout,
            "gnn_type": self.gnn_type,
            "num_heads": self.num_heads,
            "pooling": self.pooling,
            "norm_type": self.norm_type,
            "skip_connections": self.skip_connections,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Architecture":
        return cls(**d)


@dataclass
class SearchResult:
    """Results from architecture search."""

    best_architecture: Architecture
    best_score: float
    all_architectures: List[Architecture]
    all_scores: List[float]
    search_history: List[dict] = field(default_factory=list)


class ArchitectureGenerator:
    """
    Generate architectures from search space.

    Args:
        search_space: The search space to sample from
        seed: Random seed
    """

    def __init__(self, search_space: SearchSpace, seed: int = 42):
        self.search_space = search_space
        self.rng = random.Random(seed)

    def random_architecture(self) -> Architecture:
        """Generate a random architecture."""
        num_layers = self.rng.choice(self.search_space.num_layers)
        hidden_dims = [
            self.rng.choice(self.search_space.hidden_dims)
            for _ in range(num_layers)
        ]

        return Architecture(
            hidden_dims=hidden_dims,
            activation=self.rng.choice(self.search_space.activation_fns),
            dropout=self.rng.choice(self.search_space.dropout_rates),
            gnn_type=self.rng.choice(self.search_space.gnn_types),
            num_heads=self.rng.choice(self.search_space.num_heads),
            pooling=self.rng.choice(self.search_space.pooling_types),
            norm_type=self.rng.choice(self.search_space.norm_types),
            skip_connections=self.rng.choice(self.search_space.skip_connections),
        )

    def mutate_architecture(
        self,
        arch: Architecture,
        mutation_rate: float = 0.3,
    ) -> Architecture:
        """Mutate an architecture."""
        new_arch = copy.deepcopy(arch)

        if self.rng.random() < mutation_rate:
            # Mutate hidden dimensions
            idx = self.rng.randint(0, len(new_arch.hidden_dims) - 1)
            new_arch.hidden_dims[idx] = self.rng.choice(self.search_space.hidden_dims)

        if self.rng.random() < mutation_rate:
            new_arch.activation = self.rng.choice(self.search_space.activation_fns)

        if self.rng.random() < mutation_rate:
            new_arch.dropout = self.rng.choice(self.search_space.dropout_rates)

        if self.rng.random() < mutation_rate:
            new_arch.gnn_type = self.rng.choice(self.search_space.gnn_types)

        if self.rng.random() < mutation_rate:
            new_arch.num_heads = self.rng.choice(self.search_space.num_heads)

        if self.rng.random() < mutation_rate:
            new_arch.pooling = self.rng.choice(self.search_space.pooling_types)

        if self.rng.random() < mutation_rate:
            new_arch.norm_type = self.rng.choice(self.search_space.norm_types)

        if self.rng.random() < mutation_rate:
            new_arch.skip_connections = self.rng.choice(
                self.search_space.skip_connections
            )

        return new_arch

    def crossover(
        self,
        arch1: Architecture,
        arch2: Architecture,
    ) -> Architecture:
        """Crossover two architectures."""
        # Take random attributes from each parent
        hidden_dims = arch1.hidden_dims if self.rng.random() > 0.5 else arch2.hidden_dims

        return Architecture(
            hidden_dims=hidden_dims,
            activation=arch1.activation if self.rng.random() > 0.5 else arch2.activation,
            dropout=arch1.dropout if self.rng.random() > 0.5 else arch2.dropout,
            gnn_type=arch1.gnn_type if self.rng.random() > 0.5 else arch2.gnn_type,
            num_heads=arch1.num_heads if self.rng.random() > 0.5 else arch2.num_heads,
            pooling=arch1.pooling if self.rng.random() > 0.5 else arch2.pooling,
            norm_type=arch1.norm_type if self.rng.random() > 0.5 else arch2.norm_type,
            skip_connections=(
                arch1.skip_connections if self.rng.random() > 0.5
                else arch2.skip_connections
            ),
        )


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
    }
    return activations.get(name, nn.ReLU())


class SearchableMLPBlock(nn.Module):
    """MLP block with configurable architecture."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "relu",
        dropout: float = 0.1,
        norm_type: str = "batch",
        skip_connection: bool = True,
    ):
        super().__init__()

        self.skip_connection = skip_connection and (in_dim == out_dim)

        self.linear = nn.Linear(in_dim, out_dim)

        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = nn.Identity()

        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        if self.skip_connection:
            out = out + x

        return out


class SearchableMLP(nn.Module):
    """
    MLP with searchable architecture.

    Args:
        in_channels: Input dimension
        out_channels: Output dimension
        architecture: Architecture specification
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        architecture: Architecture,
    ):
        super().__init__()

        self.architecture = architecture

        layers = []
        current_dim = in_channels

        for hidden_dim in architecture.hidden_dims:
            layers.append(
                SearchableMLPBlock(
                    current_dim,
                    hidden_dim,
                    activation=architecture.activation,
                    dropout=architecture.dropout,
                    norm_type=architecture.norm_type,
                    skip_connection=architecture.skip_connections,
                )
            )
            current_dim = hidden_dim

        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(current_dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return self.output(x)


class RandomSearch:
    """
    Random search for neural architecture.

    Args:
        search_space: Architecture search space
        model_builder: Function to build model from architecture
        evaluator: Function to evaluate model
        num_samples: Number of architectures to sample
    """

    def __init__(
        self,
        search_space: SearchSpace,
        model_builder: Callable[[Architecture], nn.Module],
        evaluator: Callable[[nn.Module], float],
        num_samples: int = 100,
    ):
        self.search_space = search_space
        self.model_builder = model_builder
        self.evaluator = evaluator
        self.num_samples = num_samples
        self.generator = ArchitectureGenerator(search_space)

    def search(self) -> SearchResult:
        """Run random search."""
        all_architectures = []
        all_scores = []
        history = []

        best_arch = None
        best_score = float("-inf")

        for i in range(self.num_samples):
            arch = self.generator.random_architecture()

            try:
                model = self.model_builder(arch)
                score = self.evaluator(model)
            except Exception as e:
                logger.warning(f"Failed to evaluate architecture: {e}")
                score = float("-inf")

            all_architectures.append(arch)
            all_scores.append(score)
            history.append({
                "iteration": i,
                "architecture": arch.to_dict(),
                "score": score,
            })

            if score > best_score:
                best_score = score
                best_arch = arch
                logger.info(f"New best: {score:.4f}")

            if (i + 1) % 10 == 0:
                logger.info(f"Searched {i + 1}/{self.num_samples} architectures")

        return SearchResult(
            best_architecture=best_arch,
            best_score=best_score,
            all_architectures=all_architectures,
            all_scores=all_scores,
            search_history=history,
        )


class EvolutionarySearch:
    """
    Evolutionary search for neural architecture.

    Uses genetic algorithms to evolve architectures.

    Args:
        search_space: Architecture search space
        model_builder: Function to build model from architecture
        evaluator: Function to evaluate model
        population_size: Size of population
        num_generations: Number of generations
        mutation_rate: Probability of mutation
        elite_ratio: Fraction of population to keep as elite
    """

    def __init__(
        self,
        search_space: SearchSpace,
        model_builder: Callable[[Architecture], nn.Module],
        evaluator: Callable[[nn.Module], float],
        population_size: int = 20,
        num_generations: int = 50,
        mutation_rate: float = 0.3,
        elite_ratio: float = 0.2,
    ):
        self.search_space = search_space
        self.model_builder = model_builder
        self.evaluator = evaluator
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.generator = ArchitectureGenerator(search_space)

    def search(self) -> SearchResult:
        """Run evolutionary search."""
        # Initialize population
        population = [
            self.generator.random_architecture()
            for _ in range(self.population_size)
        ]

        all_architectures = []
        all_scores = []
        history = []

        best_arch = None
        best_score = float("-inf")

        for gen in range(self.num_generations):
            # Evaluate population
            scores = []
            for arch in population:
                try:
                    model = self.model_builder(arch)
                    score = self.evaluator(model)
                except Exception as e:
                    logger.warning(f"Evaluation failed: {e}")
                    score = float("-inf")

                scores.append(score)
                all_architectures.append(arch)
                all_scores.append(score)

                if score > best_score:
                    best_score = score
                    best_arch = arch

            history.append({
                "generation": gen,
                "best_score": max(scores),
                "avg_score": np.mean([s for s in scores if s > float("-inf")]),
            })

            logger.info(
                f"Generation {gen + 1}: Best={max(scores):.4f}, "
                f"Avg={np.mean([s for s in scores if s > float('-inf')]):.4f}"
            )

            # Selection
            sorted_indices = np.argsort(scores)[::-1]
            elite_size = int(self.population_size * self.elite_ratio)
            elite = [population[i] for i in sorted_indices[:elite_size]]

            # Create new population
            new_population = elite.copy()

            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_select(population, scores)
                parent2 = self._tournament_select(population, scores)

                # Crossover and mutation
                child = self.generator.crossover(parent1, parent2)
                child = self.generator.mutate_architecture(child, self.mutation_rate)

                new_population.append(child)

            population = new_population

        return SearchResult(
            best_architecture=best_arch,
            best_score=best_score,
            all_architectures=all_architectures,
            all_scores=all_scores,
            search_history=history,
        )

    def _tournament_select(
        self,
        population: List[Architecture],
        scores: List[float],
        tournament_size: int = 3,
    ) -> Architecture:
        """Select architecture via tournament."""
        indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(indices, key=lambda i: scores[i])
        return population[best_idx]


class BayesianOptimizationSearch:
    """
    Bayesian optimization for neural architecture search.

    Uses Gaussian Process surrogate to guide search.

    Args:
        search_space: Architecture search space
        model_builder: Function to build model
        evaluator: Function to evaluate model
        num_iterations: Number of BO iterations
        initial_samples: Number of random initial samples
    """

    def __init__(
        self,
        search_space: SearchSpace,
        model_builder: Callable[[Architecture], nn.Module],
        evaluator: Callable[[nn.Module], float],
        num_iterations: int = 50,
        initial_samples: int = 10,
    ):
        self.search_space = search_space
        self.model_builder = model_builder
        self.evaluator = evaluator
        self.num_iterations = num_iterations
        self.initial_samples = initial_samples
        self.generator = ArchitectureGenerator(search_space)

    def _encode_architecture(self, arch: Architecture) -> np.ndarray:
        """Encode architecture as feature vector."""
        features = []

        # Encode hidden dims (mean, std, num layers)
        features.append(np.mean(arch.hidden_dims))
        features.append(np.std(arch.hidden_dims) if len(arch.hidden_dims) > 1 else 0)
        features.append(len(arch.hidden_dims))

        # Encode categorical as one-hot or ordinal
        activation_map = {"relu": 0, "gelu": 1, "swish": 2, "tanh": 3}
        features.append(activation_map.get(arch.activation, 0))

        features.append(arch.dropout)

        gnn_map = {"gcn": 0, "gat": 1, "gin": 2, "sage": 3}
        features.append(gnn_map.get(arch.gnn_type, 0))

        features.append(arch.num_heads)

        pooling_map = {"mean": 0, "max": 1, "sum": 2, "attention": 3}
        features.append(pooling_map.get(arch.pooling, 0))

        norm_map = {"batch": 0, "layer": 1, "none": 2}
        features.append(norm_map.get(arch.norm_type, 0))

        features.append(float(arch.skip_connections))

        return np.array(features)

    def search(self) -> SearchResult:
        """Run Bayesian optimization search."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel

        all_architectures = []
        all_scores = []
        X_observed = []
        y_observed = []

        best_arch = None
        best_score = float("-inf")

        # Initial random sampling
        for i in range(self.initial_samples):
            arch = self.generator.random_architecture()
            try:
                model = self.model_builder(arch)
                score = self.evaluator(model)
            except Exception:
                score = float("-inf")

            all_architectures.append(arch)
            all_scores.append(score)

            if score > float("-inf"):
                X_observed.append(self._encode_architecture(arch))
                y_observed.append(score)

            if score > best_score:
                best_score = score
                best_arch = arch

        # Bayesian optimization iterations
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

        for i in range(self.num_iterations - self.initial_samples):
            if len(X_observed) < 2:
                # Not enough data, sample randomly
                arch = self.generator.random_architecture()
            else:
                # Fit GP
                X = np.array(X_observed)
                y = np.array(y_observed)
                gp.fit(X, y)

                # Sample candidates and select best acquisition
                candidates = [
                    self.generator.random_architecture()
                    for _ in range(100)
                ]
                candidate_features = np.array([
                    self._encode_architecture(c) for c in candidates
                ])

                # Expected Improvement acquisition
                mu, sigma = gp.predict(candidate_features, return_std=True)
                best_so_far = np.max(y_observed)

                # EI calculation
                with np.errstate(divide='ignore', invalid='ignore'):
                    Z = (mu - best_so_far) / (sigma + 1e-8)
                    from scipy.stats import norm
                    ei = (mu - best_so_far) * norm.cdf(Z) + sigma * norm.pdf(Z)
                    ei[sigma < 1e-8] = 0

                best_candidate_idx = np.argmax(ei)
                arch = candidates[best_candidate_idx]

            # Evaluate
            try:
                model = self.model_builder(arch)
                score = self.evaluator(model)
            except Exception:
                score = float("-inf")

            all_architectures.append(arch)
            all_scores.append(score)

            if score > float("-inf"):
                X_observed.append(self._encode_architecture(arch))
                y_observed.append(score)

            if score > best_score:
                best_score = score
                best_arch = arch
                logger.info(f"New best: {score:.4f}")

            if (i + 1) % 10 == 0:
                logger.info(f"BO iteration {i + 1}")

        return SearchResult(
            best_architecture=best_arch,
            best_score=best_score,
            all_architectures=all_architectures,
            all_scores=all_scores,
        )


class EfficientNAS:
    """
    Efficient Neural Architecture Search using weight sharing.

    Args:
        search_space: Architecture search space
        supernet: Pre-built supernet with all possible paths
        evaluator: Function to evaluate subnet
        num_samples: Number of architectures to sample
    """

    def __init__(
        self,
        search_space: SearchSpace,
        supernet: nn.Module,
        evaluator: Callable[[nn.Module], float],
        num_samples: int = 100,
    ):
        self.search_space = search_space
        self.supernet = supernet
        self.evaluator = evaluator
        self.num_samples = num_samples
        self.generator = ArchitectureGenerator(search_space)

    def search(self) -> SearchResult:
        """Run efficient NAS with weight sharing."""
        all_architectures = []
        all_scores = []

        best_arch = None
        best_score = float("-inf")

        for i in range(self.num_samples):
            arch = self.generator.random_architecture()

            # Extract subnet from supernet
            subnet = self._extract_subnet(arch)

            # Evaluate subnet
            try:
                score = self.evaluator(subnet)
            except Exception:
                score = float("-inf")

            all_architectures.append(arch)
            all_scores.append(score)

            if score > best_score:
                best_score = score
                best_arch = arch

        return SearchResult(
            best_architecture=best_arch,
            best_score=best_score,
            all_architectures=all_architectures,
            all_scores=all_scores,
        )

    def _extract_subnet(self, arch: Architecture) -> nn.Module:
        """Extract subnet from supernet based on architecture."""
        # This is a simplified version - real implementation would
        # properly share weights from supernet
        return self.supernet


def run_nas(
    search_space: SearchSpace,
    model_builder: Callable[[Architecture], nn.Module],
    train_fn: Callable[[nn.Module], float],
    method: str = "random",
    **kwargs,
) -> SearchResult:
    """
    Run neural architecture search.

    Args:
        search_space: The search space
        model_builder: Function to build model from architecture
        train_fn: Function to train and evaluate model (returns score)
        method: Search method ("random", "evolutionary", "bayesian")
        **kwargs: Additional arguments for search method

    Returns:
        SearchResult with best architecture
    """
    if method == "random":
        searcher = RandomSearch(
            search_space, model_builder, train_fn,
            num_samples=kwargs.get("num_samples", 100),
        )
    elif method == "evolutionary":
        searcher = EvolutionarySearch(
            search_space, model_builder, train_fn,
            population_size=kwargs.get("population_size", 20),
            num_generations=kwargs.get("num_generations", 50),
        )
    elif method == "bayesian":
        searcher = BayesianOptimizationSearch(
            search_space, model_builder, train_fn,
            num_iterations=kwargs.get("num_iterations", 50),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return searcher.search()
