"""Hyperparameter optimization using Optuna for molecular property prediction models."""

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Define the hyperparameter search space."""

    # Learning rate
    lr_min: float = 1e-5
    lr_max: float = 1e-2

    # Batch size options
    batch_sizes: list[int] = field(default_factory=lambda: [32, 64, 128, 256])

    # Dropout
    dropout_min: float = 0.1
    dropout_max: float = 0.5

    # Hidden dimensions
    hidden_dim_options: list[int] = field(default_factory=lambda: [128, 256, 512, 1024])

    # Number of layers
    num_layers_min: int = 2
    num_layers_max: int = 6

    # Weight decay
    weight_decay_min: float = 1e-6
    weight_decay_max: float = 1e-2

    # Optimizer choices
    optimizers: list[str] = field(default_factory=lambda: ["adam", "adamw", "sgd"])


class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimizer for molecular property prediction.

    Args:
        model_class: Model class to instantiate
        train_fn: Training function that takes model and hyperparams, returns metric
        hyperparameter_space: Search space configuration
        direction: Optimization direction ('maximize' or 'minimize')
        study_name: Name for the Optuna study
        storage: Optional database URL for persistent storage
        n_trials: Number of optimization trials
        timeout: Optional timeout in seconds
        n_jobs: Number of parallel jobs (-1 for all CPUs)
    """

    def __init__(
        self,
        model_class: type,
        train_fn: Callable,
        hyperparameter_space: Optional[HyperparameterSpace] = None,
        direction: str = "maximize",
        study_name: str = "mol_property_optimization",
        storage: Optional[str] = None,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        pruner_type: str = "median",
    ):
        self.model_class = model_class
        self.train_fn = train_fn
        self.space = hyperparameter_space or HyperparameterSpace()
        self.direction = direction
        self.study_name = study_name
        self.storage = storage
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs

        # Select pruner
        if pruner_type == "median":
            self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        else:
            self.pruner = HyperbandPruner(min_resource=1, max_resource=100)

        self.study = None
        self.best_params = None
        self.best_value = None

    def suggest_hyperparameters(self, trial: Trial) -> dict:
        """
        Suggest hyperparameters for a trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                self.space.lr_min,
                self.space.lr_max,
                log=True
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size",
                self.space.batch_sizes
            ),
            "dropout": trial.suggest_float(
                "dropout",
                self.space.dropout_min,
                self.space.dropout_max
            ),
            "hidden_dim": trial.suggest_categorical(
                "hidden_dim",
                self.space.hidden_dim_options
            ),
            "num_layers": trial.suggest_int(
                "num_layers",
                self.space.num_layers_min,
                self.space.num_layers_max
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay",
                self.space.weight_decay_min,
                self.space.weight_decay_max,
                log=True
            ),
            "optimizer": trial.suggest_categorical(
                "optimizer",
                self.space.optimizers
            ),
        }
        return params

    def objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Metric value to optimize
        """
        # Suggest hyperparameters
        params = self.suggest_hyperparameters(trial)

        # Train model and get metric
        try:
            metric = self.train_fn(trial, params, self.model_class)
            return metric
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()

    def optimize(self) -> dict:
        """
        Run hyperparameter optimization.

        Returns:
            Dictionary with best parameters and value
        """
        sampler = TPESampler(seed=42)

        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=sampler,
            pruner=self.pruner,
            storage=self.storage,
            load_if_exists=True,
        )

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )

        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        logger.info(f"Best trial value: {self.best_value}")
        logger.info(f"Best parameters: {self.best_params}")

        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": len(self.study.trials),
        }

    def get_optimization_history(self) -> list[dict]:
        """Get the optimization history."""
        if self.study is None:
            return []

        history = []
        for trial in self.study.trials:
            history.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
            })
        return history

    def save_results(self, path: str) -> None:
        """Save optimization results to file."""
        results = {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "history": self.get_optimization_history(),
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {path}")

    def get_param_importances(self) -> dict[str, float]:
        """
        Calculate parameter importances using fANOVA.

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if self.study is None:
            return {}

        try:
            importances = optuna.importance.get_param_importances(self.study)
            return dict(importances)
        except Exception as e:
            logger.warning(f"Could not compute parameter importances: {e}")
            return {}


class MLPHyperparameterSpace(HyperparameterSpace):
    """Hyperparameter space specifically for MLP models."""

    def __init__(self):
        super().__init__()
        self.hidden_dim_options = [256, 512, 1024, 2048]
        self.num_layers_min = 2
        self.num_layers_max = 5
        self.fingerprint_sizes = [1024, 2048, 4096]
        self.fingerprint_radii = [2, 3, 4]


class GNNHyperparameterSpace(HyperparameterSpace):
    """Hyperparameter space specifically for GNN models."""

    def __init__(self):
        super().__init__()
        self.hidden_dim_options = [128, 256, 512]
        self.num_layers_min = 2
        self.num_layers_max = 6
        self.conv_types = ["gcn", "gat"]
        self.num_heads_options = [2, 4, 8]
        self.pooling_types = ["mean", "max", "sum", "attention"]


class AttentiveFPHyperparameterSpace(HyperparameterSpace):
    """Hyperparameter space specifically for AttentiveFP models."""

    def __init__(self):
        super().__init__()
        self.hidden_dim_options = [128, 256, 512]
        self.num_layers_min = 2
        self.num_layers_max = 4
        self.num_timesteps_options = [2, 3, 4, 5]


def suggest_mlp_params(trial: Trial, space: MLPHyperparameterSpace) -> dict:
    """Suggest hyperparameters for MLP model."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", space.lr_min, space.lr_max, log=True),
        "batch_size": trial.suggest_categorical("batch_size", space.batch_sizes),
        "dropout": trial.suggest_float("dropout", space.dropout_min, space.dropout_max),
        "hidden_dim": trial.suggest_categorical("hidden_dim", space.hidden_dim_options),
        "num_layers": trial.suggest_int("num_layers", space.num_layers_min, space.num_layers_max),
        "weight_decay": trial.suggest_float("weight_decay", space.weight_decay_min, space.weight_decay_max, log=True),
        "fingerprint_size": trial.suggest_categorical("fingerprint_size", space.fingerprint_sizes),
        "fingerprint_radius": trial.suggest_categorical("fingerprint_radius", space.fingerprint_radii),
    }


def suggest_gnn_params(trial: Trial, space: GNNHyperparameterSpace) -> dict:
    """Suggest hyperparameters for GNN model."""
    conv_type = trial.suggest_categorical("conv_type", space.conv_types)

    params = {
        "learning_rate": trial.suggest_float("learning_rate", space.lr_min, space.lr_max, log=True),
        "batch_size": trial.suggest_categorical("batch_size", space.batch_sizes),
        "dropout": trial.suggest_float("dropout", space.dropout_min, space.dropout_max),
        "hidden_channels": trial.suggest_categorical("hidden_channels", space.hidden_dim_options),
        "num_layers": trial.suggest_int("num_layers", space.num_layers_min, space.num_layers_max),
        "weight_decay": trial.suggest_float("weight_decay", space.weight_decay_min, space.weight_decay_max, log=True),
        "conv_type": conv_type,
        "pooling": trial.suggest_categorical("pooling", space.pooling_types),
    }

    if conv_type == "gat":
        params["num_heads"] = trial.suggest_categorical("num_heads", space.num_heads_options)

    return params


def suggest_attentivefp_params(trial: Trial, space: AttentiveFPHyperparameterSpace) -> dict:
    """Suggest hyperparameters for AttentiveFP model."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", space.lr_min, space.lr_max, log=True),
        "batch_size": trial.suggest_categorical("batch_size", space.batch_sizes),
        "dropout": trial.suggest_float("dropout", space.dropout_min, space.dropout_max),
        "hidden_channels": trial.suggest_categorical("hidden_channels", space.hidden_dim_options),
        "num_layers": trial.suggest_int("num_layers", space.num_layers_min, space.num_layers_max),
        "weight_decay": trial.suggest_float("weight_decay", space.weight_decay_min, space.weight_decay_max, log=True),
        "num_timesteps": trial.suggest_categorical("num_timesteps", space.num_timesteps_options),
    }


class EarlyStooppingCallback:
    """Optuna callback for early stopping based on no improvement."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = None
        self.no_improvement_count = 0

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        if trial.value is None:
            return

        if self.best_value is None:
            self.best_value = trial.value
        elif study.direction == optuna.study.StudyDirection.MAXIMIZE:
            if trial.value > self.best_value + self.min_delta:
                self.best_value = trial.value
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
        else:  # MINIMIZE
            if trial.value < self.best_value - self.min_delta:
                self.best_value = trial.value
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            study.stop()
