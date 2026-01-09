"""Configuration validation utilities."""

from typing import Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Container for a validation error."""

    path: str  # Config path (e.g., "training.batch_size")
    message: str
    severity: str = "error"  # "error", "warning"


@dataclass
class ValidationResult:
    """Container for validation results."""

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    def add_error(self, path: str, message: str) -> None:
        """Add an error."""
        self.errors.append(ValidationError(path, message, "error"))
        self.is_valid = False

    def add_warning(self, path: str, message: str) -> None:
        """Add a warning."""
        self.warnings.append(ValidationError(path, message, "warning"))

    def __str__(self) -> str:
        """String representation."""
        lines = [f"Valid: {self.is_valid}"]
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  - [{e.path}] {e.message}")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  - [{w.path}] {w.message}")
        return "\n".join(lines)


class ConfigSchema:
    """
    Schema definition for configuration validation.

    Defines expected types, ranges, and requirements for config values.
    """

    # Expected schema for molecular property prediction config
    SCHEMA = {
        "data": {
            "type": "dict",
            "required": True,
            "children": {
                "dataset": {"type": "str", "required": True},
                "test_size": {"type": "float", "min": 0.0, "max": 1.0},
                "val_size": {"type": "float", "min": 0.0, "max": 1.0},
                "random_state": {"type": "int", "min": 0},
                "num_tasks": {"type": "int", "min": 1, "max": 100},
            },
        },
        "training": {
            "type": "dict",
            "required": True,
            "children": {
                "batch_size": {"type": "int", "min": 1, "max": 4096},
                "learning_rate": {"type": "float", "min": 1e-8, "max": 1.0},
                "weight_decay": {"type": "float", "min": 0.0, "max": 1.0},
                "epochs": {"type": "int", "min": 1, "max": 10000},
                "early_stopping_patience": {"type": "int", "min": 1, "max": 1000},
                "gradient_clip_norm": {"type": "float", "min": 0.0},
                "scheduler": {"type": "str", "choices": [
                    "cosine_annealing", "step", "exponential", "reduce_on_plateau",
                    "one_cycle", "cosine_warmup", "constant"
                ]},
                "optimizer": {"type": "str", "choices": ["adam", "adamw", "sgd", "rmsprop"]},
            },
        },
        "mlp": {
            "type": "dict",
            "children": {
                "fingerprint_size": {"type": "int", "choices": [512, 1024, 2048, 4096]},
                "fingerprint_radius": {"type": "int", "min": 1, "max": 6},
                "hidden_sizes": {"type": "list", "item_type": "int", "min_length": 1},
                "dropout": {"type": "float", "min": 0.0, "max": 0.9},
                "epochs": {"type": "int", "min": 1},
            },
        },
        "gnn": {
            "type": "dict",
            "children": {
                "hidden_channels": {"type": "int", "min": 32, "max": 1024},
                "num_layers": {"type": "int", "min": 1, "max": 20},
                "dropout": {"type": "float", "min": 0.0, "max": 0.9},
                "conv_type": {"type": "str", "choices": ["gcn", "gat", "gin", "sage"]},
                "epochs": {"type": "int", "min": 1},
                "pretrained": {"type": "bool"},
                "freeze_encoder": {"type": "bool"},
            },
        },
        "attentivefp": {
            "type": "dict",
            "children": {
                "hidden_channels": {"type": "int", "min": 32, "max": 1024},
                "num_layers": {"type": "int", "min": 1, "max": 10},
                "num_timesteps": {"type": "int", "min": 1, "max": 10},
                "dropout": {"type": "float", "min": 0.0, "max": 0.9},
                "epochs": {"type": "int", "min": 1},
            },
        },
        "ensemble": {
            "type": "dict",
            "children": {
                "weights": {
                    "type": "dict",
                    "children": {
                        "mlp": {"type": "float", "min": 0.0, "max": 1.0},
                        "gcn": {"type": "float", "min": 0.0, "max": 1.0},
                        "attentivefp": {"type": "float", "min": 0.0, "max": 1.0},
                    },
                },
            },
        },
        "features": {
            "type": "dict",
            "children": {
                "morgan": {"type": "dict"},
                "gcn_node": {"type": "dict"},
                "attentivefp_node": {"type": "dict"},
                "attentivefp_edge": {"type": "dict"},
            },
        },
        "paths": {
            "type": "dict",
            "children": {
                "data_dir": {"type": "str"},
                "models_dir": {"type": "str"},
                "results_dir": {"type": "str"},
            },
        },
        "device": {
            "type": "dict",
            "children": {
                "prefer_gpu": {"type": "bool"},
                "gpu_id": {"type": "int", "min": 0},
            },
        },
    }


class ConfigValidator:
    """
    Validate configuration files against a schema.

    Args:
        schema: Schema definition (defaults to ConfigSchema.SCHEMA)
    """

    def __init__(self, schema: Optional[dict] = None):
        self.schema = schema or ConfigSchema.SCHEMA

    def validate(self, config: dict) -> ValidationResult:
        """
        Validate a configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        self._validate_dict(config, self.schema, "", result)
        return result

    def validate_file(self, config_path: str) -> ValidationResult:
        """
        Validate a configuration file.

        Args:
            config_path: Path to YAML config file

        Returns:
            ValidationResult
        """
        path = Path(config_path)

        if not path.exists():
            result = ValidationResult(is_valid=False)
            result.add_error("", f"Config file not found: {config_path}")
            return result

        try:
            with open(path) as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            result = ValidationResult(is_valid=False)
            result.add_error("", f"YAML parsing error: {e}")
            return result

        return self.validate(config)

    def _validate_dict(
        self,
        config: dict,
        schema: dict,
        path_prefix: str,
        result: ValidationResult,
    ) -> None:
        """Recursively validate a dictionary."""
        for key, spec in schema.items():
            full_path = f"{path_prefix}.{key}" if path_prefix else key

            if key not in config:
                if spec.get("required", False):
                    result.add_error(full_path, "Required field missing")
                continue

            value = config[key]
            self._validate_value(value, spec, full_path, result)

        # Check for unknown keys
        for key in config:
            if key not in schema:
                full_path = f"{path_prefix}.{key}" if path_prefix else key
                result.add_warning(full_path, "Unknown configuration key")

    def _validate_value(
        self,
        value: Any,
        spec: dict,
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate a single value."""
        expected_type = spec.get("type")

        # Type check
        if expected_type == "dict":
            if not isinstance(value, dict):
                result.add_error(path, f"Expected dict, got {type(value).__name__}")
                return
            if "children" in spec:
                self._validate_dict(value, spec["children"], path, result)

        elif expected_type == "list":
            if not isinstance(value, list):
                result.add_error(path, f"Expected list, got {type(value).__name__}")
                return
            if "min_length" in spec and len(value) < spec["min_length"]:
                result.add_error(path, f"List too short (min: {spec['min_length']})")
            if "item_type" in spec:
                for i, item in enumerate(value):
                    item_path = f"{path}[{i}]"
                    if not self._check_type(item, spec["item_type"]):
                        result.add_error(item_path, f"Expected {spec['item_type']}")

        elif expected_type == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                result.add_error(path, f"Expected int, got {type(value).__name__}")
                return
            self._validate_numeric(value, spec, path, result)

        elif expected_type == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                result.add_error(path, f"Expected float, got {type(value).__name__}")
                return
            self._validate_numeric(value, spec, path, result)

        elif expected_type == "str":
            if not isinstance(value, str):
                result.add_error(path, f"Expected str, got {type(value).__name__}")
                return
            if "choices" in spec and value not in spec["choices"]:
                result.add_error(path, f"Invalid value. Choices: {spec['choices']}")

        elif expected_type == "bool":
            if not isinstance(value, bool):
                result.add_error(path, f"Expected bool, got {type(value).__name__}")

    def _validate_numeric(
        self,
        value: Union[int, float],
        spec: dict,
        path: str,
        result: ValidationResult,
    ) -> None:
        """Validate numeric constraints."""
        if "min" in spec and value < spec["min"]:
            result.add_error(path, f"Value {value} below minimum {spec['min']}")
        if "max" in spec and value > spec["max"]:
            result.add_error(path, f"Value {value} above maximum {spec['max']}")
        if "choices" in spec and value not in spec["choices"]:
            result.add_error(path, f"Invalid value. Choices: {spec['choices']}")

    def _check_type(self, value: Any, type_name: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            "int": (int,),
            "float": (int, float),
            "str": (str,),
            "bool": (bool,),
            "dict": (dict,),
            "list": (list,),
        }
        return isinstance(value, type_map.get(type_name, (object,)))


def validate_config(config_path: str = "config.yaml") -> ValidationResult:
    """
    Convenience function to validate a config file.

    Args:
        config_path: Path to config file

    Returns:
        ValidationResult
    """
    validator = ConfigValidator()
    return validator.validate_file(config_path)


def check_config_consistency(config: dict) -> list[str]:
    """
    Check for logical consistency issues in config.

    Args:
        config: Configuration dictionary

    Returns:
        List of warning messages
    """
    warnings = []

    # Check ensemble weights sum to ~1
    if "ensemble" in config and "weights" in config["ensemble"]:
        weights = config["ensemble"]["weights"]
        total = sum(weights.values())
        if not (0.99 <= total <= 1.01):
            warnings.append(f"Ensemble weights sum to {total}, expected ~1.0")

    # Check validation size is smaller than test size
    if "data" in config:
        test_size = config["data"].get("test_size", 0.2)
        val_size = config["data"].get("val_size", 0.125)
        if val_size >= test_size:
            warnings.append("Validation size should typically be smaller than test size")

    # Check learning rate is reasonable for chosen optimizer
    if "training" in config:
        lr = config["training"].get("learning_rate", 0.001)
        optimizer = config["training"].get("optimizer", "adam")
        if optimizer == "sgd" and lr > 0.1:
            warnings.append("Learning rate may be too high for SGD optimizer")

    return warnings
