"""Configuration loader utility for centralized hyperparameter management."""

from pathlib import Path
from typing import Any, Optional
import yaml


class Config:
    """
    Configuration manager for molecular property prediction.

    Loads settings from config.yaml and provides easy access to hyperparameters.

    Usage:
        config = Config()
        batch_size = config.training.batch_size
        mlp_dropout = config.mlp.dropout

        # Or use get method with defaults
        lr = config.get('training.learning_rate', 0.001)
    """

    _instance: Optional['Config'] = None
    _config: dict = {}

    def __new__(cls, config_path: Optional[Path] = None):
        """Singleton pattern - only load config once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: Optional[Path] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            # Default to project root config.yaml
            config_path = Path(__file__).parent.parent.parent / 'config.yaml'

        if config_path.exists():
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            print(f"Warning: Config file not found at {config_path}")
            self._config = {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Dot-separated key path (e.g., 'training.batch_size')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to top-level config sections."""
        if name.startswith('_'):
            return super().__getattribute__(name)

        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value

        raise AttributeError(f"Config has no section '{name}'")

    @property
    def raw(self) -> dict:
        """Get the raw configuration dictionary."""
        return self._config.copy()

    @classmethod
    def reload(cls, config_path: Optional[Path] = None) -> 'Config':
        """Force reload the configuration."""
        cls._instance = None
        return cls(config_path)


class ConfigSection:
    """
    Wrapper for config sections to allow attribute-style access.

    Usage:
        config = Config()
        print(config.training.batch_size)
        print(config.mlp.hidden_sizes)
    """

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return super().__getattribute__(name)

        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value

        raise AttributeError(f"Config section has no key '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with optional default."""
        return self._data.get(key, default)

    def __repr__(self) -> str:
        return f"ConfigSection({self._data})"


def get_config(config_path: Optional[Path] = None) -> Config:
    """
    Get the global configuration instance.

    Args:
        config_path: Optional path to config file (only used on first call)

    Returns:
        Config instance
    """
    return Config(config_path)


# Convenience function for quick access
def get_training_config() -> ConfigSection:
    """Get training configuration section."""
    return Config().training


def get_model_config(model_name: str) -> ConfigSection:
    """
    Get configuration for a specific model.

    Args:
        model_name: One of 'mlp', 'gnn', 'attentivefp'

    Returns:
        ConfigSection for the specified model
    """
    config = Config()
    return getattr(config, model_name)
