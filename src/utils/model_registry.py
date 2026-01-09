"""Model registry for tracking and managing trained models."""

import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, asdict, field
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""

    model_id: str
    model_name: str
    model_class: str
    version: str
    created_at: str
    description: str = ""

    # Performance metrics
    metrics: dict = field(default_factory=dict)

    # Training configuration
    training_config: dict = field(default_factory=dict)

    # Model architecture details
    architecture: dict = field(default_factory=dict)

    # File paths
    checkpoint_path: Optional[str] = None
    onnx_path: Optional[str] = None

    # Additional info
    tags: list[str] = field(default_factory=list)
    num_parameters: int = 0
    file_size_mb: float = 0.0


class ModelRegistry:
    """
    Registry for tracking and managing trained models.

    Provides version control, metadata storage, and model comparison
    capabilities for molecular property prediction models.

    Args:
        registry_dir: Directory for storing registry data
    """

    def __init__(self, registry_dir: str = "models/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.registry_dir / "index.json"
        self.models_dir = self.registry_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self._load_index()

    def _load_index(self) -> None:
        """Load or create registry index."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                self.index = json.load(f)
        else:
            self.index = {
                "models": {},
                "latest": {},
                "created_at": datetime.now().isoformat(),
            }
            self._save_index()

    def _save_index(self) -> None:
        """Save registry index."""
        self.index["updated_at"] = datetime.now().isoformat()
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)

    def _generate_model_id(self, model_name: str, version: str) -> str:
        """Generate unique model ID."""
        unique_str = f"{model_name}:{version}:{datetime.now().isoformat()}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:12]

    def register(
        self,
        model: nn.Module,
        model_name: str,
        version: str,
        metrics: Optional[dict] = None,
        training_config: Optional[dict] = None,
        description: str = "",
        tags: Optional[list[str]] = None,
        set_as_latest: bool = True,
    ) -> ModelMetadata:
        """
        Register a trained model.

        Args:
            model: PyTorch model
            model_name: Name for the model (e.g., "mlp", "gcn")
            version: Version string (e.g., "1.0.0", "2024-01-15")
            metrics: Performance metrics dictionary
            training_config: Training configuration used
            description: Model description
            tags: List of tags
            set_as_latest: Whether to set as latest version

        Returns:
            ModelMetadata for registered model
        """
        model_id = self._generate_model_id(model_name, version)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())

        # Get architecture info
        architecture = {
            "class": model.__class__.__name__,
            "num_parameters": num_params,
            "layers": str(model),
        }

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_class=model.__class__.__name__,
            version=version,
            created_at=datetime.now().isoformat(),
            description=description,
            metrics=metrics or {},
            training_config=training_config or {},
            architecture=architecture,
            tags=tags or [],
            num_parameters=num_params,
        )

        # Save model checkpoint
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)

        checkpoint_path = model_dir / "model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "metadata": asdict(metadata),
        }, checkpoint_path)

        metadata.checkpoint_path = str(checkpoint_path)
        metadata.file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        # Update index
        if model_name not in self.index["models"]:
            self.index["models"][model_name] = {}

        self.index["models"][model_name][version] = {
            "model_id": model_id,
            "created_at": metadata.created_at,
            "metrics": metadata.metrics,
        }

        if set_as_latest:
            self.index["latest"][model_name] = {
                "version": version,
                "model_id": model_id,
            }

        self._save_index()

        logger.info(f"Registered model: {model_name} v{version} (ID: {model_id})")
        return metadata

    def get(
        self,
        model_name: str,
        version: Optional[str] = None,
    ) -> Optional[ModelMetadata]:
        """
        Get model metadata.

        Args:
            model_name: Model name
            version: Version string (None for latest)

        Returns:
            ModelMetadata or None if not found
        """
        if model_name not in self.index["models"]:
            return None

        if version is None:
            if model_name not in self.index["latest"]:
                return None
            version = self.index["latest"][model_name]["version"]

        if version not in self.index["models"][model_name]:
            return None

        model_id = self.index["models"][model_name][version]["model_id"]
        metadata_path = self.models_dir / model_id / "metadata.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            data = json.load(f)

        return ModelMetadata(**data)

    def load(
        self,
        model_class: type,
        model_name: str,
        version: Optional[str] = None,
        **model_kwargs,
    ) -> Optional[nn.Module]:
        """
        Load a registered model.

        Args:
            model_class: Model class to instantiate
            model_name: Model name
            version: Version string (None for latest)
            **model_kwargs: Additional arguments for model constructor

        Returns:
            Loaded model or None if not found
        """
        metadata = self.get(model_name, version)
        if metadata is None:
            logger.warning(f"Model not found: {model_name} v{version}")
            return None

        if metadata.checkpoint_path is None:
            logger.warning(f"No checkpoint found for model: {metadata.model_id}")
            return None

        checkpoint = torch.load(metadata.checkpoint_path, map_location="cpu")

        model = model_class(**model_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(f"Loaded model: {model_name} v{version}")
        return model

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self.index["models"].keys())

    def list_versions(self, model_name: str) -> list[str]:
        """List all versions of a model."""
        if model_name not in self.index["models"]:
            return []
        return list(self.index["models"][model_name].keys())

    def compare(
        self,
        model_name: str,
        versions: Optional[list[str]] = None,
        metric: str = "mean_auc_roc",
    ) -> list[dict]:
        """
        Compare different versions of a model.

        Args:
            model_name: Model name
            versions: List of versions to compare (None for all)
            metric: Metric to sort by

        Returns:
            List of version info sorted by metric
        """
        if model_name not in self.index["models"]:
            return []

        if versions is None:
            versions = list(self.index["models"][model_name].keys())

        results = []
        for version in versions:
            info = self.index["models"][model_name].get(version)
            if info:
                metric_value = info.get("metrics", {}).get(metric, 0)
                results.append({
                    "version": version,
                    "model_id": info["model_id"],
                    "created_at": info["created_at"],
                    metric: metric_value,
                    "metrics": info.get("metrics", {}),
                })

        # Sort by metric (descending)
        results.sort(key=lambda x: x.get(metric, 0), reverse=True)
        return results

    def delete(self, model_name: str, version: str) -> bool:
        """
        Delete a registered model version.

        Args:
            model_name: Model name
            version: Version to delete

        Returns:
            True if deleted successfully
        """
        if model_name not in self.index["models"]:
            return False

        if version not in self.index["models"][model_name]:
            return False

        model_id = self.index["models"][model_name][version]["model_id"]

        # Delete files
        model_dir = self.models_dir / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)

        # Update index
        del self.index["models"][model_name][version]

        # Update latest if necessary
        if (model_name in self.index["latest"] and
            self.index["latest"][model_name]["version"] == version):
            # Set new latest
            remaining = list(self.index["models"][model_name].keys())
            if remaining:
                self.index["latest"][model_name] = {
                    "version": remaining[-1],
                    "model_id": self.index["models"][model_name][remaining[-1]]["model_id"],
                }
            else:
                del self.index["latest"][model_name]

        self._save_index()

        logger.info(f"Deleted model: {model_name} v{version}")
        return True

    def export_lineage(self, model_name: str) -> dict:
        """
        Export model lineage information.

        Args:
            model_name: Model name

        Returns:
            Dictionary with model lineage
        """
        if model_name not in self.index["models"]:
            return {}

        lineage = {
            "model_name": model_name,
            "versions": [],
        }

        for version in sorted(self.index["models"][model_name].keys()):
            metadata = self.get(model_name, version)
            if metadata:
                lineage["versions"].append({
                    "version": version,
                    "model_id": metadata.model_id,
                    "created_at": metadata.created_at,
                    "metrics": metadata.metrics,
                    "num_parameters": metadata.num_parameters,
                })

        if model_name in self.index["latest"]:
            lineage["latest_version"] = self.index["latest"][model_name]["version"]

        return lineage
