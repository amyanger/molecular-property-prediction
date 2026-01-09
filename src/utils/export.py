"""Model export utilities for deployment (ONNX, TorchScript, etc.)."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class ModelExporter:
    """
    Export PyTorch models to various formats for deployment.

    Supports ONNX, TorchScript, and SavedModel formats.

    Args:
        model: PyTorch model to export
        model_name: Name for the exported model
    """

    def __init__(self, model: nn.Module, model_name: str = "model"):
        self.model = model
        self.model_name = model_name

    def export_onnx(
        self,
        output_path: str,
        input_shape: Union[Tuple[int, ...], dict],
        input_names: Optional[list[str]] = None,
        output_names: Optional[list[str]] = None,
        dynamic_axes: Optional[dict] = None,
        opset_version: int = 14,
        do_constant_folding: bool = True,
        verbose: bool = False,
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            output_path: Path for output ONNX file
            input_shape: Shape of input tensor(s) or dict of shapes
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes for variable batch size
            opset_version: ONNX opset version
            do_constant_folding: Whether to fold constants
            verbose: Print export details

        Returns:
            Path to exported ONNX file
        """
        self.model.eval()

        # Create dummy input
        if isinstance(input_shape, dict):
            dummy_inputs = {
                name: torch.randn(shape)
                for name, shape in input_shape.items()
            }
            dummy_input = tuple(dummy_inputs.values())
            if input_names is None:
                input_names = list(input_shape.keys())
        else:
            dummy_input = torch.randn(input_shape)
            if input_names is None:
                input_names = ["input"]

        if output_names is None:
            output_names = ["output"]

        # Set up dynamic axes for batch dimension
        if dynamic_axes is None:
            dynamic_axes = {
                name: {0: "batch_size"} for name in input_names
            }
            dynamic_axes.update({
                name: {0: "batch_size"} for name in output_names
            })

        # Export
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
        )

        logger.info(f"Model exported to ONNX: {output_path}")

        # Verify the export
        if self._verify_onnx(str(output_path)):
            logger.info("ONNX model verification passed")
        else:
            logger.warning("ONNX model verification failed")

        return str(output_path)

    def _verify_onnx(self, onnx_path: str) -> bool:
        """Verify ONNX model is valid."""
        try:
            import onnx
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            return True
        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")
            return False

    def export_torchscript(
        self,
        output_path: str,
        method: str = "trace",
        example_inputs: Optional[tuple] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        strict: bool = True,
    ) -> str:
        """
        Export model to TorchScript format.

        Args:
            output_path: Path for output file
            method: Export method ('trace' or 'script')
            example_inputs: Example inputs for tracing
            input_shape: Shape for dummy input if example_inputs not provided
            strict: Strict mode for scripting

        Returns:
            Path to exported TorchScript file
        """
        self.model.eval()

        if method == "trace":
            if example_inputs is None:
                if input_shape is None:
                    raise ValueError("Either example_inputs or input_shape required for tracing")
                example_inputs = (torch.randn(input_shape),)

            scripted_model = torch.jit.trace(self.model, example_inputs)
        elif method == "script":
            scripted_model = torch.jit.script(self.model)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        scripted_model.save(str(output_path))
        logger.info(f"Model exported to TorchScript: {output_path}")

        return str(output_path)

    def export_state_dict(
        self,
        output_path: str,
        include_metadata: bool = True,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Export model state dict with metadata.

        Args:
            output_path: Path for output file
            include_metadata: Whether to include model metadata
            metadata: Additional metadata to include

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "state_dict": self.model.state_dict(),
        }

        if include_metadata:
            save_dict["model_name"] = self.model_name
            save_dict["model_class"] = self.model.__class__.__name__

            # Try to get model config
            if hasattr(self.model, "config"):
                save_dict["config"] = self.model.config

            # Count parameters
            save_dict["num_parameters"] = sum(
                p.numel() for p in self.model.parameters()
            )
            save_dict["num_trainable"] = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

        if metadata:
            save_dict["metadata"] = metadata

        torch.save(save_dict, str(output_path))
        logger.info(f"Model state dict exported: {output_path}")

        return str(output_path)


class ONNXInference:
    """
    Run inference using ONNX Runtime.

    Args:
        model_path: Path to ONNX model
        providers: ONNX Runtime providers (e.g., ['CUDAExecutionProvider'])
    """

    def __init__(
        self,
        model_path: str,
        providers: Optional[list[str]] = None,
    ):
        import onnxruntime as ort

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def predict(self, *inputs) -> list:
        """
        Run inference.

        Args:
            *inputs: Input tensors (numpy arrays)

        Returns:
            List of output arrays
        """
        import numpy as np

        # Prepare input dict
        input_dict = {
            name: inp.astype(np.float32) if isinstance(inp, np.ndarray) else inp
            for name, inp in zip(self.input_names, inputs)
        }

        # Run inference
        outputs = self.session.run(self.output_names, input_dict)

        return outputs

    def get_input_shapes(self) -> dict:
        """Get expected input shapes."""
        return {
            inp.name: inp.shape
            for inp in self.session.get_inputs()
        }

    def get_output_shapes(self) -> dict:
        """Get expected output shapes."""
        return {
            out.name: out.shape
            for out in self.session.get_outputs()
        }


def export_for_mlp(
    model: nn.Module,
    output_dir: str,
    fingerprint_size: int = 2048,
    num_tasks: int = 12,
) -> dict:
    """
    Export MLP model for deployment.

    Args:
        model: MLP model
        output_dir: Output directory
        fingerprint_size: Input fingerprint size
        num_tasks: Number of output tasks

    Returns:
        Dictionary with export paths
    """
    exporter = ModelExporter(model, "mlp_toxicity")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # ONNX export
    paths["onnx"] = exporter.export_onnx(
        output_dir / "mlp_model.onnx",
        input_shape=(1, fingerprint_size),
        input_names=["fingerprint"],
        output_names=["predictions"],
    )

    # TorchScript export
    paths["torchscript"] = exporter.export_torchscript(
        output_dir / "mlp_model.pt",
        method="trace",
        input_shape=(1, fingerprint_size),
    )

    # State dict with metadata
    paths["state_dict"] = exporter.export_state_dict(
        output_dir / "mlp_weights.pth",
        metadata={
            "fingerprint_size": fingerprint_size,
            "num_tasks": num_tasks,
        },
    )

    return paths


def export_for_gnn(
    model: nn.Module,
    output_dir: str,
    num_node_features: int = 141,
    num_tasks: int = 12,
    max_nodes: int = 100,
    max_edges: int = 200,
) -> dict:
    """
    Export GNN model for deployment.

    Note: GNN models are more complex to export due to variable graph sizes.
    This creates TorchScript and state dict exports.

    Args:
        model: GNN model
        output_dir: Output directory
        num_node_features: Number of node features
        num_tasks: Number of output tasks
        max_nodes: Maximum number of nodes
        max_edges: Maximum number of edges

    Returns:
        Dictionary with export paths
    """
    exporter = ModelExporter(model, "gnn_toxicity")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Create example inputs for GNN
    example_x = torch.randn(max_nodes, num_node_features)
    example_edge_index = torch.randint(0, max_nodes, (2, max_edges))
    example_batch = torch.zeros(max_nodes, dtype=torch.long)

    # TorchScript export
    try:
        paths["torchscript"] = exporter.export_torchscript(
            output_dir / "gnn_model.pt",
            method="trace",
            example_inputs=(example_x, example_edge_index, example_batch),
        )
    except Exception as e:
        logger.warning(f"TorchScript export failed: {e}")

    # State dict with metadata
    paths["state_dict"] = exporter.export_state_dict(
        output_dir / "gnn_weights.pth",
        metadata={
            "num_node_features": num_node_features,
            "num_tasks": num_tasks,
        },
    )

    return paths


def quantize_model(
    model: nn.Module,
    calibration_data: Optional[torch.Tensor] = None,
    backend: str = "qnnpack",
) -> nn.Module:
    """
    Quantize model for faster inference.

    Args:
        model: PyTorch model
        calibration_data: Data for calibration (post-training quantization)
        backend: Quantization backend

    Returns:
        Quantized model
    """
    model.eval()

    # Set quantization backend
    torch.backends.quantized.engine = backend

    if calibration_data is not None:
        # Post-training static quantization
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        model_prepared = torch.quantization.prepare(model)

        # Calibrate with data
        with torch.no_grad():
            model_prepared(calibration_data)

        quantized_model = torch.quantization.convert(model_prepared)
    else:
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )

    logger.info("Model quantized successfully")
    return quantized_model
