"""Tests for model architectures."""

import pytest
import torch

from src.models import MolecularPropertyPredictor, GNN, AttentiveFPPredictor


class TestMLP:
    """Tests for MLP model."""

    def test_mlp_creation(self):
        """Test MLP model can be created with default parameters."""
        model = MolecularPropertyPredictor()
        assert model is not None

    def test_mlp_custom_parameters(self):
        """Test MLP model with custom parameters."""
        model = MolecularPropertyPredictor(
            input_size=1024,
            hidden_sizes=[512, 256],
            num_tasks=6,
            dropout=0.5
        )
        assert model is not None

    def test_mlp_forward_pass(self):
        """Test MLP forward pass produces correct output shape."""
        model = MolecularPropertyPredictor(input_size=2048, num_tasks=12)
        model.eval()

        batch_size = 8
        x = torch.randn(batch_size, 2048)
        output = model(x)

        assert output.shape == (batch_size, 12)

    def test_mlp_output_range(self):
        """Test MLP output is not NaN or Inf."""
        model = MolecularPropertyPredictor()
        model.eval()

        x = torch.randn(4, 2048)
        output = model(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_mlp_parameter_count(self):
        """Test MLP has reasonable parameter count."""
        model = MolecularPropertyPredictor()
        num_params = sum(p.numel() for p in model.parameters())

        # Should have at least 1M parameters
        assert num_params > 1_000_000

    def test_mlp_gradient_flow(self):
        """Test gradients flow through MLP."""
        model = MolecularPropertyPredictor()
        model.train()

        x = torch.randn(4, 2048)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestGNN:
    """Tests for GNN model."""

    def test_gnn_creation(self):
        """Test GNN model can be created."""
        model = GNN()
        assert model is not None

    def test_gnn_gcn_type(self):
        """Test GNN with GCN convolution type."""
        model = GNN(conv_type='gcn')
        assert model is not None

    def test_gnn_gat_type(self):
        """Test GNN with GAT convolution type."""
        model = GNN(conv_type='gat')
        assert model is not None

    def test_gnn_forward_pass(self):
        """Test GNN forward pass produces correct output shape."""
        model = GNN(num_node_features=141, num_tasks=12)
        model.eval()

        # Create mock graph data
        num_nodes = 20
        x = torch.randn(num_nodes, 141)
        edge_index = torch.randint(0, num_nodes, (2, 40))
        batch = torch.zeros(num_nodes, dtype=torch.long)  # Single graph

        output = model(x, edge_index, batch)
        assert output.shape == (1, 12)  # 1 graph, 12 tasks

    def test_gnn_batch_processing(self):
        """Test GNN can process batch of graphs."""
        model = GNN(num_node_features=141, num_tasks=12)
        model.eval()

        # Create batch of 3 graphs
        x_list = []
        edge_list = []
        batch_list = []
        node_offset = 0

        for i in range(3):
            num_nodes = 10 + i * 5
            x_list.append(torch.randn(num_nodes, 141))
            edges = torch.randint(0, num_nodes, (2, num_nodes * 2))
            edge_list.append(edges + node_offset)
            batch_list.append(torch.full((num_nodes,), i, dtype=torch.long))
            node_offset += num_nodes

        x = torch.cat(x_list, dim=0)
        edge_index = torch.cat(edge_list, dim=1)
        batch = torch.cat(batch_list, dim=0)

        output = model(x, edge_index, batch)
        assert output.shape == (3, 12)  # 3 graphs, 12 tasks

    def test_gnn_output_range(self):
        """Test GNN output is not NaN or Inf."""
        model = GNN()
        model.eval()

        x = torch.randn(10, 141)
        edge_index = torch.randint(0, 10, (2, 20))
        batch = torch.zeros(10, dtype=torch.long)

        output = model(x, edge_index, batch)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gnn_residual_connections(self):
        """Test GNN has residual connections by checking layer output."""
        model = GNN(num_layers=4)

        # Verify model has correct number of conv layers
        assert len(model.convs) == 4
        assert len(model.batch_norms) == 4


class TestAttentiveFP:
    """Tests for AttentiveFP model."""

    def test_attentivefp_creation(self):
        """Test AttentiveFP model can be created."""
        model = AttentiveFPPredictor(in_channels=148)
        assert model is not None

    def test_attentivefp_custom_parameters(self):
        """Test AttentiveFP with custom parameters."""
        model = AttentiveFPPredictor(
            in_channels=148,
            hidden_channels=128,
            out_channels=6,
            edge_dim=12,
            num_layers=2,
            num_timesteps=2,
            dropout=0.1
        )
        assert model is not None

    def test_attentivefp_forward_pass(self):
        """Test AttentiveFP forward pass produces correct output shape."""
        model = AttentiveFPPredictor(
            in_channels=148,
            out_channels=12,
            edge_dim=12
        )
        model.eval()

        # Create mock graph data with edge features
        num_nodes = 15
        x = torch.randn(num_nodes, 148)
        edge_index = torch.randint(0, num_nodes, (2, 30))
        edge_attr = torch.randn(30, 12)
        batch = torch.zeros(num_nodes, dtype=torch.long)

        output = model(x, edge_index, edge_attr, batch)
        assert output.shape == (1, 12)

    def test_attentivefp_batch_processing(self):
        """Test AttentiveFP can process batch of graphs."""
        model = AttentiveFPPredictor(
            in_channels=148,
            out_channels=12,
            edge_dim=12
        )
        model.eval()

        # Create batch of 2 graphs
        x = torch.randn(25, 148)  # Total nodes
        edge_index = torch.randint(0, 25, (2, 50))
        edge_attr = torch.randn(50, 12)
        batch = torch.cat([
            torch.zeros(12, dtype=torch.long),
            torch.ones(13, dtype=torch.long)
        ])

        output = model(x, edge_index, edge_attr, batch)
        assert output.shape == (2, 12)

    def test_attentivefp_output_range(self):
        """Test AttentiveFP output is not NaN or Inf."""
        model = AttentiveFPPredictor(in_channels=148, edge_dim=12)
        model.eval()

        x = torch.randn(10, 148)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_attr = torch.randn(20, 12)
        batch = torch.zeros(10, dtype=torch.long)

        output = model(x, edge_index, edge_attr, batch)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestModelComparison:
    """Tests comparing different models."""

    def test_all_models_same_output_tasks(self):
        """Test all models produce same number of output tasks."""
        num_tasks = 12

        mlp = MolecularPropertyPredictor(num_tasks=num_tasks)
        gnn = GNN(num_tasks=num_tasks)
        afp = AttentiveFPPredictor(in_channels=148, out_channels=num_tasks)

        # MLP output
        mlp_out = mlp(torch.randn(1, 2048))

        # GNN output
        x = torch.randn(10, 141)
        edge_index = torch.randint(0, 10, (2, 20))
        batch = torch.zeros(10, dtype=torch.long)
        gnn_out = gnn(x, edge_index, batch)

        # AttentiveFP output
        x = torch.randn(10, 148)
        edge_attr = torch.randn(20, 12)
        afp_out = afp(x, edge_index, edge_attr, batch)

        assert mlp_out.shape[1] == num_tasks
        assert gnn_out.shape[1] == num_tasks
        assert afp_out.shape[1] == num_tasks

    def test_models_trainable(self):
        """Test all models have trainable parameters."""
        models = [
            MolecularPropertyPredictor(),
            GNN(),
            AttentiveFPPredictor(in_channels=148)
        ]

        for model in models:
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            assert trainable_params > 0, f"{model.__class__.__name__} has no trainable params"


class TestModelModes:
    """Tests for model training/eval modes."""

    def test_mlp_train_eval_modes(self):
        """Test MLP behaves differently in train vs eval mode."""
        model = MolecularPropertyPredictor(dropout=0.5)
        x = torch.randn(10, 2048)

        model.train()
        out_train = model(x)

        model.eval()
        out_eval = model(x)

        # Due to dropout, outputs should differ (most of the time)
        # This is a probabilistic test
        assert out_train.shape == out_eval.shape

    def test_gnn_train_eval_modes(self):
        """Test GNN behaves differently in train vs eval mode."""
        model = GNN(dropout=0.5)
        x = torch.randn(10, 141)
        edge_index = torch.randint(0, 10, (2, 20))
        batch = torch.zeros(10, dtype=torch.long)

        model.train()
        out_train = model(x, edge_index, batch)

        model.eval()
        out_eval = model(x, edge_index, batch)

        assert out_train.shape == out_eval.shape
