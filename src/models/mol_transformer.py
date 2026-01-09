"""Molecular Transformer architecture for property prediction."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for molecular graphs.

    Encodes node positions using random walk-based positional encoding.

    Args:
        dim: Encoding dimension
        max_len: Maximum sequence length
    """

    def __init__(self, dim: int, max_len: int = 500):
        super().__init__()
        self.encoding = nn.Embedding(max_len, dim)

    def forward(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        return self.encoding(positions).unsqueeze(0).expand(batch_size, -1, -1)


class GraphPositionalEncoding(nn.Module):
    """
    Graph-based positional encoding using Laplacian eigenvectors.

    Args:
        dim: Output dimension
        num_eigenvectors: Number of eigenvectors to use
    """

    def __init__(self, dim: int, num_eigenvectors: int = 8):
        super().__init__()
        self.num_eigenvectors = num_eigenvectors
        self.linear = nn.Linear(num_eigenvectors, dim)

    def forward(self, lap_eigvec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lap_eigvec: Laplacian eigenvectors [N, num_eigenvectors]

        Returns:
            Positional encoding [N, dim]
        """
        # Sign invariance: use absolute values or random sign flip
        sign = 2 * (torch.rand(lap_eigvec.size(-1), device=lap_eigvec.device) > 0.5).float() - 1
        lap_eigvec = lap_eigvec * sign.unsqueeze(0)
        return self.linear(lap_eigvec)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention for molecular graphs.

    Args:
        dim: Input/output dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        bias: Whether to use bias
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, D]
            mask: Attention mask [B, N] (True = valid)
            edge_attr: Optional edge features [B, N, N, E]

        Returns:
            Output tensor [B, N, D]
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: [B, H, N, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            # [B, N] -> [B, 1, 1, N]
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.out_proj(out)

        return out


class EdgeAwareAttention(nn.Module):
    """
    Edge-aware multi-head attention incorporating bond information.

    Args:
        dim: Input dimension
        edge_dim: Edge feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int,
        edge_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Edge bias projection
        self.edge_proj = nn.Linear(edge_dim, num_heads)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [B, N, D]
            edge_attr: Edge features [B, N, N, E]
            mask: Attention mask [B, N]

        Returns:
            Output tensor [B, N, D]
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores with edge bias
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add edge bias
        edge_bias = self.edge_proj(edge_attr)  # [B, N, N, H]
        edge_bias = edge_bias.permute(0, 3, 1, 2)  # [B, H, N, N]
        attn = attn + edge_bias

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.out_proj(out)

        return out


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer for molecular graphs.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        ffn_dim: Feed-forward network dimension
        dropout: Dropout rate
        use_edge_features: Whether to use edge-aware attention
        edge_dim: Edge feature dimension
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        use_edge_features: bool = False,
        edge_dim: int = 12,
    ):
        super().__init__()

        if use_edge_features:
            self.self_attn = EdgeAwareAttention(dim, edge_dim, num_heads, dropout)
        else:
            self.self_attn = MultiHeadAttention(dim, num_heads, dropout)

        self.use_edge_features = use_edge_features

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        if self.use_edge_features and edge_attr is not None:
            attn_out = self.self_attn(x, edge_attr, mask)
        else:
            attn_out = self.self_attn(x, mask)

        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class MolecularTransformer(nn.Module):
    """
    Transformer architecture for molecular property prediction.

    Uses self-attention to capture long-range atomic interactions
    without the locality constraints of GNNs.

    Args:
        in_channels: Number of input node features
        hidden_channels: Model dimension
        out_channels: Number of output tasks
        edge_dim: Edge feature dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        ffn_dim: Feed-forward network dimension
        dropout: Dropout rate
        use_edge_features: Whether to incorporate edge features
        pooling: Graph pooling method ('mean', 'sum', 'max', 'cls')
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 12,
        edge_dim: int = 12,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        use_edge_features: bool = True,
        pooling: str = "mean",
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.pooling = pooling
        self.use_edge_features = use_edge_features

        # Input projection
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Edge encoder for dense edge matrix
        if use_edge_features:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, hidden_channels // 2),
                nn.GELU(),
                nn.Linear(hidden_channels // 2, edge_dim),
            )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_channels)

        # Optional CLS token for classification
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_channels))

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim=hidden_channels,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                use_edge_features=use_edge_features,
                edge_dim=edge_dim,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_channels)

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, E_dim]
            batch: Batch assignment [N]

        Returns:
            Predictions [B, out_channels]
        """
        # Convert sparse graph to dense batch
        x_dense, mask = to_dense_batch(x, batch)  # [B, max_N, F]
        batch_size, max_nodes, _ = x_dense.shape

        # Node encoding
        x_dense = self.node_encoder(x_dense)

        # Add positional encoding
        pos_enc = self.pos_encoding(batch_size, max_nodes, x.device)
        x_dense = x_dense + pos_enc

        # Create dense edge features if needed
        edge_attr_dense = None
        if self.use_edge_features and edge_attr is not None:
            edge_attr_dense = self._create_dense_edge_attr(
                edge_index, edge_attr, batch, max_nodes
            )
            edge_attr_dense = self.edge_encoder(edge_attr_dense)

        # Add CLS token if using CLS pooling
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x_dense = torch.cat([cls_tokens, x_dense], dim=1)
            mask = F.pad(mask, (1, 0), value=True)

            if edge_attr_dense is not None:
                # Expand edge features for CLS token
                edge_attr_dense = F.pad(
                    edge_attr_dense, (0, 0, 1, 0, 1, 0), value=0
                )

        # Apply transformer layers
        for layer in self.layers:
            x_dense = layer(x_dense, mask, edge_attr_dense)

        x_dense = self.final_norm(x_dense)

        # Pooling
        if self.pooling == "cls":
            graph_emb = x_dense[:, 0]  # CLS token
        elif self.pooling == "mean":
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).float()
            graph_emb = (x_dense * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        elif self.pooling == "max":
            x_dense = x_dense.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            graph_emb = x_dense.max(dim=1)[0]
        else:  # sum
            mask_expanded = mask.unsqueeze(-1).float()
            graph_emb = (x_dense * mask_expanded).sum(dim=1)

        # Prediction
        return self.predictor(graph_emb)

    def _create_dense_edge_attr(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        max_nodes: int,
    ) -> torch.Tensor:
        """Convert sparse edge attributes to dense format."""
        batch_size = batch.max().item() + 1

        # Initialize dense edge matrix
        dense_edge = torch.zeros(
            batch_size, max_nodes, max_nodes, edge_attr.size(-1),
            device=edge_attr.device, dtype=edge_attr.dtype
        )

        # Get node offsets per batch
        node_counts = torch.bincount(batch, minlength=batch_size)
        node_offsets = torch.cat([
            torch.zeros(1, device=batch.device, dtype=torch.long),
            node_counts.cumsum(0)[:-1]
        ])

        # Assign edge features
        src, dst = edge_index
        batch_idx = batch[src]

        # Adjust indices within each graph
        local_src = src - node_offsets[batch_idx]
        local_dst = dst - node_offsets[batch_idx]

        dense_edge[batch_idx, local_src, local_dst] = edge_attr

        return dense_edge


class MolecularTransformerWithVirtualNode(MolecularTransformer):
    """
    Molecular Transformer with virtual node for global communication.

    Adds a virtual node connected to all atoms for better global
    information aggregation.

    Args:
        Same as MolecularTransformer
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Virtual node embedding
        self.virtual_node = nn.Parameter(
            torch.zeros(1, 1, self.hidden_channels)
        )

        # Virtual node update
        self.virtual_update = nn.Sequential(
            nn.Linear(self.hidden_channels * 2, self.hidden_channels),
            nn.GELU(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # Convert to dense
        x_dense, mask = to_dense_batch(x, batch)
        batch_size, max_nodes, _ = x_dense.shape

        # Node encoding
        x_dense = self.node_encoder(x_dense)
        pos_enc = self.pos_encoding(batch_size, max_nodes, x.device)
        x_dense = x_dense + pos_enc

        # Add virtual node
        vn = self.virtual_node.expand(batch_size, -1, -1)
        x_dense = torch.cat([vn, x_dense], dim=1)
        mask = F.pad(mask, (1, 0), value=True)

        # Dense edge features
        edge_attr_dense = None
        if self.use_edge_features and edge_attr is not None:
            edge_attr_dense = self._create_dense_edge_attr(
                edge_index, edge_attr, batch, max_nodes
            )
            edge_attr_dense = self.edge_encoder(edge_attr_dense)
            edge_attr_dense = F.pad(edge_attr_dense, (0, 0, 1, 0, 1, 0), value=0)

        # Apply transformer layers with virtual node updates
        for layer in self.layers:
            x_dense = layer(x_dense, mask, edge_attr_dense)

            # Update virtual node
            vn = x_dense[:, 0:1]
            nodes = x_dense[:, 1:]
            node_mask = mask[:, 1:].unsqueeze(-1).float()
            node_agg = (nodes * node_mask).sum(dim=1, keepdim=True) / node_mask.sum(dim=1, keepdim=True).clamp(min=1)

            vn_update = self.virtual_update(torch.cat([vn, node_agg], dim=-1))
            x_dense = torch.cat([vn + vn_update, nodes], dim=1)

        x_dense = self.final_norm(x_dense)

        # Use virtual node as graph embedding
        graph_emb = x_dense[:, 0]

        return self.predictor(graph_emb)


class PreLayerNormTransformer(MolecularTransformer):
    """
    Molecular Transformer with Pre-LayerNorm for better training stability.

    Applies layer normalization before attention and FFN instead of after.

    Args:
        Same as MolecularTransformer
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace layers with pre-norm versions
        self.layers = nn.ModuleList([
            PreNormTransformerLayer(
                dim=self.hidden_channels,
                num_heads=kwargs.get("num_heads", 8),
                ffn_dim=kwargs.get("ffn_dim", 1024),
                dropout=kwargs.get("dropout", 0.1),
                use_edge_features=self.use_edge_features,
                edge_dim=kwargs.get("edge_dim", 12),
            )
            for _ in range(kwargs.get("num_layers", 4))
        ])


class PreNormTransformerLayer(nn.Module):
    """Pre-LayerNorm transformer layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        use_edge_features: bool = False,
        edge_dim: int = 12,
    ):
        super().__init__()

        if use_edge_features:
            self.self_attn = EdgeAwareAttention(dim, edge_dim, num_heads, dropout)
        else:
            self.self_attn = MultiHeadAttention(dim, num_heads, dropout)

        self.use_edge_features = use_edge_features

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        normed = self.norm1(x)
        if self.use_edge_features and edge_attr is not None:
            attn_out = self.self_attn(normed, edge_attr, mask)
        else:
            attn_out = self.self_attn(normed, mask)
        x = x + self.dropout(attn_out)

        # Pre-norm FFN
        normed = self.norm2(x)
        x = x + self.ffn(normed)

        return x
