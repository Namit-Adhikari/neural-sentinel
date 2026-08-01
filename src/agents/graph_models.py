"""GNN model definitions for the Graph Agent.

Kept in a separate module so that ``graph_agent.py`` stays within the
500-line limit required by AGENTS.md §10.1.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from torch_geometric.nn import GATConv, SAGEConv
    _PYG_AVAILABLE = True
except ImportError:  # pragma: no cover
    GATConv = None  # type: ignore[assignment]
    SAGEConv = None  # type: ignore[assignment]
    _PYG_AVAILABLE = False


class _GraphSAGEModel(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Two-layer GraphSAGE node classifier.

    Args:
        in_channels: Dimensionality of the input node feature vector.
        hidden_channels: Hidden layer width.
        dropout: Dropout probability applied between the two SAGE layers.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
        """Forward pass returning per-node logits (pre-sigmoid fraud scores).

        Args:
            x: Node feature matrix (n_nodes, in_channels).
            edge_index: Edge index tensor (2, n_edges) in COO format.

        Returns:
            Per-node logits (n_nodes,).
        """
        x = self.act(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x.squeeze(-1)


class _GATModel(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Two-layer Graph Attention Network node classifier.

    GAT attention weights provide interpretable neighbour contributions,
    making it easier to explain which accounts pushed a node's score high.

    Args:
        in_channels: Dimensionality of the input node feature vector.
        hidden_channels: Hidden layer width (per head).
        heads: Number of attention heads in the first GAT layer.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, 1, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ELU()

    def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
        """Forward pass returning per-node logits (pre-sigmoid fraud scores).

        Args:
            x: Node feature matrix (n_nodes, in_channels).
            edge_index: Edge index tensor (2, n_edges) in COO format.

        Returns:
            Per-node logits (n_nodes,).
        """
        x = self.act(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x.squeeze(-1)
