"""Graph-based fraud detection agent using GraphSAGE (primary) and GAT (comparison).

The agent builds a directed transaction graph where nodes are accounts and
directed edges represent transactions (sender_account → receiver_account).
Node features are derived from account-level aggregates plus graph-structural
metrics (degree, PageRank). A GraphSAGE or GAT model then learns a per-node
fraud score that is propagated back to individual transactions.

AML subgraph patterns targeted:
- Fan-in  : many accounts sending to a single receiver.
- Fan-out : a single account distributing to many receivers.
- Chains  : multi-hop A→B→C→D fund movements (layering).
- Cycles  : circular flows A→B→C→A (round-tripping).
- Dense communities : mule-account clusters.

When PyTorch Geometric (PyG) is not installed the agent falls back to a
lightweight heuristic that uses in-degree / out-degree ratios derived from
NetworkX (or a pure-pandas approximation if NetworkX is also absent).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.utils.config import get_config

# ---------------------------------------------------------------------------
# Optional heavy dependencies – fail gracefully on CPU-only test environments
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from torch_geometric.data import Data as PyGData
    from torch_geometric.nn import GATConv, SAGEConv
    _PYG_AVAILABLE = True
except ImportError:  # pragma: no cover
    PyGData = None  # type: ignore[assignment]
    GATConv = None  # type: ignore[assignment]
    SAGEConv = None  # type: ignore[assignment]
    _PYG_AVAILABLE = False

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:  # pragma: no cover
    nx = None  # type: ignore[assignment]
    _NX_AVAILABLE = False


# ---------------------------------------------------------------------------
# GNN model definitions (defined in graph_models.py to respect 500-line limit)
# ---------------------------------------------------------------------------
from src.agents.graph_models import _GATModel, _GraphSAGEModel

# ---------------------------------------------------------------------------
# Graph construction utilities (defined in graph_builders.py to respect
# 500-line limit)
# ---------------------------------------------------------------------------
from src.agents.graph_builders import (
    build_edge_index,
    build_graph,
    build_node_features,
    resolve_account_columns,
)


# ---------------------------------------------------------------------------
# Graph Agent
# ---------------------------------------------------------------------------

class GraphAgent(BaseAgent):
    """Score transactions using graph-structural anomaly detection.

    Nodes are bank accounts; directed edges are transactions weighted by amount.
    A GNN (GraphSAGE by default, GAT as challenger) assigns per-node risk scores
    (logits internally, sigmoid at inference) that are mapped back to individual
    transactions.

    Args:
        config: Configuration mapping.  Recognised keys:

            - ``graph_model_type``: ``"graphsage"`` (default) or ``"gat"``.
            - ``graph_hidden_channels``: Hidden dimension (default 64).
            - ``graph_gat_heads``: Number of GAT attention heads (default 4).
            - ``graph_dropout``: Dropout probability (default 0.3).
            - ``graph_epochs``: Training epochs (default 20).
            - ``graph_lr``: Learning rate (default 1e-3).
            - ``graph_alert_threshold``: Alert threshold (default 0.65).
            - ``random_seed``: Global random seed (default 42).

        logger: Structured logger.
    """

    agent_name = "graph"

    def __init__(self, config: Any = None, logger: logging.Logger | None = None) -> None:
        """Initialise graph-model configuration and runtime state."""
        super().__init__(config or get_config(), logger)
        self.model_type: str = str(
            self.config.get("graph_model_type", "graphsage")
        ).lower()
        self.hidden_channels: int = int(self.config.get("graph_hidden_channels", 64))
        self.gat_heads: int = int(self.config.get("graph_gat_heads", 4))
        self.dropout: float = float(self.config.get("graph_dropout", 0.3))
        self.epochs: int = int(self.config.get("graph_epochs", 20))
        self.lr: float = float(self.config.get("graph_lr", 1e-3))
        self.random_seed: int = int(self.config.get("random_seed", 42))

        # Populated during fit
        self._model: Any = None
        self._device: Any = None
        self._node_index: dict[str, int] = {}   # account_id → integer node index
        self._node_scores: dict[str, float] = {}  # account_id → risk score
        self._in_features: int = 0
        self._explanations: dict[str, str] = {}

    def _build_model(self, in_channels: int) -> Any:
        """Instantiate the selected GNN model.

        Args:
            in_channels: Number of input node feature dimensions.

        Returns:
            Initialised GNN model on the detected device.
        """
        if self.model_type == "gat":
            model = _GATModel(
                in_channels=in_channels,
                hidden_channels=self.hidden_channels // self.gat_heads,
                heads=self.gat_heads,
                dropout=self.dropout,
            )
        else:
            model = _GraphSAGEModel(
                in_channels=in_channels,
                hidden_channels=self.hidden_channels,
                dropout=self.dropout,
            )
        return model.to(self._device)

    def _train_gnn(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Train the GNN on the node-classification task.

        Args:
            node_features: Float32 array (n_nodes, n_features).
            edge_index: Int64 COO edge array (2, n_edges).
            labels: Float32 array (n_nodes,) — node-level fraud flags.
        """
        x = torch.tensor(node_features, dtype=torch.float32, device=self._device)
        ei = torch.tensor(edge_index, dtype=torch.long, device=self._device)
        y = torch.tensor(labels, dtype=torch.float32, device=self._device)

        n_pos = float(y.sum().item())
        n_neg = float(len(y)) - n_pos
        # pos_ratio clips to [1, 99] so extremely rare fraud doesn't dominate
        # to the point of instability.
        pos_ratio = float(np.clip(n_neg / max(n_pos, 1.0), 1.0, 99.0))

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        self._model.train()
        pos_weight = torch.tensor([pos_ratio], device=self._device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            out = self._model(x, ei)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                self.logger.info(
                    "GraphAgent (%s) epoch %d/%d — loss: %.4f",
                    self.model_type.upper(),
                    epoch + 1,
                    self.epochs,
                    loss.item(),
                )

        self._model.eval()

    def _heuristic_scores(
        self,
        node_index: dict[str, int],
        node_features: np.ndarray,
    ) -> dict[str, float]:
        """Lightweight heuristic when GNN dependencies are absent.

        Uses fan-in/fan-out imbalance and cross-border fraction as proxies.

        Args:
            node_index: Account → node integer mapping.
            node_features: Feature matrix (n_nodes, 11).

        Returns:
            Mapping from account_id to heuristic risk score.
        """
        fan_out = np.clip(node_features[:, 6] / (node_features[:, 6].max() + 1e-9), 0, 1)
        fan_in = np.clip(node_features[:, 7] / (node_features[:, 7].max() + 1e-9), 0, 1)
        cross_border = node_features[:, 8]
        fraud_nbr = node_features[:, 9]
        scores = np.clip(
            0.3 * fan_out
            + 0.3 * fan_in
            + 0.2 * cross_border
            + 0.2 * fraud_nbr,
            0.0,
            1.0,
        )
        return {acc: float(scores[idx]) for acc, idx in node_index.items()}

    def _gnn_scores(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
    ) -> np.ndarray:
        """Run GNN inference and return per-node scores.

        Args:
            node_features: Float32 array (n_nodes, n_features).
            edge_index: Int64 COO edge array (2, n_edges).

        Returns:
            Float32 array (n_nodes,) of sigmoid risk scores.
        """
        x = torch.tensor(node_features, dtype=torch.float32, device=self._device)
        ei = torch.tensor(edge_index, dtype=torch.long, device=self._device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(x, ei).cpu().numpy()
        return torch.sigmoid(torch.tensor(logits)).numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "GraphAgent":
        """Build the transaction graph and train the GNN.

        If GNN dependencies (PyTorch + PyG) are not available the agent
        initialises in heuristic mode and ``is_fitted`` is still set to ``True``
        so that ``predict()`` can run.

        Args:
            data: Transaction DataFrame.

        Returns:
            self for fluent composition.
        """
        if not self.require_columns(data, ("transaction_id",)):
            self.is_fitted = True
            return self

        sender_col, receiver_col = resolve_account_columns(data)
        if sender_col is None or receiver_col is None:
            self.logger.warning(
                "GraphAgent.fit(): no account columns found; heuristic mode only."
            )
            self.is_fitted = True
            return self

        node_index, node_features, edge_index, labels = build_graph(
            data, sender_col, receiver_col
        )
        self._node_index = node_index
        self._in_features = node_features.shape[1]

        if not _TORCH_AVAILABLE or not _PYG_AVAILABLE:
            self.logger.warning(
                "PyTorch / PyG not available — GraphAgent running in heuristic mode."
            )
            self._node_scores = self._heuristic_scores(node_index, node_features)
            self.is_fitted = True
            return self

        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

        try:
            from src.utils.device_utils import get_device
            self._device = get_device()
        except Exception:
            self._device = torch.device("cpu")

        self._model = self._build_model(self._in_features)

        if labels is not None and labels.sum() >= 2:
            self.logger.info(
                "Training GraphAgent (%s) on %d nodes, %d edges, %d fraud nodes.",
                self.model_type.upper(),
                len(node_index),
                edge_index.shape[1] if edge_index.ndim == 2 else 0,
                int(labels.sum()),
            )
            self._train_gnn(node_features, edge_index, labels)
        else:
            self.logger.info(
                "GraphAgent: insufficient fraud labels — model initialised but not trained; "
                "heuristic fallback will be used."
            )
            self._node_scores = self._heuristic_scores(node_index, node_features)
            self.is_fitted = True
            return self

        # Cache GNN-inferred scores for all nodes
        raw_scores = self._gnn_scores(node_features, edge_index)
        self._node_scores = {acc: float(raw_scores[idx]) for acc, idx in node_index.items()}

        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return per-transaction graph risk scores.

        Transaction score = risk score of its sender account node.  Unseen
        accounts (not in the training graph) receive the global mean score.

        Args:
            data: Transaction DataFrame.

        Returns:
            Canonical agent output DataFrame.
        """
        if not self.require_columns(data, ("transaction_id",)):
            return self.empty_predictions()

        if not self.is_fitted:
            self.logger.warning(
                "GraphAgent.predict() called before fit(); fitting now."
            )
            self.fit(data)

        sender_col, receiver_col = resolve_account_columns(data)

        if not self._node_scores:
            # No graph built yet — re-derive heuristic scores from this data
            if sender_col and receiver_col:
                node_index, node_features, _, _ = build_graph(
                    data, sender_col, receiver_col
                )
                self._node_scores = self._heuristic_scores(node_index, node_features)
            else:
                scores = np.zeros(len(data), dtype=np.float32)
                reasons = np.full(len(data), "NO_GRAPH_FEATURES", dtype=object)
                self._explanations = dict(
                    zip(data["transaction_id"].astype(str), reasons)
                )
                result = self.build_predictions(data, scores, reason_code="GRAPH_ANALYSIS")
                result["reason_code"] = reasons
                result["explanation"] = [
                    "Graph analysis: no account columns available." for _ in reasons
                ]
                return result

        default_score = float(np.mean(list(self._node_scores.values()))) if self._node_scores else 0.0

        if sender_col:
            scores = np.array(
                [self._node_scores.get(str(acct), default_score) for acct in data[sender_col].astype(str)],
                dtype=np.float32,
            )
        else:
            scores = np.full(len(data), default_score, dtype=np.float32)

        # Reason codes based on structural signals
        if sender_col and receiver_col:
            # Fan-in: receiver has more in-edges than out-edges
            reasons = []
            for acct in data[sender_col].astype(str):
                score = self._node_scores.get(acct, default_score)
                if score >= self.alert_threshold:
                    reasons.append("GRAPH_ANOMALY")
                else:
                    reasons.append("NORMAL_GRAPH")
            reasons_arr = np.array(reasons)
        else:
            reasons_arr = np.where(scores >= self.alert_threshold, "GRAPH_ANOMALY", "NORMAL_GRAPH")

        self._explanations = dict(
            zip(data["transaction_id"].astype(str), reasons_arr.astype(str))
        )
        result = self.build_predictions(data, scores, reason_code="GRAPH_ANALYSIS")
        result["reason_code"] = reasons_arr
        result["explanation"] = [
            f"Graph analysis ({self.model_type.upper()}): {r.replace('_', ' ').lower()}."
            for r in reasons_arr
        ]
        return result

    def explain(self, transaction_id: str) -> str:
        """Return a human-readable explanation for a previously scored transaction.

        Args:
            transaction_id: The transaction to explain.

        Returns:
            Explanation string referencing the graph structure.
        """
        reason = self._explanations.get(
            str(transaction_id),
            "no graph score is available for this transaction",
        )
        return (
            f"Transaction {transaction_id}: {reason.replace('_', ' ').lower()} "
            f"(model: {self.model_type.upper()})."
        )
