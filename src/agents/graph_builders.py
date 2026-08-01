"""Graph construction utilities for the Graph Agent.

Extracted from ``graph_agent.py`` to keep that module within the 500-line
limit required by AGENTS.md §10.1.  These are pure functions (no state) that
receive everything they need as arguments.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def resolve_account_columns(
    data: pd.DataFrame,
) -> tuple[str | None, str | None]:
    """Resolve sender / receiver account column names.

    Args:
        data: Transaction DataFrame.

    Returns:
        Tuple of (sender_column_name, receiver_column_name). Either may be
        ``None`` if the column is absent.
    """
    sender = next(
        (c for c in ("sender_account_id", "sender_account") if c in data.columns),
        None,
    )
    receiver = next(
        (c for c in ("receiver_account_id", "receiver_account") if c in data.columns),
        None,
    )
    return sender, receiver


def build_node_features(
    data: pd.DataFrame,
    sender_col: str,
    receiver_col: str,
    node_index: dict[str, int],
) -> np.ndarray:
    """Construct per-node feature matrix from transaction aggregates.

    Features per node:
    - ``out_degree``       : number of outgoing transactions.
    - ``in_degree``        : number of incoming transactions.
    - ``out_amount_sum``   : total NPR sent.
    - ``in_amount_sum``    : total NPR received.
    - ``out_amount_mean``  : mean NPR per outgoing transaction.
    - ``in_amount_mean``   : mean NPR per incoming transaction.
    - ``fan_out_ratio``    : out_degree / (in_degree + 1).
    - ``fan_in_ratio``     : in_degree / (out_degree + 1).
    - ``is_cross_border``  : fraction of transactions that are cross-border.
    - ``is_fraud_neighbor``: fraction of neighbouring transactions flagged (0 if labels absent).
    - ``pagerank``         : PageRank score (via NetworkX if available, else uniform).

    Args:
        data: Transaction DataFrame.
        sender_col: Name of the sender account column.
        receiver_col: Name of the receiver account column.
        node_index: Mapping from account_id string to integer index.

    Returns:
        Float32 numpy array of shape (n_nodes, 11).
    """
    n = len(node_index)
    feat = np.zeros((n, 11), dtype=np.float32)

    amount_col = "amount_npr" if "amount_npr" in data.columns else "amount"
    amounts = pd.to_numeric(
        data.get(amount_col, pd.Series(0.0, index=data.index)), errors="coerce"
    ).fillna(0.0)
    is_cross = pd.to_numeric(
        data.get("is_cross_border", pd.Series(0, index=data.index)), errors="coerce"
    ).fillna(0.0)
    is_fraud_col = pd.to_numeric(
        data.get("is_fraud", pd.Series(0, index=data.index)), errors="coerce"
    ).fillna(0.0)

    senders = data[sender_col].astype(str).map(node_index)
    receivers = data[receiver_col].astype(str).map(node_index)

    for s_idx, r_idx, amt, cross, fraud in zip(
        senders, receivers, amounts, is_cross, is_fraud_col
    ):
        if pd.notna(s_idx):
            s_idx = int(s_idx)
            feat[s_idx, 0] += 1          # out_degree
            feat[s_idx, 2] += amt        # out_amount_sum
            feat[s_idx, 8] += cross      # cross_border accumulator
            feat[s_idx, 9] += fraud      # fraud_neighbor accumulator
        if pd.notna(r_idx):
            r_idx = int(r_idx)
            feat[r_idx, 1] += 1          # in_degree
            feat[r_idx, 3] += amt        # in_amount_sum

    # Derive mean amounts
    out_deg = feat[:, 0]
    in_deg = feat[:, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        feat[:, 4] = np.where(out_deg > 0, feat[:, 2] / out_deg, 0.0)   # out_amount_mean
        feat[:, 5] = np.where(in_deg > 0, feat[:, 3] / in_deg, 0.0)    # in_amount_mean
        feat[:, 6] = out_deg / (in_deg + 1)                              # fan_out_ratio
        feat[:, 7] = in_deg / (out_deg + 1)                             # fan_in_ratio
        feat[:, 8] = np.where(out_deg > 0, feat[:, 8] / out_deg, 0.0)  # is_cross_border fraction
        feat[:, 9] = np.where(
            (out_deg + in_deg) > 0,
            feat[:, 9] / (out_deg + in_deg),
            0.0,
        )  # fraud_neighbor fraction

    # PageRank via NetworkX
    try:
        import networkx as nx  # local import to keep optional-dep guard
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        valid_edges = [
            (int(s), int(r))
            for s, r in zip(senders, receivers)
            if pd.notna(s) and pd.notna(r)
        ]
        g.add_edges_from(valid_edges)
        pr = nx.pagerank(g, alpha=0.85, max_iter=100)
        for node_id, score in pr.items():
            feat[node_id, 10] = float(score)
    except Exception as exc:  # pragma: no cover
        logger.warning("PageRank computation failed: %s", exc)

    # Log-scale normalisation for amount columns to dampen extreme values
    for col_idx in (2, 3, 4, 5):
        feat[:, col_idx] = np.log1p(feat[:, col_idx])

    return feat


def build_edge_index(
    data: pd.DataFrame,
    sender_col: str,
    receiver_col: str,
    node_index: dict[str, int],
) -> np.ndarray:
    """Build COO-format edge index array from transactions.

    Args:
        data: Transaction DataFrame.
        sender_col: Sender account column name.
        receiver_col: Receiver account column name.
        node_index: Account → node integer mapping.

    Returns:
        Int64 array of shape (2, n_edges).
    """
    senders = data[sender_col].astype(str).map(node_index)
    receivers = data[receiver_col].astype(str).map(node_index)
    mask = senders.notna() & receivers.notna()
    src = senders[mask].astype(int).to_numpy()
    dst = receivers[mask].astype(int).to_numpy()
    return np.stack([src, dst], axis=0)


def build_graph(
    data: pd.DataFrame,
    sender_col: str,
    receiver_col: str,
) -> tuple[dict[str, int], np.ndarray, np.ndarray, np.ndarray | None]:
    """Construct node index, feature matrix, and edge index.

    Args:
        data: Transaction DataFrame.
        sender_col: Sender account column.
        receiver_col: Receiver account column.

    Returns:
        Tuple of (node_index, node_features, edge_index, labels_or_None).
    """
    all_accounts = pd.concat([
        data[sender_col].astype(str),
        data[receiver_col].astype(str),
    ]).dropna().unique()
    node_index = {acc: i for i, acc in enumerate(all_accounts)}

    node_features = build_node_features(data, sender_col, receiver_col, node_index)
    edge_index = build_edge_index(data, sender_col, receiver_col, node_index)

    # Build per-node labels (max fraud flag among transactions of that account)
    labels = None
    if "is_fraud" in data.columns:
        node_labels = np.zeros(len(node_index), dtype=np.float32)
        fraud_vals = pd.to_numeric(data["is_fraud"], errors="coerce").fillna(0)
        senders = data[sender_col].astype(str).map(node_index)
        for idx, fraud in zip(senders, fraud_vals):
            if pd.notna(idx):
                node_labels[int(idx)] = max(node_labels[int(idx)], float(fraud))
        labels = node_labels

    return node_index, node_features, edge_index, labels
