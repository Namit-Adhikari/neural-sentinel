"""PyTorch sequence models for the Behaviour Agent.

Defines :class:`SequenceModel` — a shared GRU/LSTM backbone used by
:class:`BehaviourAgent` in :mod:`src.agents.behaviour.agent`.  The torch
import is guarded so this module (and the behaviour package) can be imported
in CPU-only test environments without a hard torch dependency.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("neural_sentinel.behaviour.model")

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - minimal CPU environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


__all__ = ["SequenceModel", "_TORCH_AVAILABLE", "torch", "nn"]


if _TORCH_AVAILABLE:

    class SequenceModel(nn.Module):  # type: ignore[valid-type]
        """Shared backbone for GRU and LSTM behaviour models.

        Architecture:
            Embedding(transaction_type) · Embedding(channel) · scaled numeric
            → recurrent stack → dropout → Linear → Sigmoid → risk score
        """

        def __init__(
            self,
            num_transaction_types: int,
            num_channels: int,
            embedding_dim: int = 16,
            numeric_dim: int = 2,
            hidden_dim: int = 64,
            num_layers: int = 2,
            dropout: float = 0.3,
            model_type: str = "gru",
        ) -> None:
            super().__init__()
            self.type_embedding = nn.Embedding(
                num_transaction_types + 1, embedding_dim, padding_idx=0
            )
            self.channel_embedding = nn.Embedding(
                num_channels + 1, embedding_dim, padding_idx=0
            )
            self.input_dim = 2 * embedding_dim + numeric_dim
            self.model_type = model_type.lower()
            rnn_kwargs: dict[str, Any] = dict(
                input_size=self.input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            if self.model_type == "lstm":
                self.rnn: nn.Module = nn.LSTM(**rnn_kwargs)
            else:
                self.rnn = nn.GRU(**rnn_kwargs)

            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(
            self,
            type_ids: "torch.Tensor",
            channel_ids: "torch.Tensor",
            numerics: "torch.Tensor",
        ) -> "torch.Tensor":
            """Forward pass.

            Args:
                type_ids: Long (batch, seq_len) transaction-type indices.
                channel_ids: Long (batch, seq_len) channel indices.
                numerics: Float (batch, seq_len, numeric_dim).

            Returns:
                Float (batch,) sigmoid risk scores in ``[0, 1]``.
            """
            type_emb = self.type_embedding(type_ids)
            chan_emb = self.channel_embedding(channel_ids)
            x = torch.cat([type_emb, chan_emb, numerics], dim=-1)
            out, _ = self.rnn(x)
            last = out[:, -1, :]
            last = self.dropout(last)
            return self.sigmoid(self.fc(last)).squeeze(-1)

else:  # pragma: no cover - torch unavailable
    SequenceModel = None  # type: ignore[assignment,misc]
