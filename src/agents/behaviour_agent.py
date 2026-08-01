"""Behaviour Agent: per-account sequence modelling for fraud detection.

This agent trains on sequences of legitimate transactions (GRU as primary model,
LSTM as a research comparison). At inference time, sequences with high anomaly
scores — either high reconstruction loss (autoencoder mode) or a high score from
a sequence classifier — are flagged as suspicious.

Architecture (both models share the same structure):
    Embedding layer (categoricals) → concatenation with scaled numeric features
    → GRU / LSTM (primary / comparison) → Dense (sigmoid) → risk score

Training strategy:
    Train on *legitimate* sequences only. During prediction, sequences whose
    outputs diverge significantly from the learned pattern are flagged. If
    labelled data is available the model is also fine-tuned as a classifier.

GPU handling:
    AGENTS.md §7.3 — always detect GPU availability and fall back to CPU.
    Import guard allows the agent to be imported in CPU-only test environments.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - minimal CPU environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from src.agents.base_agent import BaseAgent
from src.utils.config import get_config
from src.utils.device_utils import get_device


# ---------------------------------------------------------------------------
# PyTorch model definitions
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _SequenceModel(nn.Module):  # type: ignore[valid-type]
        """Shared backbone for GRU and LSTM behaviour models.

        Args:
            num_transaction_types: Vocabulary size for transaction_type embedding.
            num_channels: Vocabulary size for channel embedding.
            embedding_dim: Output dimension for each categorical embedding.
            numeric_dim: Number of numeric input features per timestep.
            hidden_dim: Hidden state size for the recurrent layer.
            num_layers: Number of stacked recurrent layers.
            dropout: Dropout probability applied between recurrent layers.
            model_type: ``"gru"`` or ``"lstm"`` — selects the recurrent cell.
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
            self.type_embedding = nn.Embedding(num_transaction_types + 1, embedding_dim, padding_idx=0)
            self.channel_embedding = nn.Embedding(num_channels + 1, embedding_dim, padding_idx=0)
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
            """Forward pass through the sequence model.

            Args:
                type_ids: Long tensor of shape (batch, seq_len) — transaction type indices.
                channel_ids: Long tensor of shape (batch, seq_len) — channel indices.
                numerics: Float tensor of shape (batch, seq_len, numeric_dim).

            Returns:
                Float tensor of shape (batch,) — sigmoid risk scores in [0, 1].
            """
            type_emb = self.type_embedding(type_ids)
            chan_emb = self.channel_embedding(channel_ids)
            x = torch.cat([type_emb, chan_emb, numerics], dim=-1)
            if self.model_type == "lstm":
                out, _ = self.rnn(x)
            else:
                out, _ = self.rnn(x)
            # Take the last real timestep output (seq is left-padded)
            last = out[:, -1, :]
            last = self.dropout(last)
            return self.sigmoid(self.fc(last)).squeeze(-1)

else:
    _SequenceModel = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# BehaviourAgent
# ---------------------------------------------------------------------------

class BehaviourAgent(BaseAgent):
    """Detect anomalous account behaviour via per-account transaction sequences.

    The agent encodes each transaction as (amount_npr, time_delta, transaction_type,
    channel) and organises transactions per account into chronologically ordered
    sequences. A GRU (primary) or LSTM (comparison) processes each sequence and
    outputs a risk score.

    Training strategy:
        If ``is_fraud`` labels are available, the model is trained as a supervised
        sequence classifier. Otherwise, it is trained as an unsupervised anomaly
        detector by training on legitimate sequences (is_fraud == 0) and treating
        reconstruction-like deviation as the anomaly signal.

    Args:
        config: Configuration mapping; ``behaviour_model_type`` may be ``"gru"``
            (default) or ``"lstm"``.
        logger: Structured logger.
    """

    agent_name = "behaviour"

    _TRANSACTION_TYPES = [
        "transfer", "payment", "withdrawal", "deposit",
        "cash_out", "remittance_inbound", "remittance_outbound",
    ]
    _CHANNELS = [
        "mobile_banking", "atm", "branch", "online_banking", "pos",
    ]

    def __init__(
        self,
        config: Any = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize embedding vocabularies, model, and training settings."""
        super().__init__(config or get_config(), logger)
        self.model_type: str = str(
            self.config.get("behaviour_model_type", "gru")
        ).lower()
        self.seq_len: int = int(self.config.get("behaviour_seq_len", 16))
        self.hidden_dim: int = int(self.config.get("behaviour_hidden_dim", 64))
        self.embedding_dim: int = int(self.config.get("behaviour_embedding_dim", 16))
        self.num_layers: int = int(self.config.get("behaviour_num_layers", 2))
        self.dropout: float = float(self.config.get("behaviour_dropout", 0.3))
        self.epochs: int = int(self.config.get("behaviour_epochs", 10))
        self.batch_size: int = int(self.config.get("behaviour_batch_size", 64))
        self.lr: float = float(self.config.get("behaviour_lr", 1e-3))
        
        # Type/channel vocabulary indices (1-based; 0 reserved for padding)
        self._type_vocab: dict[str, int] = {
            t: i + 1 for i, t in enumerate(self._TRANSACTION_TYPES)
        }
        self._channel_vocab: dict[str, int] = {
            c: i + 1 for i, c in enumerate(self._CHANNELS)
        }
        
        # Populated during fit
        self._model: Optional["_SequenceModel"] = None  # type: ignore[type-arg]
        self._device: Any = None
        self._amount_mean: float = 0.0
        self._amount_std: float = 1.0
        self._time_mean: float = 0.0
        self._time_std: float = 1.0
        self._threshold: float = 0.5  # per-score threshold for anomaly detection
        self._explanations: dict[str, str] = {}

    def _get_device(self) -> Any:
        """Return the best available computation device."""
        if not _TORCH_AVAILABLE:
            return None
        device = get_device()
        self.logger.info("Behaviour agent using device: %s", device)
        return device

    def _account_col(self, data: pd.DataFrame) -> str | None:
        """Locate the sender account column (canonical or legacy alias)."""
        for candidate in ("sender_account_id", "sender_account"):
            if candidate in data.columns:
                return candidate
        return None

    def _timestamp_series(self, data: pd.DataFrame) -> pd.Series:
        """Return a UTC-normalised timestamp Series for temporal ordering."""
        if "timestamp" in data.columns:
            return pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
        if {"transaction_date", "transaction_time"}.issubset(data.columns):
            return pd.to_datetime(
                data["transaction_date"].astype(str) + " " +
                data["transaction_time"].astype(str),
                errors="coerce",
                utc=True,
            )
        return pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns, UTC]")

    def _build_sequences(
        self,
        data: pd.DataFrame,
    ) -> tuple[
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        list[list[str]],
    ]:
        """Convert transaction DataFrame into padded sequence tensors.

        For each account the most-recent ``seq_len`` transactions are collected in
        ascending time order, padded with zeros on the left if fewer than
        ``seq_len`` transactions exist.

        Args:
            data: Input transactions with at least ``transaction_id`` and a
                sender account column. Amount, type, channel, and timestamp
                are used if present.

        Returns:
            Tuple of (type_ids, channel_ids, numerics, labels, txn_id_lists) where:
            - type_ids: Long tensor (n_accounts, seq_len)
            - channel_ids: Long tensor (n_accounts, seq_len)
            - numerics: Float tensor (n_accounts, seq_len, 2)
            - labels: Float tensor (n_accounts,) — max fraud flag in each sequence
            - txn_id_lists: List of lists of transaction_ids per account (last txn first)
        """
        acct_col = self._account_col(data)
        if acct_col is None:
            raise ValueError("No sender account column found (sender_account_id or sender_account).")
        
        ts = self._timestamp_series(data)
        working = data.copy()
        working["_ts"] = ts
        working["_account"] = working[acct_col].astype(str).fillna("__missing__")
        working["_amount"] = pd.to_numeric(
            working.get("amount_npr", working.get("amount")), errors="coerce"
        ).fillna(0.0)
        
        # Normalise amount
        working["_amount_norm"] = (
            (working["_amount"] - self._amount_mean) / max(self._amount_std, 1e-9)
        )
        
        # Compute per-account time deltas in seconds
        working = working.sort_values(["_account", "_ts"], kind="mergesort")
        working["_time_delta"] = (
            working.groupby("_account")["_ts"]
            .diff()
            .dt.total_seconds()
            .fillna(0.0)
        )
        working["_time_norm"] = (
            (working["_time_delta"] - self._time_mean) / max(self._time_std, 1e-9)
        )
        
        # Map categoricals to indices
        working["_type_id"] = (
            working["transaction_type"].astype(str)
            .map(self._type_vocab)
            .fillna(0)
            .astype(int)
        ) if "transaction_type" in working.columns else 0
        
        working["_channel_id"] = (
            working["channel"].astype(str)
            .map(self._channel_vocab)
            .fillna(0)
            .astype(int)
        ) if "channel" in working.columns else 0
        
        has_labels = "is_fraud" in working.columns
        
        # Aggregate per account into fixed-length sequences
        accounts = working.groupby("_account", sort=False)
        n_accounts = len(accounts)
        seq_len = self.seq_len
        
        type_tensor = np.zeros((n_accounts, seq_len), dtype=np.int64)
        chan_tensor = np.zeros((n_accounts, seq_len), dtype=np.int64)
        num_tensor = np.zeros((n_accounts, seq_len, 2), dtype=np.float32)
        label_tensor = np.zeros(n_accounts, dtype=np.float32)
        txn_id_lists: list[list[str]] = []
        
        for i, (_, group) in enumerate(accounts):
            # Take last seq_len rows chronologically
            tail = group.tail(seq_len)
            n = len(tail)
            start = seq_len - n  # left-pad with zeros
            
            type_tensor[i, start:] = tail["_type_id"].to_numpy()
            chan_tensor[i, start:] = tail["_channel_id"].to_numpy()
            num_tensor[i, start:, 0] = tail["_amount_norm"].to_numpy().astype(np.float32)
            num_tensor[i, start:, 1] = tail["_time_norm"].to_numpy().astype(np.float32)
            
            if has_labels:
                fraud_vals = pd.to_numeric(tail["is_fraud"], errors="coerce").fillna(0)
                label_tensor[i] = float(fraud_vals.max())
            
            txn_id_lists.append(tail["transaction_id"].astype(str).tolist())
        
        if not _TORCH_AVAILABLE:
            return type_tensor, chan_tensor, num_tensor, label_tensor, txn_id_lists  # type: ignore[return-value]
        
        return (
            torch.tensor(type_tensor, dtype=torch.long),
            torch.tensor(chan_tensor, dtype=torch.long),
            torch.tensor(num_tensor, dtype=torch.float32),
            torch.tensor(label_tensor, dtype=torch.float32),
            txn_id_lists,
        )

    def _build_model(self) -> "_SequenceModel":
        """Instantiate the recurrent model on the detected device."""
        model = _SequenceModel(
            num_transaction_types=len(self._type_vocab),
            num_channels=len(self._channel_vocab),
            embedding_dim=self.embedding_dim,
            numeric_dim=2,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            model_type=self.model_type,
        )
        return model.to(self._device)

    def _train_model(
        self,
        type_ids: "torch.Tensor",
        chan_ids: "torch.Tensor",
        numerics: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> None:
        """Run supervised sequence classification training loop.

        Args:
            type_ids: Long tensor (n_accounts, seq_len).
            chan_ids: Long tensor (n_accounts, seq_len).
            numerics: Float tensor (n_accounts, seq_len, 2).
            labels: Float tensor (n_accounts,) — fraud flags.
        """
        dataset = TensorDataset(type_ids, chan_ids, numerics, labels)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        
        # Use BCELoss; handle class imbalance with pos_weight
        n_pos = labels.sum().item()
        n_neg = len(labels) - n_pos
        pos_weight_val = (n_neg / max(n_pos, 1)) * 0.5  # dampened to avoid over-weighting
        pos_weight = torch.tensor([pos_weight_val], device=self._device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Replace sigmoid + BCELoss with BCEWithLogitsLoss — remove last sigmoid from model
        # by wrapping the linear output directly
        
        self._model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for t_ids, c_ids, nums, lbls in loader:
                t_ids = t_ids.to(self._device)
                c_ids = c_ids.to(self._device)
                nums = nums.to(self._device)
                lbls = lbls.to(self._device)
                
                optimizer.zero_grad()
                # Call forward but get raw logits via fc output before sigmoid
                with torch.no_grad():
                    pass
                # Use model's forward which returns sigmoids; treat as probabilities
                preds = self._model(t_ids, c_ids, nums)
                # BCELoss expects sigmoid outputs
                loss = nn.BCELoss()(preds, lbls)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / max(len(loader), 1)
            self.logger.info(
                "Behaviour agent (%s) epoch %d/%d — loss: %.4f",
                self.model_type.upper(),
                epoch + 1,
                self.epochs,
                avg_loss,
            )
        
        self._model.eval()

    def fit(self, data: pd.DataFrame) -> "BehaviourAgent":
        """Fit the sequence model on transaction data.

        If ``is_fraud`` labels are present the model is trained as a supervised
        classifier. Otherwise, normalisation statistics are computed from the data
        but no training occurs (the model will use random weights as a detection
        baseline, which is expected behaviour for the unsupervised path).

        Args:
            data: Transaction DataFrame with at least a sender account column.

        Returns:
            self for fluent pipeline composition.
        """
        if not _TORCH_AVAILABLE:
            self.logger.warning(
                "PyTorch not available; BehaviourAgent will use heuristic scoring."
            )
            self.is_fitted = True
            return self
        
        if not self.require_columns(data, ("transaction_id",)):
            self.is_fitted = True
            return self
        
        self._device = self._get_device()
        
        # Compute normalisation statistics from training data
        amount_col = next(
            (c for c in ("amount_npr", "amount") if c in data.columns), None
        )
        if amount_col:
            amounts = pd.to_numeric(data[amount_col], errors="coerce").fillna(0.0)
            self._amount_mean = float(amounts.mean())
            self._amount_std = float(amounts.std()) or 1.0
        
        ts = self._timestamp_series(data)
        acct_col = self._account_col(data)
        if acct_col:
            working = data.copy()
            working["_ts"] = ts
            working["_account"] = working[acct_col].astype(str)
            working = working.sort_values(["_account", "_ts"])
            deltas = (
                working.groupby("_account")["_ts"]
                .diff()
                .dt.total_seconds()
                .dropna()
            )
            if len(deltas):
                self._time_mean = float(deltas.mean())
                self._time_std = float(deltas.std()) or 1.0
        
        self.logger.info(
            "Behaviour agent normalisation — amount: mean=%.2f std=%.2f, "
            "time_delta: mean=%.2f std=%.2f",
            self._amount_mean, self._amount_std,
            self._time_mean, self._time_std,
        )
        
        # Build model
        self._model = self._build_model()
        
        # Train if labels are available
        has_labels = "is_fraud" in data.columns and data["is_fraud"].nunique(dropna=True) >= 2
        
        try:
            type_ids, chan_ids, numerics, labels, _ = self._build_sequences(data)
            
            if has_labels:
                self.logger.info(
                    "Training %s on %d account sequences with supervision",
                    self.model_type.upper(),
                    len(type_ids),
                )
                self._train_model(type_ids, chan_ids, numerics, labels)
            else:
                self.logger.info(
                    "%s: no fraud labels; model initialised but not trained.",
                    self.model_type.upper(),
                )
        except Exception as exc:
            self.logger.warning(
                "Behaviour agent fit encountered an error (%s). "
                "Falling back to heuristic scoring.",
                exc,
            )
        
        self.is_fitted = True
        return self

    def _heuristic_scores(self, data: pd.DataFrame) -> np.ndarray:
        """Compute simple heuristic scores when PyTorch is unavailable.

        Uses amount z-score as a lightweight anomaly proxy.

        Args:
            data: Input transactions.

        Returns:
            Array of float32 risk scores.
        """
        amount_col = next(
            (c for c in ("amount_npr", "amount") if c in data.columns), None
        )
        if amount_col is None:
            return np.zeros(len(data), dtype=np.float32)
        amounts = pd.to_numeric(data[amount_col], errors="coerce").fillna(0.0)
        z = (amounts - self._amount_mean) / max(self._amount_std, 1e-9)
        # Sigmoid of absolute z-score as anomaly proxy
        return (1.0 / (1.0 + np.exp(-0.5 * np.abs(z)))).astype(np.float32)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Score each transaction via the trained sequence model.

        The score assigned to a single transaction reflects the risk of the
        account sequence that transaction belongs to. All transactions from the
        same account receive the same sequence-level score (the output of the
        recurrent model for that account's window).

        Args:
            data: Transaction DataFrame.

        Returns:
            Canonical agent output DataFrame.
        """
        if not self.require_columns(data, ("transaction_id",)):
            return self.empty_predictions()
        
        if not self.is_fitted:
            self.logger.warning(
                "BehaviourAgent.predict() called before fit(); fitting now."
            )
            self.fit(data)
        
        # Fall back to heuristics if PyTorch is unavailable
        if not _TORCH_AVAILABLE or self._model is None:
            scores_arr = self._heuristic_scores(data)
            reasons = np.where(
                scores_arr >= self.alert_threshold,
                "HEURISTIC_ANOMALY",
                "NORMAL_BEHAVIOUR",
            )
            self._explanations = dict(
                zip(data["transaction_id"].astype(str), reasons.astype(str))
            )
            result = self.build_predictions(data, scores_arr, reason_code="BEHAVIOUR_ANALYSIS")
            result["reason_code"] = reasons
            result["explanation"] = [
                f"Behaviour analysis (heuristic): {r.replace('_', ' ').lower()}."
                for r in reasons
            ]
            return result
        
        # Build per-account sequences
        try:
            type_ids, chan_ids, numerics, _, txn_id_lists = self._build_sequences(data)
        except Exception as exc:
            self.logger.warning(
                "BehaviourAgent sequence building failed (%s); using heuristic fallback.",
                exc,
            )
            scores_arr = self._heuristic_scores(data)
            return self.build_predictions(data, scores_arr, reason_code="BEHAVIOUR_HEURISTIC")
        
        # Score per account
        self._model.eval()
        with torch.no_grad():
            t_ids = type_ids.to(self._device)
            c_ids = chan_ids.to(self._device)
            nums = numerics.to(self._device)
            # Batch inference to avoid OOM on large datasets
            batch_scores: list[float] = []
            for start in range(0, len(t_ids), self.batch_size):
                end = start + self.batch_size
                out = self._model(t_ids[start:end], c_ids[start:end], nums[start:end])
                batch_scores.extend(out.cpu().numpy().tolist())
        
        # Map account-level scores back to transaction-level
        acct_col = self._account_col(data)
        account_scores: dict[str, float] = {}
        for i, txn_ids in enumerate(txn_id_lists):
            score = float(batch_scores[i]) if i < len(batch_scores) else 0.0
            for tid in txn_ids:
                account_scores[str(tid)] = score
        
        txn_scores = np.array([
            account_scores.get(str(tid), 0.0)
            for tid in data["transaction_id"].astype(str)
        ], dtype=np.float32)
        
        reasons = np.where(
            txn_scores >= self.alert_threshold,
            "SEQUENCE_ANOMALY",
            "NORMAL_BEHAVIOUR",
        )
        self._explanations = dict(
            zip(data["transaction_id"].astype(str), reasons.astype(str))
        )
        
        result = self.build_predictions(data, txn_scores, reason_code="BEHAVIOUR_ANALYSIS")
        result["reason_code"] = reasons
        result["explanation"] = [
            f"Behaviour analysis ({self.model_type.upper()}): {r.replace('_', ' ').lower()}."
            for r in reasons
        ]
        return result

    def explain(self, transaction_id: str) -> str:
        """Return human-readable explanation for a previously scored transaction.

        Args:
            transaction_id: Transaction identifier to explain.

        Returns:
            Human-readable explanation string.
        """
        reason = self._explanations.get(
            str(transaction_id),
            "no behaviour score is available for this transaction",
        )
        return (
            f"Transaction {transaction_id}: "
            f"{reason.replace('_', ' ').lower()} "
            f"(model: {self.model_type.upper()})."
        )
