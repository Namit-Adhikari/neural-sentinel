"""Behaviour Agent: per-account sequence anomaly detection.

Splits the former monolithic ``behaviour_agent.py`` into manageable helpers,
each shorter than 50 lines.  Importable even when PyTorch is unavailable
(heuristic scoring is used as a silent fallback in that environment).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.agents.behaviour.model import (
    SequenceModel,
    _TORCH_AVAILABLE,
    nn,
    torch,
)
from src.utils.config import get_config
from src.utils.device_utils import get_device

logger = logging.getLogger("neural_sentinel.agents.behaviour")

__all__ = ["BehaviourAgent"]


# ---------------------------------------------------------------------------
# BehaviourAgent
# ---------------------------------------------------------------------------


class BehaviourAgent(BaseAgent):
    """Detect anomalous account behaviour via GRU/LSTM per-account sequences."""

    agent_name = "behaviour"

    _TRANSACTION_TYPES = [
        "transfer",
        "payment",
        "withdrawal",
        "deposit",
        "cash_out",
        "remittance_inbound",
        "remittance_outbound",
    ]
    _CHANNELS = [
        "mobile_banking",
        "atm",
        "branch",
        "online_banking",
        "pos",
    ]

    def __init__(
        self,
        config: Any = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(config or get_config(), logger)
        self.model_type: str = str(
            self.config.get("behaviour_model_type", "gru")
        ).lower()
        self.seq_len: int = int(self.config.get("behaviour_seq_len", 16))
        self.hidden_dim: int = int(self.config.get("behaviour_hidden_dim", 64))
        self.embedding_dim: int = int(
            self.config.get("behaviour_embedding_dim", 16)
        )
        self.num_layers: int = int(self.config.get("behaviour_num_layers", 2))
        self.dropout: float = float(self.config.get("behaviour_dropout", 0.3))
        self.epochs: int = int(self.config.get("behaviour_epochs", 10))
        self.batch_size: int = int(self.config.get("behaviour_batch_size", 64))
        self.lr: float = float(self.config.get("behaviour_lr", 1e-3))

        self._type_vocab: dict[str, int] = {
            t: i + 1 for i, t in enumerate(self._TRANSACTION_TYPES)
        }
        self._channel_vocab: dict[str, int] = {
            c: i + 1 for i, c in enumerate(self._CHANNELS)
        }

        self._model: Optional[SequenceModel] = None  # type: ignore[type-arg]
        self._device: Any = None
        self._amount_mean: float = 0.0
        self._amount_std: float = 1.0
        self._time_mean: float = 0.0
        self._time_std: float = 1.0
        self._threshold: float = 0.5
        self._explanations: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers (each < 50 lines)
    # ------------------------------------------------------------------

    def _get_device(self) -> Any:
        if not _TORCH_AVAILABLE:
            return None
        device = get_device()
        self.logger.info("Behaviour agent using device: %s", device)
        return device

    @staticmethod
    def _account_col(data: pd.DataFrame) -> str | None:
        for candidate in ("sender_account_id", "sender_account"):
            if candidate in data.columns:
                return candidate
        return None

    @staticmethod
    def _timestamp_series(data: pd.DataFrame) -> pd.Series:
        if "timestamp" in data.columns:
            return pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
        if {"transaction_date", "transaction_time"}.issubset(data.columns):
            return pd.to_datetime(
                data["transaction_date"].astype(str)
                + " "
                + data["transaction_time"].astype(str),
                errors="coerce",
                utc=True,
            )
        return pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns, UTC]")

    def _build_working_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Attach normalized working columns (_amount, _ts, _account, etc.)."""
        acct_col = self._account_col(data)
        if acct_col is None:
            raise ValueError(
                "No sender account column found (sender_account_id or sender_account)."
            )
        ts = self._timestamp_series(data)
        w = data.copy()
        w["_ts"] = ts
        w["_account"] = w[acct_col].astype(str).fillna("__missing__")
        w["_amount"] = pd.to_numeric(
            w.get("amount_npr", w.get("amount")), errors="coerce"
        ).fillna(0.0)
        w["_amount_norm"] = (w["_amount"] - self._amount_mean) / max(
            self._amount_std, 1e-9
        )
        w = w.sort_values(["_account", "_ts"], kind="mergesort")
        w["_time_delta"] = (
            w.groupby("_account")["_ts"]
            .diff()
            .dt.total_seconds()
            .fillna(0.0)
        )
        w["_time_norm"] = (w["_time_delta"] - self._time_mean) / max(
            self._time_std, 1e-9
        )
        w["_type_id"] = (
            w["transaction_type"].astype(str).map(self._type_vocab).fillna(0).astype(int)
        ) if "transaction_type" in w.columns else 0
        w["_channel_id"] = (
            w["channel"].astype(str).map(self._channel_vocab).fillna(0).astype(int)
        ) if "channel" in w.columns else 0
        return w

    def _build_tensors_from_groups(
        self,
        working: pd.DataFrame,
        has_labels: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[str]]]:
        """Convert grouped per-account frames to numpy tensors."""
        accounts = working.groupby("_account", sort=False)
        n_accounts = len(accounts)
        seq_len = self.seq_len
        t_tens = np.zeros((n_accounts, seq_len), dtype=np.int64)
        c_tens = np.zeros((n_accounts, seq_len), dtype=np.int64)
        n_tens = np.zeros((n_accounts, seq_len, 2), dtype=np.float32)
        l_tens = np.zeros(n_accounts, dtype=np.float32)
        txn_ids: list[list[str]] = []

        for i, (_, group) in enumerate(accounts):
            tail = group.tail(seq_len)
            n = len(tail)
            start = seq_len - n
            t_tens[i, start:] = tail["_type_id"].to_numpy()
            c_tens[i, start:] = tail["_channel_id"].to_numpy()
            n_tens[i, start:, 0] = tail["_amount_norm"].to_numpy().astype(np.float32)
            n_tens[i, start:, 1] = tail["_time_norm"].to_numpy().astype(np.float32)
            if has_labels:
                fraud_vals = pd.to_numeric(tail["is_fraud"], errors="coerce").fillna(0)
                l_tens[i] = float(fraud_vals.max())
            txn_ids.append(tail["transaction_id"].astype(str).tolist())
        return t_tens, c_tens, n_tens, l_tens, txn_ids

    def _build_sequences(
        self,
        data: pd.DataFrame,
    ) -> tuple[Any, Any, Any, Any, list[list[str]]]:
        """Transform DataFrame → (type_ids, channel_ids, numerics, labels, txn_ids)."""
        working = self._build_working_frame(data)
        has_labels = "is_fraud" in working.columns
        t_arr, c_arr, n_arr, l_arr, txn_ids = self._build_tensors_from_groups(
            working, has_labels
        )
        if not _TORCH_AVAILABLE:
            return t_arr, c_arr, n_arr, l_arr, txn_ids
        return (
            torch.tensor(t_arr, dtype=torch.long),
            torch.tensor(c_arr, dtype=torch.long),
            torch.tensor(n_arr, dtype=torch.float32),
            torch.tensor(l_arr, dtype=torch.float32),
            txn_ids,
        )

    def _build_model(self) -> "SequenceModel":
        model = SequenceModel(
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

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _train_model(
        self,
        type_ids: "torch.Tensor",
        chan_ids: "torch.Tensor",
        numerics: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> None:
        """Run supervised BCELoss training for ``self.epochs`` epochs."""
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(type_ids, chan_ids, numerics, labels)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        bce = nn.BCELoss()
        self._model.train()
        for epoch in range(self.epochs):
            epoch_loss = self._run_one_epoch(loader, optimizer, bce)
            avg = epoch_loss / max(len(loader), 1)
            self.logger.info(
                "Behaviour agent (%s) epoch %d/%d — loss: %.4f",
                self.model_type.upper(),
                epoch + 1,
                self.epochs,
                avg,
            )
        self._model.eval()

    def _run_one_epoch(
        self,
        loader: Any,
        optimizer: Any,
        criterion: Any,
    ) -> float:
        """One training pass; returns accumulated loss."""
        epoch_loss = 0.0
        for t_ids, c_ids, nums, lbls in loader:
            t_ids = t_ids.to(self._device)
            c_ids = c_ids.to(self._device)
            nums = nums.to(self._device)
            lbls = lbls.to(self._device)
            optimizer.zero_grad()
            preds = self._model(t_ids, c_ids, nums)
            loss = criterion(preds, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss

    # ------------------------------------------------------------------
    # fit / predict / explain (BaseAgent contract)
    # ------------------------------------------------------------------

    def _compute_normalization(self, data: pd.DataFrame) -> None:
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
            w = data.copy()
            w["_ts"] = ts
            w["_account"] = w[acct_col].astype(str)
            w = w.sort_values(["_account", "_ts"])
            deltas = (
                w.groupby("_account")["_ts"]
                .diff()
                .dt.total_seconds()
                .dropna()
            )
            if len(deltas):
                self._time_mean = float(deltas.mean())
                self._time_std = float(deltas.std()) or 1.0
        self.logger.info(
            "Normalisation — amount(mean=%.2f, std=%.2f) time_delta(mean=%.2f, std=%.2f)",
            self._amount_mean, self._amount_std,
            self._time_mean, self._time_std,
        )

    def fit(self, data: pd.DataFrame) -> "BehaviourAgent":
        if not _TORCH_AVAILABLE:
            self.logger.warning(
                "PyTorch not available; BehaviourAgent uses heuristic scoring."
            )
            self.is_fitted = True
            return self
        if not self.require_columns(data, ("transaction_id",)):
            self.is_fitted = True
            return self
        self._device = self._get_device()
        self._compute_normalization(data)
        self._model = self._build_model()
        has_labels = (
            "is_fraud" in data.columns
            and data["is_fraud"].nunique(dropna=True) >= 2
        )
        try:
            type_ids, chan_ids, numerics, labels, _ = self._build_sequences(data)
            if has_labels:
                self.logger.info(
                    "Training %s on %d account sequences (supervised)",
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
                "Behaviour fit error (%s); falling back to heuristics.", exc
            )
        self.is_fitted = True
        return self

    def _heuristic_scores(self, data: pd.DataFrame) -> np.ndarray:
        amount_col = next(
            (c for c in ("amount_npr", "amount") if c in data.columns), None
        )
        if amount_col is None:
            return np.zeros(len(data), dtype=np.float32)
        amounts = pd.to_numeric(data[amount_col], errors="coerce").fillna(0.0)
        z = (amounts - self._amount_mean) / max(self._amount_std, 1e-9)
        return (1.0 / (1.0 + np.exp(-0.5 * np.abs(z)))).astype(np.float32)

    def _batch_inference(
        self,
        type_ids: "torch.Tensor",
        chan_ids: "torch.Tensor",
        numerics: "torch.Tensor",
    ) -> list[float]:
        """Run model inference in self.batch_size chunks to avoid OOM."""
        self._model.eval()
        batch_scores: list[float] = []
        with torch.no_grad():
            t_ids = type_ids.to(self._device)
            c_ids = chan_ids.to(self._device)
            nums = numerics.to(self._device)
            for start in range(0, len(t_ids), self.batch_size):
                end = start + self.batch_size
                out = self._model(t_ids[start:end], c_ids[start:end], nums[start:end])
                batch_scores.extend(out.cpu().numpy().tolist())
        return batch_scores

    def _assign_transaction_scores(
        self,
        data: pd.DataFrame,
        batch_scores: list[float],
        txn_id_lists: list[list[str]],
    ) -> np.ndarray:
        """Broadcast account-level scores back to individual transactions."""
        account_scores: dict[str, float] = {}
        for i, txn_ids in enumerate(txn_id_lists):
            score = float(batch_scores[i]) if i < len(batch_scores) else 0.0
            for tid in txn_ids:
                account_scores[str(tid)] = score
        return np.array(
            [account_scores.get(str(tid), 0.0) for tid in data["transaction_id"].astype(str)],
            dtype=np.float32,
        )

    def _finalize_predictions(
        self,
        data: pd.DataFrame,
        txn_scores: np.ndarray,
        model_tag: str,
    ) -> pd.DataFrame:
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
            f"Behaviour analysis ({model_tag}): {r.replace('_', ' ').lower()}."
            for r in reasons
        ]
        return result

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.require_columns(data, ("transaction_id",)):
            return self.empty_predictions()
        if not self.is_fitted:
            self.logger.warning("Behaviour.predict() called before fit(); fitting now.")
            self.fit(data)

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
            result = self.build_predictions(
                data, scores_arr, reason_code="BEHAVIOUR_ANALYSIS"
            )
            result["reason_code"] = reasons
            result["explanation"] = [
                f"Behaviour analysis (heuristic): {r.replace('_', ' ').lower()}."
                for r in reasons
            ]
            return result

        try:
            type_ids, chan_ids, numerics, _, txn_id_lists = self._build_sequences(data)
        except Exception as exc:
            self.logger.warning(
                "Behaviour sequence build failed (%s); heuristic fallback.", exc
            )
            scores_arr = self._heuristic_scores(data)
            return self.build_predictions(
                data, scores_arr, reason_code="BEHAVIOUR_HEURISTIC"
            )

        batch_scores = self._batch_inference(type_ids, chan_ids, numerics)
        txn_scores = self._assign_transaction_scores(
            data, batch_scores, txn_id_lists
        )
        return self._finalize_predictions(
            data, txn_scores, self.model_type.upper()
        )

    def explain(self, transaction_id: str) -> str:
        reason = self._explanations.get(
            str(transaction_id),
            "no behaviour score is available for this transaction",
        )
        return (
            f"Transaction {transaction_id}: "
            f"{reason.replace('_', ' ').lower()} "
            f"(model: {self.model_type.upper()})."
        )
