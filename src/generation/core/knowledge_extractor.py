"""
src/generation/knowledge_extractor.py
--------------------------------------
Phase 3 — Banking Knowledge Extraction.

Learns how the real banking system behaves from cleaned data and saves
structured knowledge to data/interim/knowledge_base/.

Outputs
-------
knowledge_base/behavior_profiles.parquet   — per-account statistical profiles
knowledge_base/institution_mapping.json    — institution → branches → cities
knowledge_base/branch_mapping.json         — branch → city + institution
knowledge_base/country_mapping.json        — country frequencies + risk scores
knowledge_base/graph_statistics.json       — graph-level metrics (degree, PageRank, etc.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class KnowledgeExtractor:
    """Extract and persist banking domain knowledge from cleaned data.

    Parameters
    ----------
    project_root : Path
        Absolute path to the repository root.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.interim_dir = self.project_root / "data" / "interim"
        self.kb_dir = self.interim_dir / "knowledge_base"
        self.kb_dir.mkdir(parents=True, exist_ok=True)

        # Will be populated by extract()
        self.behavior_profiles: pd.DataFrame | None = None
        self.institution_mapping: dict = {}
        self.branch_mapping: dict = {}
        self.country_mapping: dict = {}
        self.graph_statistics: dict = {}
        self.amount_distribution: dict = {}
        self.temporal_distribution: dict = {}
        self.currency_distribution: dict = {}
        self.payment_type_distribution: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        transactions: pd.DataFrame,
        accounts: pd.DataFrame,
    ) -> "KnowledgeExtractor":
        """Run all extraction steps and populate internal state.

        Parameters
        ----------
        transactions : pd.DataFrame
            Canonical transactions (from ``data/interim/transactions.parquet``).
        accounts : pd.DataFrame
            Canonical accounts (from ``data/interim/accounts.parquet``).
        """
        logger.info("=== Phase 3: Banking Knowledge Extraction ===")
        logger.info("Transactions: %d rows | Accounts: %d rows", len(transactions), len(accounts))

        self._extract_institution_knowledge(accounts)
        self._extract_geographic_knowledge(transactions, accounts)
        self._extract_transaction_distributions(transactions)
        self._extract_customer_profiles(transactions, accounts)
        self._extract_graph_knowledge(transactions)

        logger.info("Knowledge extraction complete.")
        return self

    def save(self) -> None:
        """Persist all extracted knowledge to ``knowledge_base/``."""
        # Behavior profiles (parquet)
        if self.behavior_profiles is not None:
            self.behavior_profiles.to_parquet(
                self.kb_dir / "behavior_profiles.parquet", index=True
            )
            logger.info("Saved behavior_profiles.parquet — %d accounts", len(self.behavior_profiles))

        # JSON blobs
        _save_json(self.institution_mapping, self.kb_dir / "institution_mapping.json")
        _save_json(self.branch_mapping, self.kb_dir / "branch_mapping.json")
        _save_json(self.country_mapping, self.kb_dir / "country_mapping.json")
        _save_json(self.graph_statistics, self.kb_dir / "graph_statistics.json")
        _save_json(self.amount_distribution, self.kb_dir / "amount_distribution.json")
        _save_json(self.temporal_distribution, self.kb_dir / "temporal_distribution.json")
        _save_json(self.currency_distribution, self.kb_dir / "currency_distribution.json")
        _save_json(self.payment_type_distribution, self.kb_dir / "payment_type_distribution.json")

        logger.info("All knowledge saved to: %s", self.kb_dir)

    # ------------------------------------------------------------------
    # Institution Knowledge
    # ------------------------------------------------------------------

    def _extract_institution_knowledge(self, accounts: pd.DataFrame) -> None:
        """Build institution → branch → city hierarchy from accounts."""
        logger.info("Extracting institution knowledge...")

        raw_accounts = _try_load_raw_accounts(self.project_root)
        if raw_accounts is None:
            logger.warning("Raw accounts.csv not found; skipping raw institution extraction")
            return

        inst_map: dict[str, dict[str, Any]] = {}
        branch_map: dict[str, dict[str, Any]] = {}

        for _, row in raw_accounts.iterrows():
            inst = str(row.get("institution", "UNKNOWN")).strip()
            branch = str(row.get("branch", "UNKNOWN")).strip()
            city = str(row.get("city", "Kathmandu")).strip()

            if inst not in inst_map:
                inst_map[inst] = {"branches": {}, "frequency": 0}
            inst_map[inst]["frequency"] += 1

            if branch not in inst_map[inst]["branches"]:
                inst_map[inst]["branches"][branch] = {"city": city, "count": 0}
            inst_map[inst]["branches"][branch]["count"] += 1

            branch_map[branch] = {"institution": inst, "city": city}

        # Compute institution weights (for sampling)
        total = sum(v["frequency"] for v in inst_map.values())
        for inst in inst_map:
            inst_map[inst]["weight"] = inst_map[inst]["frequency"] / total

        self.institution_mapping = inst_map
        self.branch_mapping = branch_map
        logger.info("Learned %d institutions, %d branches", len(inst_map), len(branch_map))

    # ------------------------------------------------------------------
    # Geographic Knowledge
    # ------------------------------------------------------------------

    def _extract_geographic_knowledge(
        self, transactions: pd.DataFrame, accounts: pd.DataFrame
    ) -> None:
        """Learn country/city frequency and risk mappings."""
        logger.info("Extracting geographic knowledge...")

        sender_col = _col_pick(transactions, ["sender_country", "Sender_bank_location"])
        recv_col = _col_pick(transactions, ["receiver_country", "Receiver_bank_location"])

        all_countries: list[str] = []
        if sender_col:
            all_countries += transactions[sender_col].dropna().tolist()
        if recv_col:
            all_countries += transactions[recv_col].dropna().tolist()

        country_freq = _value_counts_dict(all_countries)

        # City distribution from accounts
        city_col = _col_pick(accounts, ["city"])
        city_freq: dict[str, int] = {}
        if city_col:
            city_freq = accounts[city_col].dropna().value_counts().to_dict()

        # Cross-border rate
        cb_col = _col_pick(transactions, ["is_cross_border", "cross_border_flag"])
        cross_border_rate = float(transactions[cb_col].mean()) if cb_col else 0.15

        # Country risk index (from nepal_context)
        from src.utils.nepal_context import REMITTANCE_CORRIDORS, CORRIDOR_RISK_SCORES
        country_risk: dict[str, float] = {}
        for corridor, tier in REMITTANCE_CORRIDORS.items():
            country = corridor.split("->")[0]
            country_risk[country] = CORRIDOR_RISK_SCORES[tier]
        country_risk["Nepal"] = 0.1  # Domestic = low risk

        self.country_mapping = {
            "country_frequencies": country_freq,
            "city_frequencies": {str(k): int(v) for k, v in city_freq.items()},
            "cross_border_rate": cross_border_rate,
            "country_risk": country_risk,
        }
        logger.info("Learned %d countries, %d cities", len(country_freq), len(city_freq))

    # ------------------------------------------------------------------
    # Transaction Distributions
    # ------------------------------------------------------------------

    def _extract_transaction_distributions(self, transactions: pd.DataFrame) -> None:
        """Fit distributions to amount, time, currency, payment type."""
        logger.info("Extracting transaction distributions...")

        amount_col = _col_pick(transactions, ["amount_npr", "amount_local_npr", "Amount"])
        if amount_col:
            amounts = transactions[amount_col].dropna().clip(lower=1.0)
            log_amounts = np.log(amounts)
            quantiles = np.quantile(amounts, np.arange(0.05, 1.0, 0.05)).tolist()
            self.amount_distribution = {
                "log_mean": float(log_amounts.mean()),
                "log_std": float(log_amounts.std()),
                "min": float(amounts.min()),
                "max": float(amounts.max()),
                "quantiles_05_to_95": quantiles,
                "p25": float(np.percentile(amounts, 25)),
                "p50": float(np.percentile(amounts, 50)),
                "p75": float(np.percentile(amounts, 75)),
                "p90": float(np.percentile(amounts, 90)),
                "p95": float(np.percentile(amounts, 95)),
                "p99": float(np.percentile(amounts, 99)),
            }

        # Temporal distributions
        hour_col = _col_pick(transactions, ["hour_of_day"])
        dow_col = _col_pick(transactions, ["day_of_week"])
        month_col = _col_pick(transactions, ["month"])

        self.temporal_distribution = {}
        if hour_col:
            hc = transactions[hour_col].value_counts().sort_index()
            total_h = hc.sum()
            self.temporal_distribution["hour_weights"] = {
                str(int(h)): float(c / total_h) for h, c in hc.items()
            }
        if dow_col:
            dc = transactions[dow_col].value_counts().sort_index()
            total_d = dc.sum()
            self.temporal_distribution["dow_weights"] = {
                str(int(d)): float(c / total_d) for d, c in dc.items()
            }
        if month_col:
            mc = transactions[month_col].value_counts().sort_index()
            total_m = mc.sum()
            self.temporal_distribution["month_weights"] = {
                str(int(m)): float(c / total_m) for m, c in mc.items()
            }

        # Currency distribution
        curr_col = _col_pick(transactions, ["original_currency", "Payment_currency"])
        if curr_col:
            cc = transactions[curr_col].value_counts()
            total_c = cc.sum()
            self.currency_distribution = {str(k): float(v / total_c) for k, v in cc.items()}

        # Payment type distribution
        pt_col = _col_pick(transactions, ["transaction_type", "Payment_type"])
        if pt_col:
            ptc = transactions[pt_col].value_counts()
            total_pt = ptc.sum()
            self.payment_type_distribution = {str(k): float(v / total_pt) for k, v in ptc.items()}

    # ------------------------------------------------------------------
    # Customer Profiles (Per-Account)
    # ------------------------------------------------------------------

    def _extract_customer_profiles(
        self, transactions: pd.DataFrame, accounts: pd.DataFrame
    ) -> None:
        """Compute per-account behavioral statistics."""
        logger.info("Extracting customer behavior profiles...")

        sender_col = _col_pick(transactions, ["sender_account_id", "Sender_account"])
        amount_col = _col_pick(transactions, ["amount_npr", "amount_local_npr", "Amount"])
        date_col = _col_pick(transactions, ["transaction_date", "Date"])
        hour_col = _col_pick(transactions, ["hour_of_day"])
        dow_col = _col_pick(transactions, ["day_of_week"])
        cb_col = _col_pick(transactions, ["is_cross_border", "cross_border_flag"])
        pt_col = _col_pick(transactions, ["transaction_type", "Payment_type"])

        if not sender_col or not amount_col:
            logger.warning("Missing sender/amount columns; skipping profile extraction")
            return

        tx = transactions.copy()
        tx["_sender"] = tx[sender_col].astype(str)
        tx["_amount"] = pd.to_numeric(tx[amount_col], errors="coerce").fillna(0)

        profiles: list[dict] = []

        for acct_id, grp in tx.groupby("_sender"):
            grp = grp.sort_values(date_col) if date_col in grp.columns else grp
            amounts = grp["_amount"]

            profile: dict[str, Any] = {
                "account_id": str(acct_id),
                "tx_count": int(len(grp)),
                "amount_mean": float(amounts.mean()),
                "amount_median": float(amounts.median()),
                "amount_std": float(amounts.std(ddof=0)),
                "amount_min": float(amounts.min()),
                "amount_max": float(amounts.max()),
            }

            # Temporal preferences
            if hour_col in grp.columns:
                vc = grp[hour_col].value_counts()
                profile["preferred_hour"] = int(vc.idxmax()) if len(vc) > 0 else 10
                profile["hour_weights"] = {str(int(k)): float(v / len(grp)) for k, v in vc.items()}
            if dow_col in grp.columns:
                vc = grp[dow_col].value_counts()
                profile["preferred_dow"] = int(vc.idxmax()) if len(vc) > 0 else 1
            if pt_col in grp.columns:
                vc = grp[pt_col].value_counts()
                profile["preferred_payment_type"] = str(vc.idxmax()) if len(vc) > 0 else "transfer"
                profile["payment_type_weights"] = {str(k): float(v / len(grp)) for k, v in vc.items()}

            # Cross-border frequency
            if cb_col in grp.columns:
                profile["cross_border_rate"] = float(grp[cb_col].mean())
            else:
                profile["cross_border_rate"] = 0.0

            # Daily frequency
            if date_col in grp.columns:
                dates = pd.to_datetime(grp[date_col], errors="coerce").dropna()
                if len(dates) > 1:
                    date_range = (dates.max() - dates.min()).days + 1
                    profile["daily_tx_frequency"] = float(len(grp) / max(date_range, 1))
                    # Average interval between transactions
                    diffs = dates.sort_values().diff().dropna()
                    profile["avg_interval_hours"] = float(diffs.dt.total_seconds().mean() / 3600)
                else:
                    profile["daily_tx_frequency"] = 1.0
                    profile["avg_interval_hours"] = 24.0

            profiles.append(profile)

        self.behavior_profiles = pd.DataFrame(profiles).set_index("account_id")
        logger.info("Built profiles for %d accounts", len(self.behavior_profiles))

    # ------------------------------------------------------------------
    # Graph Knowledge
    # ------------------------------------------------------------------

    def _extract_graph_knowledge(self, transactions: pd.DataFrame) -> None:
        """Build transaction graph and compute structural metrics."""
        logger.info("Extracting graph knowledge (building transaction graph)...")

        sender_col = _col_pick(transactions, ["sender_account_id", "Sender_account"])
        recv_col = _col_pick(transactions, ["receiver_account_id", "Receiver_account"])
        amount_col = _col_pick(transactions, ["amount_npr", "amount_local_npr", "Amount"])

        if not sender_col or not recv_col:
            logger.warning("Missing sender/receiver columns; skipping graph extraction")
            return

        G = nx.DiGraph()

        for _, row in transactions.iterrows():
            src = str(row[sender_col])
            dst = str(row[recv_col])
            amt = float(row[amount_col]) if amount_col else 1.0
            if G.has_edge(src, dst):
                G[src][dst]["weight"] += amt
                G[src][dst]["count"] += 1
            else:
                G.add_edge(src, dst, weight=amt, count=1)

        # Compute node-level metrics on a sample (full graph may be large)
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        logger.info("Graph: %d nodes, %d edges", n_nodes, n_edges)

        # Degree distribution
        out_degrees = [d for _, d in G.out_degree()]
        in_degrees = [d for _, d in G.in_degree()]

        # Top fan-out accounts (one sender → many receivers)
        top_fanout = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:20]
        # Top fan-in accounts (many senders → one receiver)
        top_fanin = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:20]

        # PageRank (top 20 nodes)
        try:
            pr = nx.pagerank(G, max_iter=100, tol=1e-4)
            top_pagerank = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:20]
        except Exception:
            top_pagerank = []

        # Communities via weakly connected components (Louvain not in std networkx)
        try:
            undirected = G.to_undirected()
            communities = list(nx.connected_components(undirected))
            community_sizes = sorted([len(c) for c in communities], reverse=True)
        except Exception:
            community_sizes = []

        self.graph_statistics = {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "avg_out_degree": float(np.mean(out_degrees)) if out_degrees else 0.0,
            "max_out_degree": int(max(out_degrees)) if out_degrees else 0,
            "avg_in_degree": float(np.mean(in_degrees)) if in_degrees else 0.0,
            "max_in_degree": int(max(in_degrees)) if in_degrees else 0,
            "out_degree_p50": float(np.percentile(out_degrees, 50)) if out_degrees else 0.0,
            "out_degree_p90": float(np.percentile(out_degrees, 90)) if out_degrees else 0.0,
            "out_degree_p99": float(np.percentile(out_degrees, 99)) if out_degrees else 0.0,
            "top_fanout_accounts": [[str(a), int(d)] for a, d in top_fanout],
            "top_fanin_accounts": [[str(a), int(d)] for a, d in top_fanin],
            "top_pagerank_accounts": [[str(a), float(p)] for a, p in top_pagerank],
            "n_communities": len(communities) if community_sizes else 0,
            "top10_community_sizes": community_sizes[:10],
        }
        logger.info(
            "Graph stats: avg_out_degree=%.2f, max_fan_out=%d, n_communities=%d",
            self.graph_statistics["avg_out_degree"],
            self.graph_statistics["max_out_degree"],
            self.graph_statistics["n_communities"],
        )


# ------------------------------------------------------------------
# Convenience: load knowledge base back from disk
# ------------------------------------------------------------------

def load_knowledge_base(project_root: Path) -> dict:
    """Load all knowledge base files from disk into a single dict.

    Returns
    -------
    dict with keys: institution_mapping, branch_mapping, country_mapping,
    graph_statistics, amount_distribution, temporal_distribution,
    currency_distribution, payment_type_distribution, behavior_profiles.
    """
    kb_dir = Path(project_root) / "data" / "interim" / "knowledge_base"
    knowledge: dict = {}

    json_files = [
        "institution_mapping",
        "branch_mapping",
        "country_mapping",
        "graph_statistics",
        "amount_distribution",
        "temporal_distribution",
        "currency_distribution",
        "payment_type_distribution",
    ]
    for name in json_files:
        path = kb_dir / f"{name}.json"
        if path.exists():
            with open(path) as f:
                knowledge[name] = json.load(f)
        else:
            knowledge[name] = {}

    profiles_path = kb_dir / "behavior_profiles.parquet"
    if profiles_path.exists():
        knowledge["behavior_profiles"] = pd.read_parquet(profiles_path)
    else:
        knowledge["behavior_profiles"] = pd.DataFrame()

    return knowledge


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _col_pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name found in df.columns."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _value_counts_dict(items: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for x in items:
        k = str(x)
        counts[k] = counts.get(k, 0) + 1
    return counts


def _save_json(obj: Any, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    logger.info("Saved %s", path.name)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def _try_load_raw_accounts(project_root: Path) -> pd.DataFrame | None:
    path = project_root / "data" / "original" / "accounts.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    tx = pd.read_parquet(root / "data" / "interim" / "transactions.parquet")
    acc = pd.read_parquet(root / "data" / "interim" / "accounts.parquet")

    extractor = KnowledgeExtractor(root)
    extractor.extract(tx, acc)
    extractor.save()
