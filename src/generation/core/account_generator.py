"""
src/generation/account_generator.py
-------------------------------------
Phase 4 — Synthetic Account Generation.

Generates realistic synthetic bank accounts using the knowledge base
extracted in Phase 3. Each generated account represents a new customer
entering the banking system.

Outputs
-------
data/interim/synthetic_accounts.csv
"""

from __future__ import annotations

import logging
import random
import string
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.nepal_context import NEPALI_CITIES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ACCOUNT_TYPES = ["savings", "current", "salary", "fixed_deposit"]
VALID_RISK_GRADES = ["low", "medium", "high"]

# Realistic default frequencies when knowledge base is absent
_DEFAULT_ACCT_TYPE_WEIGHTS = {
    "savings": 0.55,
    "current": 0.25,
    "salary": 0.15,
    "fixed_deposit": 0.05,
}
_DEFAULT_RISK_WEIGHTS = {"low": 0.75, "medium": 0.20, "high": 0.05}

# Probability of PEP flag / sanctions hit (realistic low base rates)
_PEP_RATE = 0.002        # 0.2%
_SANCTIONS_RATE = 0.0005  # 0.05%
_MULE_RATE = 0.02         # 2% of accounts are injected mules

# Realistic account opening date range (before transaction period)
_OPEN_DATE_START = pd.Timestamp("2010-01-01")
_OPEN_DATE_END = pd.Timestamp("2022-09-30")

# Reference date for account_age_days calculation
_REF_DATE = pd.Timestamp("2024-01-01")


class AccountGenerator:
    """Generate synthetic bank accounts using learned banking knowledge.

    Parameters
    ----------
    knowledge : dict
        Output of ``knowledge_extractor.load_knowledge_base()``.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, knowledge: dict, seed: int = 42) -> None:
        self.knowledge = knowledge
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        # Parse institution / branch / city structure from knowledge base
        self._institutions: list[str] = []
        self._institution_weights: list[float] = []
        self._inst_branch_city: dict[str, list[tuple[str, str]]] = {}  # inst → [(branch, city)]
        self._cities: list[str] = list(NEPALI_CITIES)

        self._parse_institution_knowledge()

        # Learnt account type / risk distributions (fall back to defaults)
        self._acct_type_weights = _DEFAULT_ACCT_TYPE_WEIGHTS.copy()
        self._risk_weights = _DEFAULT_RISK_WEIGHTS.copy()

        # Track generated IDs to guarantee uniqueness
        self._used_account_ids: set[int] = set()
        self._used_account_numbers: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n: int) -> pd.DataFrame:
        """Generate *n* synthetic accounts.

        Parameters
        ----------
        n : int
            Number of accounts to generate.

        Returns
        -------
        pd.DataFrame with canonical account columns.
        """
        logger.info("Generating %d synthetic accounts...", n)
        records: list[dict] = []

        for i in range(n):
            record = self._generate_one(is_mule=(i < int(n * _MULE_RATE)))
            records.append(record)
            if (i + 1) % 10_000 == 0:
                logger.info("  Generated %d / %d accounts", i + 1, n)

        df = pd.DataFrame(records)
        logger.info("Account generation complete: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Single account generation
    # ------------------------------------------------------------------

    def _generate_one(self, is_mule: bool = False) -> dict[str, Any]:
        # Institution + branch + city (must be internally consistent)
        inst, branch, city = self._sample_institution_branch_city()

        # Account type
        acct_type = self._sample_categorical(VALID_ACCOUNT_TYPES, self._acct_type_weights)

        # Risk grade: mule accounts are elevated
        if is_mule:
            risk_grade = self.rng.choice(["medium", "high"], p=[0.4, 0.6])
        else:
            risk_grade = self._sample_categorical(VALID_RISK_GRADES, self._risk_weights)

        # PEP / sanctions: mule accounts have higher rates
        pep_rate = 0.05 if is_mule else _PEP_RATE
        sanc_rate = 0.01 if is_mule else _SANCTIONS_RATE
        pep_flag = int(self.rng.random() < pep_rate)
        sanctions_hit = int(self.rng.random() < sanc_rate)

        # KYC: mule accounts sometimes unverified
        kyc_verified = int(self.rng.random() > (0.3 if is_mule else 0.02))

        # Opening date: mule accounts often newer
        if is_mule:
            open_start = pd.Timestamp("2020-01-01")
        else:
            open_start = _OPEN_DATE_START
        opened = self._random_date(open_start, _OPEN_DATE_END)

        account_age_days = int((_REF_DATE - pd.Timestamp(opened)).days)

        # Unique IDs
        account_id = self._unique_account_id()
        account_number = self._unique_account_number()

        # Name & tax number
        name = self._fake_name()
        tax_number = str(self.rng.integers(100_000_000, 999_999_999))

        # is_person: corporations less common
        is_person = int(self.rng.random() < 0.88)

        return {
            "account_id": account_id,
            "account_number": account_number,
            "institution": inst,
            "branch": branch,
            "account_type": acct_type,
            "risk_grade": risk_grade,
            "is_person": is_person,
            "name": name,
            "tax_number": tax_number,
            "pep_flag": pep_flag,
            "sanctions_hit": sanctions_hit,
            "city": city,
            "opened": opened,
            "kyc_verified": kyc_verified,
            "account_age_days": account_age_days,
            "is_mule": int(is_mule),
        }

    # ------------------------------------------------------------------
    # Institution / Branch / City sampling
    # ------------------------------------------------------------------

    def _parse_institution_knowledge(self) -> None:
        inst_map: dict = self.knowledge.get("institution_mapping", {})
        if not inst_map:
            # Fall back: use a hardcoded list of known Nepali banks
            self._institutions = [
                "HBL", "NCC", "ADBL", "CITIZENS", "GLOBAL_IME",
                "NIC_ASIA", "PRABHU", "SANIMA", "LAXMI", "NABIL",
            ]
            n = len(self._institutions)
            self._institution_weights = [1.0 / n] * n
            for inst in self._institutions:
                self._inst_branch_city[inst] = [
                    (f"BR_{str(i).zfill(3)}", city)
                    for i, city in enumerate(NEPALI_CITIES)
                ]
            return

        for inst, data in inst_map.items():
            self._institutions.append(inst)
            self._institution_weights.append(data.get("weight", 0.1))
            branches = data.get("branches", {})
            self._inst_branch_city[inst] = [
                (br, info.get("city", "Kathmandu"))
                for br, info in branches.items()
            ] or [("BR_001", "Kathmandu")]

        # Normalize weights
        total = sum(self._institution_weights)
        if total > 0:
            self._institution_weights = [w / total for w in self._institution_weights]
        else:
            n = len(self._institutions)
            self._institution_weights = [1.0 / n] * n

    def _sample_institution_branch_city(self) -> tuple[str, str, str]:
        inst = self.rng.choice(
            self._institutions,
            p=self._institution_weights,
        )
        branches = self._inst_branch_city.get(inst, [("BR_001", "Kathmandu")])
        branch, city = branches[self.rng.integers(len(branches))]
        return str(inst), str(branch), str(city)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_categorical(self, choices: list[str], weights: dict | list) -> str:
        if isinstance(weights, dict):
            keys = [k for k in choices if k in weights]
            probs = [weights[k] for k in keys]
            total = sum(probs)
            probs = [p / total for p in probs]
            idx = self.rng.choice(len(keys), p=probs)
            return keys[idx]
        else:
            probs = np.array(weights, dtype=float)
            probs /= probs.sum()
            idx = self.rng.choice(len(choices), p=probs)
            return choices[idx]

    def _random_date(self, start: pd.Timestamp, end: pd.Timestamp) -> str:
        delta_days = (end - start).days
        offset = int(self.rng.integers(0, max(delta_days, 1)))
        dt = start + pd.Timedelta(days=offset)
        return dt.strftime("%Y-%m-%d")

    def _unique_account_id(self) -> int:
        while True:
            aid = int(self.rng.integers(1_000_000_000, 9_999_999_999))
            if aid not in self._used_account_ids:
                self._used_account_ids.add(aid)
                return aid

    def _unique_account_number(self) -> str:
        while True:
            digits = "".join([str(self.rng.integers(0, 10)) for _ in range(20)])
            num = f"NP{digits}"
            if num not in self._used_account_numbers:
                self._used_account_numbers.add(num)
                return num

    def _fake_name(self) -> str:
        # Common Nepali first and last names
        first_names = [
            "Ram", "Sita", "Hari", "Gita", "Krishna", "Laxmi", "Bishnu",
            "Sarita", "Deepak", "Sunita", "Rajesh", "Anita", "Sunil", "Puja",
            "Binod", "Kamala", "Narayan", "Shanti", "Mohan", "Pratima",
            "Arun", "Sangita", "Mahesh", "Rekha", "Ganesh", "Rita",
        ]
        last_names = [
            "Sharma", "Thapa", "Gurung", "Tamang", "Rai", "Limbu", "Shrestha",
            "Adhikari", "Poudel", "Karki", "Bhandari", "Koirala", "Joshi",
            "Bhatta", "Regmi", "Khadka", "Oli", "Dahal", "Gautam", "Basnet",
        ]
        fn = first_names[self.rng.integers(len(first_names))]
        ln = last_names[self.rng.integers(len(last_names))]
        return f"{fn} {ln}"


# ------------------------------------------------------------------
# Convenience runner
# ------------------------------------------------------------------

def generate_accounts(
    knowledge: dict,
    n: int = 10_000,
    seed: int = 42,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Generate *n* synthetic accounts and optionally save to CSV.

    Parameters
    ----------
    knowledge : dict
        Loaded knowledge base from ``load_knowledge_base()``.
    n : int
        Number of accounts to generate.
    seed : int
        Random seed.
    output_path : Path, optional
        If provided, save the DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
    """
    gen = AccountGenerator(knowledge, seed=seed)
    df = gen.generate(n)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Accounts saved to: %s", output_path)
    return df


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.generation.core.knowledge_extractor import load_knowledge_base
    kb = load_knowledge_base(root)
    df = generate_accounts(kb, n=5000, output_path=root / "data" / "interim" / "synthetic_accounts.csv")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Mule accounts: {df['is_mule'].sum()} ({df['is_mule'].mean():.1%})")
