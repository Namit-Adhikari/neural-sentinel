"""
src/cleaning.py
---------------
Canonical data cleaning and preprocessing for Neural Sentinel.

Produces:
  data/interim/transactions.parquet           — canonical schema (base)
  data/interim/accounts.parquet
  data/interim/schema2_location.parquet       — Schema 2: location consolidation
  data/interim/schema3_label_encoded.parquet  — Schema 3: label encoding
  data/interim/schema4_quantile_amount.parquet — Schema 4: quantile binning of amount
  data/interim/schema5_cbrt_amount.parquet    — Schema 5: cubic root transform of amount

Schema variants follow the methodology in the reference paper (TechRxiv preprint)
for generator benchmarking — each schema is a standalone copy of the canonical
dataframe with one targeted transformation applied, ready to feed into SDV/custom
generator pipelines.
"""

import os
import uuid
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root resolution (works locally and on Kaggle)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Lookup maps
# ---------------------------------------------------------------------------

ACCT_TYPE_MAP = {
    "FIXED":   "fixed_deposit",
    "SAVINGS": "savings",
    "CURRENT": "current",
    "NOSTRO":  "current",
    "SALARY":  "salary",
}

RISK_MAP = {
    "RISK-LOW":    "low",
    "RISK-MED":    "medium",
    "RISK-MEDIUM": "medium",
    "RISK-HIGH":   "high",
}

CURRENCY_MAP = {
    "UK pounds":   "GBP",
    "US Dollar":   "USD",
    "Euro":        "EUR",
    "Qatari Rial": "QAR",
    "Indian Rupee":"INR",
    "Saudi Riyal": "SAR",
    "UAE Dirham":  "AED",
    "NPR":         "NPR",
    "Nepalese Rupee": "NPR",
}

# transmode_code → canonical channel
# A=Cash, B=Branch cheque, E=Electronic/Online, F=SWIFT wire,
# J=Internal journal, P=POS, Z=ATM
TRANSMODE_CHANNEL_MAP = {
    "A": "branch",         # Cash at branch counter
    "B": "branch",         # Branch cheque
    "E": "online_banking", # Electronic / internet banking
    "F": "online_banking", # SWIFT wire (treated as online)
    "J": "mobile_banking", # Internal / mobile journal transfer
    "P": "pos",            # Point-of-sale
    "Z": "atm",            # ATM
}

# Payment_type → canonical transaction_type
PAYMENT_TYPE_MAP = {
    "Cash Deposit":     "deposit",
    "Cash Withdrawal":  "cash_out",
    "Cross-border":     "remittance_outbound",  # refined per sender/receiver below
    "Wire Transfer":    "transfer",
    "Payment":          "payment",
    "ATM":              "withdrawal",
    "POS":              "payment",
    "Internal":         "transfer",
}

# Known NRB remittance corridors (top corridors to/from Nepal)
KNOWN_CORRIDORS = {
    frozenset(["Qatar",  "Nepal"]): "Qatar->Nepal",
    frozenset(["India",  "Nepal"]): "India->Nepal",
    frozenset(["UAE",    "Nepal"]): "UAE->Nepal",
    frozenset(["Malaysia","Nepal"]): "Malaysia->Nepal",
    frozenset(["Saudi Arabia","Nepal"]): "Saudi Arabia->Nepal",
    frozenset(["USA",    "Nepal"]): "USA->Nepal",
    frozenset(["UK",     "Nepal"]): "UK->Nepal",
    frozenset(["Japan",  "Nepal"]): "Japan->Nepal",
    frozenset(["South Korea","Nepal"]): "South Korea->Nepal",
    frozenset(["Australia","Nepal"]): "Australia->Nepal",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_tx_type(row: pd.Series) -> str:
    """Derive canonical transaction_type from Payment_type + cross_border_flag."""
    ptype = str(row.get("Payment_type", "")).strip()
    is_cb = int(row.get("cross_border_flag", 0))
    sender_loc = str(row.get("Sender_bank_location", "Nepal"))
    recv_loc   = str(row.get("Receiver_bank_location", "Nepal"))

    if ptype == "Cash Withdrawal":
        return "cash_out"
    if ptype == "Cash Deposit":
        return "deposit"
    if ptype == "ATM":
        return "withdrawal"
    if ptype in ("POS", "Payment"):
        return "payment"
    if is_cb:
        if recv_loc.strip().lower() == "nepal":
            return "remittance_inbound"
        return "remittance_outbound"
    return PAYMENT_TYPE_MAP.get(ptype, "transfer")


def _get_corridor(sender_loc: str, recv_loc: str, is_cross_border: int) -> str | None:
    """Return a corridor string for cross-border transactions."""
    if not is_cross_border:
        return None
    s = sender_loc.strip()
    r = recv_loc.strip()
    key = frozenset([s, r])
    if key in KNOWN_CORRIDORS:
        return KNOWN_CORRIDORS[key]
    if s == r:
        return None  # Same-country — not truly cross-border
    return f"{s}->{r}"


def _derive_fraud_labels_from_xml(reports_dir: Path, n_rows: int) -> np.ndarray:
    """
    Parse all XML STR reports and return a boolean array indexed by row_index.
    Returns an integer ndarray of length n_rows with 1 for flagged rows.
    """
    flagged: set[int] = set()
    xml_files = list(reports_dir.glob("*.xml"))
    logger.info("Parsing %d XML STR reports for fraud label cross-check...", len(xml_files))
    for xf in xml_files:
        try:
            tree = ET.parse(xf)
            root = tree.getroot()
            for tx in root.findall(".//transaction"):
                ref = tx.findtext("internal_ref_number") or ""
                parts = ref.split("-")
                if len(parts) == 3:
                    try:
                        flagged.add(int(parts[2]))
                    except ValueError:
                        pass
        except ET.ParseError:
            logger.warning("Could not parse %s", xf.name)
    arr = np.zeros(n_rows, dtype=np.int8)
    valid = [i for i in flagged if 0 <= i < n_rows]
    arr[valid] = 1
    logger.info("XML cross-check: %d flagged row indices → %d valid", len(flagged), len(valid))
    return arr


# ---------------------------------------------------------------------------
# Accounts cleaning
# ---------------------------------------------------------------------------

def _clean_accounts(acc_df: pd.DataFrame, tx_df: pd.DataFrame) -> pd.DataFrame:
    """Map raw accounts CSV to canonical accounts schema."""
    out = pd.DataFrame()
    out["account_id"]       = acc_df["account_id"].astype(str)
    out["account_type"]     = acc_df["acct_type"].map(ACCT_TYPE_MAP).fillna("current")
    out["account_open_date"] = pd.to_datetime(acc_df["opened"], errors="coerce")

    ref_date = pd.Timestamp("2024-01-01")
    out["account_age_days"] = (ref_date - out["account_open_date"]).dt.days.clip(lower=0).fillna(0).astype(int)
    out["account_open_date"] = out["account_open_date"].dt.date

    out["kyc_verified"]   = 1
    out["kyc_risk_grade"] = acc_df["risk_grade"].map(RISK_MAP).fillna("medium")
    out["is_pep"]         = acc_df["pep_flag"].fillna(0).astype(int)
    out["is_sanctioned"]  = acc_df["sanctions_hit"].fillna(0).astype(int)

    # Derive monthly volume/count from transactions
    vol = (
        tx_df.groupby("Sender_account")["amount_local_npr"]
        .sum()
        .reset_index()
        .rename(columns={"Sender_account": "account_id", "amount_local_npr": "total_vol"})
    )
    cnt = (
        tx_df.groupby("Sender_account")
        .size()
        .reset_index(name="total_count")
        .rename(columns={"Sender_account": "account_id"})
    )
    vol["account_id"] = vol["account_id"].astype(str)
    cnt["account_id"] = cnt["account_id"].astype(str)

    out = out.merge(vol, on="account_id", how="left")
    out = out.merge(cnt, on="account_id", how="left")

    # Estimate monthly averages (dataset spans ~12 months of 2022)
    n_months = 12.0
    out["average_monthly_volume"] = (out["total_vol"].fillna(0) / n_months).round(2)
    out["average_monthly_count"]  = (out["total_count"].fillna(0) / n_months).astype(int)
    out.drop(columns=["total_vol", "total_count"], inplace=True)

    out["country"] = "Nepal"
    out["city"]    = acc_df["city"].fillna("Kathmandu")
    out["is_mule"] = 0

    return out


# ---------------------------------------------------------------------------
# Transactions cleaning (canonical base)
# ---------------------------------------------------------------------------

def _clean_transactions(
    tx_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    reports_dir: Path,
) -> pd.DataFrame:
    """Map raw transactions CSV + ml_features to canonical transactions schema."""
    n = len(tx_df)
    out = pd.DataFrame()

    # ── Identifiers ────────────────────────────────────────────────────────
    out["transaction_id"]       = [str(uuid.uuid4()) for _ in range(n)]
    out["transaction_date"]     = pd.to_datetime(tx_df["Date"], errors="coerce").dt.date
    out["transaction_time"]     = pd.to_datetime(tx_df["Time"], errors="coerce").dt.time
    out["sender_account_id"]    = tx_df["Sender_account"].astype(str)
    out["receiver_account_id"]  = tx_df["Receiver_account"].astype(str)

    # ── Transaction type ────────────────────────────────────────────────────
    out["transaction_type"] = tx_df.apply(_map_tx_type, axis=1)

    # ── Amount / currency ───────────────────────────────────────────────────
    out["amount_npr"]         = tx_df["amount_local_npr"].clip(lower=0.01)
    out["original_currency"]  = tx_df["Payment_currency"].map(CURRENCY_MAP).fillna("NPR")
    out["exchange_rate"]      = tx_df["fx_rate_to_npr"].clip(lower=0.01)

    # ── Channel (derived from transmode_code, not random) ───────────────────
    out["channel"] = tx_df["transmode_code"].map(TRANSMODE_CHANNEL_MAP).fillna("branch")

    # ── Geography ───────────────────────────────────────────────────────────
    out["sender_country"]   = tx_df["Sender_bank_location"].fillna("Nepal")
    out["receiver_country"] = tx_df["Receiver_bank_location"].fillna("Nepal")
    out["is_cross_border"]  = tx_df["cross_border_flag"].fillna(0).astype(int)

    out["remittance_corridor"] = tx_df.apply(
        lambda r: _get_corridor(
            r["Sender_bank_location"], r["Receiver_bank_location"], r["cross_border_flag"]
        ),
        axis=1,
    )

    # ── Device / IP (not in raw data — derive best approximation) ───────────
    # ATM and POS channels imply physical device; others are digital
    channel_device_map = {
        "atm":            "atm_card",
        "pos":            "pos_terminal",
        "mobile_banking": "mobile",
        "online_banking": "desktop",
        "branch":         "branch_terminal",
    }
    out["merchant_category"] = None
    out["device_type"]       = out["channel"].map(channel_device_map).fillna("unknown")
    out["ip_address"]        = None   # Not available in raw data
    out["ip_country"]        = out["sender_country"]
    out["ip_is_vpn"]         = 0

    # ── Fraud label (primary: ml_features.csv is_suspicious_tx) ─────────────
    if "is_suspicious_tx" in ml_df.columns:
        out["is_fraud"] = ml_df["is_suspicious_tx"].values.astype(int)
        n_fraud_primary = out["is_fraud"].sum()
        logger.info("Fraud label from ml_features.csv: %d / %d suspicious", n_fraud_primary, n)

        # Cross-validate against XML reports
        if reports_dir.exists():
            xml_labels = _derive_fraud_labels_from_xml(reports_dir, n)
            n_xml = xml_labels.sum()
            logger.info("XML cross-check flags %d rows (overlap: %d)",
                        n_xml,
                        int(np.logical_and(out["is_fraud"].values, xml_labels).sum()))
            # Union: flag any row marked by either source
            out["is_fraud"] = np.maximum(out["is_fraud"].values, xml_labels).astype(int)
            logger.info("After union: %d / %d rows flagged", out["is_fraud"].sum(), n)
    else:
        logger.warning("is_suspicious_tx not found in ml_features — falling back to XML only")
        reports_dir_path = reports_dir if reports_dir.exists() else Path("data/original/reports")
        out["is_fraud"] = _derive_fraud_labels_from_xml(reports_dir_path, n)

    # ── Fraud type (derive from transmode_comment in XMLs where possible) ───
    # For now use a rule-based assignment consistent with is_fraud
    def _assign_fraud_type(row):
        if not row["is_fraud"]:
            return None
        # Use available signals to assign a plausible fraud type
        if row["is_cross_border"] and row.get("amount_npr", 0) > 500_000:
            return "aml_layering"
        if row.get("amount_npr", 0) < 1_000_000 and row["is_cross_border"]:
            return "aml_structuring"
        return "transaction_fraud"

    out["fraud_type"]         = out.apply(_assign_fraud_type, axis=1)
    out["aml_risk_indicator"] = (
        out["fraud_type"].isin(["aml_layering", "aml_structuring", "aml_mule_network"])
    ).astype(int)

    # ── Preserve useful engineered features from raw data ───────────────────
    # These are used by agents but excluded from generator training columns
    out["hour_of_day"]              = tx_df["hour_of_day"].fillna(12).astype(int)
    out["day_of_week"]              = tx_df["day_of_week"].fillna(0).astype(int)
    out["is_weekend"]               = tx_df["is_weekend"].fillna(0).astype(int)
    out["month"]                    = tx_df["month"].fillna(1).astype(int)
    out["above_1M_NPR"]             = tx_df["above_1M_NPR"].fillna(0).astype(int)
    out["above_10M_NPR"]            = tx_df["above_10M_NPR"].fillna(0).astype(int)
    out["velocity_sum_10tx"]        = tx_df["velocity_sum_10tx"].fillna(0)
    out["tx_count_10"]              = tx_df["tx_count_10"].fillna(0)
    out["tx_count_30"]              = tx_df["tx_count_30"].fillna(0)
    out["currency_mismatch"]        = tx_df["currency_mismatch"].fillna(0).astype(int)
    out["sender_country_risk"]      = tx_df["sender_country_risk"].fillna(0.5)
    out["receiver_country_risk"]    = tx_df["receiver_country_risk"].fillna(0.5)
    out["sender_pep"]               = tx_df["sender_pep"].fillna(0).astype(int)
    out["sender_sanctions"]         = tx_df["sender_sanctions"].fillna(0).astype(int)
    out["receiver_pep"]             = tx_df["receiver_pep"].fillna(0).astype(int)
    out["receiver_sanctions"]       = tx_df["receiver_sanctions"].fillna(0).astype(int)
    out["sender_risk_grade"]        = tx_df["sender_risk_grade"].map(RISK_MAP).fillna("medium")
    out["sender_account_age_days"]  = tx_df["sender_account_age_days"].fillna(0).astype(int)
    out["receiver_account_age_days"]= tx_df["receiver_account_age_days"].fillna(0).astype(int)

    return out


# ---------------------------------------------------------------------------
# Schema variant builders (Schemas 2–5 from reference paper)
# ---------------------------------------------------------------------------

def build_schema2_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Schema 2: Location Consolidation.
    Merge sender_country, receiver_country, and ip_country into a single
    categorical `location` string to prevent the generator from creating
    non-existent location combinations.

    Format: "{sender_country}::{receiver_country}"
    """
    out = df.copy()
    out["location"] = (
        out["sender_country"].fillna("Unknown")
        + "::"
        + out["receiver_country"].fillna("Unknown")
    )
    # Drop the individual geography columns that are now encoded in `location`
    out.drop(columns=["sender_country", "receiver_country", "ip_country",
                       "remittance_corridor"], errors="ignore", inplace=True)
    # Mark as categorical for SDV metadata
    out["location"] = out["location"].astype("category")
    return out


def build_schema3_label_encoded(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Schema 3: Label Encoding of Categorical Columns.
    Apply sklearn LabelEncoder to categorical columns so SDV generators
    correctly treat them as categorical and cannot generate out-of-vocabulary values.

    Returns (encoded_df, encoders_dict) so inverse transform is possible.
    Encoded columns are stored as strings (not int) per the paper's guidance.
    """
    out = df.copy()
    categorical_cols = [
        "transaction_type",
        "channel",
        "original_currency",
        "sender_risk_grade",
        "fraud_type",       # nullable — fill None first
        "is_fraud",         # binary but treated as categorical for SDV
    ]
    encoders: dict[str, LabelEncoder] = {}

    for col in categorical_cols:
        if col not in out.columns:
            continue
        le = LabelEncoder()
        # Fill nulls with a sentinel before encoding
        filled = out[col].fillna("__NONE__").astype(str)
        le.fit(filled)
        # Store as string dtype (paper requirement: keeps SDV metadata categorical)
        out[col] = le.transform(filled).astype(str)
        encoders[col] = le

    return out, encoders


def build_schema4_quantile_amount(df: pd.DataFrame, n_bins: int = 10) -> tuple[pd.DataFrame, QuantileTransformer]:
    """
    Schema 4: Quantile Transformation of Amount.
    Log-transform amount_npr then bin into `n_bins` quantile bins stored as
    a string category. This addresses the extreme right-skew of transaction amounts.

    Returns (encoded_df, qt) so inverse transform is possible.
    """
    out = df.copy()
    # Log-transform first (avoids log(0) — amount is clipped ≥ 0.01)
    log_amount = np.log1p(out["amount_npr"].values.reshape(-1, 1))

    qt = QuantileTransformer(n_quantiles=n_bins, output_distribution="uniform", random_state=42)
    quantile_vals = qt.fit_transform(log_amount).flatten()

    # Bin into n_bins buckets and label as strings
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_labels = [f"Q{i+1}" for i in range(n_bins)]
    out["amount_npr"] = pd.cut(
        quantile_vals, bins=bin_edges, labels=bin_labels, include_lowest=True
    ).astype(str)

    return out, qt


def build_schema5_cbrt_amount(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Schema 5: Cubic Root Transformation of Amount.
    Standardise amount_npr to zero mean / unit variance then apply cube root.
    This is a softer non-Gaussianity correction than binning — less information
    loss, easier inverse transform.

    Returns (encoded_df, params) where params = {"mean": float, "std": float}.
    """
    out = df.copy()
    mu  = out["amount_npr"].mean()
    sig = out["amount_npr"].std()
    if sig == 0:
        sig = 1.0
    standardised = (out["amount_npr"] - mu) / sig
    out["amount_npr"] = np.cbrt(standardised)
    params = {"mean": mu, "std": sig}
    return out, params


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def clean_data(project_root: Path | None = None) -> None:
    """Run the full cleaning pipeline and save all outputs to data/interim/."""
    if project_root is None:
        project_root = PROJECT_ROOT

    data_dir    = project_root / "data" / "original"
    interim_dir = project_root / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    # ── Load raw files ──────────────────────────────────────────────────────
    logger.info("Loading raw data...")
    tx_df  = pd.read_csv(data_dir / "transactions.csv")
    acc_df = pd.read_csv(data_dir / "accounts.csv")
    ml_df  = pd.read_csv(data_dir / "ml_features.csv")

    logger.info("Raw transactions : %s", tx_df.shape)
    logger.info("Raw accounts     : %s", acc_df.shape)
    logger.info("ML features      : %s", ml_df.shape)

    # ── Basic quality checks ────────────────────────────────────────────────
    tx_df  = tx_df.drop_duplicates().dropna(how="all")
    acc_df = acc_df.drop_duplicates().dropna(how="all")
    ml_df  = ml_df.drop_duplicates().dropna(how="all")

    # Keep only positive-amount transactions
    tx_df = tx_df[tx_df["Amount"] > 0].reset_index(drop=True)
    ml_df = ml_df.iloc[: len(tx_df)].reset_index(drop=True)

    # ── Canonical accounts ──────────────────────────────────────────────────
    logger.info("Building canonical accounts table...")
    final_acc = _clean_accounts(acc_df, tx_df)

    # ── Canonical transactions (base schema) ────────────────────────────────
    logger.info("Building canonical transactions table...")
    reports_dir = data_dir / "reports"
    final_tx = _clean_transactions(tx_df, ml_df, reports_dir)

    # ── Save base canonical outputs ─────────────────────────────────────────
    logger.info("Saving base canonical parquets...")
    final_acc.to_parquet(interim_dir / "accounts.parquet", index=False)
    final_tx.to_parquet(interim_dir / "transactions.parquet", index=False)
    logger.info("✓ accounts.parquet   → %s rows", len(final_acc))
    logger.info("✓ transactions.parquet → %s rows", len(final_tx))

    # ── Schema 2: Location consolidation ───────────────────────────────────
    logger.info("Building Schema 2 (location consolidation)...")
    s2 = build_schema2_location(final_tx)
    s2.to_parquet(interim_dir / "schema2_location.parquet", index=False)
    logger.info("✓ schema2_location.parquet → %s cols", len(s2.columns))

    # ── Schema 3: Label encoding ────────────────────────────────────────────
    logger.info("Building Schema 3 (label encoding)...")
    s3, _encoders = build_schema3_label_encoded(final_tx)
    s3.to_parquet(interim_dir / "schema3_label_encoded.parquet", index=False)
    logger.info("✓ schema3_label_encoded.parquet")

    # ── Schema 4: Quantile amount binning ──────────────────────────────────
    logger.info("Building Schema 4 (quantile amount binning)...")
    s4, _qt = build_schema4_quantile_amount(final_tx, n_bins=10)
    s4.to_parquet(interim_dir / "schema4_quantile_amount.parquet", index=False)
    logger.info("✓ schema4_quantile_amount.parquet")

    # ── Schema 5: Cubic root transformation ────────────────────────────────
    logger.info("Building Schema 5 (cubic root amount transform)...")
    s5, _params = build_schema5_cbrt_amount(final_tx)
    s5.to_parquet(interim_dir / "schema5_cbrt_amount.parquet", index=False)
    logger.info("✓ schema5_cbrt_amount.parquet")

    logger.info("=" * 60)
    logger.info("Cleaning complete. Fraud rate: %.4f%%", final_tx["is_fraud"].mean() * 100)
    logger.info("Files written to: %s", interim_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    clean_data()
