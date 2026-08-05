"""
src/evaluation/report.py
--------------------------
Produce final comparison tables in CTAB-GAN+ paper format.

Outputs
-------
final_table.csv   — machine-readable results table
final_table.md    — markdown table matching Tables 3/4 layout from the paper
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.benchmark_runner import SynthesizerResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column display names (paper Table 3/4 layout)
# ---------------------------------------------------------------------------

_TABLE_COLUMNS = [
    ("avg_accuracy_diff", "Accuracy (%)"),
    ("avg_f1_diff",       "F1-score"),
    ("avg_auc_diff",      "AUC"),
    ("avg_jsd",           "Avg JSD"),
    ("avg_wd",            "Avg WD"),
    ("diff_corr",         "Diff. Corr."),
]


def build_final_table(
    results: dict[str, SynthesizerResult],
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Build and optionally save the final comparison table.

    Parameters
    ----------
    results : dict
        Mapping synthesizer_name → SynthesizerResult (output of BenchmarkRunner.run).
    output_dir : Path, optional
        If provided, saves ``final_table.csv`` and ``final_table.md`` here.

    Returns
    -------
    pd.DataFrame — the comparison table with "mean ± std" string values.
    """
    rows: list[dict[str, str]] = []

    for name, result in results.items():
        agg = result.aggregate()
        row: dict[str, Any] = {"Method": name}
        for metric_key, display_name in _TABLE_COLUMNS:
            row[display_name] = agg.get(metric_key, "N/A")
        rows.append(row)

    if not rows:
        logger.warning("No results to tabulate.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("Method")

    logger.info("\n%s", _format_markdown_table(df))

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "final_table.csv"
        df.to_csv(csv_path)
        logger.info("Saved final_table.csv → %s", csv_path)

        md_path = output_dir / "final_table.md"
        with open(md_path, "w") as f:
            f.write("# Synthesizer Benchmark — Final Results\n\n")
            f.write("> Lower values indicate better synthetic data quality.\n\n")
            f.write(_format_markdown_table(df))
            f.write("\n")
        logger.info("Saved final_table.md → %s", md_path)

    return df


def _format_markdown_table(df: pd.DataFrame) -> str:
    """Format a DataFrame as a GitHub-flavoured markdown table."""
    cols = ["Method"] + list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows_md = []
    for idx, row in df.iterrows():
        cells = [str(idx)] + [str(row[c]) for c in df.columns]
        rows_md.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator] + rows_md) + "\n"


def print_results_summary(results: dict[str, SynthesizerResult]) -> None:
    """Print a quick text summary of all benchmark results to the logger."""
    logger.info("\n%s", "=" * 70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("%s", "=" * 70)
    for name, result in results.items():
        agg = result.aggregate()
        logger.info(
            "  %-20s | AUC diff: %-15s | JSD: %-15s | WD: %-15s | Corr: %s",
            name,
            agg.get("avg_auc_diff", "N/A"),
            agg.get("avg_jsd", "N/A"),
            agg.get("avg_wd", "N/A"),
            agg.get("diff_corr", "N/A"),
        )
    logger.info("%s", "=" * 70)
