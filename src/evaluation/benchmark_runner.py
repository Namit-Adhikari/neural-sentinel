"""
src/evaluation/benchmark_runner.py
-------------------------------------
End-to-end synthesizer benchmark runner.

Implements the full evaluation pipeline from Benchmark_pipeline.md, which
follows the CTAB-GAN+ paper methodology (Tables 3/4 format).

Workflow per synthesizer
------------------------
1.  Load real dataset
2.  Detect feature types (continuous / categorical)
3.  Stratified 80/20 train/test split
4.  Fit synthesizer on train set
5.  Generate synthetic set (size = train set size)
6.  Evaluate:
    a. ML Utility  — 5 classifiers, Accuracy / F1 / AUC
    b. Avg JSD     — Jensen-Shannon divergence (categorical)
    c. Avg WD      — Wasserstein distance (continuous)
    d. Diff Corr   — Frobenius norm of correlation matrix difference
7.  Repeat N seeds, report mean ± std
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.evaluation.ml_utility import compute_full_utility
from src.evaluation.statistical_metrics import (
    compute_avg_jsd,
    compute_avg_wasserstein,
    compute_correlation_distance,
)
from src.generation.synthesizers.base_generator import BaseGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SynthesizerResult:
    """Aggregated results for one synthesizer across multiple seeds."""
    synthesizer_name: str
    seeds_run: list[int] = field(default_factory=list)
    raw: list[dict] = field(default_factory=list)  # one dict per seed

    def aggregate(self) -> dict[str, str]:
        """Return mean ± std for every metric.

        Returns
        -------
        dict mapping metric_name → "mean ± std" string.
        """
        if not self.raw:
            return {}

        all_keys = set().union(*[r.keys() for r in self.raw])
        agg: dict[str, str] = {}
        for key in sorted(all_keys):
            vals = [r[key] for r in self.raw if key in r and r[key] is not None]
            numeric = []
            for v in vals:
                try:
                    numeric.append(float(v))
                except (TypeError, ValueError):
                    pass
            if numeric:
                mean = np.mean(numeric)
                std  = np.std(numeric, ddof=0)
                agg[key] = f"{mean:.4f} ± {std:.4f}"
            else:
                agg[key] = "N/A"
        return agg


# ---------------------------------------------------------------------------
# Feature type detection
# ---------------------------------------------------------------------------

def detect_feature_types(
    df: pd.DataFrame,
    target_col: str,
    categorical_threshold: int = 20,
) -> tuple[list[str], list[str]]:
    """Auto-detect continuous and categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_col : str
        Target column (excluded from feature lists).
    categorical_threshold : int
        If a numeric column has fewer unique values than this, treat as categorical.

    Returns
    -------
    (continuous_cols, categorical_cols)
    """
    continuous, categorical = [], []
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            categorical.append(col)
        elif df[col].nunique() <= categorical_threshold:
            categorical.append(col)
        else:
            continuous.append(col)
    return continuous, categorical


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Run the full synthesizer benchmark pipeline.

    Parameters
    ----------
    real_data : pd.DataFrame
        The original real dataset (must contain target_col).
    target_col : str
        Binary classification target column.
    seeds : list[int]
        Random seeds for repeated experiments (default: [0, 1, 2]).
    test_size : float
        Fraction of real data held out for evaluation.
    output_dir : Path, optional
        Directory to save per-synthesizer outputs.
    """

    def __init__(
        self,
        real_data: pd.DataFrame,
        target_col: str,
        seeds: list[int] | None = None,
        test_size: float = 0.2,
        output_dir: Path | None = None,
    ) -> None:
        self.real_data  = real_data.copy()
        self.target_col = target_col
        self.seeds      = seeds or [42, 0, 1]
        self.test_size  = test_size
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Detect feature types once
        self.continuous_cols, self.categorical_cols = detect_feature_types(
            real_data, target_col
        )
        self.feature_cols = self.continuous_cols + self.categorical_cols
        logger.info(
            "BenchmarkRunner: %d rows, target=%s, %d continuous, %d categorical",
            len(real_data), target_col, len(self.continuous_cols), len(self.categorical_cols),
        )

    def run(self, synthesizers: dict[str, BaseGenerator]) -> dict[str, SynthesizerResult]:
        """Run benchmark for all synthesizers.

        Parameters
        ----------
        synthesizers : dict mapping name → fitted or unfitted BaseGenerator instance.
            The runner will call .fit() on each generator per seed.

        Returns
        -------
        dict mapping synthesizer_name → SynthesizerResult
        """
        results: dict[str, SynthesizerResult] = {}

        for name, generator in synthesizers.items():
            logger.info("=" * 60)
            logger.info("Benchmarking: %s", name)
            result = SynthesizerResult(synthesizer_name=name)

            for seed in self.seeds:
                logger.info("  Seed %d ...", seed)
                seed_metrics = self._run_one_seed(generator, name, seed)
                if seed_metrics:
                    result.raw.append(seed_metrics)
                    result.seeds_run.append(seed)

            results[name] = result
            logger.info("  %s complete: %d seeds run", name, len(result.seeds_run))

            # Save per-synthesizer outputs
            if self.output_dir:
                self._save_synthesizer_output(name, result)

        return results

    def _run_one_seed(
        self, generator: BaseGenerator, name: str, seed: int
    ) -> dict[str, Any] | None:
        """Run one complete fit→sample→evaluate cycle."""
        t0 = time.time()
        try:
            # 1. Stratified train/test split
            target = self.real_data[self.target_col]
            X = self.real_data.drop(columns=[self.target_col])
            X_train, X_test, y_train, y_test = train_test_split(
                X, target,
                test_size=self.test_size,
                random_state=seed,
                stratify=target if target.nunique() > 1 else None,
            )
            train_df = X_train.copy()
            train_df[self.target_col] = y_train
            test_df = X_test.copy()
            test_df[self.target_col] = y_test

            n_train = len(train_df)

            # 2. Fit synthesizer on training set
            logger.info("    Fitting %s on %d rows...", name, n_train)
            generator.fit(train_df)

            # 3. Generate synthetic data (same size as training set)
            logger.info("    Sampling %d synthetic rows...", n_train)
            synthetic = generator.generate(n_train)

            # Align columns
            synthetic = _align_columns(synthetic, train_df)

            # 4. ML Utility (5-model suite)
            logger.info("    Computing ML utility...")
            utility = compute_full_utility(
                real_data=self.real_data,
                synthetic_data=synthetic,
                target_col=self.target_col,
                feature_cols=self.feature_cols,
                seed=seed,
            )

            # 5. Statistical similarity
            logger.info("    Computing statistical metrics...")
            avg_jsd = compute_avg_jsd(
                train_df, synthetic, self.categorical_cols
            )
            avg_wd = compute_avg_wasserstein(
                train_df, synthetic, self.continuous_cols
            )
            diff_corr = compute_correlation_distance(
                train_df, synthetic,
                self.continuous_cols,
                self.categorical_cols,
            )

            elapsed = time.time() - t0
            metrics: dict[str, Any] = {
                "seed":        seed,
                "elapsed_s":   round(elapsed, 1),
                # ML utility (averages over 5 models)
                "avg_accuracy_diff": utility.get("avg_accuracy"),
                "avg_f1_diff":       utility.get("avg_f1"),
                "avg_auc_diff":      utility.get("avg_auc"),
                "real_auc":          utility.get("real_auc"),
                "synthetic_auc":     utility.get("synthetic_auc"),
                # Statistical
                "avg_jsd":    avg_jsd,
                "avg_wd":     avg_wd,
                "diff_corr":  diff_corr,
            }
            logger.info(
                "    Seed %d — AUC diff=%.4f, JSD=%.4f, WD=%.4f, DiffCorr=%.4f (%.1fs)",
                seed, metrics["avg_auc_diff"] or 0,
                avg_jsd, avg_wd, diff_corr, elapsed,
            )
            return metrics

        except Exception as e:
            logger.error("  Seed %d FAILED for %s: %s", seed, name, e)
            logger.debug(traceback.format_exc())
            return None

    def _save_synthesizer_output(self, name: str, result: SynthesizerResult) -> None:
        """Save metrics.json and aggregated results for one synthesizer."""
        synth_dir = self.output_dir / name
        synth_dir.mkdir(exist_ok=True)

        # Raw seed results
        with open(synth_dir / "metrics.json", "w") as f:
            json.dump({"synthesizer": name, "seeds": result.raw}, f, indent=2, default=str)

        # Aggregated
        agg = result.aggregate()
        with open(synth_dir / "results_aggregated.json", "w") as f:
            json.dump({"synthesizer": name, "aggregated": agg}, f, indent=2)

        logger.info("  Saved outputs to %s/", name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _align_columns(synthetic: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Align synthetic columns to match reference (fill missing with 0)."""
    for col in reference.columns:
        if col not in synthetic.columns:
            synthetic[col] = 0
    return synthetic[[c for c in reference.columns if c in synthetic.columns]]
