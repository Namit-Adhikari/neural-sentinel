# Explanation and evaluation design

This document records the explanation and evaluation decisions for this branch.
The data-generation and EDA work documented elsewhere remains outside this set of
changes.

## Design decisions

### SHAP plus deterministic templates

The explanation agent uses SHAP values when the fitted meta-learner exposes
them, then renders transaction facts and upstream reason codes through a
deterministic template. SHAP was selected over a free-form LLM explanation
because Lundberg and Lee define additive, locally faithful feature attribution
with desirable consistency properties in their NeurIPS paper: [A Unified
Approach to Interpreting Model Predictions](https://papers.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html).
The template is the audit layer: it remains available when the optional `shap`
dependency is absent and makes every sentence traceable to stored fields.

### PR-AUC, ROC-AUC, F1, Precision@k, and Brier score

The evaluator reports ROC-AUC for broad ranking comparison, but also PR-AUC and
Precision@k because suspicious transactions are expected to be rare and the
operational queue is top-ranked. Saito and Rehmsmeier show why precision-recall
plots are more informative than ROC plots for imbalanced binary classification:
[The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating
Binary Classifiers on Imbalanced Datasets](https://journals.plos.org/plosone/article/citation?id=10.1371/journal.pone.0118432).

Brier score is included because a risk score is used as a probability-like
priority signal, not only as a ranking. Calibration is operationally important
when a bank chooses a review threshold; Guo et al. study calibration and show
why calibrated probability estimates matter in modern neural networks: [On
Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html).

### Optimal threshold selection

The evaluation code reports the threshold that maximizes F1 on the observed
scores instead of hard-coding 0.5. That keeps the metric aligned with the queue
review objective and avoids pretending that a fixed probability cut-point is
always the best operating choice.

### Interpretation framing

The explanations must stay within the suspicious-activity framing described in
`AGENTS.md`. They are operational triage notes, not criminal findings. That is
why the template says the alert is for review and explicitly avoids language
that would overstate what the score means.

## Limitations

- The explanation agent is deterministic by design, so it is auditable but not
  free-form.
- SHAP is optional and falls back cleanly when unavailable.
- The evaluator is only as strong as the label quality in the input data.
- The notebook should stay self-contained and import reusable logic from `src/`
  rather than duplicating implementation details.
