# Phase 1 EDA: research and design decisions

## Decision record

### Keep EDA source-faithful and defer canonicalization

The source data contains both operational inputs and model-engineered columns.
Phase 1 profiles both, but does not use the engineered columns as canonical
facts. This protects the project’s canonical data contract from feature
leakage and leaves all mappings and derivations auditable in Phase 2.

The primary sources are therefore `transactions.csv` and `accounts.csv`; the
edge list is used for structural profiling; and `ml_features.csv` is used only
to describe the supplied suspicious-activity label and to check alignment.

### Profile suspicious activity separately and avoid causal language

The source documentation labels `is_suspicious_tx`, and AGENTS.md defines the
project target as a suspicious-transaction flag rather than proven crime. The
notebooks use stratified descriptive summaries and explicitly avoid treating a
feature difference as proof of fraud. This is consistent with the risk-based
approach, where institutions assess and understand their specific risks and
apply proportionate mitigation rather than use a single universal rule
([FATF banking-sector guidance](https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Risk-based-approach-banking-sector.html)).

### Use a six-notebook, question-oriented EDA sequence

The repository contract requires six EDA notebooks. Each has one limited
question: structure, distributions, label associations, time, geography, and
AML/network indicators. They share a compact setup/loading pattern and do not
duplicate later production code. This makes them clean-kernel Kaggle runnable
and gives Phase 2 an auditable evidence trail.

### Treat the source as structural seed data, not Nepal ground truth

The supplied README identifies the dataset as fully synthetic, and the EDA
finds UK-heavy locations/currencies. The data is useful for validating a
pipeline and discovering schema gaps, but it cannot independently justify
claims about Nepal’s transaction distribution. Nepal-specific fields and
controls will be added only under the documented canonical/generation work,
not retroactively asserted by EDA.

## Citation-ready sources

1. Karst, F. S., Chong, S.-Y., Antenor, A. A., Lin, E., Li, M. M., &
   Leimeister, J. M. (2024). *Generative AI for Banks: Benchmarks and
   Algorithms for Synthetic Financial Transaction Data*. arXiv:2412.14730.
   https://doi.org/10.48550/arXiv.2412.14730
   ([open record](https://arxiv.org/abs/2412.14730)).

   Relevance: a directly related financial-transaction benchmark that compares
   CTGAN, TVAE, GAN, diffusion, and sequential approaches across fidelity,
   utility, privacy, efficiency, and graph structure. It supports benchmarking
   rather than selecting a generator by intuition in Phase 3. Its reported
   graph-structure limitation reinforces keeping graph typology injection as a
   distinct later pipeline step.

2. Financial Action Task Force. (2014). *Guidance for a Risk-Based Approach:
   The Banking Sector*. FATF.
   ([open guidance](https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Risk-based-approach-banking-sector.html)).

   Relevance: establishes the risk-based framing for prioritizing higher-risk
   activity and explains why the project reports calibrated risk evidence
   rather than a binary criminal conclusion.

3. Nepal Rastra Bank, Financial Intelligence Unit Nepal. (2026). *Annual
   Report 2024/25*.
   ([open PDF](https://www.nrb.org.np/contents/uploads/2026/05/Annual-Report-2024-25-English.pdf)).

   Relevance: current official Nepal context. The report documents FIU-Nepal’s
   role in analysing suspicious transaction/activity reports and is the
   appropriate primary source to cite for the project’s Nepal-specific AML
   framing.

4. DataCebo. (n.d.). *Synthetic Data Vault (SDV)*.
   ([official documentation](https://sdv.dev/)).

   Relevance: official documentation for the planned CTGAN, TVAE, and Gaussian
   Copula implementation ecosystem. This supports implementation provenance;
   the empirical generator recommendation remains contingent on the Phase 3
   benchmark.

## References not used as evidence

The bundled TechRxiv PDF remains a project-provided reference and should be
cited from its own bibliographic metadata if used in the final report. It was
not used here to manufacture findings about the supplied source bundle. The
four sources above have public, working opening links as checked on 2026-08-02.
