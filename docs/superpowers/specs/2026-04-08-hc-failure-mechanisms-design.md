# HC Failure Mechanisms — Research Design Spec

**Date:** 2026-04-08
**Dataset:** amazon_compound (135 queries, 80K docs, compound category+price queries)
**Goal:** Define when HC works and why — produce a paper section with intuitive analytical arguments backed by targeted experiments

---

## Global Research Question

Higher Criticism (HC) adaptively determines retrieval depth by detecting statistical signal in candidate pools. Previous investigation revealed two key phenomena:

1. A tradeoff between p-value uniformity and signal detection as the candidate pool grows
2. 30% of queries (41/135) produce negative HC statistics, causing zero retrieval

This research investigates the mechanisms behind both phenomena and tests whether they share a common root cause: the interaction between per-query z-scoring and a global null distribution.

## Framework

Three research directions that build on each other:

- **Direction 1** characterizes the uniformity–detection tradeoff across pool sizes (macro view)
- **Direction 2** zooms into failing queries to understand what breaks at each pipeline layer (micro view)
- **Direction 3** tests whether the per-query z-scoring × global null interaction is the root cause of both

Output per experiment: results JSON and diagnostic plots. Analytical arguments are woven into interpretation, not standalone theory.

## Global Parameters

- **Gamma:** Always adaptive — `gamma = 43 / pool_size` where 43 is the global maximum relevant docs across all queries. This keeps the HC search window at a fixed 43 positions regardless of pool size.
- **z_score_fraction:** Treated as a variable — `[0.5, 0.6, 0.7, 0.8]` explored where specified.
- **Pool sizes:** 1K–10K, step 1K (10 values) where pool is varied; pool=1000 where fixed.

---

## Direction 1: Is the Uniformity–Detection Tradeoff Real?

### Research Question

As pool size grows, per-query uniformity degrades and aggregate signal detection improves. Is this a smooth, monotonic trend or does it only appear at extremes? And is the "improved signal detection" genuine or an artifact of averaging over miscalibrated queries?

### Connection to Global Goal

If HC requires both uniformity and signal detection, and they're fundamentally in tension, that defines a hard constraint on when HC can work. But if the "tradeoff" is an averaging artifact driven by anti-conservative miscalibration, the real constraint is different.

### Experiment H1.1 — Monotonicity Test

**What it does:**
- At each pool size (1K–10K, step 1K) and each z_score_fraction (0.5, 0.6, 0.7, 0.8), build a pool-matched null and run the 4-test validation gate (split-half stability, per-query uniformity, signal detection, HC-K correlation)
- Record per cell: uniformity pass rate, signal detection rate, mean KS stat, mean HC stat

**What it validates:**
- Whether the tradeoff is continuous and inherent, or only appears at extreme pool sizes
- Whether a different z_score_fraction mitigates the tradeoff at larger pools
- A smooth inverse curve means the tradeoff is structural; plateaus or non-monotonic behavior means additional mechanisms are at play

**Output:** 2D grid (pool_size × z_score_fraction) of uniformity and signal detection rates

### Experiment H1.2 — Artifact Test

**What it does:**
- At each pool size and z_score_fraction, use per-query KS test results from H1.1 to split queries into "uniform" (KS pass) and "non-uniform" (KS fail)
- Compute signal detection rate separately for each group

**What it validates:**
- Whether the aggregate signal detection improvement is real (both groups benefit) or driven by the miscalibrated group
- If the uniform group shows flat or declining signal detection, the "tradeoff" is a mirage — the aggregate improvement comes from anti-conservative miscalibration, not genuine detection power

**Output:** Two signal-detection curves per z_score_fraction (uniform group vs non-uniform group)

---

## Direction 2: What Causes HC to Fail at the Query Level?

### Research Question

41 queries (30%) have negative HC stat. What characterizes these queries? Is it the queries themselves, the similarity landscape they produce, the pool composition, the z-scoring behavior, or the null fit? Is the signal absent, or present but too sparse for HC to detect?

### Connection to Global Goal

Most direct answer to "when does HC work." If HC failure is predictable from query properties (Layer 1), HC has an identifiable scope of applicability. If it's caused by normalization or null fit (Layers 4–5), it's potentially fixable. Understanding which layer breaks identifies the preconditions HC requires.

### Experiment H2.1 — Query Characterization

**What it does:**
- Partition all 135 queries into negative-HC (41) vs positive-HC (94)
- Compare: number of relevant docs in ground truth, recall ceiling (how many relevant are in the pool), query text properties (length, presence of numeric/price terms)

**What it validates:**
- Whether HC failure is predictable from query properties alone, before any retrieval happens
- If yes, HC has a hard boundary of applicability defined by query type

**Output:** Summary statistics table and distributions for both groups

### Experiment H2.2 — Similarity Landscape Shapes

**What it does:**
- For each query, extract the full cosine similarity vector of top-1000 candidates
- Separate relevant vs non-relevant candidates
- Compute: mean/std/min/max similarity for each group, overlap between distributions (fraction of relevant docs whose similarity falls within the non-relevant interquartile range), distribution shape (variance, skewness)

**What it validates:**
- Whether failure is "no signal exists" (complete overlap between relevant and non-relevant similarities) vs "signal exists but is subtle" (partial overlap, some relevant at top)
- Distinguishes embedding-model limitations from HC-method limitations

**Output:** Per-query similarity profiles, aggregate comparison between negative-HC and positive-HC groups

### Experiment H2.3 — Rank Position and Clustering

**What it does:**
- For each query, record exact rank positions (1–1000) of every relevant doc in the pool
- Compute: count in top-50, top-100, top-200; clustering metric (mean pairwise distance between relevant doc positions, ratio of range spanned vs count)
- Per-query rank histograms for negative-HC vs positive-HC

**What it validates:**
- Whether HC fails because relevant docs are absent from the top, or present but scattered
- A query with 10 relevant docs scattered at positions 5, 47, 130, 280, 510 is fundamentally harder for thresholding than one with 10 docs at positions 3–12
- Identifies the minimum concentration HC needs

**Output:** Position distributions and clustering scores for both groups

### Experiment H2.4 — Z-Scoring Contamination

**What it does:**
- For each query, identify which candidates in the bottom 80% (used for mu/sigma estimation) are actually relevant docs
- Compute: fraction of estimation set that is relevant, mu and sigma with vs without relevant docs, resulting z-scores and HC stat with vs without contamination
- Report contamination at each z_score_fraction (0.5, 0.6, 0.7, 0.8) to show how contamination scales

**What it validates:**
- Whether the normalization step is self-sabotaging for weak-signal queries
- If relevant docs in the estimation set inflate sigma and this flips HC negative for some queries, the z_score_fraction is a tunable lever
- If removing contamination flips some queries from negative to positive HC, the failure is partly a normalization artifact

**Output:** Per-query contamination rates, delta in HC stat when contamination is removed, contamination at each fraction level

### Experiment H2.5 — Per-Query Null Fit

**What it does:**
- For each query, compute KS statistic of its non-relevant z-scores against the global null
- Compare the global null's shape to each query's actual non-relevant z-score distribution (std, skewness, tail weight)

**What it validates:**
- Whether the global null is systematically too wide for negative-HC queries (conservative p-values) vs too narrow for positive-HC queries (anti-conservative)
- Combined with H3.2, closes the loop: z-score spread → null mismatch → conservative p-values → negative HC

**Output:** Per-query KS stats partitioned by HC sign, distribution shape comparisons

---

## Direction 3: Per-Query Z-Scoring × Global Null Interaction

### Research Question

Z-scoring is per-query (each query estimates its own mu/sigma) but the null is global (shared). Does this mismatch cause both the uniformity-detection tradeoff (Direction 1) and the negative HC phenomenon (Direction 2)? Or is the problem in the z-scoring step itself?

### Connection to Global Goal

This is the "why." If verified, it provides a single mechanistic explanation for both observed problems — the strongest form of understanding for the paper. It also points toward solutions: if the mismatch is the cause, per-query nulls or different normalization could fix it. If z-scoring itself is the problem, the fix is different.

### Experiment H3.1 — Z-Score Heterogeneity Grows with Pool Size

**What it does:**
- At each pool size (1K–10K, step 1K), compute per-query z-score statistics: std, skewness, kurtosis of all z-scores within each query's pool
- Measure cross-query variance of each statistic (variance of the 135 per-query stds, etc.)

**What it validates:**
- The premise that the global null's job gets harder as pool grows
- If cross-query z-score heterogeneity increases monotonically with pool, it directly explains why uniformity degrades — the global null is fitting an increasingly diverse population
- Provides the mechanism behind Direction 1's tradeoff

**Output:** Cross-query variance curves vs pool size

### Experiment H3.2 — Z-Score Spread Predicts Calibration Direction

**What it does:**
- At pool=1000, compute each query's z-score std and compare to the global null's std
- Compute the ratio (query z-score std / null std) for each query
- Correlate this ratio with: HC stat sign, HC stat magnitude, per-query KS stat (uniformity), per-query signal detection rate

**What it validates:**
- That the direction of miscalibration is predictable from the z-score mismatch
- If ratio < 1 → conservative (negative HC) and ratio > 1 → anti-conservative (inflated HC), this directly links the z-scoring × null interaction to both Direction 2 (negative HC) and Direction 1 H1.2 (miscalibrated signal detection)

**Output:** Scatter plots and correlation coefficients

### Experiment H3.3 — Isolating the Cause

**What it does:**
- Two dimensions: 3 null conditions × 4 z_score_fractions
- Null conditions:
  - (a) Per-query z-scoring + global null (current pipeline)
  - (b) Raw similarities + global null (skip z-scoring — build null from raw similarities, compute p-values from raw similarities)
  - (c) Per-query z-scoring + per-query null (each query gets its own null from its own non-relevant z-scores)
- z_score_fractions: 0.5, 0.6, 0.7, 0.8
- For each cell: per-query uniformity pass rate, signal detection rate, mean HC stat, number of negative-HC queries, mean F1

**What it validates:**
- Whether the problem is the z-scoring, the global null, or their interaction
- If (c) restores uniformity and reduces negative HC without sacrificing signal detection → the mismatch is the root cause
- If (b) also restores uniformity → z-scoring itself is the problem
- If neither helps → the issue is more fundamental
- The z_score_fraction dimension reveals whether fraction tuning resolves issues under any null condition

**Output:** 3×4 comparison table with all metrics

---

## Experiment Organization

### Scripts

| Script | Experiments | Pool sizes | z_score_fractions |
|---|---|---|---|
| `exp_h1_uniformity_detection_tradeoff.py` | H1.1, H1.2 | 1K–10K, step 1K | 0.5, 0.6, 0.7, 0.8 |
| `exp_h2_query_level_failure.py` | H2.1–H2.5 | 1000 only | 0.8 default; H2.4 reports at all 4 |
| `exp_h3_zscore_null_interaction.py` | H3.1, H3.2, H3.3 | H3.1: 1K–10K; H3.2–H3.3: 1000 | H3.1: 0.8; H3.3: all 4 |

### Dependencies

- H1 and H2 can run independently
- H3.1 can run in the same pass as H1 (same pool-size loop) but is a separate script for clarity
- H3.2 uses the same pool=1000 data as H2 but analyzes differently
- H3.3 is fully independent (builds new null variants)
- Interpretation ties them together — H3 explains the mechanisms found in H1 and H2

### Output Structure

```
results/amazon_compound/null_research/
  h1_uniformity_detection/
    h1_monotonicity.json        # 2D grid: pool_size × z_score_fraction
    h1_artifact_test.json       # Signal detection split by uniformity group
  h2_query_failure/
    h2_query_characterization.json
    h2_similarity_landscapes.json
    h2_rank_positions.json
    h2_zscore_contamination.json
    h2_null_fit.json
  h3_zscore_null_interaction/
    h3_heterogeneity.json       # Cross-query variance vs pool size
    h3_spread_prediction.json   # Z-score std ratio correlations
    h3_isolation.json           # 3×4 null condition × fraction grid
```

### Existing Code Reuse

- Pool null building: `experiments/scripts/pool_null/build_pool_null.py`
- Validation logic: existing 4-test gate from `exp_null_deep_analysis.py`
- HC computation: `src/hc_rag/hc/higher_criticism.py`
- FAISS retrieval: `src/hc_rag/retrieval/hc_retrieval.py`

### Config

- Base config: `experiments/configs/amazon_compound.yaml`
- Gamma: derived as `43 / pool_size` (adaptive, not configured)
- z_score_fraction: `[0.5, 0.6, 0.7, 0.8]` where varied; 0.8 where fixed
- Pool sizes: `[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]`
