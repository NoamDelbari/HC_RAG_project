# Pool Size Diagnostics — Design Spec

**Date:** 2026-03-30
**Dataset:** amazon_compound (135 queries, 80K docs)
**Goal:** Investigate why HC F1 decreases with larger pool sizes despite theory predicting stable or improved performance. 6 diagnostics targeting p-value calibration, z-score estimation, null shape, and HC behavior.

---

## The Problem

From Experiment D, HC F1 drops monotonically as pool_size increases (0.176 at 1000 → 0.150 at 2000 → 0.086 at 10000). Mean K explodes (34 → 2000). HC theory predicts that adding more noise around the same ~30 relevant docs should not cause HC to declare more docs as significant. Something is systematically wrong with the p-values at larger pool sizes.

---

## Diagnostics

### D1: P-Value Distributions at Each Pool Size

**Question:** Are non-relevant p-values uniform at each pool size?

**Method:** For each pool_size in [1000, 2000, 4000, 6000, 10000]:
1. Build pool-matched null at that pool_size
2. For each query: FAISS top pool_size, z-score using bottom 80%, compute p-values of non-relevant docs against the null
3. Aggregate: KS stat, mean p-value, fraction of non-relevant p-values < 0.05

If non-relevant p-values skew left (too many small p-values) at larger pools, HC will over-retrieve.

**Output:** Table: pool_size → mean KS stat, mean p-value, fraction p < 0.05 for non-relevant docs

### D2: Gamma Capped at Max Relevant K

**Question:** If we restrict HC's search window to only where signal could plausibly exist, does F1 recover?

**Method:** For the amazon_compound dataset, max relevant docs per query is ~43 (from the data).
For each pool_size in [1000, 2000, 4000, 6000, 10000]:
1. Set gamma = min(43 / pool_size, 0.5) — caps the HC search window at 43 positions
   - pool=1000: gamma=0.043
   - pool=2000: gamma=0.0215
   - pool=4000: gamma=0.01075
   - pool=6000: gamma=0.00717
   - pool=10000: gamma=0.0043
2. Run HC retrieval with this capped gamma
3. Measure Recall, Precision, F1, Mean K

**Output:** Table: pool_size → F1, Recall, Precision, Mean K with capped gamma. Compare to Experiment D results.

### D3: Z-Score Estimation — Absolute vs Percentage

**Question:** Does using a fixed absolute number of bottom docs for z-score estimation (instead of a percentage) produce better calibration at larger pool sizes?

**Method:** For each pool_size in [1000, 2000, 4000, 6000, 10000], compare 4 z-score estimation strategies:
- **(a) Current 80%:** Bottom 80% of pool (800, 1600, 3200, 4800, 8000 docs)
- **(b) Fixed 800:** Always bottom 800 docs (matches pool=1000 at 80%)
- **(c) Fixed 1000:** Always bottom 1000 docs
- **(d) Ground truth oracle:** mu/sigma from ALL non-relevant docs in corpus (requires qrels, not available at inference)

For each (pool_size, strategy):
1. Compute mu_est, sigma_est per query
2. Compare to ground-truth mu_gt, sigma_gt: absolute error |mu_est - mu_gt|, |sigma_est - sigma_gt|
3. Z-score candidates, compute p-values against pool-matched null (built with same strategy)
4. Measure KS uniformity of non-relevant p-values
5. Run HC retrieval, measure F1, Mean K

**Output:** Table: (pool_size, strategy) → mu_error, sigma_error, KS stat, F1, Mean K

### D4: Null Distribution Shape Comparison

**Question:** How does the null distribution shape change with pool size, and how far does it drift from the "correct" distribution?

**Method:** For each pool_size in [1000, 2000, 4000, 6000, 10000]:
1. Build pool-matched null at that pool_size
2. Build "oracle null" at that pool_size: same FAISS top-K candidates, but z-score using ground-truth mu/sigma from ALL non-relevant docs, pool only non-relevant z-scores
3. Compare distributions:
   - Mean, std, skewness, kurtosis of each null
   - 2-sample KS test between pool-matched null and oracle null
   - Quantile comparison (5th, 25th, 50th, 75th, 95th percentiles)

**Output:** Table: pool_size → null stats (mean, std, skew, kurtosis) for both pool-matched and oracle nulls, plus KS distance between them.

### D5: Per-Query K Tracking Across Pool Sizes

**Question:** Does HC-selected k stay stable as pool_size grows, or does it grow with pool_size?

**Method:** For each query (all 135), run HC at pool_size=[1000, 2000, 4000, 6000, 10000] with fixed gamma=0.1.
Record HC-selected k for each (query, pool_size).

**Analysis:**
- Per-query: plot k vs pool_size (should be flat if HC works correctly)
- Aggregate: mean k, median k, std k at each pool_size
- Correlation: k(pool=N) vs k(pool=1000) — does the ordering preserve?
- Growth rate: mean(k(pool=N) / k(pool=1000)) — if >1 consistently, k grows with pool

**Output:** Table + per-query tracking data

### D6: HC Statistic Curve Tracing

**Question:** How does the HC_i curve shape change with pool size? Where does the argmax fall?

**Method:** Select 5 representative queries:
- 1 with small true_k (k~5)
- 1 with medium true_k (k~20)
- 1 with large true_k (k~40)
- 1 where HC performs well at pool=1000
- 1 where HC performs poorly at pool=1000

For each query, at each pool_size=[1000, 2000, 4000, 10000]:
1. Compute the full HC_i values for i=1..gamma*pool_size
2. Record: argmax position, max HC value, HC curve shape
3. Also record: which positions correspond to relevant vs non-relevant docs

**Output:** Per-query HC curves at each pool size, argmax positions, relevant doc positions

---

## Implementation

All diagnostics in a single script: `experiments/scripts/pool_null/exp_d2_pool_diagnostics.py`

The script should:
1. Pre-compute all query embeddings once
2. Pre-compute FAISS results at max pool_size (10000) once, truncate for smaller sizes
3. Pre-compute ground-truth mu/sigma for each query once (sims to all docs)
4. Run all 6 diagnostics
5. Save results to `results/amazon_compound/null_research/pool_diagnostics/`

CLI: `python -m experiments.scripts.pool_null.exp_d2_pool_diagnostics --config experiments/configs/amazon_compound.yaml`

---

## What Success Looks Like

These diagnostics should answer:
1. **Is it a p-value calibration problem?** (D1) — if non-relevant p-values aren't uniform at large pools, the null is miscalibrated
2. **Is it a gamma problem?** (D2) — if capping gamma fixes F1, the issue is gamma scaling not null calibration
3. **Is it a z-score estimation problem?** (D3) — if fixed-800 outperforms 80%, the percentage-based approach breaks at scale
4. **Is it a null shape problem?** (D4) — if the null drifts far from oracle, the pool-matched construction fails at scale
5. **Is k growing with pool size?** (D5) — directly confirms the over-retrieval hypothesis
6. **Where does the HC curve break?** (D6) — shows mechanistically what happens to the argmax
