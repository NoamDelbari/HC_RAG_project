# Null Distribution Research Plan

**Date:** 2026-03-30
**Dataset:** amazon_compound (135 queries, 80K docs, compound category+price queries)
**Goal:** Systematically investigate 7 identified issues with the null distribution calibration and determine the best approach for HC-based adaptive retrieval.

---

## Issues Under Investigation

1. **80% z-score fraction is arbitrary** — no theoretical justification, needs ablation
2. **compare_nulls.py uses different code path than retrieval pipeline** — manual z-scoring vs HCRetrieval internal z-scoring
3. **Z-scoring imperfectly standardizes across queries** — 32% of queries fail per-query uniformity
4. **Signal detection is low (46.8%)** — embedding model can't distinguish price constraints
5. **Original null passes validation via miscalibration** — inflated z-scores from distribution mismatch
6. **Pool size and gamma are coupled** — changing one requires re-tuning the other
7. **Build-time vs inference-time tension** — ground truth available at build time but not at inference

---

## Experiments

### Experiment A: Z-Score Fraction Sweep (Issue 1)

**Question:** What is the optimal z_score_fraction for the pool-matched null?

**Method:** For each fraction in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
1. Build pool-matched null with that fraction
2. Run 4-test validation
3. Run HC retrieval, measure Recall, Precision, F1, Mean K

**Parameters:** pool_size and gamma fixed at best values from Experiment D (run after D).

**Output:** Table: fraction -> validation pass/fail + retrieval metrics

### Experiment B: Code Path Consistency (Issue 2)

**Question:** Does the manual z-scoring in compare_nulls.py produce different results than HCRetrieval with use_zscore=True?

**Method:**
1. Run HC retrieval using HCRetrieval class directly (same as run_retrieval.py) with pool-matched null
2. Run HC retrieval using manual z-scoring (same as compare_nulls.py) with pool-matched null
3. Compare per-query k selections and metrics

**Parameters:** pool_size=1000, gamma=0.1, z_score_fraction=0.8

**Output:** Per-query k difference, aggregate metric comparison, flag whether the two paths diverge.

### Experiment C: Embedding Recall Ceiling (Issue 4)

**Question:** What fraction of relevant documents appear in the top-K pool at various K values?

**Method:** For each K in [50, 100, 500, 1000, 2000, 4000, 6000, 8000, 10000]:
1. FAISS search for top-K per query
2. Count how many relevant docs are in the pool (using ground truth qrels)
3. Compute recall@K = |relevant in pool| / |total relevant|

**Output:** K -> recall@K curve. This is the ceiling for any adaptive retrieval method at each pool size. HC can never exceed recall@K.

### Experiment D: Pool Size x Gamma Grid (Issue 6)

**Question:** How do pool_size and gamma interact? What's the optimal pairing?

**Method:** Grid sweep:
- pool_size: [2000, 4000, 6000, 8000, 10000]
- gamma: [0.05, 0.1, 0.2, 0.5]

For each (pool_size, gamma) pair:
1. Build pool-matched null (z_score_fraction=0.8)
2. Run HC retrieval on all 135 queries
3. Measure: Recall, Precision, F1, Mean K, Mean HC stat

**Output:** 5x4 grid of retrieval metrics. Identify optimal (pool_size, gamma) pairing.

### Experiment E: Null Construction Methods (Issues 3, 5, 7)

**Question:** How much does the null construction method matter?

**Method:** Compare 3 null variants (at fixed pool_size/gamma):
- **(a) Original:** Random non-relevant docs from full corpus, z-scored with ground-truth mu/sigma from ALL non-relevant docs
- **(b) Pool-matched:** Top-K pool, z-scored with bottom-fraction estimated mu/sigma
- **(c) Hybrid:** Top-K pool docs, but z-scored with ground-truth mu/sigma from ALL non-relevant docs

Variant (c) isolates whether the issue is:
- The pool sampling (which docs go into the null), or
- The mu/sigma estimation method (ground truth vs bottom fraction)

For each variant:
1. Build null
2. Run 4-test validation
3. Run HC retrieval, measure metrics
4. Measure per-query KS uniformity of non-relevant p-values

**Output:** 3-way comparison table of validation results + retrieval metrics + uniformity diagnostics.

### Experiment F: Alternative Normalization (Issue 3)

**Question:** Is z-scoring the right normalization? Would alternatives work better?

**Method:** Compare 3 normalization approaches:
- **(a) Z-score + empirical CDF:** Current approach
- **(b) Rank-based:** p_value = rank / pool_size (no null distribution needed, distribution-free)
- **(c) Raw CDF:** Empirical CDF of raw cosine similarities against a null of raw similarities (no z-scoring)

For each approach:
1. Run HC retrieval on all 135 queries
2. Measure Recall, Precision, F1, Mean K
3. Measure per-query p-value uniformity for non-relevant docs

**Output:** 3-way comparison of retrieval metrics + uniformity diagnostics.

---

## Execution Order

1. **Experiment C** (embedding ceiling) — fastest, no null building, establishes upper bound
2. **Experiment B** (code path check) — quick, validates our comparison methodology
3. **Experiment E** (null construction methods) — core investigation, 3-way comparison
4. **Experiment D** (pool size x gamma grid) — systematic sweep at larger pool sizes
5. **Experiment A** (fraction sweep) — run at best pool_size from Experiment D
6. **Experiment F** (alternative normalization) — exploratory

---

## Config

All experiments use:
- Dataset: amazon_compound (135 queries, 80K docs)
- Embedding model: text-embedding-3-small (1536d)
- Vector DB: FAISS IndexFlatIP (cosine similarity)
- Existing artifacts: artifacts/amazon_compound/

## Output

All results saved to: results/amazon_compound/null_research/experiment_{A-F}/
