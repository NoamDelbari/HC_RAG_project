# Pool-Matched Null Distribution — Design Spec

**Date**: 2026-03-29
**Goal**: Fix the calibration mismatch between null distribution build-time and HC inference-time by building the null using the same FAISS top-K retrieval procedure that HC uses at inference.

---

## 1. The Problem

The current null distribution is built from z-scores of **random non-relevant documents across the full corpus**. At inference time, HC sees only the **top-K most similar candidates** from FAISS, z-scored using the bottom 80% of that pool. These are fundamentally different distributions:

- **Build time**: `mean_q` from ALL ~80K non-relevant docs (mostly very low similarity)
- **Inference time**: `mu_est` from bottom 80% of top-1000 candidates (all high similarity)

After z-scoring, the distributions only align if cosine similarities are Gaussian. They likely aren't (bounded, skewed, possibly gamma-distributed). This means p-values at inference time are **systematically miscalibrated** against the global null.

---

## 2. The Fix: Pool-Matched Null

Build the null using the exact same procedure HC uses at inference:

```
For each query:
  1. Embed query, FAISS search for top-K (same pool_size as config)
  2. Sort candidates by similarity descending
  3. Take bottom 80% of pool → compute mu_est, sigma_est
     (includes relevant docs if present — matches inference)
  4. Z-score ALL K candidates using mu_est, sigma_est
  5. Identify non-relevant docs using ground truth qrels
  6. Add ONLY non-relevant z-scores to the global pool

Pool all non-relevant z-scores across queries → global pool-matched null
```

**Key design decisions:**
- Z-score estimation (step 3) includes relevant docs in bottom 80% — matches inference where ground truth is unavailable
- Only non-relevant z-scores enter the null (step 6) — the null must represent non-relevant distribution only
- pool_size and z_score_fraction must match what HC will use at inference — changing either requires rebuilding

---

## 3. File Structure

All new code goes in `experiments/scripts/pool_null/`. Existing code is untouched.

```
experiments/scripts/
├── build_null.py              # existing — untouched
├── validate_null.py           # existing — untouched
├── pool_null/                 # NEW
│   ├── __init__.py
│   ├── build_pool_null.py     # builds null using FAISS top-K procedure
│   ├── validate_pool_null.py  # validates using same top-K procedure
│   └── compare_nulls.py       # runs both nulls side-by-side, compares
```

Output artifacts go to `{artifacts_dir}/pool_null/` (separate from existing `{artifacts_dir}/global_null.pkl`).

---

## 4. Script Specifications

### 4.1 `build_pool_null.py`

**Input:** Config YAML (same as existing pipeline)
**Output:** `{artifacts_dir}/pool_null/global_pool_null.pkl`

**Procedure:**
1. Load config, adapter, vector DB, embedding model
2. Load queries and qrels
3. For each query:
   a. Embed query
   b. FAISS search for top `config.hc.pool_size` candidates (cosine similarity)
   c. Sort similarities descending
   d. Compute `n_null_est = int(pool_size * config.null.z_score_fraction)`
   e. `mu_est = mean(similarities[-n_null_est:])`, `sigma_est = std(similarities[-n_null_est:])`
   f. Skip query if `sigma_est < 1e-12`
   g. Z-score all candidates: `z = (sim - mu_est) / sigma_est`
   h. Identify which candidates are non-relevant (using qrels)
   i. Append non-relevant z-scores to global pool
4. Concatenate all z-scores → NullDistribution
5. Save to `{artifacts_dir}/pool_null/global_pool_null.pkl`

**CLI:** `python -m experiments.scripts.pool_null.build_pool_null --config experiments/configs/amazon_compound.yaml [--force]`

### 4.2 `validate_pool_null.py`

**Input:** Config YAML, pool-matched null distribution
**Output:** `{artifacts_dir}/pool_null/validation_report.json`

Same 4-test hard gate as existing validator, but all tests use the pool-matched procedure:

**Test 1: Split-Half KS Uniformity**
- Same as existing — split the null's z-scores in half, check internal consistency
- No change needed (operates on the null itself, not on queries)

**Test 2: Per-Query Uniformity (pool-matched)**
- For each sampled query: FAISS search for top-K, z-score using bottom 80% of pool
- Compute p-values of non-relevant z-scores against pool-matched null
- KS test for uniformity
- Pass: >= 70% of queries pass (KS p > 0.05)

**Test 3: Signal Detection (pool-matched)**
- For each sampled query: FAISS search for top-K, z-score using bottom 80% of pool
- Compute p-values of relevant z-scores against pool-matched null
- Pass: >= 60% of relevant docs have p < 0.05

**Test 4: HC K Correlation (pool-matched)**
- For each query: FAISS search for top-K, z-score, run HC with pool-matched null
- Pass: mean HC stat > 1.0 AND unique k >= 3

**CLI:** `python -m experiments.scripts.pool_null.validate_pool_null --config experiments/configs/amazon_compound.yaml`

Exit code 0 on pass, 1 on fail (same as existing).

### 4.3 `compare_nulls.py`

**Input:** Config YAML (both nulls must exist)
**Output:** `{results_dir}/null_comparison/` with JSON results and diagnostic info

**Procedure:**
1. Load both null distributions (original + pool-matched)
2. For each query:
   a. Run HC retrieval with original null → k_old, retrieved docs
   b. Run HC retrieval with pool-matched null → k_new, retrieved docs
   c. Compute retrieval metrics for each (using qrels)
   d. Compute p-value distributions for non-relevant docs under each null
3. Output:
   - Aggregate retrieval metrics (Recall, Precision, F1, Mean K) for each null
   - Per-query comparison: k_old vs k_new
   - P-value distribution diagnostics: KS statistic between the two null's p-values
   - Summary: which null produces more uniform non-relevant p-values

**CLI:** `python -m experiments.scripts.pool_null.compare_nulls --config experiments/configs/amazon_compound.yaml`

---

## 5. Integration with Existing Pipeline

The pool-matched null is **standalone** — it does not modify or replace the existing pipeline. To use the pool-matched null with `run_retrieval.py` or `run_eval.py`, you would manually point to the pool-matched null file. This is intentionally NOT automated until results confirm the approach works.

---

## 6. What Success Looks Like

1. **Validation passes**: Pool-matched null passes all 4 tests
2. **Better p-value calibration**: Non-relevant p-values under pool-matched null are more uniform than under original null (measurable via KS statistic in compare_nulls.py)
3. **Retrieval improvement**: HC with pool-matched null achieves equal or better Recall/F1 than with original null
4. **Adaptive k improves**: HC-selected k correlates better with oracle k under pool-matched null
