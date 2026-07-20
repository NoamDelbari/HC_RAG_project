# HC Pipeline Trace (H4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trace the full HC pipeline per query to classify why each failing query produces negative HC, and test whether targeted interventions can rescue them.

**Architecture:** Single script (`exp_h4_pipeline_trace.py`) with one FAISS retrieval loop producing per-query traces, then four analysis functions (H4.1–H4.4) that consume the traces. Follows the same patterns as existing H2/H3 scripts.

**Tech Stack:** Python, NumPy, SciPy, FAISS (via VectorDatabase), existing HC/null modules.

---

## File Structure

- Create: `experiments/scripts/pool_null/exp_h4_pipeline_trace.py`
- Read: `src/hc_rag/hc/higher_criticism.py` (HC computation patterns)
- Read: `src/hc_rag/hc/null_distribution.py` (NullDistribution dataclass)
- Read: `experiments/scripts/pool_null/exp_h2_query_level_failure.py` (code patterns)
- Output: `results/amazon_compound/null_research/h4_pipeline_trace/` (4 JSON files)

---

### Task 1: Create the script with imports, constants, and helpers

**Files:**
- Create: `experiments/scripts/pool_null/exp_h4_pipeline_trace.py`

- [ ] **Step 1: Create the script file with module docstring, imports, and constants**

```python
"""Direction 4: Per-query pipeline trace and failure mode classification.

Four analyses from a single FAISS retrieval loop:
  H4.1 — Full pipeline trace (similarity → z-score → p-value → HC curve)
  H4.2 — Relevant document p-value analysis
  H4.3 — Failure mode classification of irreducible-failure queries
  H4.4 — Targeted fix tests (contamination removal + detection boundary)

Usage:
    python -m experiments.scripts.pool_null.exp_h4_pipeline_trace \
        --config experiments/configs/amazon_compound.yaml [--force]
"""

import argparse
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

from hc_rag.embeddings.embedding_model import create_embedding_model
from hc_rag.embeddings.vector_database import VectorDatabase
from hc_rag.hc.null_distribution import NullDistribution
from hc_rag.hc.higher_criticism import HigherCriticism
from experiments.lib.config import load_config
from experiments.lib.registry import get_adapter
from experiments.lib.results_io import save_results_json
import experiments.datasets  # noqa: F401


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POOL_SIZE = 1000
MAX_RELEVANT = 43
GAMMA = MAX_RELEVANT / POOL_SIZE  # 0.043
Z_SCORE_FRACTION = 0.8
TOP_N_STORE = 100  # Store full trace for top-100 candidates per query
```

- [ ] **Step 2: Add helper functions**

```python
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def zscore_candidates(candidate_sims, z_score_fraction):
    """Z-score candidates using bottom fraction. Returns (z_scores, mu, sigma) or None."""
    sorted_desc = np.sort(candidate_sims)[::-1]
    n = len(sorted_desc)
    n_null_est = max(int(n * z_score_fraction), 10)
    bottom = sorted_desc[-n_null_est:]
    mu = float(np.mean(bottom))
    sigma = float(np.std(bottom))
    if sigma < 1e-10:
        return None
    z_scores = (candidate_sims - mu) / sigma
    return z_scores, mu, sigma


def compute_pvalues_empirical(values, null_dist):
    """Compute p-values of values against a NullDistribution (empirical CDF)."""
    null_sorted = np.sort(null_dist.similarities)
    N_null = len(null_sorted)
    indices = np.searchsorted(null_sorted, values, side="left")
    n_greater_equal = N_null - indices
    p_values = (n_greater_equal + 1.0) / (N_null + 1.0)
    eps = 1.0 / (N_null + 1.0)
    p_values = np.clip(p_values, eps, 1.0 - eps)
    return p_values


def compute_hc_curve(p_values_sorted_asc, n_gamma, n):
    """Compute HC_i at each position i=1..n_gamma.

    Args:
        p_values_sorted_asc: p-values sorted ascending (smallest first).
        n_gamma: Number of positions to evaluate.
        n: Total number of candidates.

    Returns:
        Array of HC_i values (length n_gamma).
    """
    hc_values = np.zeros(n_gamma)
    for i in range(1, n_gamma + 1):
        p_i = p_values_sorted_asc[i - 1]
        expected = i / n
        denom = np.sqrt(p_i * (1 - p_i) / n)
        if denom > 0:
            hc_values[i - 1] = (expected - p_i) / denom
    return hc_values


def compute_metrics(retrieved_ids, relevant_ids):
    """Recall / precision / F1 for a single query."""
    if len(relevant_ids) == 0:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    hits = len(retrieved_set & relevant_set)
    recall = hits / len(relevant_set) if relevant_set else 0.0
    precision = hits / len(retrieved_set) if retrieved_set else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}
```

- [ ] **Step 3: Verify the file runs without errors**

Run: `python -c "import experiments.scripts.pool_null.exp_h4_pipeline_trace"`
Expected: No import errors.

- [ ] **Step 4: Commit**

```bash
git add experiments/scripts/pool_null/exp_h4_pipeline_trace.py
git commit -m "feat(h4): add pipeline trace script skeleton with helpers"
```

---

### Task 2: Implement H4.1 — Full Pipeline Trace (data collection)

This is the core data collection loop. All other analyses (H4.2–H4.4) consume its output.

**Files:**
- Modify: `experiments/scripts/pool_null/exp_h4_pipeline_trace.py`

- [ ] **Step 1: Add the `h4_1_pipeline_traces` function**

```python
# ---------------------------------------------------------------------------
# H4.1 — Full Pipeline Trace
# ---------------------------------------------------------------------------

def h4_1_pipeline_traces(queries, qrels, all_faiss_results, global_null):
    """Trace the full HC pipeline for every query.

    Returns list of per-query trace dicts (one per query).
    """
    n_gamma = int(np.floor(GAMMA * POOL_SIZE))  # 43
    traces = []

    for qi, query in enumerate(queries):
        relevant_ids = qrels.get(query.query_id, set())
        faiss_results = all_faiss_results[qi][:POOL_SIZE]

        cids = [cid for cid, _ in faiss_results]
        csims = np.array([sim for _, sim in faiss_results])

        # Sort descending by similarity
        sort_idx = np.argsort(csims)[::-1]
        csims = csims[sort_idx]
        cids = [cids[j] for j in sort_idx]

        # Label relevant/non-relevant
        rel_mask = np.array([cid in relevant_ids for cid in cids])
        n_rel_in_pool = int(rel_mask.sum())

        # Z-score
        zs_result = zscore_candidates(csims, Z_SCORE_FRACTION)
        if zs_result is None:
            # Degenerate query — skip
            continue
        z_scores, mu_est, sigma_est = zs_result

        # Sort z-scores descending (same order as csims since zscore is monotonic)
        # z_scores are in the same order as csims (descending similarity)

        # P-values against global null
        pvals_global = compute_pvalues_empirical(z_scores, global_null)

        # Build per-query null from this query's non-relevant z-scores
        nonrel_z = z_scores[~rel_mask]
        if len(nonrel_z) < 10:
            continue
        per_query_null = NullDistribution(
            similarities=nonrel_z,
            mean=float(np.mean(nonrel_z)),
            std=float(np.std(nonrel_z)),
            min_val=float(np.min(nonrel_z)),
            max_val=float(np.max(nonrel_z)),
            n_samples=len(nonrel_z),
        )
        pvals_perquery = compute_pvalues_empirical(z_scores, per_query_null)

        # HC curves — need p-values sorted ascending (most significant first)
        # Sort by p-value ascending for HC computation
        pval_order_global = np.argsort(pvals_global)
        pvals_sorted_global = pvals_global[pval_order_global]
        hc_curve_global = compute_hc_curve(pvals_sorted_global, n_gamma, POOL_SIZE)
        hc_stat_global = float(np.max(hc_curve_global))
        k_global = int(np.argmax(hc_curve_global) + 1) if hc_stat_global > 0 else 0

        pval_order_pq = np.argsort(pvals_perquery)
        pvals_sorted_pq = pvals_perquery[pval_order_pq]
        hc_curve_pq = compute_hc_curve(pvals_sorted_pq, n_gamma, POOL_SIZE)
        hc_stat_pq = float(np.max(hc_curve_pq))
        k_pq = int(np.argmax(hc_curve_pq) + 1) if hc_stat_pq > 0 else 0

        # Which candidates are relevant in the p-value ordering?
        rel_in_pval_order_global = rel_mask[pval_order_global]
        rel_in_pval_order_pq = rel_mask[pval_order_pq]

        trace = {
            "query_id": query.query_id,
            "query_text": query.text,
            "true_k": len(relevant_ids),
            "n_relevant_in_pool": n_rel_in_pool,
            "mu_est": mu_est,
            "sigma_est": sigma_est,
            # Top-100 candidate details (manageable size)
            "top100_sims": csims[:TOP_N_STORE].tolist(),
            "top100_is_relevant": rel_mask[:TOP_N_STORE].tolist(),
            "top100_zscores": z_scores[:TOP_N_STORE].tolist(),
            "top100_pvals_global": pvals_global[:TOP_N_STORE].tolist(),
            "top100_pvals_perquery": pvals_perquery[:TOP_N_STORE].tolist(),
            # Full HC curves (43 values each)
            "hc_curve_global": hc_curve_global.tolist(),
            "hc_curve_perquery": hc_curve_pq.tolist(),
            "hc_stat_global": hc_stat_global,
            "hc_stat_perquery": hc_stat_pq,
            "k_global": k_global,
            "k_perquery": k_pq,
            # Relevant doc p-values (variable length, all relevant docs)
            "relevant_pvals_global": pvals_global[rel_mask].tolist(),
            "relevant_pvals_perquery": pvals_perquery[rel_mask].tolist(),
            "relevant_zscores": z_scores[rel_mask].tolist(),
            "relevant_sims": csims[rel_mask].tolist(),
            # Relevant doc positions in p-value ordering (what HC sees)
            "relevant_pval_ranks_global": np.where(rel_in_pval_order_global)[0].tolist(),
            "relevant_pval_ranks_perquery": np.where(rel_in_pval_order_pq)[0].tolist(),
            # Summary stats for all 1000 candidates
            "all_sims_mean": float(np.mean(csims)),
            "all_sims_std": float(np.std(csims)),
            "all_zscores_mean": float(np.mean(z_scores)),
            "all_zscores_std": float(np.std(z_scores)),
            "nonrel_zscores_std": float(np.std(nonrel_z)),
            "std_ratio": float(np.std(nonrel_z) / global_null.std) if global_null.std > 0 else 0.0,
        }
        traces.append(trace)

    return traces
```

- [ ] **Step 2: Verify the function is syntactically correct**

Run: `python -c "from experiments.scripts.pool_null.exp_h4_pipeline_trace import h4_1_pipeline_traces; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add experiments/scripts/pool_null/exp_h4_pipeline_trace.py
git commit -m "feat(h4): implement H4.1 full pipeline trace data collection"
```

---

### Task 3: Implement H4.2 — Relevant Document P-Value Analysis

**Files:**
- Modify: `experiments/scripts/pool_null/exp_h4_pipeline_trace.py`

- [ ] **Step 1: Add the `h4_2_relevant_pvalues` function**

```python
# ---------------------------------------------------------------------------
# H4.2 — Relevant Document P-Value Analysis
# ---------------------------------------------------------------------------

def h4_2_relevant_pvalues(traces):
    """Analyze p-values of relevant documents across three query groups.

    Groups:
      - positive_hc: HC stat > 0 under global null (80 queries expected)
      - fixable: HC negative under global, positive under per-query (10 expected)
      - irreducible: HC negative under both (45 expected)
    """
    n_gamma = int(np.floor(GAMMA * POOL_SIZE))

    groups = {"positive_hc": [], "fixable": [], "irreducible": []}
    for t in traces:
        if t["hc_stat_global"] > 0:
            groups["positive_hc"].append(t)
        elif t["hc_stat_perquery"] > 0:
            groups["fixable"].append(t)
        else:
            groups["irreducible"].append(t)

    result = {"groups": {}}
    for group_name, group_traces in groups.items():
        if not group_traces:
            result["groups"][group_name] = {"n": 0}
            continue

        per_query = []
        all_best_pvals_global = []
        all_best_pvals_pq = []
        all_frac_sig_global = []
        all_frac_sig_pq = []
        all_n_rel_in_gamma_global = []
        all_n_rel_in_gamma_pq = []

        for t in group_traces:
            rpg = np.array(t["relevant_pvals_global"])
            rpp = np.array(t["relevant_pvals_perquery"])
            ranks_global = np.array(t["relevant_pval_ranks_global"])
            ranks_pq = np.array(t["relevant_pval_ranks_perquery"])

            n_rel = len(rpg)
            if n_rel == 0:
                continue

            best_pval_global = float(np.min(rpg))
            best_pval_pq = float(np.min(rpp))
            frac_sig_global = float(np.mean(rpg < 0.05))
            frac_sig_pq = float(np.mean(rpp < 0.05))
            frac_sig01_global = float(np.mean(rpg < 0.01))
            frac_sig01_pq = float(np.mean(rpp < 0.01))
            n_in_gamma_global = int(np.sum(ranks_global < n_gamma))
            n_in_gamma_pq = int(np.sum(ranks_pq < n_gamma))

            all_best_pvals_global.append(best_pval_global)
            all_best_pvals_pq.append(best_pval_pq)
            all_frac_sig_global.append(frac_sig_global)
            all_frac_sig_pq.append(frac_sig_pq)
            all_n_rel_in_gamma_global.append(n_in_gamma_global)
            all_n_rel_in_gamma_pq.append(n_in_gamma_pq)

            per_query.append({
                "query_id": t["query_id"],
                "n_relevant_in_pool": t["n_relevant_in_pool"],
                "best_pval_global": best_pval_global,
                "best_pval_perquery": best_pval_pq,
                "frac_significant_global": frac_sig_global,
                "frac_significant_perquery": frac_sig_pq,
                "frac_sig01_global": frac_sig01_global,
                "frac_sig01_perquery": frac_sig01_pq,
                "n_rel_in_gamma_global": n_in_gamma_global,
                "n_rel_in_gamma_perquery": n_in_gamma_pq,
                "median_pval_global": float(np.median(rpg)),
                "median_pval_perquery": float(np.median(rpp)),
            })

        result["groups"][group_name] = {
            "n": len(group_traces),
            "aggregate": {
                "mean_best_pval_global": float(np.mean(all_best_pvals_global)) if all_best_pvals_global else None,
                "mean_best_pval_perquery": float(np.mean(all_best_pvals_pq)) if all_best_pvals_pq else None,
                "mean_frac_significant_global": float(np.mean(all_frac_sig_global)) if all_frac_sig_global else None,
                "mean_frac_significant_perquery": float(np.mean(all_frac_sig_pq)) if all_frac_sig_pq else None,
                "mean_n_rel_in_gamma_global": float(np.mean(all_n_rel_in_gamma_global)) if all_n_rel_in_gamma_global else None,
                "mean_n_rel_in_gamma_perquery": float(np.mean(all_n_rel_in_gamma_pq)) if all_n_rel_in_gamma_pq else None,
            },
            "per_query": per_query,
        }

    return result
```

- [ ] **Step 2: Verify the function is syntactically correct**

Run: `python -c "from experiments.scripts.pool_null.exp_h4_pipeline_trace import h4_2_relevant_pvalues; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add experiments/scripts/pool_null/exp_h4_pipeline_trace.py
git commit -m "feat(h4): implement H4.2 relevant document p-value analysis"
```

---

### Task 4: Implement H4.3 — Failure Mode Classification

**Files:**
- Modify: `experiments/scripts/pool_null/exp_h4_pipeline_trace.py`

- [ ] **Step 1: Add the `h4_3_failure_modes` function**

```python
# ---------------------------------------------------------------------------
# H4.3 — Failure Mode Classification
# ---------------------------------------------------------------------------

def h4_3_failure_modes(traces, global_null):
    """Classify each irreducible-failure query into a failure mode.

    Modes (applied in order — first match wins):
      1. Weak signal — best relevant p-value (per-query null) > 0.05
      2. Too sparse — relevant docs in gamma window < 3
      3. Z-score distortion — clean z-scoring flips HC positive
      4. Borderline — HC stat (per-query null) > -1.0
      5. Too scattered — remaining cases
    """
    n_gamma = int(np.floor(GAMMA * POOL_SIZE))

    # Identify irreducible queries: negative HC under BOTH global and per-query null
    irreducible = [t for t in traces
                   if t["hc_stat_global"] <= 0 and t["hc_stat_perquery"] <= 0]

    classifications = []
    mode_counts = {"weak_signal": 0, "too_sparse": 0, "zscore_distortion": 0,
                   "borderline": 0, "too_scattered": 0}

    for t in irreducible:
        rpq = np.array(t["relevant_pvals_perquery"])
        ranks_pq = np.array(t["relevant_pval_ranks_perquery"])
        n_rel = len(rpq)

        # Defaults
        best_pval_pq = float(np.min(rpq)) if n_rel > 0 else 1.0
        n_rel_in_gamma = int(np.sum(ranks_pq < n_gamma)) if n_rel > 0 else 0
        n_sig = int(np.sum(rpq < 0.05)) if n_rel > 0 else 0

        # Mode 4 check: clean z-scoring
        # Need to redo z-scoring with relevant docs removed from estimation set
        # We need the raw data — reconstruct from trace
        # Use top100 data (which covers the gamma window) plus summary stats
        # For clean z-scoring we need the full candidate set, but we only stored top-100
        # So we'll flag this and compute it in the data collection loop instead
        # For now, set a placeholder that will be filled by the main loop
        clean_flips_hc = False  # Will be set by caller

        # Classification (order matters)
        if n_rel == 0 or best_pval_pq > 0.05:
            mode = "weak_signal"
        elif n_rel_in_gamma < 3:
            mode = "too_sparse"
        elif clean_flips_hc:
            mode = "zscore_distortion"
        elif t["hc_stat_perquery"] > -1.0:
            mode = "borderline"
        else:
            mode = "too_scattered"

        # Concentration ratio: fraction of top-gamma positions that are relevant
        concentration = n_rel_in_gamma / n_gamma if n_gamma > 0 else 0.0

        mode_counts[mode] += 1
        classifications.append({
            "query_id": t["query_id"],
            "query_text": t["query_text"],
            "mode": mode,
            "hc_stat_global": t["hc_stat_global"],
            "hc_stat_perquery": t["hc_stat_perquery"],
            "best_pval_perquery": best_pval_pq,
            "n_relevant_in_pool": t["n_relevant_in_pool"],
            "n_rel_in_gamma": n_rel_in_gamma,
            "n_significant_005": n_sig,
            "concentration_ratio": concentration,
            "clean_flips_hc": clean_flips_hc,
        })

    return {
        "n_irreducible": len(irreducible),
        "mode_counts": mode_counts,
        "classifications": classifications,
    }
```

- [ ] **Step 2: Verify the function is syntactically correct**

Run: `python -c "from experiments.scripts.pool_null.exp_h4_pipeline_trace import h4_3_failure_modes; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add experiments/scripts/pool_null/exp_h4_pipeline_trace.py
git commit -m "feat(h4): implement H4.3 failure mode classification"
```

---

### Task 5: Implement H4.4 — Targeted Fix Tests

**Files:**
- Modify: `experiments/scripts/pool_null/exp_h4_pipeline_trace.py`

- [ ] **Step 1: Add the `h4_4_targeted_fixes` function**

This function needs access to the raw FAISS results (not just traces) because it recomputes z-scores with clean estimation. It also computes the detection boundary analysis for all 135 queries.

```python
# ---------------------------------------------------------------------------
# H4.4 — Targeted Fix Tests
# ---------------------------------------------------------------------------

def h4_4_targeted_fixes(queries, qrels, all_faiss_results, global_null, traces):
    """Test whether specific interventions rescue failing queries.

    Fix 1: Z-score contamination removal (oracle clean z-scoring)
    Fix 2: Detection boundary analysis (Donoho-Jin phase diagram)
    """
    n_gamma = int(np.floor(GAMMA * POOL_SIZE))

    # Build query_id -> trace index mapping
    trace_by_qid = {t["query_id"]: t for t in traces}

    # Identify irreducible queries
    irreducible_qids = {t["query_id"] for t in traces
                        if t["hc_stat_global"] <= 0 and t["hc_stat_perquery"] <= 0}

    # ------ Fix 1: Contamination removal ------
    fix1_results = []
    fix1_flips = 0

    for qi, query in enumerate(queries):
        if query.query_id not in irreducible_qids:
            continue

        relevant_ids = qrels.get(query.query_id, set())
        faiss_results = all_faiss_results[qi][:POOL_SIZE]

        cids = [cid for cid, _ in faiss_results]
        csims = np.array([sim for _, sim in faiss_results])
        sort_idx = np.argsort(csims)[::-1]
        csims = csims[sort_idx]
        cids = [cids[j] for j in sort_idx]
        rel_mask = np.array([cid in relevant_ids for cid in cids])

        # Standard z-scoring (contaminated — includes relevant in estimation set)
        zs_result = zscore_candidates(csims, Z_SCORE_FRACTION)
        if zs_result is None:
            continue
        z_scores_dirty, mu_dirty, sigma_dirty = zs_result

        # Clean z-scoring (exclude relevant docs from estimation set)
        sorted_desc = np.sort(csims)[::-1]
        n = len(sorted_desc)
        n_null_est = max(int(n * Z_SCORE_FRACTION), 10)
        # Bottom fraction indices in sorted-descending order
        bottom_indices = np.arange(n - n_null_est, n)
        # Which of these are relevant? Use rel_mask on the sorted array
        # csims is already sorted descending, so bottom_indices correspond to
        # the lowest-similarity candidates
        bottom_rel = rel_mask[bottom_indices]
        bottom_clean = sorted_desc[bottom_indices][~bottom_rel]

        if len(bottom_clean) < 10:
            continue

        mu_clean = float(np.mean(bottom_clean))
        sigma_clean = float(np.std(bottom_clean))
        if sigma_clean < 1e-10:
            continue

        z_scores_clean = (csims - mu_clean) / sigma_clean

        # Build per-query null from clean non-relevant z-scores
        nonrel_z_clean = z_scores_clean[~rel_mask]
        if len(nonrel_z_clean) < 10:
            continue
        clean_pq_null = NullDistribution(
            similarities=nonrel_z_clean,
            mean=float(np.mean(nonrel_z_clean)),
            std=float(np.std(nonrel_z_clean)),
            min_val=float(np.min(nonrel_z_clean)),
            max_val=float(np.max(nonrel_z_clean)),
            n_samples=len(nonrel_z_clean),
        )

        # HC with clean z-scores + clean per-query null
        pvals_clean = compute_pvalues_empirical(z_scores_clean, clean_pq_null)
        pvals_sorted = np.sort(pvals_clean)
        hc_curve_clean = compute_hc_curve(pvals_sorted, n_gamma, POOL_SIZE)
        hc_stat_clean = float(np.max(hc_curve_clean))

        t = trace_by_qid[query.query_id]
        hc_stat_original = t["hc_stat_perquery"]
        flipped = hc_stat_original <= 0 and hc_stat_clean > 0
        if flipped:
            fix1_flips += 1

        fix1_results.append({
            "query_id": query.query_id,
            "hc_stat_original_pq": hc_stat_original,
            "hc_stat_clean_pq": hc_stat_clean,
            "delta": hc_stat_clean - hc_stat_original,
            "flipped": flipped,
            "mu_dirty": mu_dirty,
            "mu_clean": mu_clean,
            "sigma_dirty": sigma_dirty,
            "sigma_clean": sigma_clean,
            "n_relevant_in_estimation": int(bottom_rel.sum()),
        })

    # ------ Fix 2: Detection boundary analysis ------
    boundary_results = []

    for t in traces:
        rpq = np.array(t["relevant_pvals_perquery"])
        ranks_pq = np.array(t["relevant_pval_ranks_perquery"])
        n_rel = len(rpq)

        # Signal fraction: relevant docs in gamma window / gamma*n
        n_rel_in_gamma = int(np.sum(ranks_pq < n_gamma)) if n_rel > 0 else 0
        epsilon = n_rel_in_gamma / n_gamma if n_gamma > 0 else 0.0

        # Signal strength: z-score separation
        rel_z = np.array(t["relevant_zscores"])
        # Need non-relevant z-scores — use all_zscores stats
        # We have nonrel_zscores_std and can approximate nonrel mean ≈ all_zscores_mean
        # (since non-relevant are ~99% of candidates)
        # Better: compute from the z-scores of relevant vs overall
        if n_rel > 0 and n_rel_in_gamma > 0:
            # Use the relevant z-scores vs the overall distribution
            # mu_signal = mean z-score of relevant docs in gamma window
            # We need to identify which relevant z-scores are in the gamma window
            # Relevant docs at pval_ranks < n_gamma → those are the ones in the gamma window
            in_gamma_mask = ranks_pq < n_gamma
            rel_z_in_gamma = rel_z[in_gamma_mask] if len(in_gamma_mask) > 0 else np.array([])

            if len(rel_z_in_gamma) > 0:
                nonrel_std = t["nonrel_zscores_std"]
                # Signal strength: how many std devs above the non-relevant mean
                # Non-relevant mean z-score ≈ 0 after z-scoring (by construction)
                # So signal strength ≈ mean(rel_z_in_gamma) / nonrel_std
                mu_signal = float(np.mean(rel_z_in_gamma)) / nonrel_std if nonrel_std > 0 else 0.0
            else:
                mu_signal = 0.0
        else:
            mu_signal = 0.0

        # Detection boundary: mu > sqrt(2 * log(1/epsilon))
        if epsilon > 0:
            boundary_threshold = np.sqrt(2 * np.log(1.0 / epsilon))
            above_boundary = mu_signal > boundary_threshold
        else:
            boundary_threshold = float("inf")
            above_boundary = False

        # Query group
        if t["hc_stat_global"] > 0:
            group = "positive_hc"
        elif t["hc_stat_perquery"] > 0:
            group = "fixable"
        else:
            group = "irreducible"

        boundary_results.append({
            "query_id": t["query_id"],
            "group": group,
            "epsilon": epsilon,
            "mu_signal": mu_signal,
            "boundary_threshold": boundary_threshold,
            "above_boundary": above_boundary,
            "n_rel_in_gamma": n_rel_in_gamma,
            "hc_stat_perquery": t["hc_stat_perquery"],
        })

    # Aggregate boundary stats by group
    boundary_summary = {}
    for group_name in ["positive_hc", "fixable", "irreducible"]:
        group_items = [b for b in boundary_results if b["group"] == group_name]
        if group_items:
            boundary_summary[group_name] = {
                "n": len(group_items),
                "n_above_boundary": sum(1 for b in group_items if b["above_boundary"]),
                "n_below_boundary": sum(1 for b in group_items if not b["above_boundary"]),
                "mean_epsilon": float(np.mean([b["epsilon"] for b in group_items])),
                "mean_mu_signal": float(np.mean([b["mu_signal"] for b in group_items])),
            }

    return {
        "fix1_contamination_removal": {
            "n_tested": len(fix1_results),
            "n_flipped": fix1_flips,
            "per_query": fix1_results,
        },
        "fix2_detection_boundary": {
            "boundary_formula": "mu > sqrt(2 * log(1/epsilon))",
            "summary_by_group": boundary_summary,
            "per_query": boundary_results,
        },
    }
```

- [ ] **Step 2: Verify the function is syntactically correct**

Run: `python -c "from experiments.scripts.pool_null.exp_h4_pipeline_trace import h4_4_targeted_fixes; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add experiments/scripts/pool_null/exp_h4_pipeline_trace.py
git commit -m "feat(h4): implement H4.4 targeted fix tests (contamination + boundary)"
```

---

### Task 6: Implement main() with FAISS loop, integrate H4.1–H4.4, and integrate clean z-scoring into H4.3

**Files:**
- Modify: `experiments/scripts/pool_null/exp_h4_pipeline_trace.py`

- [ ] **Step 1: Add the main function that ties everything together**

The main function: loads config/data, pre-caches FAISS results, runs H4.1 (trace collection), then runs H4.2–H4.4 (analyses on the traces), and saves all results.

The clean z-scoring check for H4.3 Mode 4 needs the raw FAISS data, so we integrate it by running H4.4 Fix 1 first, then passing the flip results to H4.3.

```python
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="H4: Per-query pipeline trace")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    args = parser.parse_args()

    config = load_config(args.config)
    adapter = get_adapter(config.dataset.name)
    queries, qrels, _corpus = adapter.load_dataset(config.dataset.data_dir)

    results_dir = Path(config.dataset.results_dir) / "null_research" / "h4_pipeline_trace"
    out_files = {
        "traces": results_dir / "h4_pipeline_traces.json",
        "pvalues": results_dir / "h4_relevant_pvalues.json",
        "modes": results_dir / "h4_failure_modes.json",
        "fixes": results_dir / "h4_targeted_fixes.json",
    }

    if all(f.exists() for f in out_files.values()) and not args.force:
        print(f"Results exist in {results_dir}. Use --force to rerun.")
        return

    # Load artifacts
    artifacts_dir = Path(config.dataset.artifacts_dir)
    vector_db = VectorDatabase.load(str(artifacts_dir / "vector_db"))
    model = create_embedding_model(
        config.embedding.model,
        normalize_embeddings=config.embedding.normalize,
    )
    global_null = NullDistribution.load(
        str(artifacts_dir / "pool_null" / "global_pool_null")
    )

    print(f"Loaded {len(queries)} queries, null with {global_null.n_samples} samples")
    print(f"Global null: mean={global_null.mean:.4f}, std={global_null.std:.4f}")

    # Pre-cache FAISS results
    print("Running FAISS search...")
    all_faiss_results = []
    for i, query in enumerate(queries):
        query_emb = model.embed_query(query.text)
        candidates = vector_db.search(query_emb, k=POOL_SIZE)
        all_faiss_results.append([(c.doc_id, c.similarity) for c in candidates])
        if (i + 1) % 20 == 0:
            print(f"  FAISS: {i + 1}/{len(queries)}")
    print(f"  FAISS complete: {len(queries)} queries")

    config_dict = {
        "dataset": config.dataset.name,
        "pool_size": POOL_SIZE,
        "z_score_fraction": Z_SCORE_FRACTION,
        "gamma": GAMMA,
        "max_relevant": MAX_RELEVANT,
        "n_queries": len(queries),
        "top_n_stored": TOP_N_STORE,
    }

    # H4.1 — Full pipeline traces
    print("\n=== H4.1: Full Pipeline Trace ===")
    traces = h4_1_pipeline_traces(queries, qrels, all_faiss_results, global_null)
    n_neg_global = sum(1 for t in traces if t["hc_stat_global"] <= 0)
    n_neg_pq = sum(1 for t in traces if t["hc_stat_perquery"] <= 0)
    print(f"  Traced {len(traces)} queries")
    print(f"  Negative HC (global null): {n_neg_global}")
    print(f"  Negative HC (per-query null): {n_neg_pq}")

    save_results_json({"config": config_dict, "traces": traces}, str(out_files["traces"]))
    print(f"  Saved: {out_files['traces']}")

    # H4.4 — Targeted fixes (run BEFORE H4.3 so we can feed clean-flip results)
    print("\n=== H4.4: Targeted Fix Tests ===")
    fixes = h4_4_targeted_fixes(queries, qrels, all_faiss_results, global_null, traces)
    print(f"  Fix 1 (contamination removal): {fixes['fix1_contamination_removal']['n_flipped']}/{fixes['fix1_contamination_removal']['n_tested']} flipped")
    for grp, stats in fixes["fix2_detection_boundary"]["summary_by_group"].items():
        print(f"  Fix 2 ({grp}): {stats['n_above_boundary']}/{stats['n']} above detection boundary")

    save_results_json({"config": config_dict, **fixes}, str(out_files["fixes"]))
    print(f"  Saved: {out_files['fixes']}")

    # Build clean-flip lookup for H4.3
    clean_flip_qids = {r["query_id"] for r in fixes["fix1_contamination_removal"]["per_query"]
                       if r["flipped"]}

    # H4.3 — Failure mode classification (uses clean-flip results from H4.4)
    print("\n=== H4.3: Failure Mode Classification ===")
    modes = h4_3_failure_modes(traces, global_null)
    # Patch in clean z-score flip results
    for c in modes["classifications"]:
        if c["query_id"] in clean_flip_qids:
            c["clean_flips_hc"] = True
    # Re-classify with the patched data
    # Re-run classification logic with clean_flips_hc set
    mode_counts = {"weak_signal": 0, "too_sparse": 0, "zscore_distortion": 0,
                   "borderline": 0, "too_scattered": 0}
    for c in modes["classifications"]:
        if c["mode"] != "zscore_distortion" and c["clean_flips_hc"]:
            # Check if this should be reclassified as zscore_distortion
            # Only reclassify if it wasn't already weak_signal or too_sparse
            # (those modes take priority in the classification order)
            if c["mode"] not in ("weak_signal", "too_sparse"):
                c["mode"] = "zscore_distortion"
        mode_counts[c["mode"]] += 1
    modes["mode_counts"] = mode_counts

    for mode, count in modes["mode_counts"].items():
        print(f"  {mode}: {count}")

    save_results_json({"config": config_dict, **modes}, str(out_files["modes"]))
    print(f"  Saved: {out_files['modes']}")

    # H4.2 — Relevant document p-value analysis
    print("\n=== H4.2: Relevant Document P-Value Analysis ===")
    pvalues = h4_2_relevant_pvalues(traces)
    for grp, stats in pvalues["groups"].items():
        if stats["n"] > 0 and stats.get("aggregate"):
            agg = stats["aggregate"]
            print(f"  {grp} (n={stats['n']}): "
                  f"best_pval_pq={agg['mean_best_pval_perquery']:.4f}, "
                  f"frac_sig_pq={agg['mean_frac_significant_perquery']:.3f}, "
                  f"rel_in_gamma={agg['mean_n_rel_in_gamma_perquery']:.1f}")

    save_results_json({"config": config_dict, **pvalues}, str(out_files["pvalues"]))
    print(f"  Saved: {out_files['pvalues']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full script**

Run: `python -m experiments.scripts.pool_null.exp_h4_pipeline_trace --config experiments/configs/amazon_compound.yaml`

Expected output:
- "Loaded 135 queries, null with 132K samples"
- H4.1: "Traced 135 queries", negative HC counts
- H4.4: contamination flip count, boundary analysis per group
- H4.3: failure mode distribution
- H4.2: p-value statistics per group
- 4 JSON files saved in `results/amazon_compound/null_research/h4_pipeline_trace/`

- [ ] **Step 3: Verify output files exist and are valid JSON**

Run: `python -c "import json; [print(f, len(json.load(open(f)))) for f in ['results/amazon_compound/null_research/h4_pipeline_trace/h4_pipeline_traces.json', 'results/amazon_compound/null_research/h4_pipeline_trace/h4_relevant_pvalues.json', 'results/amazon_compound/null_research/h4_pipeline_trace/h4_failure_modes.json', 'results/amazon_compound/null_research/h4_pipeline_trace/h4_targeted_fixes.json']]"`

Expected: 4 lines, each showing filename and number of top-level keys (2-3 per file).

- [ ] **Step 4: Commit**

```bash
git add experiments/scripts/pool_null/exp_h4_pipeline_trace.py
git add results/amazon_compound/null_research/h4_pipeline_trace/
git commit -m "feat(h4): implement main loop, run all H4 experiments, save results"
```

---

### Task 7: Verify results against spec expectations

**Files:**
- Read: `results/amazon_compound/null_research/h4_pipeline_trace/*.json`
- Read: `docs/superpowers/specs/2026-04-11-hc-pipeline-trace-design.md`

- [ ] **Step 1: Verify H4.1 trace structure**

Run: `python -c "
import json
traces = json.load(open('results/amazon_compound/null_research/h4_pipeline_trace/h4_pipeline_traces.json'))
t0 = traces['traces'][0]
required = ['query_id','query_text','true_k','n_relevant_in_pool','mu_est','sigma_est',
            'top100_sims','top100_is_relevant','top100_zscores','top100_pvals_global',
            'top100_pvals_perquery','hc_curve_global','hc_curve_perquery',
            'hc_stat_global','hc_stat_perquery','k_global','k_perquery',
            'relevant_pvals_global','relevant_pvals_perquery','relevant_zscores',
            'relevant_sims','relevant_pval_ranks_global','relevant_pval_ranks_perquery',
            'all_sims_mean','all_sims_std','std_ratio']
missing = [k for k in required if k not in t0]
print(f'Traces: {len(traces[\"traces\"])} queries')
print(f'HC curve length: {len(t0[\"hc_curve_global\"])}')
print(f'Missing fields: {missing or \"none\"}')
print(f'Top-100 stored: {len(t0[\"top100_sims\"])}')
"
`

Expected: 135 queries, HC curve length 43, no missing fields, 100 top candidates stored.

- [ ] **Step 2: Verify H4.2 group counts**

Run: `python -c "
import json
pv = json.load(open('results/amazon_compound/null_research/h4_pipeline_trace/h4_relevant_pvalues.json'))
for g in ['positive_hc','fixable','irreducible']:
    n = pv['groups'][g]['n']
    print(f'{g}: {n}')
print(f'Total: {sum(pv[\"groups\"][g][\"n\"] for g in [\"positive_hc\",\"fixable\",\"irreducible\"])}')
"
`

Expected: positive_hc ~80, fixable ~10, irreducible ~45, total ~135.

- [ ] **Step 3: Verify H4.3 mode distribution**

Run: `python -c "
import json
modes = json.load(open('results/amazon_compound/null_research/h4_pipeline_trace/h4_failure_modes.json'))
print(f'Irreducible: {modes[\"n_irreducible\"]}')
for mode, count in modes['mode_counts'].items():
    print(f'  {mode}: {count}')
print(f'Total classified: {sum(modes[\"mode_counts\"].values())}')
"
`

Expected: n_irreducible = 45, mode counts sum to 45.

- [ ] **Step 4: Verify H4.4 results**

Run: `python -c "
import json
fixes = json.load(open('results/amazon_compound/null_research/h4_pipeline_trace/h4_targeted_fixes.json'))
f1 = fixes['fix1_contamination_removal']
print(f'Fix 1: {f1[\"n_flipped\"]}/{f1[\"n_tested\"]} flipped')
f2 = fixes['fix2_detection_boundary']
for g, s in f2['summary_by_group'].items():
    print(f'Fix 2 ({g}): {s[\"n_above_boundary\"]}/{s[\"n\"]} above boundary')
"
`

Expected: Fix 1 shows some number of flips (possibly 0-10). Fix 2 shows boundary classification per group.

- [ ] **Step 5: Commit verification (no code changes, just confirming results)**

No commit needed — results were already committed in Task 6.
