# HC Failure Mechanisms — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement three experiment scripts that investigate when HC works and why, per the spec at `docs/superpowers/specs/2026-04-08-hc-failure-mechanisms-design.md`.

**Architecture:** Three independent experiment scripts following the existing `experiments/scripts/pool_null/` pattern. Each script loads config, embeds queries, runs FAISS, and performs analysis. Shared helpers (z-scoring, metrics, null building) are inlined per the existing codebase pattern. Results saved as JSON to `results/amazon_compound/null_research/`.

**Tech Stack:** Python, numpy, scipy.stats, FAISS (via VectorDatabase), existing `hc_rag` package and `experiments.lib` utilities.

---

## File Structure

| File | Responsibility |
|---|---|
| `experiments/scripts/pool_null/exp_h1_uniformity_detection_tradeoff.py` | Direction 1: H1.1 monotonicity + H1.2 artifact test across pool sizes and z_score_fractions |
| `experiments/scripts/pool_null/exp_h2_query_level_failure.py` | Direction 2: H2.1–H2.5 layered query-level failure analysis at pool=1000 |
| `experiments/scripts/pool_null/exp_h3_zscore_null_interaction.py` | Direction 3: H3.1 heterogeneity + H3.2 spread prediction + H3.3 cause isolation |

---

## Task 1: H1 — Uniformity–Detection Tradeoff

**Files:**
- Create: `experiments/scripts/pool_null/exp_h1_uniformity_detection_tradeoff.py`
- Read: `experiments/scripts/pool_null/exp_null_deep_analysis.py` (pattern reference)
- Read: `experiments/configs/amazon_compound.yaml`

**Output:**
- `results/amazon_compound/null_research/h1_uniformity_detection/h1_monotonicity.json`
- `results/amazon_compound/null_research/h1_uniformity_detection/h1_artifact_test.json`

- [ ] **Step 1: Write the experiment script**

Create `experiments/scripts/pool_null/exp_h1_uniformity_detection_tradeoff.py`:

```python
"""Direction 1: Uniformity–detection tradeoff characterization.

H1.1 — Monotonicity test: Run 4-test validation at pool sizes 1K–10K (step 1K)
       and z_score_fractions [0.5, 0.6, 0.7, 0.8]. Produces a 2D grid of
       uniformity pass rate and signal detection rate.

H1.2 — Artifact test: At each (pool_size, z_score_fraction), split queries into
       "uniform" vs "non-uniform" by per-query KS test, compute signal detection
       rate for each group separately.

Usage:
    python -m experiments.scripts.pool_null.exp_h1_uniformity_detection_tradeoff \
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
MAX_RELEVANT = 43  # global max relevant docs across all queries
POOL_SIZES = list(range(1000, 10001, 1000))
Z_SCORE_FRACTIONS = [0.5, 0.6, 0.7, 0.8]


# ---------------------------------------------------------------------------
# Helpers (inlined from exp_null_deep_analysis.py pattern)
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


def build_pool_null_at_size(queries, qrels, all_faiss_results, pool_size, z_score_fraction):
    """Build pool-matched null at a given pool_size from pre-computed FAISS results."""
    all_z_scores = []
    for i, query in enumerate(queries):
        candidates = all_faiss_results[i][:pool_size]
        if len(candidates) == 0:
            continue
        candidate_ids = [c[0] for c in candidates]
        candidate_sims = np.array([c[1] for c in candidates])
        result = zscore_candidates(candidate_sims, z_score_fraction)
        if result is None:
            continue
        z_scores, _, _ = result
        relevant = qrels.get(query.query_id, set())
        nonrel_mask = np.array([cid not in relevant for cid in candidate_ids])
        nonrel_z = z_scores[nonrel_mask]
        if len(nonrel_z) > 0:
            all_z_scores.append(nonrel_z)
    pooled = np.concatenate(all_z_scores)
    return NullDistribution(
        similarities=pooled,
        mean=float(np.mean(pooled)),
        std=float(np.std(pooled)),
        min_val=float(np.min(pooled)),
        max_val=float(np.max(pooled)),
        n_samples=len(pooled),
    )


# ---------------------------------------------------------------------------
# H1.1: Monotonicity test — 4-test validation at each (pool_size, z_frac)
# ---------------------------------------------------------------------------

def run_validation(null_dist, queries, qrels, all_faiss_results,
                   pool_size, z_score_fraction, gamma):
    """Run the 4-test validation gate. Returns dict with per-test results."""
    null_sorted = np.sort(null_dist.similarities)
    N_null = len(null_sorted)
    hc_module = HigherCriticism(null_distribution=null_dist)

    # --- Test 1: Split-half KS ---
    sims = null_dist.similarities
    ks_stats, ks_pvals = [], []
    for seed in range(20):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(sims))
        half = len(idx) // 2
        ref_sorted = np.sort(sims[idx[:half]])
        test_half = sims[idx[half:]]
        pv = np.clip(1.0 - np.searchsorted(ref_sorted, test_half) / len(ref_sorted), 1e-10, 1.0)
        ks_stat, ks_p = sp_stats.kstest(pv, "uniform")
        ks_stats.append(ks_stat)
        ks_pvals.append(ks_p)
    t1_passed = bool(np.median(ks_pvals) > 0.05 and np.median(ks_stats) < 0.1)
    t1 = {"test": "split_half_ks", "passed": t1_passed,
           "median_ks_stat": float(np.median(ks_stats)),
           "median_ks_pval": float(np.median(ks_pvals))}

    # --- Test 2: Per-query uniformity (all queries, not sampled) ---
    # Also collect per-query KS results for H1.2
    per_query_ks = []  # list of (query_index, ks_pval, passed_bool)
    for qi, query in enumerate(queries):
        relevant = qrels.get(query.query_id, set())
        candidates = all_faiss_results[qi][:pool_size]
        if len(candidates) < 10:
            continue
        cids = [c[0] for c in candidates]
        csims = np.array([c[1] for c in candidates])
        res = zscore_candidates(csims, z_score_fraction)
        if res is None:
            continue
        z_scores, _, _ = res
        nonrel_mask = np.array([cid not in relevant for cid in cids])
        nonrel_z = z_scores[nonrel_mask]
        if len(nonrel_z) < 5:
            continue
        pv = np.clip((N_null - np.searchsorted(null_sorted, nonrel_z, side="left") + 1.0) / (N_null + 1.0), 1e-10, 1.0)
        _, ks_p = sp_stats.kstest(pv, "uniform")
        per_query_ks.append((qi, float(ks_p), bool(ks_p > 0.05)))

    n_tested = len(per_query_ks)
    n_passed = sum(1 for _, _, p in per_query_ks if p)
    pass_rate = n_passed / n_tested if n_tested > 0 else 0.0
    t2 = {"test": "per_query_uniformity", "passed": bool(pass_rate >= 0.70),
           "pass_rate": float(pass_rate), "n_queries_tested": n_tested}

    # --- Test 3: Signal detection (all queries) ---
    all_rel_pvals = []
    per_query_signal = []  # for H1.2: (query_index, signal_rate)
    for qi, query in enumerate(queries):
        relevant = qrels.get(query.query_id, set())
        if not relevant:
            continue
        candidates = all_faiss_results[qi][:pool_size]
        if len(candidates) < 10:
            continue
        cids = [c[0] for c in candidates]
        csims = np.array([c[1] for c in candidates])
        res = zscore_candidates(csims, z_score_fraction)
        if res is None:
            continue
        z_scores, _, _ = res
        rel_mask = np.array([cid in relevant for cid in cids])
        rel_z = z_scores[rel_mask]
        if len(rel_z) == 0:
            per_query_signal.append((qi, 0.0))
            continue
        pv = np.clip((N_null - np.searchsorted(null_sorted, rel_z, side="left") + 1.0) / (N_null + 1.0), 1e-10, 1.0)
        rate = float(np.mean(pv < 0.05))
        per_query_signal.append((qi, rate))
        all_rel_pvals.extend(pv.tolist())

    signal_rate = float(np.mean(np.array(all_rel_pvals) < 0.05)) if all_rel_pvals else 0.0
    t3 = {"test": "signal_detection", "passed": bool(signal_rate >= 0.60),
           "signal_rate": signal_rate, "n_relevant_docs": len(all_rel_pvals)}

    # --- Test 4: HC K correlation ---
    hc_stats, hc_ks = [], []
    for qi, query in enumerate(queries):
        candidates = all_faiss_results[qi][:pool_size]
        if len(candidates) < 10:
            continue
        cids = [c[0] for c in candidates]
        csims = np.array([c[1] for c in candidates])
        res = zscore_candidates(csims, z_score_fraction)
        if res is None:
            continue
        z_scores, _, _ = res
        r = hc_module.compute_hc_threshold(z_scores, gamma=gamma)
        hc_stats.append(r.hc_statistic)
        hc_ks.append(r.k)
    mean_hc = float(np.mean(hc_stats)) if hc_stats else 0.0
    unique_ks = len(set(hc_ks))
    t4 = {"test": "hc_k_correlation", "passed": bool(mean_hc > 1.0 and unique_ks >= 3),
           "mean_hc_statistic": mean_hc, "unique_k_values": unique_ks,
           "n_negative_hc": int(sum(1 for s in hc_stats if s < 0))}

    all_passed = all(t["passed"] for t in [t1, t2, t3, t4])

    return {
        "all_passed": all_passed,
        "tests": [t1, t2, t3, t4],
        "per_query_ks": per_query_ks,
        "per_query_signal": per_query_signal,
    }


# ---------------------------------------------------------------------------
# H1.2: Artifact test — signal detection split by uniformity group
# ---------------------------------------------------------------------------

def compute_artifact_test(validation_result):
    """Split signal detection by uniform vs non-uniform queries."""
    per_query_ks = validation_result["per_query_ks"]
    per_query_signal = validation_result["per_query_signal"]

    # Build lookup: query_index -> ks_passed
    ks_map = {qi: passed for qi, _, passed in per_query_ks}
    # Build lookup: query_index -> signal_rate
    sig_map = {qi: rate for qi, rate in per_query_signal}

    # Only include queries that appear in both
    common_qi = set(ks_map.keys()) & set(sig_map.keys())

    uniform_rates = [sig_map[qi] for qi in common_qi if ks_map.get(qi, False)]
    non_uniform_rates = [sig_map[qi] for qi in common_qi if not ks_map.get(qi, False)]

    return {
        "uniform_group": {
            "n_queries": len(uniform_rates),
            "mean_signal_rate": float(np.mean(uniform_rates)) if uniform_rates else 0.0,
            "median_signal_rate": float(np.median(uniform_rates)) if uniform_rates else 0.0,
        },
        "non_uniform_group": {
            "n_queries": len(non_uniform_rates),
            "mean_signal_rate": float(np.mean(non_uniform_rates)) if non_uniform_rates else 0.0,
            "median_signal_rate": float(np.median(non_uniform_rates)) if non_uniform_rates else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="H1: Uniformity-detection tradeoff")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--force", action="store_true", help="Rerun even if output exists")
    args = parser.parse_args()

    config = load_config(args.config)
    adapter = get_adapter(config.dataset.name)
    artifacts_dir = Path(config.dataset.artifacts_dir)
    results_dir = Path(config.dataset.results_dir) / "null_research" / "h1_uniformity_detection"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_mono = results_dir / "h1_monotonicity.json"
    out_artifact = results_dir / "h1_artifact_test.json"

    if out_mono.exists() and out_artifact.exists() and not args.force:
        print(f"Results exist in {results_dir}. Use --force to rerun.")
        return

    # Load data
    queries, qrels, _ = adapter.load_dataset(config.dataset.data_dir)
    print(f"Loaded {len(queries)} queries")

    vector_db = VectorDatabase.load(str(artifacts_dir / "vector_db"))
    print(f"Vector DB: {vector_db.get_num_documents():,} documents")

    model = create_embedding_model(
        config.embedding.model, normalize_embeddings=config.embedding.normalize
    )

    # Pre-compute FAISS at max pool size
    max_pool = max(POOL_SIZES)
    print(f"\nPre-computing FAISS at K={max_pool} for {len(queries)} queries...")
    all_faiss_results = []
    for i, query in enumerate(queries):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Query {i+1}/{len(queries)}")
        query_emb = model.embed_query(query.text)
        candidates = vector_db.search(query_emb, k=max_pool)
        all_faiss_results.append([(c.doc_id, c.similarity) for c in candidates])
    print(f"  Cached {len(all_faiss_results)} queries")

    # ===== H1.1: Monotonicity — 2D grid (pool_size x z_score_fraction) =====
    print("\n" + "=" * 80)
    print("H1.1: MONOTONICITY TEST")
    print("=" * 80)

    mono_results = {}
    artifact_results = {}

    for z_frac in Z_SCORE_FRACTIONS:
        print(f"\n--- z_score_fraction = {z_frac} ---")
        mono_results[str(z_frac)] = {}
        artifact_results[str(z_frac)] = {}

        for ps in POOL_SIZES:
            gamma = MAX_RELEVANT / ps
            print(f"  pool={ps}, gamma={gamma:.4f} ... ", end="", flush=True)

            # Build pool-matched null at this (pool_size, z_frac)
            null_dist = build_pool_null_at_size(
                queries, qrels, all_faiss_results, ps, z_frac
            )

            # Run full validation
            val = run_validation(
                null_dist, queries, qrels, all_faiss_results,
                ps, z_frac, gamma
            )

            # Store H1.1 results (strip per-query data for JSON)
            mono_results[str(z_frac)][str(ps)] = {
                "all_passed": val["all_passed"],
                "tests": val["tests"],
            }

            # Store H1.2 artifact test
            artifact = compute_artifact_test(val)
            artifact_results[str(z_frac)][str(ps)] = artifact

            # Print summary
            t2_rate = val["tests"][1]["pass_rate"]
            t3_rate = val["tests"][2]["signal_rate"]
            n_neg = val["tests"][3]["n_negative_hc"]
            u_n = artifact["uniform_group"]["n_queries"]
            u_sig = artifact["uniform_group"]["mean_signal_rate"]
            nu_n = artifact["non_uniform_group"]["n_queries"]
            nu_sig = artifact["non_uniform_group"]["mean_signal_rate"]
            print(f"unif={t2_rate:.2f} sig={t3_rate:.2f} neg_hc={n_neg} | "
                  f"uniform({u_n}): sig={u_sig:.3f}, non-uniform({nu_n}): sig={nu_sig:.3f}")

    # Save
    save_results_json({
        "config": {
            "dataset": config.dataset.name,
            "pool_sizes": POOL_SIZES,
            "z_score_fractions": Z_SCORE_FRACTIONS,
            "max_relevant": MAX_RELEVANT,
            "n_queries": len(queries),
        },
        "grid": mono_results,
    }, str(out_mono))
    print(f"\nSaved: {out_mono}")

    save_results_json({
        "config": {
            "dataset": config.dataset.name,
            "pool_sizes": POOL_SIZES,
            "z_score_fractions": Z_SCORE_FRACTIONS,
            "max_relevant": MAX_RELEVANT,
            "n_queries": len(queries),
        },
        "grid": artifact_results,
    }, str(out_artifact))
    print(f"Saved: {out_artifact}")

    # ===== Summary table =====
    print("\n" + "=" * 80)
    print("SUMMARY: Uniformity pass rate / Signal detection rate")
    print("=" * 80)
    for z_frac in Z_SCORE_FRACTIONS:
        print(f"\n  z_score_fraction = {z_frac}")
        print(f"  {'Pool':>6} {'Uniformity':>12} {'SignalDet':>12} {'SplitHalf':>12} {'HC-K':>12} {'NegHC':>8}")
        print(f"  {'-'*62}")
        for ps in POOL_SIZES:
            r = mono_results[str(z_frac)][str(ps)]
            t1, t2, t3, t4 = r["tests"]
            print(f"  {ps:>6} {t2['pass_rate']:>12.3f} {t3['signal_rate']:>12.3f} "
                  f"{'PASS' if t1['passed'] else 'FAIL':>12} "
                  f"{'PASS' if t4['passed'] else 'FAIL':>12} "
                  f"{t4['n_negative_hc']:>8}")

    print("\n" + "=" * 80)
    print("H1 complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment**

Run:
```bash
~/miniconda3/envs/adaptivek/python -m experiments.scripts.pool_null.exp_h1_uniformity_detection_tradeoff --config experiments/configs/amazon_compound.yaml
```

Expected: Script runs through 10 pool sizes × 4 z_score_fractions = 40 cells. Prints summary table. Saves two JSON files to `results/amazon_compound/null_research/h1_uniformity_detection/`.

- [ ] **Step 3: Verify output files exist and contain expected structure**

Run:
```bash
ls results/amazon_compound/null_research/h1_uniformity_detection/
```

Expected: `h1_monotonicity.json` and `h1_artifact_test.json` both present.

- [ ] **Step 4: Commit**

```bash
git add experiments/scripts/pool_null/exp_h1_uniformity_detection_tradeoff.py
git add results/amazon_compound/null_research/h1_uniformity_detection/
git commit -m "feat: add H1 uniformity-detection tradeoff experiment

H1.1 monotonicity test across pool sizes 1K-10K and z_score_fractions.
H1.2 artifact test splitting signal detection by uniform vs non-uniform queries."
```

---

## Task 2: H2 — Query-Level Failure Analysis

**Files:**
- Create: `experiments/scripts/pool_null/exp_h2_query_level_failure.py`

**Output:**
- `results/amazon_compound/null_research/h2_query_failure/h2_query_characterization.json`
- `results/amazon_compound/null_research/h2_query_failure/h2_similarity_landscapes.json`
- `results/amazon_compound/null_research/h2_query_failure/h2_rank_positions.json`
- `results/amazon_compound/null_research/h2_query_failure/h2_zscore_contamination.json`
- `results/amazon_compound/null_research/h2_query_failure/h2_null_fit.json`

- [ ] **Step 1: Write the experiment script**

Create `experiments/scripts/pool_null/exp_h2_query_level_failure.py`:

```python
"""Direction 2: What causes HC to fail at the query level?

Five layered experiments at pool=1000:
  H2.1 — Query characterization (negative-HC vs positive-HC)
  H2.2 — Similarity landscape shapes
  H2.3 — Rank position and clustering of relevant docs
  H2.4 — Z-scoring contamination analysis
  H2.5 — Per-query null fit

Usage:
    python -m experiments.scripts.pool_null.exp_h2_query_level_failure \
        --config experiments/configs/amazon_compound.yaml [--force]
"""

import argparse
import re
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
Z_SCORE_FRACTIONS = [0.5, 0.6, 0.7, 0.8]
DEFAULT_Z_FRAC = 0.8


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


def compute_metrics(retrieved_ids, relevant_ids):
    """Recall / precision / F1 for a single query."""
    if len(relevant_ids) == 0:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    hits = len(set(retrieved_ids) & relevant_ids)
    recall = hits / len(relevant_ids)
    precision = hits / len(retrieved_ids) if len(retrieved_ids) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}


# ---------------------------------------------------------------------------
# Shared: compute HC stat for each query, partition into neg/pos
# ---------------------------------------------------------------------------

def compute_per_query_hc(queries, qrels, all_faiss_results, null_dist, z_score_fraction, gamma):
    """Compute HC stat + related data for every query. Returns list of per-query dicts."""
    hc_module = HigherCriticism(null_distribution=null_dist)
    null_sorted = np.sort(null_dist.similarities)
    N_null = len(null_sorted)
    per_query = []

    for qi, query in enumerate(queries):
        relevant = qrels.get(query.query_id, set())
        true_k = len(relevant)
        candidates = all_faiss_results[qi][:POOL_SIZE]
        if len(candidates) < 10:
            continue

        cids = [c[0] for c in candidates]
        csims = np.array([c[1] for c in candidates])
        res = zscore_candidates(csims, z_score_fraction)
        if res is None:
            continue
        z_scores, mu_est, sigma_est = res

        # HC computation
        hc_result = hc_module.compute_hc_threshold(z_scores, gamma=gamma)

        # Relevance masks
        rel_mask = np.array([cid in relevant for cid in cids])
        nonrel_mask = ~rel_mask

        # Rank positions of relevant docs (0-indexed)
        rel_positions = np.where(rel_mask)[0].tolist()  # positions in ranked list

        # Recall ceiling
        n_rel_in_pool = int(np.sum(rel_mask))
        recall_ceiling = n_rel_in_pool / true_k if true_k > 0 else 0.0

        per_query.append({
            "qi": qi,
            "query_id": query.query_id,
            "query_text": query.text,
            "true_k": true_k,
            "n_rel_in_pool": n_rel_in_pool,
            "recall_ceiling": recall_ceiling,
            "hc_stat": hc_result.hc_statistic,
            "hc_k": hc_result.k,
            "hc_negative": hc_result.hc_statistic < 0,
            "mu_est": mu_est,
            "sigma_est": sigma_est,
            "cids": cids,
            "csims": csims,
            "z_scores": z_scores,
            "rel_mask": rel_mask,
            "nonrel_mask": nonrel_mask,
            "rel_positions": rel_positions,
        })

    return per_query


# ---------------------------------------------------------------------------
# H2.1: Query characterization
# ---------------------------------------------------------------------------

def h2_1_query_characterization(per_query):
    """Compare negative-HC vs positive-HC query properties."""
    print("\n" + "=" * 80)
    print("H2.1: QUERY CHARACTERIZATION")
    print("=" * 80)

    neg = [q for q in per_query if q["hc_negative"]]
    pos = [q for q in per_query if not q["hc_negative"]]

    def summarize_group(group, label):
        if not group:
            return {"label": label, "n": 0}
        texts = [q["query_text"] for q in group]
        has_numeric = [bool(re.search(r'\d', t)) for t in texts]
        has_price = [bool(re.search(r'\$|price|under|over|less than|more than|cheap|expensive', t, re.I)) for t in texts]
        return {
            "label": label,
            "n": len(group),
            "mean_true_k": float(np.mean([q["true_k"] for q in group])),
            "std_true_k": float(np.std([q["true_k"] for q in group])),
            "mean_n_rel_in_pool": float(np.mean([q["n_rel_in_pool"] for q in group])),
            "mean_recall_ceiling": float(np.mean([q["recall_ceiling"] for q in group])),
            "mean_query_length": float(np.mean([len(t.split()) for t in texts])),
            "frac_has_numeric": float(np.mean(has_numeric)),
            "frac_has_price_term": float(np.mean(has_price)),
            "mean_hc_stat": float(np.mean([q["hc_stat"] for q in group])),
            "mean_hc_k": float(np.mean([q["hc_k"] for q in group])),
        }

    neg_summary = summarize_group(neg, "negative_hc")
    pos_summary = summarize_group(pos, "positive_hc")

    for s in [neg_summary, pos_summary]:
        if s["n"] == 0:
            continue
        print(f"\n  {s['label'].upper()} ({s['n']} queries):")
        for k, v in s.items():
            if k in ("label", "n"):
                continue
            print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")

    # Per-query detail for negative-HC group
    per_query_detail = []
    for q in per_query:
        per_query_detail.append({
            "query_id": q["query_id"],
            "query_text": q["query_text"],
            "true_k": q["true_k"],
            "n_rel_in_pool": q["n_rel_in_pool"],
            "recall_ceiling": q["recall_ceiling"],
            "hc_stat": q["hc_stat"],
            "hc_k": q["hc_k"],
            "hc_negative": q["hc_negative"],
            "query_length": len(q["query_text"].split()),
            "has_numeric": bool(re.search(r'\d', q["query_text"])),
            "has_price_term": bool(re.search(r'\$|price|under|over|less than|more than|cheap|expensive', q["query_text"], re.I)),
        })

    return {
        "negative_hc": neg_summary,
        "positive_hc": pos_summary,
        "per_query": per_query_detail,
    }


# ---------------------------------------------------------------------------
# H2.2: Similarity landscape shapes
# ---------------------------------------------------------------------------

def h2_2_similarity_landscapes(per_query):
    """Analyze similarity distributions for negative-HC vs positive-HC."""
    print("\n" + "=" * 80)
    print("H2.2: SIMILARITY LANDSCAPE SHAPES")
    print("=" * 80)

    per_query_landscapes = []
    for q in per_query:
        csims = q["csims"]
        rel_mask = q["rel_mask"]
        nonrel_mask = q["nonrel_mask"]

        rel_sims = csims[rel_mask]
        nonrel_sims = csims[nonrel_mask]

        # Overlap: fraction of relevant docs within non-relevant IQR
        if len(nonrel_sims) > 0 and len(rel_sims) > 0:
            q25, q75 = np.percentile(nonrel_sims, [25, 75])
            in_iqr = np.sum((rel_sims >= q25) & (rel_sims <= q75))
            overlap_frac = float(in_iqr / len(rel_sims))
        else:
            overlap_frac = None

        entry = {
            "query_id": q["query_id"],
            "hc_negative": q["hc_negative"],
            "n_rel": int(np.sum(rel_mask)),
            "n_nonrel": int(np.sum(nonrel_mask)),
        }
        if len(rel_sims) > 0:
            entry.update({
                "rel_sim_mean": float(np.mean(rel_sims)),
                "rel_sim_std": float(np.std(rel_sims)),
                "rel_sim_min": float(np.min(rel_sims)),
                "rel_sim_max": float(np.max(rel_sims)),
            })
        if len(nonrel_sims) > 0:
            entry.update({
                "nonrel_sim_mean": float(np.mean(nonrel_sims)),
                "nonrel_sim_std": float(np.std(nonrel_sims)),
                "nonrel_sim_min": float(np.min(nonrel_sims)),
                "nonrel_sim_max": float(np.max(nonrel_sims)),
                "nonrel_sim_skew": float(sp_stats.skew(nonrel_sims)),
            })
        if len(rel_sims) > 0 and len(nonrel_sims) > 0:
            entry["sim_gap"] = float(np.mean(rel_sims) - np.mean(nonrel_sims))
            entry["overlap_in_nonrel_iqr"] = overlap_frac
            # Top-100 non-relevant comparison
            nonrel_sorted = np.sort(nonrel_sims)[::-1]
            top100_nonrel_mean = float(np.mean(nonrel_sorted[:min(100, len(nonrel_sorted))]))
            entry["sim_gap_vs_top100"] = float(np.mean(rel_sims) - top100_nonrel_mean)

        per_query_landscapes.append(entry)

    # Aggregate by group
    neg_entries = [e for e in per_query_landscapes if e["hc_negative"]]
    pos_entries = [e for e in per_query_landscapes if not e["hc_negative"]]

    def agg_group(entries, label):
        if not entries:
            return {"label": label, "n": 0}
        def safe_mean(key):
            vals = [e[key] for e in entries if key in e and e[key] is not None]
            return float(np.mean(vals)) if vals else None
        return {
            "label": label, "n": len(entries),
            "mean_rel_sim_mean": safe_mean("rel_sim_mean"),
            "mean_nonrel_sim_mean": safe_mean("nonrel_sim_mean"),
            "mean_sim_gap": safe_mean("sim_gap"),
            "mean_sim_gap_vs_top100": safe_mean("sim_gap_vs_top100"),
            "mean_overlap_in_nonrel_iqr": safe_mean("overlap_in_nonrel_iqr"),
            "mean_rel_sim_std": safe_mean("rel_sim_std"),
            "mean_nonrel_sim_std": safe_mean("nonrel_sim_std"),
        }

    neg_agg = agg_group(neg_entries, "negative_hc")
    pos_agg = agg_group(pos_entries, "positive_hc")

    for agg in [neg_agg, pos_agg]:
        print(f"\n  {agg['label'].upper()} ({agg['n']} queries):")
        for k, v in agg.items():
            if k in ("label", "n") or v is None:
                continue
            print(f"    {k}: {v:.4f}")

    return {
        "aggregate": {"negative_hc": neg_agg, "positive_hc": pos_agg},
        "per_query": per_query_landscapes,
    }


# ---------------------------------------------------------------------------
# H2.3: Rank position and clustering
# ---------------------------------------------------------------------------

def h2_3_rank_positions(per_query):
    """Analyze where relevant docs sit in the ranked pool."""
    print("\n" + "=" * 80)
    print("H2.3: RANK POSITION AND CLUSTERING")
    print("=" * 80)

    per_query_ranks = []
    for q in per_query:
        positions = q["rel_positions"]  # 0-indexed positions in ranked pool
        n_rel = len(positions)

        entry = {
            "query_id": q["query_id"],
            "hc_negative": q["hc_negative"],
            "true_k": q["true_k"],
            "n_rel_in_pool": q["n_rel_in_pool"],
            "positions": positions,
        }

        if n_rel > 0:
            positions_arr = np.array(positions)
            entry["in_top_50"] = int(np.sum(positions_arr < 50))
            entry["in_top_100"] = int(np.sum(positions_arr < 100))
            entry["in_top_200"] = int(np.sum(positions_arr < 200))
            entry["mean_position"] = float(np.mean(positions_arr))
            entry["median_position"] = float(np.median(positions_arr))
            entry["min_position"] = int(np.min(positions_arr))
            entry["max_position"] = int(np.max(positions_arr))
            entry["position_range"] = int(np.max(positions_arr) - np.min(positions_arr))

            # Clustering metric: mean pairwise distance between positions
            if n_rel > 1:
                sorted_pos = np.sort(positions_arr)
                gaps = np.diff(sorted_pos)
                entry["mean_gap"] = float(np.mean(gaps))
                entry["max_gap"] = int(np.max(gaps))
                # Dispersion: range / count — lower = more clustered
                entry["dispersion"] = float(entry["position_range"] / n_rel)
            else:
                entry["mean_gap"] = 0.0
                entry["max_gap"] = 0
                entry["dispersion"] = 0.0
        else:
            entry.update({
                "in_top_50": 0, "in_top_100": 0, "in_top_200": 0,
                "mean_position": None, "median_position": None,
                "min_position": None, "max_position": None,
                "position_range": None, "mean_gap": None,
                "max_gap": None, "dispersion": None,
            })

        per_query_ranks.append(entry)

    # Aggregate
    neg = [e for e in per_query_ranks if e["hc_negative"]]
    pos = [e for e in per_query_ranks if not e["hc_negative"]]

    def agg_group(entries, label):
        if not entries:
            return {"label": label, "n": 0}
        def safe_mean(key):
            vals = [e[key] for e in entries if e.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        return {
            "label": label, "n": len(entries),
            "mean_n_rel_in_pool": safe_mean("n_rel_in_pool"),
            "mean_in_top_50": safe_mean("in_top_50"),
            "mean_in_top_100": safe_mean("in_top_100"),
            "mean_in_top_200": safe_mean("in_top_200"),
            "mean_position": safe_mean("mean_position"),
            "mean_dispersion": safe_mean("dispersion"),
            "mean_mean_gap": safe_mean("mean_gap"),
        }

    neg_agg = agg_group(neg, "negative_hc")
    pos_agg = agg_group(pos, "positive_hc")

    for agg in [neg_agg, pos_agg]:
        print(f"\n  {agg['label'].upper()} ({agg['n']} queries):")
        for k, v in agg.items():
            if k in ("label", "n") or v is None:
                continue
            print(f"    {k}: {v:.2f}" if isinstance(v, float) else f"    {k}: {v}")

    return {
        "aggregate": {"negative_hc": neg_agg, "positive_hc": pos_agg},
        "per_query": per_query_ranks,
    }


# ---------------------------------------------------------------------------
# H2.4: Z-scoring contamination
# ---------------------------------------------------------------------------

def h2_4_zscore_contamination(queries, qrels, all_faiss_results, null_dist, per_query_data):
    """Check if relevant docs in the estimation set contaminate mu/sigma."""
    print("\n" + "=" * 80)
    print("H2.4: Z-SCORING CONTAMINATION")
    print("=" * 80)

    hc_module = HigherCriticism(null_distribution=null_dist)
    gamma = MAX_RELEVANT / POOL_SIZE

    per_query_contam = []
    for q in per_query_data:
        qi = q["qi"]
        cids = q["cids"]
        csims = q["csims"]
        rel_mask = q["rel_mask"]
        relevant = qrels.get(q["query_id"], set())

        entry = {
            "query_id": q["query_id"],
            "hc_negative": q["hc_negative"],
            "true_k": q["true_k"],
        }

        # For each z_score_fraction, compute contamination
        frac_results = {}
        for z_frac in Z_SCORE_FRACTIONS:
            sorted_desc_idx = np.argsort(csims)[::-1]
            n = len(csims)
            n_null_est = max(int(n * z_frac), 10)
            # Bottom fraction indices (lowest similarity)
            bottom_idx = sorted_desc_idx[-n_null_est:]

            # How many relevant docs are in the estimation set?
            n_rel_in_bottom = int(np.sum(rel_mask[bottom_idx]))
            frac_rel_in_bottom = n_rel_in_bottom / n_null_est

            # Compute mu/sigma WITH contamination (current behavior)
            bottom_sims = csims[bottom_idx]
            mu_with = float(np.mean(bottom_sims))
            sigma_with = float(np.std(bottom_sims))

            # Compute mu/sigma WITHOUT relevant docs
            bottom_nonrel_idx = bottom_idx[~rel_mask[bottom_idx]]
            if len(bottom_nonrel_idx) < 5:
                frac_results[str(z_frac)] = {
                    "n_rel_in_estimation": n_rel_in_bottom,
                    "frac_rel_in_estimation": frac_rel_in_bottom,
                    "too_few_nonrel": True,
                }
                continue

            bottom_nonrel_sims = csims[bottom_nonrel_idx]
            mu_without = float(np.mean(bottom_nonrel_sims))
            sigma_without = float(np.std(bottom_nonrel_sims))

            # Compute HC stat with decontaminated z-scores
            if sigma_without > 1e-10:
                z_decontam = (csims - mu_without) / sigma_without
                hc_decontam = hc_module.compute_hc_threshold(z_decontam, gamma=gamma)
                hc_stat_decontam = hc_decontam.hc_statistic
                hc_k_decontam = hc_decontam.k
            else:
                hc_stat_decontam = None
                hc_k_decontam = None

            frac_results[str(z_frac)] = {
                "n_rel_in_estimation": n_rel_in_bottom,
                "frac_rel_in_estimation": frac_rel_in_bottom,
                "n_estimation_set": n_null_est,
                "mu_with": mu_with,
                "sigma_with": sigma_with,
                "mu_without": mu_without,
                "sigma_without": sigma_without,
                "delta_mu": mu_with - mu_without,
                "delta_sigma": sigma_with - sigma_without,
                "hc_stat_original": q["hc_stat"],
                "hc_stat_decontaminated": hc_stat_decontam,
                "hc_k_decontaminated": hc_k_decontam,
                "flipped_to_positive": bool(
                    q["hc_negative"] and hc_stat_decontam is not None and hc_stat_decontam >= 0
                ),
            }

        entry["by_z_frac"] = frac_results
        per_query_contam.append(entry)

    # Aggregate: how many queries flip from negative to positive?
    for z_frac in Z_SCORE_FRACTIONS:
        z_key = str(z_frac)
        neg_queries = [e for e in per_query_contam if e["hc_negative"]]
        flipped = [e for e in neg_queries
                   if z_key in e["by_z_frac"]
                   and e["by_z_frac"][z_key].get("flipped_to_positive", False)]
        print(f"\n  z_frac={z_frac}: {len(flipped)}/{len(neg_queries)} negative-HC queries "
              f"flipped to positive after decontamination")

        # Mean contamination rate
        contam_rates = [e["by_z_frac"][z_key]["frac_rel_in_estimation"]
                        for e in per_query_contam
                        if z_key in e["by_z_frac"] and "frac_rel_in_estimation" in e["by_z_frac"][z_key]]
        if contam_rates:
            print(f"    Mean contamination rate: {np.mean(contam_rates):.4f}")

    return {"per_query": per_query_contam}


# ---------------------------------------------------------------------------
# H2.5: Per-query null fit
# ---------------------------------------------------------------------------

def h2_5_null_fit(per_query, null_dist):
    """Compare each query's non-relevant z-score distribution to the global null."""
    print("\n" + "=" * 80)
    print("H2.5: PER-QUERY NULL FIT")
    print("=" * 80)

    null_sorted = np.sort(null_dist.similarities)
    null_std = null_dist.std
    null_mean = null_dist.mean

    per_query_fit = []
    for q in per_query:
        z_scores = q["z_scores"]
        nonrel_mask = q["nonrel_mask"]
        nonrel_z = z_scores[nonrel_mask]

        entry = {
            "query_id": q["query_id"],
            "hc_negative": q["hc_negative"],
            "n_nonrel": int(len(nonrel_z)),
        }

        if len(nonrel_z) >= 5:
            # KS test against global null
            # Compare this query's non-relevant z-scores to the global null distribution
            ks_stat, ks_p = sp_stats.ks_2samp(nonrel_z, null_dist.similarities)
            entry["ks_stat"] = float(ks_stat)
            entry["ks_pval"] = float(ks_p)
            entry["ks_passes"] = bool(ks_p > 0.05)

            # Shape comparison
            entry["query_nonrel_mean"] = float(np.mean(nonrel_z))
            entry["query_nonrel_std"] = float(np.std(nonrel_z))
            entry["query_nonrel_skew"] = float(sp_stats.skew(nonrel_z))
            entry["query_nonrel_kurtosis"] = float(sp_stats.kurtosis(nonrel_z))

            # Ratio of query std to null std
            entry["std_ratio"] = float(np.std(nonrel_z) / null_std) if null_std > 0 else None

            # Direction of mismatch
            if entry["std_ratio"] is not None:
                if entry["std_ratio"] < 0.8:
                    entry["mismatch_direction"] = "conservative"  # query narrower than null
                elif entry["std_ratio"] > 1.2:
                    entry["mismatch_direction"] = "anti_conservative"  # query wider than null
                else:
                    entry["mismatch_direction"] = "matched"

        per_query_fit.append(entry)

    # Aggregate
    neg = [e for e in per_query_fit if e["hc_negative"] and "ks_stat" in e]
    pos = [e for e in per_query_fit if not e["hc_negative"] and "ks_stat" in e]

    def agg_group(entries, label):
        if not entries:
            return {"label": label, "n": 0}
        return {
            "label": label, "n": len(entries),
            "mean_ks_stat": float(np.mean([e["ks_stat"] for e in entries])),
            "mean_ks_pval": float(np.mean([e["ks_pval"] for e in entries])),
            "frac_ks_passes": float(np.mean([e["ks_passes"] for e in entries])),
            "mean_std_ratio": float(np.mean([e["std_ratio"] for e in entries if e.get("std_ratio") is not None])),
            "mean_nonrel_std": float(np.mean([e["query_nonrel_std"] for e in entries])),
            "mean_nonrel_skew": float(np.mean([e["query_nonrel_skew"] for e in entries])),
            "frac_conservative": float(np.mean([e.get("mismatch_direction") == "conservative" for e in entries])),
            "frac_anti_conservative": float(np.mean([e.get("mismatch_direction") == "anti_conservative" for e in entries])),
            "frac_matched": float(np.mean([e.get("mismatch_direction") == "matched" for e in entries])),
        }

    neg_agg = agg_group(neg, "negative_hc")
    pos_agg = agg_group(pos, "positive_hc")

    for agg in [neg_agg, pos_agg]:
        print(f"\n  {agg['label'].upper()} ({agg['n']} queries):")
        for k, v in agg.items():
            if k in ("label", "n"):
                continue
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    return {
        "global_null_stats": {"mean": null_mean, "std": null_std},
        "aggregate": {"negative_hc": neg_agg, "positive_hc": pos_agg},
        "per_query": per_query_fit,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="H2: Query-level failure analysis")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--force", action="store_true", help="Rerun")
    args = parser.parse_args()

    config = load_config(args.config)
    adapter = get_adapter(config.dataset.name)
    artifacts_dir = Path(config.dataset.artifacts_dir)
    results_dir = Path(config.dataset.results_dir) / "null_research" / "h2_query_failure"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_files = {
        "h2_1": results_dir / "h2_query_characterization.json",
        "h2_2": results_dir / "h2_similarity_landscapes.json",
        "h2_3": results_dir / "h2_rank_positions.json",
        "h2_4": results_dir / "h2_zscore_contamination.json",
        "h2_5": results_dir / "h2_null_fit.json",
    }
    if all(f.exists() for f in out_files.values()) and not args.force:
        print(f"Results exist in {results_dir}. Use --force to rerun.")
        return

    # Load data
    queries, qrels, _ = adapter.load_dataset(config.dataset.data_dir)
    print(f"Loaded {len(queries)} queries")

    vector_db = VectorDatabase.load(str(artifacts_dir / "vector_db"))
    print(f"Vector DB: {vector_db.get_num_documents():,} documents")

    model = create_embedding_model(
        config.embedding.model, normalize_embeddings=config.embedding.normalize
    )

    # Load pool-matched null at pool=1000
    pool_null_path = artifacts_dir / "pool_null" / "global_pool_null"
    null_dist = NullDistribution.load(str(pool_null_path))
    print(f"Pool-matched null: {null_dist}")

    # Pre-compute FAISS at pool_size=1000
    print(f"\nPre-computing FAISS at K={POOL_SIZE} for {len(queries)} queries...")
    all_faiss_results = []
    for i, query in enumerate(queries):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Query {i+1}/{len(queries)}")
        query_emb = model.embed_query(query.text)
        candidates = vector_db.search(query_emb, k=POOL_SIZE)
        all_faiss_results.append([(c.doc_id, c.similarity) for c in candidates])
    print(f"  Cached {len(all_faiss_results)} queries")

    # Compute per-query HC data
    gamma = MAX_RELEVANT / POOL_SIZE
    print(f"\nComputing HC for all queries (gamma={gamma:.4f})...")
    per_query = compute_per_query_hc(
        queries, qrels, all_faiss_results, null_dist, DEFAULT_Z_FRAC, gamma
    )
    n_neg = sum(1 for q in per_query if q["hc_negative"])
    n_pos = sum(1 for q in per_query if not q["hc_negative"])
    print(f"  {n_neg} negative-HC, {n_pos} positive-HC queries")

    config_dict = {
        "dataset": config.dataset.name,
        "pool_size": POOL_SIZE,
        "z_score_fraction": DEFAULT_Z_FRAC,
        "gamma": gamma,
        "max_relevant": MAX_RELEVANT,
        "n_queries": len(queries),
        "n_queries_analyzed": len(per_query),
        "n_negative_hc": n_neg,
        "n_positive_hc": n_pos,
    }

    # H2.1
    h2_1_results = h2_1_query_characterization(per_query)
    save_results_json({"config": config_dict, **h2_1_results}, str(out_files["h2_1"]))
    print(f"\nSaved: {out_files['h2_1']}")

    # H2.2
    h2_2_results = h2_2_similarity_landscapes(per_query)
    save_results_json({"config": config_dict, **h2_2_results}, str(out_files["h2_2"]))
    print(f"Saved: {out_files['h2_2']}")

    # H2.3
    h2_3_results = h2_3_rank_positions(per_query)
    save_results_json({"config": config_dict, **h2_3_results}, str(out_files["h2_3"]))
    print(f"Saved: {out_files['h2_3']}")

    # H2.4
    h2_4_results = h2_4_zscore_contamination(queries, qrels, all_faiss_results, null_dist, per_query)
    save_results_json({"config": config_dict, **h2_4_results}, str(out_files["h2_4"]))
    print(f"Saved: {out_files['h2_4']}")

    # H2.5
    h2_5_results = h2_5_null_fit(per_query, null_dist)
    save_results_json({"config": config_dict, **h2_5_results}, str(out_files["h2_5"]))
    print(f"Saved: {out_files['h2_5']}")

    print("\n" + "=" * 80)
    print("H2 complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment**

Run:
```bash
~/miniconda3/envs/adaptivek/python -m experiments.scripts.pool_null.exp_h2_query_level_failure --config experiments/configs/amazon_compound.yaml
```

Expected: Script runs all 5 sub-experiments. Prints summaries for each. Saves 5 JSON files.

- [ ] **Step 3: Verify output files**

Run:
```bash
ls results/amazon_compound/null_research/h2_query_failure/
```

Expected: All 5 JSON files present.

- [ ] **Step 4: Commit**

```bash
git add experiments/scripts/pool_null/exp_h2_query_level_failure.py
git add results/amazon_compound/null_research/h2_query_failure/
git commit -m "feat: add H2 query-level failure analysis experiment

Five layered analyses: query characterization, similarity landscapes,
rank positions, z-scoring contamination, per-query null fit."
```

---

## Task 3: H3 — Z-Score × Null Interaction

**Files:**
- Create: `experiments/scripts/pool_null/exp_h3_zscore_null_interaction.py`

**Output:**
- `results/amazon_compound/null_research/h3_zscore_null_interaction/h3_heterogeneity.json`
- `results/amazon_compound/null_research/h3_zscore_null_interaction/h3_spread_prediction.json`
- `results/amazon_compound/null_research/h3_zscore_null_interaction/h3_isolation.json`

- [ ] **Step 1: Write the experiment script**

Create `experiments/scripts/pool_null/exp_h3_zscore_null_interaction.py`:

```python
"""Direction 3: Per-query z-scoring × global null interaction.

H3.1 — Z-score heterogeneity across pool sizes (1K-10K step 1K)
H3.2 — Z-score spread predicts calibration direction (pool=1000)
H3.3 — Isolating the cause: 3 null conditions × 4 z_score_fractions

Usage:
    python -m experiments.scripts.pool_null.exp_h3_zscore_null_interaction \
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
MAX_RELEVANT = 43
POOL_SIZES = list(range(1000, 10001, 1000))
Z_SCORE_FRACTIONS = [0.5, 0.6, 0.7, 0.8]
DEFAULT_Z_FRAC = 0.8


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


def build_pool_null(queries, qrels, all_faiss_results, pool_size, z_score_fraction):
    """Build pool-matched null from pre-computed FAISS results."""
    all_z_scores = []
    for i, query in enumerate(queries):
        candidates = all_faiss_results[i][:pool_size]
        if len(candidates) == 0:
            continue
        cids = [c[0] for c in candidates]
        csims = np.array([c[1] for c in candidates])
        result = zscore_candidates(csims, z_score_fraction)
        if result is None:
            continue
        z_scores, _, _ = result
        relevant = qrels.get(query.query_id, set())
        nonrel_mask = np.array([cid not in relevant for cid in cids])
        nonrel_z = z_scores[nonrel_mask]
        if len(nonrel_z) > 0:
            all_z_scores.append(nonrel_z)
    pooled = np.concatenate(all_z_scores)
    return NullDistribution(
        similarities=pooled, mean=float(np.mean(pooled)),
        std=float(np.std(pooled)), min_val=float(np.min(pooled)),
        max_val=float(np.max(pooled)), n_samples=len(pooled),
    )


def build_raw_sim_null(queries, qrels, all_faiss_results, pool_size):
    """Build null from raw similarities (no z-scoring)."""
    all_sims = []
    for i, query in enumerate(queries):
        candidates = all_faiss_results[i][:pool_size]
        if len(candidates) == 0:
            continue
        cids = [c[0] for c in candidates]
        csims = np.array([c[1] for c in candidates])
        relevant = qrels.get(query.query_id, set())
        nonrel_mask = np.array([cid not in relevant for cid in cids])
        nonrel_sims = csims[nonrel_mask]
        if len(nonrel_sims) > 0:
            all_sims.append(nonrel_sims)
    pooled = np.concatenate(all_sims)
    return NullDistribution(
        similarities=pooled, mean=float(np.mean(pooled)),
        std=float(np.std(pooled)), min_val=float(np.min(pooled)),
        max_val=float(np.max(pooled)), n_samples=len(pooled),
    )


def compute_metrics(retrieved_ids, relevant_ids):
    """Recall / precision / F1."""
    if len(relevant_ids) == 0:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    hits = len(set(retrieved_ids) & relevant_ids)
    recall = hits / len(relevant_ids)
    precision = hits / len(retrieved_ids) if len(retrieved_ids) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}


# ---------------------------------------------------------------------------
# H3.1: Z-score heterogeneity across pool sizes
# ---------------------------------------------------------------------------

def h3_1_heterogeneity(queries, qrels, all_faiss_results):
    """Measure cross-query variance of z-score statistics at each pool size."""
    print("\n" + "=" * 80)
    print("H3.1: Z-SCORE HETEROGENEITY ACROSS POOL SIZES")
    print("=" * 80)

    results = {}
    for ps in POOL_SIZES:
        per_query_stats = []
        for i, query in enumerate(queries):
            candidates = all_faiss_results[i][:ps]
            if len(candidates) < 10:
                continue
            csims = np.array([c[1] for c in candidates])
            res = zscore_candidates(csims, DEFAULT_Z_FRAC)
            if res is None:
                continue
            z_scores, mu, sigma = res
            per_query_stats.append({
                "std": float(np.std(z_scores)),
                "skewness": float(sp_stats.skew(z_scores)),
                "kurtosis": float(sp_stats.kurtosis(z_scores)),
                "mu_est": mu,
                "sigma_est": sigma,
            })

        stds = [s["std"] for s in per_query_stats]
        skews = [s["skewness"] for s in per_query_stats]
        kurts = [s["kurtosis"] for s in per_query_stats]
        sigma_ests = [s["sigma_est"] for s in per_query_stats]

        results[str(ps)] = {
            "n_queries": len(per_query_stats),
            "cross_query_variance_of_std": float(np.var(stds)),
            "cross_query_variance_of_skewness": float(np.var(skews)),
            "cross_query_variance_of_kurtosis": float(np.var(kurts)),
            "cross_query_variance_of_sigma_est": float(np.var(sigma_ests)),
            "mean_std": float(np.mean(stds)),
            "mean_skewness": float(np.mean(skews)),
            "mean_kurtosis": float(np.mean(kurts)),
            "mean_sigma_est": float(np.mean(sigma_ests)),
            "std_of_sigma_est": float(np.std(sigma_ests)),
        }

        r = results[str(ps)]
        print(f"  pool={ps}: var(std)={r['cross_query_variance_of_std']:.6f}  "
              f"var(skew)={r['cross_query_variance_of_skewness']:.6f}  "
              f"var(sigma_est)={r['cross_query_variance_of_sigma_est']:.6f}")

    return results


# ---------------------------------------------------------------------------
# H3.2: Z-score spread predicts calibration direction
# ---------------------------------------------------------------------------

def h3_2_spread_prediction(queries, qrels, all_faiss_results, null_dist):
    """Correlate per-query z-score std ratio with HC stat and calibration."""
    print("\n" + "=" * 80)
    print("H3.2: Z-SCORE SPREAD PREDICTS CALIBRATION DIRECTION")
    print("=" * 80)

    hc_module = HigherCriticism(null_distribution=null_dist)
    null_sorted = np.sort(null_dist.similarities)
    N_null = len(null_sorted)
    gamma = MAX_RELEVANT / POOL_SIZE

    per_query = []
    for i, query in enumerate(queries):
        relevant = qrels.get(query.query_id, set())
        candidates = all_faiss_results[i][:POOL_SIZE]
        if len(candidates) < 10:
            continue
        cids = [c[0] for c in candidates]
        csims = np.array([c[1] for c in candidates])
        res = zscore_candidates(csims, DEFAULT_Z_FRAC)
        if res is None:
            continue
        z_scores, mu, sigma = res

        # HC stat
        hc_result = hc_module.compute_hc_threshold(z_scores, gamma=gamma)

        # Per-query z-score statistics
        nonrel_mask = np.array([cid not in relevant for cid in cids])
        nonrel_z = z_scores[nonrel_mask]
        query_nonrel_std = float(np.std(nonrel_z)) if len(nonrel_z) > 0 else None

        # Per-query KS test (uniformity)
        if len(nonrel_z) >= 5:
            pv = np.clip((N_null - np.searchsorted(null_sorted, nonrel_z, side="left") + 1.0) / (N_null + 1.0), 1e-10, 1.0)
            ks_stat, ks_p = sp_stats.kstest(pv, "uniform")
        else:
            ks_stat, ks_p = None, None

        # Signal detection rate
        rel_mask = np.array([cid in relevant for cid in cids])
        rel_z = z_scores[rel_mask]
        if len(rel_z) > 0:
            rel_pv = np.clip((N_null - np.searchsorted(null_sorted, rel_z, side="left") + 1.0) / (N_null + 1.0), 1e-10, 1.0)
            signal_rate = float(np.mean(rel_pv < 0.05))
        else:
            signal_rate = None

        std_ratio = query_nonrel_std / null_dist.std if query_nonrel_std and null_dist.std > 0 else None

        per_query.append({
            "query_id": query.query_id,
            "hc_stat": hc_result.hc_statistic,
            "hc_k": hc_result.k,
            "hc_negative": hc_result.hc_statistic < 0,
            "query_nonrel_std": query_nonrel_std,
            "null_std": null_dist.std,
            "std_ratio": std_ratio,
            "ks_stat": ks_stat,
            "ks_pval": ks_p,
            "signal_rate": signal_rate,
            "sigma_est": sigma,
        })

    # Correlations
    valid = [q for q in per_query if q["std_ratio"] is not None]
    std_ratios = np.array([q["std_ratio"] for q in valid])
    hc_stats = np.array([q["hc_stat"] for q in valid])
    hc_signs = np.array([1 if q["hc_negative"] else 0 for q in valid])

    corr_hc_stat = float(sp_stats.pearsonr(std_ratios, hc_stats)[0]) if len(valid) > 2 else None
    corr_hc_sign = float(sp_stats.pointbiserialr(hc_signs, std_ratios)[0]) if len(valid) > 2 else None

    valid_ks = [q for q in valid if q["ks_stat"] is not None]
    ks_stats_arr = np.array([q["ks_stat"] for q in valid_ks])
    ratios_ks = np.array([q["std_ratio"] for q in valid_ks])
    corr_ks = float(sp_stats.pearsonr(ratios_ks, ks_stats_arr)[0]) if len(valid_ks) > 2 else None

    valid_sig = [q for q in valid if q["signal_rate"] is not None]
    sig_rates = np.array([q["signal_rate"] for q in valid_sig])
    ratios_sig = np.array([q["std_ratio"] for q in valid_sig])
    corr_signal = float(sp_stats.pearsonr(ratios_sig, sig_rates)[0]) if len(valid_sig) > 2 else None

    correlations = {
        "std_ratio_vs_hc_stat": corr_hc_stat,
        "std_ratio_vs_hc_negative": corr_hc_sign,
        "std_ratio_vs_ks_stat": corr_ks,
        "std_ratio_vs_signal_rate": corr_signal,
    }

    print(f"\n  Correlations:")
    for k, v in correlations.items():
        print(f"    {k}: {v:.4f}" if v is not None else f"    {k}: N/A")

    # Summary by std_ratio bins
    print(f"\n  Summary by std_ratio bins:")
    bins = [(0, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, float("inf"))]
    bin_labels = ["<0.8 (conservative)", "0.8-1.0", "1.0-1.2", ">1.2 (anti-conservative)"]
    for (lo, hi), label in zip(bins, bin_labels):
        in_bin = [q for q in valid if lo <= q["std_ratio"] < hi]
        if not in_bin:
            continue
        n_neg = sum(1 for q in in_bin if q["hc_negative"])
        mean_hc = float(np.mean([q["hc_stat"] for q in in_bin]))
        print(f"    {label}: n={len(in_bin)}, neg_hc={n_neg}, mean_hc={mean_hc:.2f}")

    return {
        "correlations": correlations,
        "null_std": null_dist.std,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# H3.3: Isolating the cause — 3 conditions × 4 z_score_fractions
# ---------------------------------------------------------------------------

def h3_3_isolation(queries, qrels, all_faiss_results):
    """Test 3 null conditions × 4 z_score_fractions at pool=1000."""
    print("\n" + "=" * 80)
    print("H3.3: ISOLATING THE CAUSE")
    print("=" * 80)

    gamma = MAX_RELEVANT / POOL_SIZE
    results = {}

    for z_frac in Z_SCORE_FRACTIONS:
        print(f"\n--- z_score_fraction = {z_frac} ---")
        results[str(z_frac)] = {}

        # Condition (a): per-query z-scoring + global null (current pipeline)
        print("  Building condition (a): z-scored + global null...")
        global_null = build_pool_null(queries, qrels, all_faiss_results, POOL_SIZE, z_frac)
        a_result = evaluate_condition(
            queries, qrels, all_faiss_results, global_null,
            gamma, z_frac, use_zscore=True, label="zscore_global_null"
        )
        results[str(z_frac)]["zscore_global_null"] = a_result

        # Condition (b): raw similarities + global null (no z-scoring)
        print("  Building condition (b): raw sims + global null...")
        raw_null = build_raw_sim_null(queries, qrels, all_faiss_results, POOL_SIZE)
        b_result = evaluate_condition(
            queries, qrels, all_faiss_results, raw_null,
            gamma, z_frac, use_zscore=False, label="raw_global_null"
        )
        results[str(z_frac)]["raw_global_null"] = b_result

        # Condition (c): per-query z-scoring + per-query null
        print("  Running condition (c): z-scored + per-query null...")
        c_result = evaluate_per_query_null_condition(
            queries, qrels, all_faiss_results, gamma, z_frac
        )
        results[str(z_frac)]["zscore_per_query_null"] = c_result

    # Print comparison table
    print("\n" + "=" * 80)
    print("H3.3 COMPARISON TABLE")
    print("=" * 80)
    conditions = ["zscore_global_null", "raw_global_null", "zscore_per_query_null"]
    print(f"\n  {'z_frac':>8} {'Condition':<25} {'Unif%':>8} {'SigDet%':>8} {'NegHC':>8} {'MeanHC':>8} {'MeanK':>8} {'F1':>8}")
    print(f"  {'-'*88}")
    for z_frac in Z_SCORE_FRACTIONS:
        for cond in conditions:
            r = results[str(z_frac)][cond]
            print(f"  {z_frac:>8.1f} {cond:<25} {r['uniformity_pass_rate']:>8.3f} "
                  f"{r['signal_detection_rate']:>8.3f} {r['n_negative_hc']:>8} "
                  f"{r['mean_hc_stat']:>8.2f} {r['mean_k']:>8.1f} {r['mean_f1']:>8.4f}")

    return results


def evaluate_condition(queries, qrels, all_faiss_results, null_dist,
                       gamma, z_score_fraction, use_zscore, label):
    """Evaluate a null condition: uniformity, signal detection, HC stats, F1."""
    hc_module = HigherCriticism(null_distribution=null_dist)
    null_sorted = np.sort(null_dist.similarities)
    N_null = len(null_sorted)

    ks_passes = 0
    ks_tested = 0
    all_rel_pvals = []
    hc_stats = []
    hc_ks = []
    f1s = []

    for i, query in enumerate(queries):
        relevant = qrels.get(query.query_id, set())
        candidates = all_faiss_results[i][:POOL_SIZE]
        if len(candidates) < 10:
            continue
        cids = [c[0] for c in candidates]
        csims = np.array([c[1] for c in candidates])

        if use_zscore:
            res = zscore_candidates(csims, z_score_fraction)
            if res is None:
                continue
            values, _, _ = res
        else:
            values = csims

        # HC
        hc_result = hc_module.compute_hc_threshold(values, gamma=gamma)
        hc_stats.append(hc_result.hc_statistic)
        hc_ks.append(hc_result.k)

        # F1
        retrieved_ids = [cids[j] for j in range(min(hc_result.k, len(cids)))]
        m = compute_metrics(retrieved_ids, relevant)
        f1s.append(m["f1"])

        # Uniformity (non-relevant)
        rel_mask = np.array([cid in relevant for cid in cids])
        nonrel_values = values[~rel_mask]
        if len(nonrel_values) >= 5:
            pv = np.clip((N_null - np.searchsorted(null_sorted, nonrel_values, side="left") + 1.0) / (N_null + 1.0), 1e-10, 1.0)
            _, ks_p = sp_stats.kstest(pv, "uniform")
            ks_tested += 1
            if ks_p > 0.05:
                ks_passes += 1

        # Signal detection (relevant)
        rel_values = values[rel_mask]
        if len(rel_values) > 0:
            pv = np.clip((N_null - np.searchsorted(null_sorted, rel_values, side="left") + 1.0) / (N_null + 1.0), 1e-10, 1.0)
            all_rel_pvals.extend(pv.tolist())

    return {
        "label": label,
        "uniformity_pass_rate": ks_passes / ks_tested if ks_tested > 0 else 0.0,
        "signal_detection_rate": float(np.mean(np.array(all_rel_pvals) < 0.05)) if all_rel_pvals else 0.0,
        "mean_hc_stat": float(np.mean(hc_stats)) if hc_stats else 0.0,
        "n_negative_hc": int(sum(1 for s in hc_stats if s < 0)),
        "mean_k": float(np.mean(hc_ks)) if hc_ks else 0.0,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "n_queries": ks_tested,
    }


def evaluate_per_query_null_condition(queries, qrels, all_faiss_results, gamma, z_score_fraction):
    """Condition (c): each query gets its own null built from its own non-relevant z-scores."""
    ks_passes = 0
    ks_tested = 0
    all_rel_pvals = []
    hc_stats = []
    hc_ks = []
    f1s = []

    for i, query in enumerate(queries):
        relevant = qrels.get(query.query_id, set())
        candidates = all_faiss_results[i][:POOL_SIZE]
        if len(candidates) < 10:
            continue
        cids = [c[0] for c in candidates]
        csims = np.array([c[1] for c in candidates])

        res = zscore_candidates(csims, z_score_fraction)
        if res is None:
            continue
        z_scores, _, _ = res

        rel_mask = np.array([cid in relevant for cid in cids])
        nonrel_z = z_scores[~rel_mask]
        rel_z = z_scores[rel_mask]

        if len(nonrel_z) < 10:
            continue

        # Build per-query null from this query's non-relevant z-scores
        per_query_null = NullDistribution(
            similarities=nonrel_z,
            mean=float(np.mean(nonrel_z)),
            std=float(np.std(nonrel_z)),
            min_val=float(np.min(nonrel_z)),
            max_val=float(np.max(nonrel_z)),
            n_samples=len(nonrel_z),
        )

        hc_module = HigherCriticism(null_distribution=per_query_null)
        null_sorted = np.sort(nonrel_z)
        N_null = len(null_sorted)

        # HC
        hc_result = hc_module.compute_hc_threshold(z_scores, gamma=gamma)
        hc_stats.append(hc_result.hc_statistic)
        hc_ks.append(hc_result.k)

        # F1
        retrieved_ids = [cids[j] for j in range(min(hc_result.k, len(cids)))]
        m = compute_metrics(retrieved_ids, relevant)
        f1s.append(m["f1"])

        # Uniformity
        if len(nonrel_z) >= 5:
            pv = np.clip((N_null - np.searchsorted(null_sorted, nonrel_z, side="left") + 1.0) / (N_null + 1.0), 1e-10, 1.0)
            _, ks_p = sp_stats.kstest(pv, "uniform")
            ks_tested += 1
            if ks_p > 0.05:
                ks_passes += 1

        # Signal detection
        if len(rel_z) > 0:
            pv = np.clip((N_null - np.searchsorted(null_sorted, rel_z, side="left") + 1.0) / (N_null + 1.0), 1e-10, 1.0)
            all_rel_pvals.extend(pv.tolist())

    return {
        "label": "zscore_per_query_null",
        "uniformity_pass_rate": ks_passes / ks_tested if ks_tested > 0 else 0.0,
        "signal_detection_rate": float(np.mean(np.array(all_rel_pvals) < 0.05)) if all_rel_pvals else 0.0,
        "mean_hc_stat": float(np.mean(hc_stats)) if hc_stats else 0.0,
        "n_negative_hc": int(sum(1 for s in hc_stats if s < 0)),
        "mean_k": float(np.mean(hc_ks)) if hc_ks else 0.0,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "n_queries": ks_tested,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="H3: Z-score × null interaction")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--force", action="store_true", help="Rerun")
    args = parser.parse_args()

    config = load_config(args.config)
    adapter = get_adapter(config.dataset.name)
    artifacts_dir = Path(config.dataset.artifacts_dir)
    results_dir = Path(config.dataset.results_dir) / "null_research" / "h3_zscore_null_interaction"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_files = {
        "h3_1": results_dir / "h3_heterogeneity.json",
        "h3_2": results_dir / "h3_spread_prediction.json",
        "h3_3": results_dir / "h3_isolation.json",
    }
    if all(f.exists() for f in out_files.values()) and not args.force:
        print(f"Results exist in {results_dir}. Use --force to rerun.")
        return

    # Load data
    queries, qrels, _ = adapter.load_dataset(config.dataset.data_dir)
    print(f"Loaded {len(queries)} queries")

    vector_db = VectorDatabase.load(str(artifacts_dir / "vector_db"))
    print(f"Vector DB: {vector_db.get_num_documents():,} documents")

    model = create_embedding_model(
        config.embedding.model, normalize_embeddings=config.embedding.normalize
    )

    # Pre-compute FAISS at max pool size (10K for H3.1)
    max_pool = max(POOL_SIZES)
    print(f"\nPre-computing FAISS at K={max_pool} for {len(queries)} queries...")
    all_faiss_results = []
    for i, query in enumerate(queries):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Query {i+1}/{len(queries)}")
        query_emb = model.embed_query(query.text)
        candidates = vector_db.search(query_emb, k=max_pool)
        all_faiss_results.append([(c.doc_id, c.similarity) for c in candidates])
    print(f"  Cached {len(all_faiss_results)} queries")

    config_dict = {
        "dataset": config.dataset.name,
        "pool_sizes": POOL_SIZES,
        "z_score_fractions": Z_SCORE_FRACTIONS,
        "max_relevant": MAX_RELEVANT,
        "n_queries": len(queries),
    }

    # H3.1: Heterogeneity
    h3_1_results = h3_1_heterogeneity(queries, qrels, all_faiss_results)
    save_results_json({"config": config_dict, "by_pool_size": h3_1_results}, str(out_files["h3_1"]))
    print(f"\nSaved: {out_files['h3_1']}")

    # H3.2: Spread prediction (needs pool null at pool=1000)
    pool_null_path = artifacts_dir / "pool_null" / "global_pool_null"
    null_dist = NullDistribution.load(str(pool_null_path))
    print(f"\nPool-matched null: {null_dist}")

    h3_2_results = h3_2_spread_prediction(queries, qrels, all_faiss_results, null_dist)
    save_results_json({"config": config_dict, **h3_2_results}, str(out_files["h3_2"]))
    print(f"Saved: {out_files['h3_2']}")

    # H3.3: Isolation
    h3_3_results = h3_3_isolation(queries, qrels, all_faiss_results)
    save_results_json({"config": config_dict, "grid": h3_3_results}, str(out_files["h3_3"]))
    print(f"Saved: {out_files['h3_3']}")

    print("\n" + "=" * 80)
    print("H3 complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment**

Run:
```bash
~/miniconda3/envs/adaptivek/python -m experiments.scripts.pool_null.exp_h3_zscore_null_interaction --config experiments/configs/amazon_compound.yaml
```

Expected: H3.1 iterates 10 pool sizes. H3.2 computes correlations at pool=1000. H3.3 runs 3×4=12 conditions. Saves 3 JSON files.

- [ ] **Step 3: Verify output files**

Run:
```bash
ls results/amazon_compound/null_research/h3_zscore_null_interaction/
```

Expected: `h3_heterogeneity.json`, `h3_spread_prediction.json`, `h3_isolation.json` all present.

- [ ] **Step 4: Commit**

```bash
git add experiments/scripts/pool_null/exp_h3_zscore_null_interaction.py
git add results/amazon_compound/null_research/h3_zscore_null_interaction/
git commit -m "feat: add H3 z-score × null interaction experiment

H3.1 heterogeneity across pool sizes, H3.2 spread prediction correlations,
H3.3 cause isolation: 3 null conditions × 4 z_score_fractions."
```

---

## Task 4: Final verification and cleanup

- [ ] **Step 1: Verify all three scripts run without error**

Run each in sequence:
```bash
~/miniconda3/envs/adaptivek/python -m experiments.scripts.pool_null.exp_h1_uniformity_detection_tradeoff --config experiments/configs/amazon_compound.yaml --force
~/miniconda3/envs/adaptivek/python -m experiments.scripts.pool_null.exp_h2_query_level_failure --config experiments/configs/amazon_compound.yaml --force
~/miniconda3/envs/adaptivek/python -m experiments.scripts.pool_null.exp_h3_zscore_null_interaction --config experiments/configs/amazon_compound.yaml --force
```

- [ ] **Step 2: Verify all output files**

Run:
```bash
ls -la results/amazon_compound/null_research/h1_uniformity_detection/
ls -la results/amazon_compound/null_research/h2_query_failure/
ls -la results/amazon_compound/null_research/h3_zscore_null_interaction/
```

Expected:
- h1: 2 files (h1_monotonicity.json, h1_artifact_test.json)
- h2: 5 files (h2_query_characterization.json, h2_similarity_landscapes.json, h2_rank_positions.json, h2_zscore_contamination.json, h2_null_fit.json)
- h3: 3 files (h3_heterogeneity.json, h3_spread_prediction.json, h3_isolation.json)

- [ ] **Step 3: Commit all results**

```bash
git add results/amazon_compound/null_research/h1_uniformity_detection/
git add results/amazon_compound/null_research/h2_query_failure/
git add results/amazon_compound/null_research/h3_zscore_null_interaction/
git commit -m "results: add HC failure mechanisms experiment results

All three directions complete with JSON output files."
```
