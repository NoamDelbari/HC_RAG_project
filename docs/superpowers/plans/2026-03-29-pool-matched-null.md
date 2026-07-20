# Pool-Matched Null Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pool-matched null distribution that uses the same FAISS top-K retrieval + z-scoring procedure as HC inference, fixing the calibration mismatch in the current approach.

**Architecture:** Three standalone scripts in `experiments/scripts/pool_null/` (build, validate, compare). All use the existing `NullDistribution` class for storage. Existing null code is untouched.

**Tech Stack:** numpy, scipy.stats, faiss (via VectorDatabase), existing hc_rag package

---

### Task 1: Create the `pool_null` package with `build_pool_null.py`

**Files:**
- Create: `experiments/scripts/pool_null/__init__.py`
- Create: `experiments/scripts/pool_null/build_pool_null.py`

- [ ] **Step 1: Create empty `__init__.py`**

```python
# experiments/scripts/pool_null/__init__.py
```

- [ ] **Step 2: Write `build_pool_null.py`**

```python
"""Build pool-matched global null distribution.

Uses the same FAISS top-K retrieval + z-scoring procedure that HC uses at
inference time, fixing the calibration mismatch in the original global null.

Key differences from build_null.py:
  - Retrieves top pool_size candidates via FAISS (not all docs)
  - Z-scores using bottom z_score_fraction of pool (not all non-relevant docs)
  - Only non-relevant z-scores enter the null (relevant docs excluded from null)

Usage: python -m experiments.scripts.pool_null.build_pool_null --config experiments/configs/amazon_compound.yaml
"""

import argparse
import numpy as np
from pathlib import Path

from hc_rag.embeddings.embedding_model import create_embedding_model
from hc_rag.embeddings.vector_database import VectorDatabase
from hc_rag.hc.null_distribution import NullDistribution

from experiments.lib.config import load_config
from experiments.lib.registry import get_adapter


def build_pool_matched_null(queries, qrels, vector_db, embedding_model, pool_size, z_score_fraction):
    """Build global null using pool-matched procedure.

    For each query:
      1. FAISS search for top pool_size candidates
      2. Z-score ALL candidates using bottom z_score_fraction of pool
         (includes relevant docs — matches inference)
      3. Add ONLY non-relevant z-scores to the global pool
    """
    all_z_scores = []
    skipped = 0
    stats = {"total_nonrel_in_pool": 0, "total_rel_in_pool": 0}

    print(f"Building pool-matched null distribution...")
    print(f"  Queries: {len(queries)}")
    print(f"  Pool size: {pool_size}")
    print(f"  Z-score fraction: {z_score_fraction}")

    for i, query in enumerate(queries):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(queries)} queries")

        relevant = qrels.get(query.query_id, set())
        query_emb = embedding_model.embed([query.text], show_progress=False)[0]

        # Step 1: FAISS search for top pool_size candidates (same as HC inference)
        results = vector_db.search(query_emb.reshape(1, -1), k=pool_size)
        if not results or not results[0]:
            skipped += 1
            continue

        candidate_ids = [r.doc_id for r in results[0]]
        candidate_sims = np.array([r.score for r in results[0]])

        # Step 2: Z-score using bottom fraction (same as HC inference)
        sorted_desc = np.sort(candidate_sims)[::-1]
        n = len(sorted_desc)
        n_null_est = max(int(n * z_score_fraction), 10)
        bottom = sorted_desc[-n_null_est:]
        mu_est = np.mean(bottom)
        sigma_est = np.std(bottom)

        if sigma_est < 1e-12:
            skipped += 1
            continue

        z_scores = (candidate_sims - mu_est) / sigma_est

        # Step 3: Only add NON-RELEVANT z-scores to the null
        is_relevant = np.array([cid in relevant for cid in candidate_ids])
        nonrel_z = z_scores[~is_relevant]
        rel_count = int(np.sum(is_relevant))
        nonrel_count = len(nonrel_z)

        stats["total_nonrel_in_pool"] += nonrel_count
        stats["total_rel_in_pool"] += rel_count

        if len(nonrel_z) > 0:
            all_z_scores.append(nonrel_z)

    if skipped > 0:
        print(f"  Skipped {skipped} queries (no results or zero std)")

    pooled = np.concatenate(all_z_scores)
    print(f"  Total pooled z-scores: {len(pooled):,}")
    print(f"  Total non-relevant in pools: {stats['total_nonrel_in_pool']:,}")
    print(f"  Total relevant in pools: {stats['total_rel_in_pool']:,}")
    print(f"  Relevant fraction in pools: {stats['total_rel_in_pool'] / (stats['total_nonrel_in_pool'] + stats['total_rel_in_pool']):.4f}")

    return NullDistribution(
        similarities=pooled,
        mean=float(np.mean(pooled)),
        std=float(np.std(pooled)),
        min_val=float(np.min(pooled)),
        max_val=float(np.max(pooled)),
        n_samples=len(pooled),
    )


def main():
    parser = argparse.ArgumentParser(description="Build pool-matched null distribution")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = parser.parse_args()

    config = load_config(args.config)
    import experiments.datasets  # noqa: F401
    adapter = get_adapter(config.dataset.name)

    artifacts_dir = Path(config.dataset.artifacts_dir)
    null_dir = artifacts_dir / "pool_null"
    null_path = null_dir / "global_pool_null"
    if Path(str(null_path) + ".pkl").exists() and not args.force:
        print(f"Pool-matched null already exists at {null_path}. Use --force to rebuild.")
        return

    queries, qrels, _ = adapter.load_dataset(config.dataset.data_dir)

    db_path = str(artifacts_dir / "vector_db")
    print(f"Loading vector database from {db_path}...")
    vector_db = VectorDatabase.load(db_path)
    print(f"  Documents: {vector_db.get_num_documents():,}")

    print(f"\nLoading embedding model: {config.embedding.model}")
    model = create_embedding_model(
        config.embedding.model,
        normalize_embeddings=config.embedding.normalize,
    )

    null_dist = build_pool_matched_null(
        queries, qrels, vector_db, model,
        pool_size=config.hc.pool_size,
        z_score_fraction=config.null.z_score_fraction,
    )

    null_dir.mkdir(parents=True, exist_ok=True)
    null_dist.save(str(null_path))

    print(f"\nPool-matched null saved to {null_path}")
    print(f"  Mean: {null_dist.mean:.4f}, Std: {null_dist.std:.4f}")
    print(f"  Samples: {null_dist.n_samples:,}")
    print(f"  Config: pool_size={config.hc.pool_size}, z_score_fraction={config.null.z_score_fraction}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify it runs (dry check — imports resolve)**

Run: `cd E:/HC_RAG_project && python -c "from experiments.scripts.pool_null.build_pool_null import build_pool_matched_null; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add experiments/scripts/pool_null/__init__.py experiments/scripts/pool_null/build_pool_null.py
git commit -m "feat: add pool-matched null distribution builder"
```

---

### Task 2: Create `validate_pool_null.py`

**Files:**
- Create: `experiments/scripts/pool_null/validate_pool_null.py`

- [ ] **Step 1: Write `validate_pool_null.py`**

```python
"""Validate pool-matched null distribution (4-test hard gate).

All tests use the same FAISS top-K + z-scoring procedure as inference,
matching how the pool-matched null was built.

Exits 0 on all-pass, 1 on any-fail.

Usage: python -m experiments.scripts.pool_null.validate_pool_null --config experiments/configs/amazon_compound.yaml
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from scipy import stats

from hc_rag.embeddings.embedding_model import create_embedding_model
from hc_rag.embeddings.vector_database import VectorDatabase
from hc_rag.hc.null_distribution import NullDistribution
from hc_rag.hc.higher_criticism import HigherCriticism

from experiments.lib.config import load_config
from experiments.lib.registry import get_adapter
from experiments.lib.results_io import save_results_json


def _pool_z_score(query, vector_db, model, pool_size, z_score_fraction):
    """Retrieve top-K from FAISS and z-score using bottom fraction.

    Returns (candidate_ids, candidate_sims, z_scores, mu_est, sigma_est)
    or None if query should be skipped.
    """
    query_emb = model.embed([query.text], show_progress=False)[0]
    results = vector_db.search(query_emb.reshape(1, -1), k=pool_size)
    if not results or not results[0]:
        return None

    candidate_ids = [r.doc_id for r in results[0]]
    candidate_sims = np.array([r.score for r in results[0]])

    sorted_desc = np.sort(candidate_sims)[::-1]
    n = len(sorted_desc)
    n_null_est = max(int(n * z_score_fraction), 10)
    bottom = sorted_desc[-n_null_est:]
    mu_est = np.mean(bottom)
    sigma_est = np.std(bottom)

    if sigma_est < 1e-12:
        return None

    z_scores = (candidate_sims - mu_est) / sigma_est
    return candidate_ids, candidate_sims, z_scores, mu_est, sigma_est


def test_split_half_ks(pool_null, n_seeds=20):
    """Test 1: Split-half KS uniformity. PASS: median KS p > 0.05 AND median KS stat < 0.1."""
    sims = pool_null.similarities
    ks_stats, ks_pvals = [], []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(sims))
        half = len(idx) // 2
        ref, test = sims[idx[:half]], sims[idx[half:]]

        ref_sorted = np.sort(ref)
        p_values = 1.0 - np.searchsorted(ref_sorted, test) / len(ref_sorted)
        p_values = np.clip(p_values, 1e-10, 1.0)

        ks_stat, ks_p = stats.kstest(p_values, "uniform")
        ks_stats.append(ks_stat)
        ks_pvals.append(ks_p)

    passed = np.median(ks_pvals) > 0.05 and np.median(ks_stats) < 0.1
    return {
        "test": "split_half_ks",
        "passed": bool(passed),
        "median_ks_stat": float(np.median(ks_stats)),
        "median_ks_pval": float(np.median(ks_pvals)),
    }


def test_per_query_uniformity(pool_null, queries, qrels, vector_db, model,
                               pool_size, z_score_fraction, n_sample=50):
    """Test 2: Per-query uniformity (pool-matched). PASS: >= 70% of queries have KS p > 0.05."""
    null_sorted = np.sort(pool_null.similarities)
    rng = np.random.RandomState(42)

    sample_queries = rng.choice(len(queries), size=min(n_sample, len(queries)), replace=False)
    pass_count = 0
    tested = 0

    for qi in sample_queries:
        query = queries[qi]
        relevant = qrels.get(query.query_id, set())
        result = _pool_z_score(query, vector_db, model, pool_size, z_score_fraction)
        if result is None:
            continue

        candidate_ids, _, z_scores, _, _ = result
        tested += 1

        # Get non-relevant z-scores from the pool
        is_relevant = np.array([cid in relevant for cid in candidate_ids])
        nonrel_z = z_scores[~is_relevant]

        if len(nonrel_z) == 0:
            continue

        # Compute p-values against pool-matched null
        p_values = 1.0 - np.searchsorted(null_sorted, nonrel_z) / len(null_sorted)
        p_values = np.clip(p_values, 1e-10, 1.0)
        _, ks_p = stats.kstest(p_values, "uniform")
        if ks_p > 0.05:
            pass_count += 1

    pass_rate = pass_count / tested if tested > 0 else 0
    return {
        "test": "per_query_uniformity_pool",
        "passed": bool(pass_rate >= 0.70),
        "pass_rate": float(pass_rate),
        "n_queries_tested": tested,
    }


def test_signal_detection(pool_null, queries, qrels, vector_db, model,
                           pool_size, z_score_fraction, n_sample=50):
    """Test 3: Signal detection (pool-matched). PASS: >= 60% of relevant docs have p < 0.05."""
    null_sorted = np.sort(pool_null.similarities)
    rng = np.random.RandomState(42)

    sample_queries = rng.choice(len(queries), size=min(n_sample, len(queries)), replace=False)
    all_rel_pvals = []

    for qi in sample_queries:
        query = queries[qi]
        relevant = qrels.get(query.query_id, set())
        if not relevant:
            continue

        result = _pool_z_score(query, vector_db, model, pool_size, z_score_fraction)
        if result is None:
            continue

        candidate_ids, _, z_scores, _, _ = result

        # Get relevant z-scores from the pool
        is_relevant = np.array([cid in relevant for cid in candidate_ids])
        rel_z = z_scores[is_relevant]

        if len(rel_z) == 0:
            continue

        p_values = 1.0 - np.searchsorted(null_sorted, rel_z) / len(null_sorted)
        p_values = np.clip(p_values, 1e-10, 1.0)
        all_rel_pvals.extend(p_values)

    signal_rate = np.mean(np.array(all_rel_pvals) < 0.05) if all_rel_pvals else 0
    return {
        "test": "signal_detection_pool",
        "passed": bool(signal_rate >= 0.60),
        "signal_rate": float(signal_rate),
        "n_relevant_docs": len(all_rel_pvals),
    }


def test_hc_k_correlation(pool_null, queries, qrels, vector_db, model,
                           pool_size, z_score_fraction):
    """Test 4: HC K correlation (pool-matched). PASS: mean HC stat > 1.0 AND >= 3 unique k values."""
    hc = HigherCriticism(null_distribution=pool_null)
    hc_stats, hc_ks = [], []

    for query in queries:
        result = _pool_z_score(query, vector_db, model, pool_size, z_score_fraction)
        if result is None:
            continue

        _, _, z_scores, _, _ = result
        hc_result = hc.compute_hc_threshold(z_scores, gamma=0.1)
        hc_stats.append(hc_result.hc_statistic)
        hc_ks.append(hc_result.k)

    mean_hc = float(np.mean(hc_stats)) if hc_stats else 0
    unique_ks = len(set(hc_ks))
    return {
        "test": "hc_k_correlation_pool",
        "passed": bool(mean_hc > 1.0 and unique_ks >= 3),
        "mean_hc_statistic": mean_hc,
        "unique_k_values": unique_ks,
        "n_queries": len(hc_stats),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate pool-matched null distribution")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--force", action="store_true", help="Re-validate even if report exists")
    args = parser.parse_args()

    config = load_config(args.config)
    import experiments.datasets  # noqa: F401
    adapter = get_adapter(config.dataset.name)

    artifacts_dir = Path(config.dataset.artifacts_dir)
    null_dir = artifacts_dir / "pool_null"
    report_path = null_dir / "validation_report.json"
    if report_path.exists() and not args.force:
        print(f"Validation report exists at {report_path}. Use --force to re-run.")
        return

    null_path = null_dir / "global_pool_null"
    if not Path(str(null_path) + ".pkl").exists():
        print(f"Pool-matched null not found at {null_path}. Run build_pool_null first.")
        sys.exit(1)

    queries, qrels, _ = adapter.load_dataset(config.dataset.data_dir)
    vector_db = VectorDatabase.load(str(artifacts_dir / "vector_db"))
    pool_null = NullDistribution.load(str(null_path))
    model = create_embedding_model(
        config.embedding.model,
        normalize_embeddings=config.embedding.normalize,
    )

    pool_size = config.hc.pool_size
    z_score_fraction = config.null.z_score_fraction

    print("=" * 70)
    print(f"Validating pool-matched null for {config.dataset.name}")
    print(f"  pool_size={pool_size}, z_score_fraction={z_score_fraction}")
    print("=" * 70)

    results = []
    test_configs = [
        (test_split_half_ks, {"pool_null": pool_null}),
        (test_per_query_uniformity, {"pool_null": pool_null, "queries": queries, "qrels": qrels,
                                      "vector_db": vector_db, "model": model,
                                      "pool_size": pool_size, "z_score_fraction": z_score_fraction}),
        (test_signal_detection, {"pool_null": pool_null, "queries": queries, "qrels": qrels,
                                  "vector_db": vector_db, "model": model,
                                  "pool_size": pool_size, "z_score_fraction": z_score_fraction}),
        (test_hc_k_correlation, {"pool_null": pool_null, "queries": queries, "qrels": qrels,
                                  "vector_db": vector_db, "model": model,
                                  "pool_size": pool_size, "z_score_fraction": z_score_fraction}),
    ]

    for test_fn, kwargs in test_configs:
        result = test_fn(**kwargs)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n  {result['test']}: {status}")
        for k, v in result.items():
            if k not in ("test", "passed"):
                print(f"    {k}: {v}")
        results.append(result)

    all_passed = all(r["passed"] for r in results)
    report = {
        "all_passed": all_passed,
        "config": {
            "pool_size": pool_size,
            "z_score_fraction": z_score_fraction,
            "null_type": "pool_matched",
        },
        "tests": results,
    }
    null_dir.mkdir(parents=True, exist_ok=True)
    save_results_json(report, str(report_path))
    print(f"\nReport saved to {report_path}")

    if all_passed:
        print("\nAll 4 tests PASSED.")
        sys.exit(0)
    else:
        failed = [r["test"] for r in results if not r["passed"]]
        print(f"\nFAILED tests: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify imports resolve**

Run: `cd E:/HC_RAG_project && python -c "from experiments.scripts.pool_null.validate_pool_null import test_split_half_ks; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add experiments/scripts/pool_null/validate_pool_null.py
git commit -m "feat: add pool-matched null validation (4-test gate)"
```

---

### Task 3: Create `compare_nulls.py`

**Files:**
- Create: `experiments/scripts/pool_null/compare_nulls.py`

- [ ] **Step 1: Write `compare_nulls.py`**

```python
"""Compare original and pool-matched null distributions side-by-side.

Runs HC retrieval with both nulls on the same queries and compares:
  - Retrieval metrics (Recall, Precision, F1, Mean K)
  - Per-query k selection
  - P-value uniformity diagnostics

Usage: python -m experiments.scripts.pool_null.compare_nulls --config experiments/configs/amazon_compound.yaml
"""

import argparse
import numpy as np
from pathlib import Path
from scipy import stats

from hc_rag.embeddings.embedding_model import create_embedding_model
from hc_rag.embeddings.vector_database import VectorDatabase
from hc_rag.hc.null_distribution import NullDistribution
from hc_rag.hc.higher_criticism import HigherCriticism

from experiments.lib.config import load_config
from experiments.lib.registry import get_adapter
from experiments.lib.results_io import save_results_json


def run_hc_on_query(query, vector_db, model, null_dist, pool_size, z_score_fraction, gamma):
    """Run HC retrieval for a single query using given null distribution.

    Returns dict with k, hc_stat, retrieved_ids, or None if skipped.
    """
    query_emb = model.embed([query.text], show_progress=False)[0]
    results = vector_db.search(query_emb.reshape(1, -1), k=pool_size)
    if not results or not results[0]:
        return None

    candidate_ids = [r.doc_id for r in results[0]]
    candidate_sims = np.array([r.score for r in results[0]])

    # Z-score using bottom fraction (same as inference)
    sorted_desc = np.sort(candidate_sims)[::-1]
    n = len(sorted_desc)
    n_null_est = max(int(n * z_score_fraction), 10)
    bottom = sorted_desc[-n_null_est:]
    mu_est = np.mean(bottom)
    sigma_est = np.std(bottom)

    if sigma_est < 1e-12:
        return None

    z_scores = (candidate_sims - mu_est) / sigma_est

    hc = HigherCriticism(null_distribution=null_dist)
    hc_result = hc.compute_hc_threshold(z_scores, gamma=gamma)

    # Get the top-k candidate IDs
    top_idx = np.argsort(candidate_sims)[::-1][:hc_result.k]
    retrieved_ids = set(candidate_ids[i] for i in top_idx)

    return {
        "k": hc_result.k,
        "hc_stat": hc_result.hc_statistic,
        "retrieved_ids": retrieved_ids,
    }


def compute_pvalue_uniformity(query, vector_db, model, null_dist, qrels,
                               pool_size, z_score_fraction):
    """Compute KS statistic for non-relevant p-value uniformity under given null."""
    query_emb = model.embed([query.text], show_progress=False)[0]
    results = vector_db.search(query_emb.reshape(1, -1), k=pool_size)
    if not results or not results[0]:
        return None

    candidate_ids = [r.doc_id for r in results[0]]
    candidate_sims = np.array([r.score for r in results[0]])

    sorted_desc = np.sort(candidate_sims)[::-1]
    n = len(sorted_desc)
    n_null_est = max(int(n * z_score_fraction), 10)
    bottom = sorted_desc[-n_null_est:]
    mu_est = np.mean(bottom)
    sigma_est = np.std(bottom)

    if sigma_est < 1e-12:
        return None

    z_scores = (candidate_sims - mu_est) / sigma_est

    relevant = qrels.get(query.query_id, set())
    is_relevant = np.array([cid in relevant for cid in candidate_ids])
    nonrel_z = z_scores[~is_relevant]

    if len(nonrel_z) == 0:
        return None

    null_sorted = np.sort(null_dist.similarities)
    p_values = 1.0 - np.searchsorted(null_sorted, nonrel_z) / len(null_sorted)
    p_values = np.clip(p_values, 1e-10, 1.0)
    ks_stat, ks_p = stats.kstest(p_values, "uniform")
    return {"ks_stat": ks_stat, "ks_p": ks_p}


def main():
    parser = argparse.ArgumentParser(description="Compare original vs pool-matched null")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    import experiments.datasets  # noqa: F401
    adapter = get_adapter(config.dataset.name)

    artifacts_dir = Path(config.dataset.artifacts_dir)
    results_dir = Path(config.dataset.results_dir) / "null_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load both nulls
    original_null_path = artifacts_dir / "global_null"
    pool_null_path = artifacts_dir / "pool_null" / "global_pool_null"

    if not Path(str(original_null_path) + ".pkl").exists():
        print(f"Original null not found at {original_null_path}")
        return
    if not Path(str(pool_null_path) + ".pkl").exists():
        print(f"Pool-matched null not found at {pool_null_path}")
        return

    original_null = NullDistribution.load(str(original_null_path))
    pool_null = NullDistribution.load(str(pool_null_path))

    queries, qrels, _ = adapter.load_dataset(config.dataset.data_dir)
    vector_db = VectorDatabase.load(str(artifacts_dir / "vector_db"))
    model = create_embedding_model(
        config.embedding.model,
        normalize_embeddings=config.embedding.normalize,
    )

    pool_size = config.hc.pool_size
    z_score_fraction = config.null.z_score_fraction
    gamma = config.hc.gamma

    print("=" * 70)
    print(f"Comparing nulls for {config.dataset.name}")
    print(f"  pool_size={pool_size}, z_score_fraction={z_score_fraction}, gamma={gamma}")
    print("=" * 70)

    per_query = []
    original_ks_stats = []
    pool_ks_stats = []

    for i, query in enumerate(queries):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(queries)} queries")

        relevant = qrels.get(query.query_id, set())
        true_k = len(relevant)

        # Run HC with both nulls
        orig_result = run_hc_on_query(query, vector_db, model, original_null,
                                       pool_size, z_score_fraction, gamma)
        pool_result = run_hc_on_query(query, vector_db, model, pool_null,
                                       pool_size, z_score_fraction, gamma)

        if orig_result is None or pool_result is None:
            continue

        # Compute retrieval metrics for each
        orig_recall = len(orig_result["retrieved_ids"] & relevant) / max(len(relevant), 1)
        orig_precision = len(orig_result["retrieved_ids"] & relevant) / max(orig_result["k"], 1)
        pool_recall = len(pool_result["retrieved_ids"] & relevant) / max(len(relevant), 1)
        pool_precision = len(pool_result["retrieved_ids"] & relevant) / max(pool_result["k"], 1)

        # P-value uniformity under each null
        orig_uni = compute_pvalue_uniformity(query, vector_db, model, original_null, qrels,
                                              pool_size, z_score_fraction)
        pool_uni = compute_pvalue_uniformity(query, vector_db, model, pool_null, qrels,
                                              pool_size, z_score_fraction)

        if orig_uni:
            original_ks_stats.append(orig_uni["ks_stat"])
        if pool_uni:
            pool_ks_stats.append(pool_uni["ks_stat"])

        per_query.append({
            "query_id": query.query_id,
            "true_k": true_k,
            "original_k": orig_result["k"],
            "pool_k": pool_result["k"],
            "original_hc_stat": float(orig_result["hc_stat"]),
            "pool_hc_stat": float(pool_result["hc_stat"]),
            "original_recall": orig_recall,
            "original_precision": orig_precision,
            "pool_recall": pool_recall,
            "pool_precision": pool_precision,
        })

    # Compute aggregates
    def f1(r, p):
        return 2 * r * p / (r + p) if (r + p) > 0 else 0

    n = len(per_query)
    orig_agg = {
        "mean_k": np.mean([q["original_k"] for q in per_query]),
        "mean_recall": np.mean([q["original_recall"] for q in per_query]),
        "mean_precision": np.mean([q["original_precision"] for q in per_query]),
        "mean_f1": np.mean([f1(q["original_recall"], q["original_precision"]) for q in per_query]),
        "mean_ks_stat": float(np.mean(original_ks_stats)) if original_ks_stats else None,
        "pct_uniform_pvals": float(np.mean([ks < 0.1 for ks in original_ks_stats])) if original_ks_stats else None,
    }
    pool_agg = {
        "mean_k": np.mean([q["pool_k"] for q in per_query]),
        "mean_recall": np.mean([q["pool_recall"] for q in per_query]),
        "mean_precision": np.mean([q["pool_precision"] for q in per_query]),
        "mean_f1": np.mean([f1(q["pool_recall"], q["pool_precision"]) for q in per_query]),
        "mean_ks_stat": float(np.mean(pool_ks_stats)) if pool_ks_stats else None,
        "pct_uniform_pvals": float(np.mean([ks < 0.1 for ks in pool_ks_stats])) if pool_ks_stats else None,
    }

    # K correlation with oracle
    orig_k_corr = float(np.corrcoef(
        [q["true_k"] for q in per_query],
        [q["original_k"] for q in per_query]
    )[0, 1]) if n > 1 else 0
    pool_k_corr = float(np.corrcoef(
        [q["true_k"] for q in per_query],
        [q["pool_k"] for q in per_query]
    )[0, 1]) if n > 1 else 0

    comparison = {
        "dataset": config.dataset.name,
        "n_queries": n,
        "config": {
            "pool_size": pool_size,
            "z_score_fraction": z_score_fraction,
            "gamma": gamma,
        },
        "original_null": {**orig_agg, "k_oracle_correlation": orig_k_corr},
        "pool_matched_null": {**pool_agg, "k_oracle_correlation": pool_k_corr},
        "per_query": per_query,
    }

    output_path = results_dir / "null_comparison.json"
    save_results_json(comparison, str(output_path))

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"COMPARISON RESULTS ({n} queries)")
    print(f"{'=' * 70}")
    print(f"\n{'Metric':<25} {'Original':>12} {'Pool-Matched':>12} {'Delta':>10}")
    print("-" * 60)

    for metric in ["mean_k", "mean_recall", "mean_precision", "mean_f1"]:
        o = orig_agg[metric]
        p = pool_agg[metric]
        delta = p - o
        print(f"  {metric:<23} {o:>12.4f} {p:>12.4f} {delta:>+10.4f}")

    print(f"  {'k-oracle correlation':<23} {orig_k_corr:>12.4f} {pool_k_corr:>12.4f} {pool_k_corr - orig_k_corr:>+10.4f}")

    if orig_agg["mean_ks_stat"] is not None and pool_agg["mean_ks_stat"] is not None:
        print(f"\n  {'mean KS stat (lower=better)':<27} {orig_agg['mean_ks_stat']:>10.4f} {pool_agg['mean_ks_stat']:>10.4f}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify imports resolve**

Run: `cd E:/HC_RAG_project && python -c "from experiments.scripts.pool_null.compare_nulls import run_hc_on_query; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add experiments/scripts/pool_null/compare_nulls.py
git commit -m "feat: add null comparison script (original vs pool-matched)"
```

---

### Task 4: Run the pipeline on amazon_compound dataset

This task runs all three scripts end-to-end on one dataset.

**Files:**
- No new files — running existing scripts

- [ ] **Step 1: Build pool-matched null**

Run: `cd E:/HC_RAG_project && python -m experiments.scripts.pool_null.build_pool_null --config experiments/configs/amazon_compound.yaml`

Expected output includes:
- `Building pool-matched null distribution...`
- `Total pooled z-scores: ...` (should be pool_size * n_queries minus relevant docs, roughly 100K-130K)
- `Pool-matched null saved to artifacts/amazon_compound/pool_null/global_pool_null`

- [ ] **Step 2: Validate pool-matched null**

Run: `cd E:/HC_RAG_project && python -m experiments.scripts.pool_null.validate_pool_null --config experiments/configs/amazon_compound.yaml`

Expected: All 4 tests should PASS. If any fail, investigate before proceeding — this is the key experiment.

- [ ] **Step 3: Compare nulls**

Run: `cd E:/HC_RAG_project && python -m experiments.scripts.pool_null.compare_nulls --config experiments/configs/amazon_compound.yaml`

Expected: Comparison table printed showing retrieval metrics for both nulls. Look for:
- Pool-matched null should have lower KS stat (better p-value calibration)
- Retrieval metrics may or may not improve — this is what we're investigating

- [ ] **Step 4: Commit results**

```bash
git add artifacts/amazon_compound/pool_null/ results/amazon_compound/null_comparison/
git commit -m "results: pool-matched null comparison on amazon_compound"
```
