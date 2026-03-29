"""Validate pool-matched null distribution (4-test hard gate).

Uses the same FAISS top-K retrieval + z-scoring procedure as inference,
ensuring the null matches the actual retrieval pipeline.

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


def _pool_z_score(query_text, vector_db, model, pool_size, z_score_fraction):
    """Shared helper: pool-matched z-scoring (same procedure as inference).

    1. Embed query
    2. FAISS search for top pool_size candidates
    3. Sort descending, take bottom fraction for mu_est/sigma_est
    4. Z-score all candidates
    5. Return (candidate_ids, candidate_sims, z_scores, mu_est, sigma_est) or None if skipped

    Args:
        query_text: Query string to embed
        vector_db: VectorDatabase instance
        model: Embedding model with .embed() method
        pool_size: Number of FAISS top-K candidates to retrieve
        z_score_fraction: Fraction of bottom candidates for null estimation

    Returns:
        Tuple of (candidate_ids, candidate_sims, z_scores, mu_est, sigma_est)
        or None if the query should be skipped (e.g., degenerate sigma).
    """
    # 1. Embed query
    query_emb = model.embed([query_text], show_progress=False)[0]

    # 2. FAISS search for top pool_size candidates
    results = vector_db.search(query_emb, k=pool_size)
    if len(results) < 10:
        return None

    candidate_ids = [r.doc_id for r in results]
    candidate_sims = np.array([r.similarity for r in results])

    # 3. Sort descending, take bottom fraction for mu_est/sigma_est
    sorted_desc = np.sort(candidate_sims)[::-1]
    n = len(sorted_desc)
    n_null_est = max(int(n * z_score_fraction), 10)
    bottom = sorted_desc[-n_null_est:]  # lowest similarities
    mu_est = float(np.mean(bottom))
    sigma_est = float(np.std(bottom))

    if sigma_est < 1e-10:
        return None

    # 4. Z-score all candidates
    z_scores = (candidate_sims - mu_est) / sigma_est

    # 5. Return tuple
    return (candidate_ids, candidate_sims, z_scores, mu_est, sigma_est)


def test_split_half_ks(global_null, n_seeds=20):
    """Test 1: Split-half KS uniformity. PASS: median KS p > 0.05 AND median KS stat < 0.1."""
    sims = global_null.similarities
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


def test_per_query_uniformity(
    global_null, queries, qrels, vector_db, model, pool_size, z_score_fraction, n_sample=50
):
    """Test 2: Per-query uniformity via pool-matched z-scoring.

    For each sampled query, retrieves top-K from FAISS, z-scores using bottom
    fraction, converts non-relevant z-scores to p-values against the pool null,
    and tests for uniformity via KS test.

    PASS: >= 70% of queries have KS p > 0.05.
    """
    null_sorted = np.sort(global_null.similarities)
    rng = np.random.RandomState(42)

    sample_queries = rng.choice(len(queries), size=min(n_sample, len(queries)), replace=False)
    pass_count = 0
    tested_count = 0

    for qi in sample_queries:
        query = queries[qi]
        relevant = qrels.get(query.query_id, set())

        pool_result = _pool_z_score(query.text, vector_db, model, pool_size, z_score_fraction)
        if pool_result is None:
            continue

        candidate_ids, candidate_sims, z_scores, mu_est, sigma_est = pool_result

        # Get non-relevant z-scores from the pool
        is_relevant = np.array([cid in relevant for cid in candidate_ids])
        nonrel_z = z_scores[~is_relevant]

        if len(nonrel_z) < 5:
            continue

        tested_count += 1

        # Compute p-values against the pool null
        p_values = 1.0 - np.searchsorted(null_sorted, nonrel_z) / len(null_sorted)
        p_values = np.clip(p_values, 1e-10, 1.0)

        _, ks_p = stats.kstest(p_values, "uniform")
        if ks_p > 0.05:
            pass_count += 1

    pass_rate = pass_count / tested_count if tested_count > 0 else 0
    return {
        "test": "per_query_uniformity",
        "passed": bool(pass_rate >= 0.70),
        "pass_rate": float(pass_rate),
        "n_queries_tested": int(tested_count),
    }


def test_signal_detection(
    global_null, queries, qrels, vector_db, model, pool_size, z_score_fraction, n_sample=50
):
    """Test 3: Signal detection via pool-matched z-scoring.

    For each sampled query, retrieves top-K from FAISS, z-scores using bottom
    fraction, converts relevant z-scores to p-values against the pool null.

    PASS: >= 60% of relevant docs (that appear in the pool) have p < 0.05.
    """
    null_sorted = np.sort(global_null.similarities)
    rng = np.random.RandomState(42)

    sample_queries = rng.choice(len(queries), size=min(n_sample, len(queries)), replace=False)
    all_rel_pvals = []

    for qi in sample_queries:
        query = queries[qi]
        relevant = qrels.get(query.query_id, set())
        if not relevant:
            continue

        pool_result = _pool_z_score(query.text, vector_db, model, pool_size, z_score_fraction)
        if pool_result is None:
            continue

        candidate_ids, candidate_sims, z_scores, mu_est, sigma_est = pool_result

        # Get relevant z-scores from the pool
        is_relevant = np.array([cid in relevant for cid in candidate_ids])
        rel_z = z_scores[is_relevant]

        if len(rel_z) == 0:
            continue

        # Compute p-values against the pool null
        p_values = 1.0 - np.searchsorted(null_sorted, rel_z) / len(null_sorted)
        p_values = np.clip(p_values, 1e-10, 1.0)
        all_rel_pvals.extend(p_values)

    signal_rate = np.mean(np.array(all_rel_pvals) < 0.05) if all_rel_pvals else 0
    return {
        "test": "signal_detection",
        "passed": bool(signal_rate >= 0.60),
        "signal_rate": float(signal_rate),
        "n_relevant_docs": len(all_rel_pvals),
    }


def test_hc_k_correlation(
    global_null, queries, qrels, vector_db, model, pool_size, z_score_fraction
):
    """Test 4: HC K correlation via pool-matched z-scoring.

    For each query, retrieves top-K from FAISS, z-scores using bottom fraction,
    runs HC with gamma=0.1 against the pool null.

    PASS: mean HC > 1.0 AND unique k >= 3.
    """
    hc = HigherCriticism(null_distribution=global_null)

    hc_stats, hc_ks = [], []
    for query in queries:
        pool_result = _pool_z_score(query.text, vector_db, model, pool_size, z_score_fraction)
        if pool_result is None:
            continue

        candidate_ids, candidate_sims, z_scores, mu_est, sigma_est = pool_result

        result = hc.compute_hc_threshold(z_scores, gamma=0.1)
        hc_stats.append(result.hc_statistic)
        hc_ks.append(result.k)

    mean_hc = float(np.mean(hc_stats)) if hc_stats else 0
    unique_ks = len(set(hc_ks))
    return {
        "test": "hc_k_correlation",
        "passed": bool(mean_hc > 1.0 and unique_ks >= 3),
        "mean_hc_statistic": mean_hc,
        "unique_k_values": unique_ks,
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
    report_path = artifacts_dir / "pool_null" / "validation_report.json"
    if report_path.exists() and not args.force:
        print(f"Validation report already exists at {report_path}. Use --force to re-run.")
        return

    # Load data
    queries, qrels, _ = adapter.load_dataset(config.dataset.data_dir)
    vector_db = VectorDatabase.load(str(artifacts_dir / "vector_db"))
    global_null = NullDistribution.load(str(artifacts_dir / "pool_null" / "global_pool_null"))
    model = create_embedding_model(
        config.embedding.model,
        normalize_embeddings=config.embedding.normalize,
    )

    # Read pool-matched parameters from config
    pool_size = config.hc.pool_size
    z_score_fraction = config.null.z_score_fraction

    print("=" * 70)
    print(f"Validating pool-matched null for {config.dataset.name}")
    print(f"  pool_size={pool_size}, z_score_fraction={z_score_fraction}")
    print("=" * 70)

    results = []
    test_fns = [
        test_split_half_ks,
        test_per_query_uniformity,
        test_signal_detection,
        test_hc_k_correlation,
    ]

    for test_fn in test_fns:
        if test_fn == test_split_half_ks:
            result = test_fn(global_null)
        else:
            result = test_fn(
                global_null, queries, qrels, vector_db, model,
                pool_size, z_score_fraction,
            )

        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n  {result['test']}: {status}")
        for k, v in result.items():
            if k not in ("test", "passed"):
                print(f"    {k}: {v}")
        results.append(result)

    all_passed = all(r["passed"] for r in results)
    report = {"all_passed": all_passed, "tests": results}
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
