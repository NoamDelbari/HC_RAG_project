"""Validate global null distribution (4-test hard gate).

Exits 0 on all-pass, 1 on any-fail.

Usage: python -m experiments.scripts.validate_null --config experiments/configs/amazon_compound.yaml
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


def test_per_query_uniformity(global_null, queries, qrels, vector_db, model, n_sample=50):
    """Test 2: Per-query uniformity. PASS: >= 70% of queries have KS p > 0.05."""
    doc_embeddings = vector_db.index.reconstruct_n(0, len(vector_db.doc_ids))
    all_doc_ids = vector_db.doc_ids
    null_sorted = np.sort(global_null.similarities)
    rng = np.random.RandomState(42)

    sample_queries = rng.choice(len(queries), size=min(n_sample, len(queries)), replace=False)
    pass_count = 0

    for qi in sample_queries:
        query = queries[qi]
        relevant = qrels.get(query.query_id, set())
        query_emb = model.embed([query.text], show_progress=False)[0]
        all_sims = np.dot(doc_embeddings, query_emb)
        is_relevant = np.array([did in relevant for did in all_doc_ids])
        nonrel_sims = all_sims[~is_relevant]

        mean_q, std_q = np.mean(nonrel_sims), np.std(nonrel_sims)
        if std_q < 1e-12:
            continue

        z_scores = (nonrel_sims - mean_q) / std_q

        if len(z_scores) > 5000:
            z_scores = rng.choice(z_scores, size=5000, replace=False)

        p_values = 1.0 - np.searchsorted(null_sorted, z_scores) / len(null_sorted)
        p_values = np.clip(p_values, 1e-10, 1.0)
        _, ks_p = stats.kstest(p_values, "uniform")
        if ks_p > 0.05:
            pass_count += 1

    pass_rate = pass_count / len(sample_queries) if sample_queries.size > 0 else 0
    return {
        "test": "per_query_uniformity",
        "passed": bool(pass_rate >= 0.70),
        "pass_rate": float(pass_rate),
        "n_queries_tested": int(len(sample_queries)),
    }


def test_signal_detection(global_null, queries, qrels, vector_db, model, n_sample=50):
    """Test 3: Signal detection. PASS: >= 60% of relevant docs have p < 0.05."""
    doc_embeddings = vector_db.index.reconstruct_n(0, len(vector_db.doc_ids))
    all_doc_ids = vector_db.doc_ids
    null_sorted = np.sort(global_null.similarities)
    rng = np.random.RandomState(42)

    sample_queries = rng.choice(len(queries), size=min(n_sample, len(queries)), replace=False)
    all_rel_pvals = []

    for qi in sample_queries:
        query = queries[qi]
        relevant = qrels.get(query.query_id, set())
        if not relevant:
            continue

        query_emb = model.embed([query.text], show_progress=False)[0]
        all_sims = np.dot(doc_embeddings, query_emb)
        is_relevant = np.array([did in relevant for did in all_doc_ids])
        nonrel_sims = all_sims[~is_relevant]
        rel_sims = all_sims[is_relevant]

        mean_q, std_q = np.mean(nonrel_sims), np.std(nonrel_sims)
        if std_q < 1e-12:
            continue

        z_rel = (rel_sims - mean_q) / std_q
        p_values = 1.0 - np.searchsorted(null_sorted, z_rel) / len(null_sorted)
        p_values = np.clip(p_values, 1e-10, 1.0)
        all_rel_pvals.extend(p_values)

    signal_rate = np.mean(np.array(all_rel_pvals) < 0.05) if all_rel_pvals else 0
    return {
        "test": "signal_detection",
        "passed": bool(signal_rate >= 0.60),
        "signal_rate": float(signal_rate),
        "n_relevant_docs": len(all_rel_pvals),
    }


def test_hc_k_correlation(global_null, queries, qrels, vector_db, model):
    """Test 4: HC K correlation. PASS: mean HC stat > 1.0 AND >= 3 unique HC K values."""
    doc_embeddings = vector_db.index.reconstruct_n(0, len(vector_db.doc_ids))
    all_doc_ids = vector_db.doc_ids
    hc = HigherCriticism(null_distribution=global_null)

    hc_stats, hc_ks = [], []
    for query in queries:
        relevant = qrels.get(query.query_id, set())
        query_emb = model.embed([query.text], show_progress=False)[0]
        all_sims = np.dot(doc_embeddings, query_emb)

        top_idx = np.argsort(all_sims)[-100:][::-1]
        top_sims = all_sims[top_idx]

        is_relevant = np.array([did in relevant for did in all_doc_ids])
        nonrel_sims = all_sims[~is_relevant]
        mean_q, std_q = np.mean(nonrel_sims), np.std(nonrel_sims)
        if std_q < 1e-12:
            continue

        z_candidates = (top_sims - mean_q) / std_q
        result = hc.compute_hc_threshold(z_candidates, gamma=0.1)
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
    parser = argparse.ArgumentParser(description="Validate global null distribution")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--force", action="store_true", help="Re-validate even if report exists")
    args = parser.parse_args()

    config = load_config(args.config)
    import experiments.datasets  # noqa: F401
    adapter = get_adapter(config.dataset.name)

    artifacts_dir = Path(config.dataset.artifacts_dir)
    report_path = artifacts_dir / "validation_report.json"
    if report_path.exists() and not args.force:
        print(f"Validation report already exists at {report_path}. Use --force to re-run.")
        return

    queries, qrels, _ = adapter.load_dataset(config.dataset.data_dir)
    vector_db = VectorDatabase.load(str(artifacts_dir / "vector_db"))
    global_null = NullDistribution.load(str(artifacts_dir / "global_null"))
    model = create_embedding_model(
        config.embedding.model,
        normalize_embeddings=config.embedding.normalize,
    )

    print("=" * 70)
    print(f"Validating global null for {config.dataset.name}")
    print("=" * 70)

    results = []
    test_fns = [test_split_half_ks, test_per_query_uniformity, test_signal_detection, test_hc_k_correlation]
    for test_fn in test_fns:
        if test_fn == test_split_half_ks:
            result = test_fn(global_null)
        else:
            result = test_fn(global_null, queries, qrels, vector_db, model)

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
