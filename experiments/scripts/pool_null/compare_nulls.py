"""Compare HC retrieval using original vs pool-matched null distributions.

Runs HC retrieval with both nulls side-by-side on the same queries and outputs
aggregate comparison metrics plus per-query details.

Usage:
    python -m experiments.scripts.pool_null.compare_nulls \
        --config experiments/configs/amazon_compound.yaml
"""

import argparse
import numpy as np
from pathlib import Path
from scipy import stats

from hc_rag.hc.higher_criticism import HigherCriticism
from hc_rag.hc.null_distribution import NullDistribution
from hc_rag.embeddings.vector_database import VectorDatabase
from hc_rag.embeddings.embedding_model import create_embedding_model

from experiments.lib.config import load_config
from experiments.lib.registry import get_adapter
from experiments.lib.results_io import save_results_json


def run_hc_on_query(query_emb, vector_db, null_dist, pool_size, z_score_fraction, gamma):
    """Run HC for a single query using a given null distribution.

    Procedure:
    1. FAISS search for top pool_size candidates
    2. Sort descending, bottom z_score_fraction for mu_est, sigma_est
    3. Z-score all candidates
    4. Run HigherCriticism.compute_hc_threshold(z_scores, gamma=gamma)
    5. Return top hc_result.k candidate IDs

    Args:
        query_emb: Query embedding vector.
        vector_db: VectorDatabase instance.
        null_dist: NullDistribution to use for HC.
        pool_size: Number of FAISS candidates to retrieve.
        z_score_fraction: Fraction of bottom scores used to estimate null params.
        gamma: HC gamma parameter.

    Returns:
        Dict with keys: k, hc_stat, retrieved_ids, retrieved_scores, z_scores,
        mu_est, sigma_est.
    """
    # Step 1: FAISS search for top pool_size candidates
    candidates = vector_db.search(query_emb, k=pool_size)
    if len(candidates) == 0:
        return {
            "k": 0,
            "hc_stat": 0.0,
            "retrieved_ids": [],
            "retrieved_scores": [],
            "z_scores": np.array([]),
            "mu_est": 0.0,
            "sigma_est": 0.0,
        }

    candidate_ids = [c.doc_id for c in candidates]
    candidate_scores = np.array([c.similarity for c in candidates])

    # Step 2: Sort descending, estimate mu/sigma from bottom fraction
    sorted_desc = np.sort(candidate_scores)[::-1]
    n = len(sorted_desc)
    n_null_est = max(int(n * z_score_fraction), 10)
    bottom = sorted_desc[-n_null_est:]
    mu_est = float(np.mean(bottom))
    sigma_est = float(np.std(bottom))
    if sigma_est < 1e-10:
        sigma_est = 1e-10

    # Step 3: Z-score all candidates
    z_scores = (candidate_scores - mu_est) / sigma_est

    # Step 4: Run HC threshold computation
    hc = HigherCriticism(null_distribution=null_dist)
    hc_result = hc.compute_hc_threshold(z_scores, gamma=gamma)

    # Step 5: Return top hc_result.k candidate IDs
    k = hc_result.k
    retrieved_ids = candidate_ids[:k]
    retrieved_scores = candidate_scores[:k].tolist()

    return {
        "k": k,
        "hc_stat": float(hc_result.hc_statistic),
        "retrieved_ids": retrieved_ids,
        "retrieved_scores": retrieved_scores,
        "z_scores": z_scores,
        "mu_est": mu_est,
        "sigma_est": sigma_est,
    }


def compute_pvalue_uniformity(query_emb, vector_db, null_dist, qrels_for_query,
                               pool_size, z_score_fraction):
    """Compute KS stat for non-relevant p-values under a given null.

    Tests whether p-values for non-relevant documents are uniformly distributed,
    which indicates the null is well-calibrated for this query.

    Args:
        query_emb: Query embedding vector.
        vector_db: VectorDatabase instance.
        null_dist: NullDistribution to use.
        qrels_for_query: Set of relevant doc IDs for this query.
        pool_size: Number of FAISS candidates to retrieve.
        z_score_fraction: Fraction of bottom scores for null estimation.

    Returns:
        Dict with keys: ks_stat, ks_pval, n_nonrel.
    """
    # Retrieve candidates
    candidates = vector_db.search(query_emb, k=pool_size)
    if len(candidates) == 0:
        return {"ks_stat": 1.0, "ks_pval": 0.0, "n_nonrel": 0}

    candidate_ids = [c.doc_id for c in candidates]
    candidate_scores = np.array([c.similarity for c in candidates])

    # Z-score standardization using bottom fraction
    sorted_desc = np.sort(candidate_scores)[::-1]
    n = len(sorted_desc)
    n_null_est = max(int(n * z_score_fraction), 10)
    bottom = sorted_desc[-n_null_est:]
    mu_est = float(np.mean(bottom))
    sigma_est = float(np.std(bottom))
    if sigma_est < 1e-10:
        sigma_est = 1e-10

    z_scores = (candidate_scores - mu_est) / sigma_est

    # Separate non-relevant z-scores
    is_relevant = np.array([cid in qrels_for_query for cid in candidate_ids])
    nonrel_z = z_scores[~is_relevant]

    if len(nonrel_z) < 5:
        return {"ks_stat": 1.0, "ks_pval": 0.0, "n_nonrel": len(nonrel_z)}

    # Compute p-values against the null
    null_sorted = np.sort(null_dist.similarities)
    p_values = 1.0 - np.searchsorted(null_sorted, nonrel_z) / len(null_sorted)
    p_values = np.clip(p_values, 1e-10, 1.0)

    # KS test for uniformity
    ks_stat, ks_pval = stats.kstest(p_values, "uniform")

    return {
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval),
        "n_nonrel": int(len(nonrel_z)),
    }


def _compute_query_metrics(retrieved_ids, relevant_ids):
    """Compute recall, precision, F1 for a single query."""
    if len(relevant_ids) == 0:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    hits = len(set(retrieved_ids) & relevant_ids)
    recall = hits / len(relevant_ids) if len(relevant_ids) > 0 else 0.0
    precision = hits / len(retrieved_ids) if len(retrieved_ids) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"recall": recall, "precision": precision, "f1": f1}


def _aggregate_metrics(prefix, per_query):
    """Compute aggregate metrics for a given null variant."""
    ks = [r[f"{prefix}_k"] for r in per_query]
    recalls = [r[f"{prefix}_recall"] for r in per_query]
    precisions = [r[f"{prefix}_precision"] for r in per_query]
    f1s = [r[f"{prefix}_f1"] for r in per_query]
    ks_stats = [r[f"{prefix}_ks_stat"] for r in per_query]

    true_ks = [r["true_k"] for r in per_query]
    # Correlation between HC k and oracle k
    if len(set(ks)) > 1 and len(set(true_ks)) > 1:
        k_oracle_corr = float(np.corrcoef(ks, true_ks)[0, 1])
    else:
        k_oracle_corr = 0.0

    return {
        "mean_k": float(np.mean(ks)),
        "median_k": float(np.median(ks)),
        "std_k": float(np.std(ks)),
        "mean_recall": float(np.mean(recalls)),
        "mean_precision": float(np.mean(precisions)),
        "mean_f1": float(np.mean(f1s)),
        "mean_ks_stat": float(np.mean(ks_stats)),
        "k_oracle_correlation": k_oracle_corr,
    }


def _print_summary(orig_agg, pool_agg):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    header = "{:<25} {:>15} {:>15}".format("Metric", "Original", "Pool-Matched")
    print(header)
    print("-" * 55)

    rows = [
        ("Mean K", "mean_k", ".2f"),
        ("Median K", "median_k", ".1f"),
        ("Std K", "std_k", ".2f"),
        ("Mean Recall", "mean_recall", ".4f"),
        ("Mean Precision", "mean_precision", ".4f"),
        ("Mean F1", "mean_f1", ".4f"),
        ("Mean KS Stat", "mean_ks_stat", ".4f"),
        ("K-Oracle Correlation", "k_oracle_correlation", ".4f"),
    ]
    for label, key, fmt in rows:
        orig_val = format(orig_agg[key], fmt)
        pool_val = format(pool_agg[key], fmt)
        print("{:<25} {:>15} {:>15}".format(label, orig_val, pool_val))

    print("-" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="Compare HC retrieval with original vs pool-matched nulls"
    )
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--force", action="store_true", help="Rerun even if output exists")
    args = parser.parse_args()

    config = load_config(args.config)
    import experiments.datasets  # noqa: F401
    adapter = get_adapter(config.dataset.name)

    artifacts_dir = Path(config.dataset.artifacts_dir)
    results_dir = Path(config.dataset.results_dir) / "null_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "null_comparison.json"

    if output_path.exists() and not args.force:
        print(f"Output already exists at {output_path}. Use --force to rerun.")
        return

    # Load dataset
    queries, qrels, corpus = adapter.load_dataset(config.dataset.data_dir)

    # Load vector DB and embedding model
    vector_db = VectorDatabase.load(str(artifacts_dir / "vector_db"))
    model = create_embedding_model(
        config.embedding.model,
        normalize_embeddings=config.embedding.normalize,
    )

    # Load both null distributions
    original_null_path = artifacts_dir / "global_null"
    pool_null_path = artifacts_dir / "pool_null" / "global_pool_null"

    if not original_null_path.with_suffix(".pkl").exists():
        print(f"ERROR: Original null not found at {original_null_path}.pkl")
        return
    if not pool_null_path.with_suffix(".pkl").exists():
        print(f"ERROR: Pool-matched null not found at {pool_null_path}.pkl")
        return

    original_null = NullDistribution.load(str(original_null_path))
    pool_null = NullDistribution.load(str(pool_null_path))

    print("=" * 70)
    print(f"Null Comparison: {config.dataset.name}")
    print("=" * 70)
    print(f"  Original null: {original_null}")
    print(f"  Pool-matched null: {pool_null}")
    print(f"  Queries: {len(queries)}")
    print(f"  Pool size: {config.hc.pool_size}")
    print(f"  Gamma: {config.hc.gamma}")
    print(f"  Z-score fraction: {config.null.z_score_fraction}")

    pool_size = config.hc.pool_size
    z_score_fraction = config.null.z_score_fraction
    gamma = config.hc.gamma

    per_query_results = []

    for i, query in enumerate(queries):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"\n  Processing query {i + 1}/{len(queries)}: {query.query_id}")

        query_emb = model.embed_query(query.text)
        relevant_ids = qrels.get(query.query_id, set())
        true_k = len(relevant_ids)

        # Run HC with original null
        orig_result = run_hc_on_query(
            query_emb, vector_db, original_null, pool_size, z_score_fraction, gamma
        )

        # Run HC with pool-matched null
        pool_result = run_hc_on_query(
            query_emb, vector_db, pool_null, pool_size, z_score_fraction, gamma
        )

        # Compute retrieval metrics for each
        orig_metrics = _compute_query_metrics(orig_result["retrieved_ids"], relevant_ids)
        pool_metrics = _compute_query_metrics(pool_result["retrieved_ids"], relevant_ids)

        # Compute p-value uniformity for non-relevant docs under each null
        orig_uniformity = compute_pvalue_uniformity(
            query_emb, vector_db, original_null, relevant_ids, pool_size, z_score_fraction
        )
        pool_uniformity = compute_pvalue_uniformity(
            query_emb, vector_db, pool_null, relevant_ids, pool_size, z_score_fraction
        )

        per_query_results.append({
            "query_id": query.query_id,
            "true_k": true_k,
            "original_k": orig_result["k"],
            "pool_k": pool_result["k"],
            "original_hc_stat": orig_result["hc_stat"],
            "pool_hc_stat": pool_result["hc_stat"],
            "original_recall": orig_metrics["recall"],
            "original_precision": orig_metrics["precision"],
            "original_f1": orig_metrics["f1"],
            "pool_recall": pool_metrics["recall"],
            "pool_precision": pool_metrics["precision"],
            "pool_f1": pool_metrics["f1"],
            "original_ks_stat": orig_uniformity["ks_stat"],
            "original_ks_pval": orig_uniformity["ks_pval"],
            "pool_ks_stat": pool_uniformity["ks_stat"],
            "pool_ks_pval": pool_uniformity["ks_pval"],
        })

    # Compute aggregate metrics
    orig_agg = _aggregate_metrics("original", per_query_results)
    pool_agg = _aggregate_metrics("pool", per_query_results)

    output = {
        "config": {
            "dataset": config.dataset.name,
            "pool_size": pool_size,
            "gamma": gamma,
            "z_score_fraction": z_score_fraction,
            "n_queries": len(queries),
        },
        "original_null": orig_agg,
        "pool_matched_null": pool_agg,
        "per_query": per_query_results,
    }

    save_results_json(output, str(output_path))

    # Print summary table
    _print_summary(orig_agg, pool_agg)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
