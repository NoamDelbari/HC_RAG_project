"""
Run HC Experiments with Large Pool Sizes on Amazon Categories Dataset

Extended pool sizes: 500, 1000, 2000, 3000, 5000
with gamma values: 0.05, 0.1, 0.15, 0.2
"""

import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embeddings.embedding_model import EmbeddingModel
from embeddings.vector_database import VectorDatabase
from hc.null_distribution import NullDistribution
from retrieval.hc_retrieval import HCRetrieval
from evaluation.evaluator import RetrievalEvaluator

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_amazon_dataset


def run_hc_experiment(
    gamma, max_candidates, queries, qrels, vector_db, global_null, embedding_model, evaluator
):
    """Run HC retrieval with given gamma and candidate pool size."""
    retriever = HCRetrieval(
        vector_db=vector_db,
        global_null_distribution=global_null,
        gamma=gamma,
        max_candidates=max_candidates,
        min_hc=0.0,
        allow_empty=False,
        embedding_model=embedding_model,
        use_zscore=True,
    )

    per_query_results = []
    for i, query in enumerate(queries):
        if (i + 1) % 20 == 0:
            print(f"    Progress: {i+1}/{len(queries)}")

        output = retriever.retrieve_from_text(query.query_id, query.text)

        relevant_ids = qrels.get(query.query_id, set())
        result = evaluator.evaluate_single(
            query_id=query.query_id,
            retrieved_ids=output.retrieved_ids,
            retrieved_scores=output.retrieved_scores,
            relevant_ids=relevant_ids,
            metadata={
                "subcategory": query.subcategory,
                "k_bucket": query.k_bucket,
                "true_k": len(relevant_ids),
                "hc_k": output.k,
                "threshold": output.threshold,
            },
        )
        per_query_results.append(result)

    aggregate = evaluator.evaluate_batch(per_query_results)
    return {
        "gamma": gamma,
        "max_candidates": max_candidates,
        "aggregate": aggregate,
        "per_query": per_query_results,
    }


def analyze_by_bucket(per_query_results):
    """Compute metrics broken down by K bucket."""
    buckets = {"small": [], "medium": [], "large": []}

    for r in per_query_results:
        bucket = r.metadata.get("k_bucket", "unknown")
        if bucket in buckets:
            buckets[bucket].append(r)

    bucket_stats = {}
    for bucket, results in buckets.items():
        if not results:
            continue
        hc_ks = [r.metadata.get("hc_k", r.k) for r in results]
        true_ks = [r.metadata.get("true_k", 0) for r in results]
        bucket_stats[bucket] = {
            "n_queries": len(results),
            "recall": float(np.mean([r.recall_at_k for r in results])),
            "precision": float(np.mean([r.precision_at_k for r in results])),
            "mrr": float(np.mean([r.reciprocal_rank for r in results])),
            "ndcg": float(np.mean([r.ndcg_at_k for r in results])),
            "map": float(np.mean([r.average_precision for r in results])),
            "mean_true_k": float(np.mean(true_ks)),
            "mean_hc_k": float(np.mean(hc_ks)),
            "median_hc_k": float(np.median(hc_ks)),
            "k_correlation": float(np.corrcoef(true_ks, hc_ks)[0, 1])
            if len(set(hc_ks)) > 1
            else 0.0,
        }

    return bucket_stats


def main():
    output_dir = PROJECT_ROOT / "datasets" / "amazon_categories"
    results_dir = PROJECT_ROOT / "results" / "amazon_categories" / "hc"
    results_dir.mkdir(parents=True, exist_ok=True)

    db_path = str(output_dir / "amazon_categories_vector_db")
    null_path = str(output_dir / "null_distributions" / "amazon_global_null")

    print("=" * 70)
    print("HC Experiments - LARGE POOL SIZES (Amazon Categories Dataset)")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    queries, qrels = load_amazon_dataset(str(output_dir))

    # Load vector DB
    print("\nLoading vector database...")
    vector_db = VectorDatabase.load(db_path)
    print(f"  Documents: {vector_db.get_num_documents()}")

    # Load global null distribution
    print("\nLoading global z-score null distribution...")
    global_null = NullDistribution.load(null_path)
    print(f"  {global_null}")

    # Load embedding model
    print("\nLoading embedding model...")
    model = EmbeddingModel(
        model_name=EmbeddingModel.BGE_MODEL, normalize_embeddings=True
    )

    evaluator = RetrievalEvaluator()

    # Extended experiment grid
    gamma_values = [0.05, 0.1, 0.15, 0.2]
    pool_sizes = [500, 1000, 2000, 3000, 5000]

    all_results = {}

    print(f"\nRunning HC experiments with large pool sizes...")
    print(f"  Gamma values: {gamma_values}")
    print(f"  Pool sizes: {pool_sizes}")
    print(f"  Total configs: {len(gamma_values) * len(pool_sizes)}")
    print("-" * 70)

    for pool in pool_sizes:
        for gamma in gamma_values:
            config_key = f"gamma={gamma}_pool={pool}"
            print(f"\n--- {config_key} ---")

            result = run_hc_experiment(
                gamma, pool, queries, qrels, vector_db, global_null, model, evaluator
            )

            agg = result["aggregate"]
            rec = agg.mean_recall_at_k_labeled
            prec = agg.mean_precision_at_k_labeled
            f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0

            print(f"  Recall: {rec:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  F1: {f1:.4f}")
            print(f"  MRR: {agg.mean_reciprocal_rank_labeled:.4f}")
            print(f"  NDCG: {agg.mean_ndcg_at_k_labeled:.4f}")
            print(f"  MAP: {agg.mean_average_precision_labeled:.4f}")
            print(
                f"  Mean K selected: {agg.mean_k:.1f} (range: {agg.min_k}-{agg.max_k})"
            )

            # Per-bucket analysis
            bucket_stats = analyze_by_bucket(result["per_query"])
            print(f"  By bucket:")
            for bucket, stats in bucket_stats.items():
                b_rec, b_prec = stats["recall"], stats["precision"]
                b_f1 = (
                    2 * b_rec * b_prec / (b_rec + b_prec)
                    if (b_rec + b_prec) > 0
                    else 0
                )
                print(
                    f"    {bucket} (n={stats['n_queries']}, true_k={stats['mean_true_k']:.0f}, "
                    f"hc_k={stats['mean_hc_k']:.1f}): R={b_rec:.3f} P={b_prec:.3f} F1={b_f1:.3f}"
                )

            # Serialize
            per_query_data = []
            for r in result["per_query"]:
                per_query_data.append(
                    {
                        "query_id": r.query_id,
                        "hc_k": r.k,
                        "true_k": len(r.relevant_ids),
                        "recall": r.recall_at_k,
                        "precision": r.precision_at_k,
                        "mrr": r.reciprocal_rank,
                        "ndcg": r.ndcg_at_k,
                        "hit_rate": r.hit_rate,
                        "map": r.average_precision,
                        "k_bucket": r.metadata.get("k_bucket", ""),
                        "subcategory": r.metadata.get("subcategory", ""),
                        "threshold": r.metadata.get("threshold", None),
                    }
                )

            all_results[config_key] = {
                "config": {"gamma": gamma, "max_candidates": pool},
                "aggregate": {
                    "recall": rec,
                    "precision": prec,
                    "f1": f1,
                    "mrr": agg.mean_reciprocal_rank_labeled,
                    "ndcg": agg.mean_ndcg_at_k_labeled,
                    "hit_rate": agg.mean_hit_rate_labeled,
                    "map": agg.mean_average_precision_labeled,
                    "mean_k": agg.mean_k,
                    "min_k": agg.min_k,
                    "max_k": agg.max_k,
                },
                "by_bucket": bucket_stats,
                "per_query": per_query_data,
            }

    # Save to separate file
    results_path = results_dir / "hc_large_pool_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Also load original results and merge for combined summary
    orig_path = results_dir / "hc_results.json"
    if orig_path.exists():
        with open(orig_path) as f:
            orig_results = json.load(f)
        # Merge (skip pool=500 from new since it's already in original)
        combined = dict(orig_results)
        for k, v in all_results.items():
            if k not in combined:
                combined[k] = v
        combined_path = results_dir / "hc_all_results.json"
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"Combined results saved to {combined_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY - All configs by F1 (large pool experiment)")
    print("=" * 80)
    print(
        f"{'Config':<25} {'Recall':<8} {'Prec':<8} {'F1':<8} {'MRR':<8} {'NDCG':<8} {'MAP':<8} {'Mean K':<8}"
    )
    print("-" * 81)

    sorted_configs = sorted(
        all_results.items(), key=lambda x: x[1]["aggregate"]["f1"], reverse=True
    )
    for config_key, data in sorted_configs:
        r = data["aggregate"]
        print(
            f"{config_key:<25} {r['recall']:<8.4f} {r['precision']:<8.4f} "
            f"{r['f1']:<8.4f} {r['mrr']:<8.4f} {r['ndcg']:<8.4f} {r['map']:<8.4f} {r['mean_k']:<8.1f}"
        )

    # Compare against baseline
    baseline_path = PROJECT_ROOT / "results" / "amazon_categories" / "baseline" / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)

        print("\n" + "=" * 80)
        print("COMPARISON: Best HC (large pool) vs Baseline")
        print("=" * 80)

        best_hc = sorted_configs[0]
        best_hc_r = best_hc[1]["aggregate"]

        # Find best baseline
        best_b_f1, best_b_k = -1, None
        for k_str, data in baseline.items():
            r = data["aggregate"]
            rec, prec = r["recall"], r["precision"]
            f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
            if f1 > best_b_f1:
                best_b_f1, best_b_k = f1, k_str

        best_b_r = baseline[best_b_k]["aggregate"]

        print(f"\n  Best HC:       {best_hc[0]}")
        print(f"  Best Baseline: k={best_b_k}")
        print(f"\n  {'Metric':<15} {'Baseline':<12} {'HC':<12} {'Delta':<10}")
        print("  " + "-" * 49)

        for metric, b_val, h_val in [
            ("Recall", best_b_r["recall"], best_hc_r["recall"]),
            ("Precision", best_b_r["precision"], best_hc_r["precision"]),
            ("F1", best_b_f1, best_hc_r["f1"]),
            ("MRR", best_b_r["mrr"], best_hc_r["mrr"]),
            ("MAP", best_b_r["map"], best_hc_r["map"]),
            ("Mean K", float(best_b_k), best_hc_r["mean_k"]),
        ]:
            delta = h_val - b_val
            sign = "+" if delta >= 0 else ""
            print(f"  {metric:<15} {b_val:<12.4f} {h_val:<12.4f} {sign}{delta:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
