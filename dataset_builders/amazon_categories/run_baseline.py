"""
Run Baseline Top-K Experiments on Amazon Categories Dataset

Tests fixed k=5,10,20,50,100 and saves per-query and aggregate results
with analysis by K bucket (small/medium/large).
"""

import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embeddings.embedding_model import EmbeddingModel
from embeddings.vector_database import VectorDatabase
from retrieval.baseline_retrieval import BaselineRetrieval
from evaluation.evaluator import RetrievalEvaluator

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_amazon_dataset


def run_baseline_at_k(k, queries, qrels, vector_db, embedding_model, evaluator):
    """Run baseline retrieval at a specific k value."""
    retriever = BaselineRetrieval(vector_db=vector_db, k=k)

    per_query_results = []
    for i, query in enumerate(queries):
        if (i + 1) % 20 == 0:
            print(f"    Progress: {i+1}/{len(queries)}")

        query_embedding = embedding_model.embed_query(query.text)
        output = retriever.retrieve(query.query_id, query_embedding)

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
            },
        )
        per_query_results.append(result)

    aggregate = evaluator.evaluate_batch(per_query_results)
    return {"k": k, "aggregate": aggregate, "per_query": per_query_results}


def analyze_by_bucket(per_query_results, qrels):
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
        bucket_stats[bucket] = {
            "n_queries": len(results),
            "recall": float(np.mean([r.recall_at_k for r in results])),
            "precision": float(np.mean([r.precision_at_k for r in results])),
            "mrr": float(np.mean([r.reciprocal_rank for r in results])),
            "ndcg": float(np.mean([r.ndcg_at_k for r in results])),
            "map": float(np.mean([r.average_precision for r in results])),
            "mean_true_k": float(
                np.mean([r.metadata.get("true_k", 0) for r in results])
            ),
        }

    return bucket_stats


def main():
    output_dir = PROJECT_ROOT / "datasets" / "amazon_categories"
    results_dir = PROJECT_ROOT / "results" / "amazon_categories" / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)

    db_path = str(output_dir / "amazon_categories_vector_db")

    print("=" * 70)
    print("Baseline Top-K Experiments (Amazon Categories Dataset)")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    queries, qrels = load_amazon_dataset(str(output_dir))

    # Load vector DB
    print("\nLoading vector database...")
    vector_db = VectorDatabase.load(db_path)
    print(f"  Documents: {vector_db.get_num_documents()}")

    # Load embedding model
    print("\nLoading embedding model...")
    model = EmbeddingModel(
        model_name=EmbeddingModel.BGE_MODEL, normalize_embeddings=True
    )

    evaluator = RetrievalEvaluator()

    # Run experiments
    k_values = [5, 10, 20, 50, 100]
    all_results = {}

    print(f"\nRunning baseline experiments for k={k_values}...")
    print("-" * 70)

    for k in k_values:
        print(f"\n--- k={k} ---")
        result = run_baseline_at_k(k, queries, qrels, vector_db, model, evaluator)

        agg = result["aggregate"]
        print(f"  Recall@{k}: {agg.mean_recall_at_k_labeled:.4f}")
        print(f"  Precision@{k}: {agg.mean_precision_at_k_labeled:.4f}")
        print(f"  MRR: {agg.mean_reciprocal_rank_labeled:.4f}")
        print(f"  NDCG@{k}: {agg.mean_ndcg_at_k_labeled:.4f}")
        print(f"  MAP@{k}: {agg.mean_average_precision_labeled:.4f}")

        # Per-bucket analysis
        bucket_stats = analyze_by_bucket(result["per_query"], qrels)
        print(f"  By bucket:")
        for bucket, stats in bucket_stats.items():
            print(
                f"    {bucket} (n={stats['n_queries']}, avg_true_k={stats['mean_true_k']:.0f}): "
                f"R={stats['recall']:.3f} P={stats['precision']:.3f} F1={2*stats['recall']*stats['precision']/(stats['recall']+stats['precision']) if (stats['recall']+stats['precision'])>0 else 0:.3f}"
            )

        # Save results
        per_query_data = []
        for r in result["per_query"]:
            per_query_data.append(
                {
                    "query_id": r.query_id,
                    "k": r.k,
                    "true_k": len(r.relevant_ids),
                    "recall": r.recall_at_k,
                    "precision": r.precision_at_k,
                    "mrr": r.reciprocal_rank,
                    "ndcg": r.ndcg_at_k,
                    "hit_rate": r.hit_rate,
                    "map": r.average_precision,
                    "k_bucket": r.metadata.get("k_bucket", ""),
                    "subcategory": r.metadata.get("subcategory", ""),
                }
            )

        all_results[str(k)] = {
            "aggregate": {
                "recall": agg.mean_recall_at_k_labeled,
                "precision": agg.mean_precision_at_k_labeled,
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

    # Save
    results_path = results_dir / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"{'k':<6} {'Recall':<10} {'Precision':<10} {'F1':<10} {'MRR':<10} {'NDCG':<10} {'MAP':<10}"
    )
    print("-" * 66)
    for k in k_values:
        r = all_results[str(k)]["aggregate"]
        rec, prec = r["recall"], r["precision"]
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
        print(
            f"{k:<6} {rec:<10.4f} {prec:<10.4f} {f1:<10.4f} "
            f"{r['mrr']:<10.4f} {r['ndcg']:<10.4f} {r['map']:<10.4f}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
