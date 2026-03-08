"""
Run Baseline Top-K Experiments on Category-Key Dataset

Tests fixed k=5,8,10,12,15,20 and saves per-query and aggregate results.
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
from data_loader import load_category_key_dataset


def run_baseline_at_k(
    k: int,
    queries,
    qrels,
    vector_db,
    embedding_model,
    evaluator
) -> dict:
    """Run baseline retrieval at a specific k value."""
    retriever = BaselineRetrieval(vector_db=vector_db, k=k)

    per_query_results = []
    for query in queries:
        # Embed query
        query_embedding = embedding_model.embed_query(query.text)

        # Retrieve
        output = retriever.retrieve(query.query_id, query_embedding)

        # Evaluate
        relevant_ids = qrels.get(query.query_id, set())
        result = evaluator.evaluate_single(
            query_id=query.query_id,
            retrieved_ids=output.retrieved_ids,
            retrieved_scores=output.retrieved_scores,
            relevant_ids=relevant_ids,
            metadata={
                "country": query.country,
                "size_bucket": query.size_bucket,
                "true_k": len(relevant_ids)
            }
        )
        per_query_results.append(result)

    # Aggregate
    aggregate = evaluator.evaluate_batch(per_query_results)

    return {
        "k": k,
        "aggregate": aggregate,
        "per_query": per_query_results
    }


def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    results_dir = Path(PROJECT_ROOT) / "results" / "category_key" / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)

    db_path = str(output_dir / "category_key_vector_db")

    print("=" * 70)
    print("Baseline Top-K Experiments (Category-Key Dataset)")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    queries, qrels = load_category_key_dataset(str(output_dir))

    # Load vector DB
    print("\nLoading vector database...")
    vector_db = VectorDatabase.load(db_path)

    # Load embedding model
    print("\nLoading embedding model...")
    model = EmbeddingModel(
        model_name=EmbeddingModel.BGE_MODEL,
        normalize_embeddings=True
    )

    evaluator = RetrievalEvaluator()

    # Run experiments at different k values
    k_values = [5, 8, 10, 12, 15, 20]
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
        print(f"  Hit Rate: {agg.mean_hit_rate_labeled:.4f}")
        print(f"  MAP@{k}: {agg.mean_average_precision_labeled:.4f}")

        # Save per-query results
        per_query_data = []
        for r in result["per_query"]:
            per_query_data.append({
                "query_id": r.query_id,
                "k": r.k,
                "true_k": len(r.relevant_ids),
                "recall": r.recall_at_k,
                "precision": r.precision_at_k,
                "mrr": r.reciprocal_rank,
                "ndcg": r.ndcg_at_k,
                "hit_rate": r.hit_rate,
                "map": r.average_precision,
                "country": r.metadata.get("country", ""),
                "size_bucket": r.metadata.get("size_bucket", ""),
                "retrieved_ids": r.retrieved_ids
            })

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
            "per_query": per_query_data
        }

    # Save all results
    results_path = results_dir / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'k':<6} {'Recall':<10} {'Precision':<10} {'MRR':<10} {'NDCG':<10} {'MAP':<10}")
    print("-" * 56)
    for k in k_values:
        r = all_results[str(k)]["aggregate"]
        print(f"{k:<6} {r['recall']:<10.4f} {r['precision']:<10.4f} "
              f"{r['mrr']:<10.4f} {r['ndcg']:<10.4f} {r['map']:<10.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
