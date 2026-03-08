"""
Run HC Experiments on Category-Key Dataset

Tests HC retrieval with gamma values 0.07, 0.1, 0.2, 0.4.
"""

import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embeddings.embedding_model import EmbeddingModel
from embeddings.vector_database import VectorDatabase
from hc.null_distribution import QueryNullDistributions
from retrieval.hc_retrieval import HCRetrieval
from evaluation.evaluator import RetrievalEvaluator
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_category_key_dataset


def run_hc_at_gamma(
    gamma: float,
    queries,
    qrels,
    vector_db,
    query_null_dists,
    embedding_model,
    evaluator,
    max_candidates: int = 100
) -> dict:
    """Run HC retrieval at a specific gamma value."""
    retriever = HCRetrieval(
        vector_db=vector_db,
        query_null_distributions=query_null_dists,
        gamma=gamma,
        max_candidates=max_candidates,
        min_hc=0.0,
        allow_empty=False,
        embedding_model=embedding_model
    )

    per_query_results = []
    for query in queries:
        # Retrieve using HC
        output = retriever.retrieve_from_text(query.query_id, query.text)

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
                "true_k": len(relevant_ids),
                "hc_k": output.k,
                "threshold": output.threshold
            }
        )
        per_query_results.append(result)

    # Aggregate
    aggregate = evaluator.evaluate_batch(per_query_results)

    return {
        "gamma": gamma,
        "aggregate": aggregate,
        "per_query": per_query_results
    }


def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    results_dir = Path(PROJECT_ROOT) / "results" / "category_key" / "hc"
    results_dir.mkdir(parents=True, exist_ok=True)

    db_path = str(output_dir / "category_key_vector_db")
    null_path = str(output_dir / "null_distributions" / "category_key_per_query_null")

    print("=" * 70)
    print("HC Experiments (Category-Key Dataset)")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    queries, qrels = load_category_key_dataset(str(output_dir))

    # Load vector DB
    print("\nLoading vector database...")
    vector_db = VectorDatabase.load(db_path)

    # Load null distributions
    print("\nLoading null distributions...")
    null_dists = QueryNullDistributions.load(null_path)
    print(f"  Loaded {len(null_dists.distributions)} distributions")

    # Load embedding model
    print("\nLoading embedding model...")
    model = EmbeddingModel(
        model_name=EmbeddingModel.BGE_MODEL,
        normalize_embeddings=True
    )

    evaluator = RetrievalEvaluator()

    # Run experiments at different gamma values
    gamma_values = [0.07, 0.1, 0.2, 0.4]
    all_results = {}

    print(f"\nRunning HC experiments for gamma={gamma_values}...")
    print("-" * 70)

    for gamma in gamma_values:
        print(f"\n--- gamma={gamma} ---")
        result = run_hc_at_gamma(
            gamma, queries, qrels, vector_db, null_dists, model, evaluator
        )

        agg = result["aggregate"]
        print(f"  Recall: {agg.mean_recall_at_k_labeled:.4f}")
        print(f"  Precision: {agg.mean_precision_at_k_labeled:.4f}")
        print(f"  MRR: {agg.mean_reciprocal_rank_labeled:.4f}")
        print(f"  NDCG: {agg.mean_ndcg_at_k_labeled:.4f}")
        print(f"  Hit Rate: {agg.mean_hit_rate_labeled:.4f}")
        print(f"  MAP: {agg.mean_average_precision_labeled:.4f}")
        print(f"  Mean K selected: {agg.mean_k:.1f} (range: {agg.min_k}-{agg.max_k})")

        # Save per-query results
        per_query_data = []
        for r in result["per_query"]:
            per_query_data.append({
                "query_id": r.query_id,
                "hc_k": r.k,
                "true_k": len(r.relevant_ids),
                "recall": r.recall_at_k,
                "precision": r.precision_at_k,
                "mrr": r.reciprocal_rank,
                "ndcg": r.ndcg_at_k,
                "hit_rate": r.hit_rate,
                "map": r.average_precision,
                "country": r.metadata.get("country", ""),
                "size_bucket": r.metadata.get("size_bucket", ""),
                "threshold": r.metadata.get("threshold", None),
                "retrieved_ids": r.retrieved_ids
            })

        all_results[str(gamma)] = {
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
    results_path = results_dir / "hc_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'gamma':<8} {'Recall':<10} {'Precision':<10} {'MRR':<10} {'NDCG':<10} {'MAP':<10} {'Mean K':<10}")
    print("-" * 68)
    for gamma in gamma_values:
        r = all_results[str(gamma)]["aggregate"]
        print(f"{gamma:<8} {r['recall']:<10.4f} {r['precision']:<10.4f} "
              f"{r['mrr']:<10.4f} {r['ndcg']:<10.4f} {r['map']:<10.4f} {r['mean_k']:<10.1f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
