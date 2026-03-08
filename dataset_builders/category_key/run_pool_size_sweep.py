"""
Sweep candidate pool sizes for HC retrieval on the category-key dataset.

Tests pool sizes from 100 up to the full corpus (726 docs).
For each pool size, gamma = max_true_k / pool_size so that the HC scan
window always covers exactly the max true K (24).

Output: results/category_key/pool_size_sweep.json
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


def run_hc(gamma, queries, qrels, vector_db, null_dists, model, evaluator, max_candidates):
    retriever = HCRetrieval(
        vector_db=vector_db, query_null_distributions=null_dists,
        gamma=gamma, max_candidates=max_candidates, min_hc=0.0,
        allow_empty=False, embedding_model=model
    )
    per_query = []
    for q in queries:
        out = retriever.retrieve_from_text(q.query_id, q.text)
        rel = qrels.get(q.query_id, set())
        r = evaluator.evaluate_single(
            query_id=q.query_id, retrieved_ids=out.retrieved_ids,
            retrieved_scores=out.retrieved_scores, relevant_ids=rel,
            metadata={"country": q.country, "size_bucket": q.size_bucket,
                      "true_k": len(rel), "hc_k": out.k, "threshold": out.threshold}
        )
        per_query.append(r)
    agg = evaluator.evaluate_batch(per_query)
    return agg, per_query


def serialize(agg, per_query):
    pq_data = []
    for r in per_query:
        pq_data.append({
            "query_id": r.query_id, "k": r.k, "true_k": len(r.relevant_ids),
            "recall": r.recall_at_k, "precision": r.precision_at_k,
            "mrr": r.reciprocal_rank, "ndcg": r.ndcg_at_k,
            "hit_rate": r.hit_rate, "map": r.average_precision,
            "country": r.metadata.get("country", ""),
            "size_bucket": r.metadata.get("size_bucket", ""),
            "hc_k": r.metadata.get("hc_k"), "threshold": r.metadata.get("threshold"),
        })
    return {
        "aggregate": {
            "recall": agg.mean_recall_at_k_labeled,
            "precision": agg.mean_precision_at_k_labeled,
            "mrr": agg.mean_reciprocal_rank_labeled,
            "ndcg": agg.mean_ndcg_at_k_labeled,
            "hit_rate": agg.mean_hit_rate_labeled,
            "map": agg.mean_average_precision_labeled,
            "mean_k": agg.mean_k, "min_k": agg.min_k, "max_k": agg.max_k,
        },
        "per_query": pq_data,
    }


def main():
    data_dir = Path(__file__).parent / "output"
    db_path = str(data_dir / "category_key_vector_db")
    null_path = str(data_dir / "null_distributions" / "category_key_per_query_null")
    out_dir = PROJECT_ROOT / "results" / "category_key"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Pool Size Sweep (Category-Key Dataset)")
    print("=" * 70)

    # Load resources
    print("\nLoading dataset...")
    queries, qrels = load_category_key_dataset(str(data_dir))

    print("Loading vector database...")
    vector_db = VectorDatabase.load(db_path)
    corpus_size = len(vector_db.doc_ids)
    print(f"  Corpus size: {corpus_size}")

    print("Loading null distributions...")
    null_dists = QueryNullDistributions.load(null_path)

    print("Loading embedding model...")
    model = EmbeddingModel(model_name=EmbeddingModel.BGE_MODEL, normalize_embeddings=True)

    evaluator = RetrievalEvaluator()

    max_true_k = max(len(qrels[q.query_id]) for q in queries)
    print(f"Max true K: {max_true_k}")

    # Pool sizes to test: 100 (baseline), then increasing up to full corpus
    pool_sizes = [100, 150, 200, 300, 400, 500, corpus_size]
    # Remove duplicates and sort
    pool_sizes = sorted(set(pool_sizes))

    all_results = {}

    print(f"\nSweeping pool sizes: {pool_sizes}")
    print("-" * 70)

    for pool_size in pool_sizes:
        gamma = max_true_k / pool_size
        max_hc_k = int(np.floor(gamma * pool_size))
        print(f"\n--- pool={pool_size}, gamma={gamma:.4f}, max_hc_k={max_hc_k} ---")

        agg, pq = run_hc(gamma, queries, qrels, vector_db, null_dists, model, evaluator, pool_size)

        f1 = 2 * agg.mean_recall_at_k_labeled * agg.mean_precision_at_k_labeled / (
            agg.mean_recall_at_k_labeled + agg.mean_precision_at_k_labeled
        ) if (agg.mean_recall_at_k_labeled + agg.mean_precision_at_k_labeled) > 0 else 0

        print(f"  Recall:    {agg.mean_recall_at_k_labeled:.4f}")
        print(f"  Precision: {agg.mean_precision_at_k_labeled:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Mean K:    {agg.mean_k:.1f} (range: {agg.min_k}-{agg.max_k})")

        all_results[str(pool_size)] = {
            "pool_size": pool_size,
            "gamma": gamma,
            "max_hc_k": max_hc_k,
            **serialize(agg, pq),
        }

    # Save
    out_path = out_dir / "pool_size_sweep.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Pool':<8} {'Gamma':<8} {'Recall':<10} {'Prec':<10} {'F1':<10} {'Mean K':<10} {'Range'}")
    print("-" * 70)
    for ps in pool_sizes:
        r = all_results[str(ps)]
        a = r["aggregate"]
        f1 = 2 * a["recall"] * a["precision"] / (a["recall"] + a["precision"]) if (a["recall"] + a["precision"]) > 0 else 0
        print(f"{ps:<8} {r['gamma']:<8.4f} {a['recall']:<10.4f} {a['precision']:<10.4f} "
              f"{f1:<10.4f} {a['mean_k']:<10.1f} {a['min_k']}-{a['max_k']}")


if __name__ == "__main__":
    main()
