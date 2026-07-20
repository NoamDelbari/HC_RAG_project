"""Run retrieval evaluation for all methods defined in config.

Saves both IR metrics and raw RetrievalOutput objects per method.

Usage: python -m experiments.scripts.run_retrieval --config experiments/configs/amazon_compound.yaml
"""

import argparse
import pickle
import numpy as np
from pathlib import Path

from hc_rag.embeddings.embedding_model import create_embedding_model
from hc_rag.embeddings.vector_database import VectorDatabase
from hc_rag.hc.null_distribution import NullDistribution
from hc_rag.retrieval import BaselineRetrieval, HCRetrieval
from hc_rag.evaluation.evaluator import RetrievalEvaluator

from experiments.lib.config import load_config
from experiments.lib.registry import get_adapter
from experiments.lib.results_io import save_results_json, analyze_by_bucket


def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--force", action="store_true", help="Rerun even if output exists")
    args = parser.parse_args()

    config = load_config(args.config)
    import experiments.datasets  # noqa: F401
    adapter = get_adapter(config.dataset.name)

    results_dir = Path(config.dataset.results_dir) / "retrieval"
    results_dir.mkdir(parents=True, exist_ok=True)

    queries, qrels, corpus = adapter.load_dataset(config.dataset.data_dir)

    artifacts_dir = Path(config.dataset.artifacts_dir)
    vector_db = VectorDatabase.load(str(artifacts_dir / "vector_db"))
    global_null = NullDistribution.load(str(artifacts_dir / "global_null"))
    model = create_embedding_model(
        config.embedding.model,
        normalize_embeddings=config.embedding.normalize,
    )

    evaluator = RetrievalEvaluator()

    for method in config.methods:
        metrics_path = results_dir / f"{method.name}_metrics.json"
        outputs_path = results_dir / f"{method.name}_outputs.pkl"

        if metrics_path.exists() and not args.force:
            print(f"Skipping {method.name} — results exist. Use --force to rerun.")
            continue

        print(f"\n{'=' * 60}")
        print(f"Running retrieval: {method.name}")
        print(f"{'=' * 60}")

        if method.type == "baseline":
            retriever = BaselineRetrieval(vector_db, k=method.k, embedding_model=model)
        else:
            retriever = HCRetrieval(
                vector_db,
                global_null_distribution=global_null,
                gamma=config.hc.gamma,
                max_candidates=config.hc.pool_size,
                min_hc=0.0,
                allow_empty=False,
                embedding_model=model,
                use_zscore=True,
            )

        all_outputs = []
        all_results = []

        for query in queries:
            query_emb = model.embed_query(query.text)
            output = retriever.retrieve(query.query_id, query_emb)
            all_outputs.append(output)

            relevant_ids = qrels.get(query.query_id, set())
            result = evaluator.evaluate_single(
                query.query_id, output.retrieved_ids, output.retrieved_scores, relevant_ids
            )

            result_dict = {
                "query_id": query.query_id,
                "k": output.k,
                "recall": result.recall_at_k,
                "precision": result.precision_at_k,
                "mrr": result.reciprocal_rank,
                "ndcg": result.ndcg_at_k,
                "map": result.average_precision,
                "hit_rate": result.hit_rate,
                "metadata": adapter.get_result_metadata(query),
            }
            if method.type == "hc":
                result_dict["hc_k"] = output.k
            all_results.append(result_dict)

        aggregate = {
            "method": method.name,
            "type": method.type,
            "n_queries": len(all_results),
            "mean_k": float(np.mean([r["k"] for r in all_results])),
            "mean_recall": float(np.mean([r["recall"] for r in all_results])),
            "mean_precision": float(np.mean([r["precision"] for r in all_results])),
            "mean_mrr": float(np.mean([r["mrr"] for r in all_results])),
            "mean_ndcg": float(np.mean([r["ndcg"] for r in all_results])),
            "mean_map": float(np.mean([r["map"] for r in all_results])),
            "mean_hit_rate": float(np.mean([r["hit_rate"] for r in all_results])),
        }

        bucket_field = adapter.get_bucket_field()
        bucket_analysis = analyze_by_bucket(all_results, bucket_field)

        save_results_json(
            {"aggregate": aggregate, "per_query": all_results, "by_bucket": bucket_analysis},
            str(metrics_path),
        )
        with open(outputs_path, "wb") as f:
            pickle.dump(all_outputs, f)

        print(f"\n  Mean recall: {aggregate['mean_recall']:.4f}")
        print(f"  Mean precision: {aggregate['mean_precision']:.4f}")
        print(f"  Mean k: {aggregate['mean_k']:.1f}")
        print(f"  Saved to {metrics_path}")


if __name__ == "__main__":
    main()
