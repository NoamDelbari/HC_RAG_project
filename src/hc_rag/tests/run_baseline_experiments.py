"""
Run Baseline Top-k Retrieval Experiments

Tests baseline top-k retrieval with different k values (5, 10, 15, 20)
and evaluates using standard IR metrics.

Usage:
    python src/tests/run_baseline_experiments.py                    # Run all queries
    python src/tests/run_baseline_experiments.py --max-queries 100  # Test subset
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Set
import numpy as np
from tqdm import tqdm
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.crag_loader import CRAGLoader
from embeddings.embedding_model import EmbeddingModel
from embeddings.vector_database import VectorDatabase
from retrieval.baseline_retrieval import BaselineRetrieval
from evaluation.evaluator import RetrievalEvaluator, RetrievalResult


def extract_ground_truth_ids(query) -> Set[str]:
    """
    Extract ground truth relevant document IDs from query.

    In CRAG, each query has search_results which are the relevant documents.
    Document IDs are created by hashing the page URL.

    Args:
        query: CRAGQuery object

    Returns:
        Set of relevant document IDs (hashed URLs)
    """
    import hashlib
    relevant_ids = set()

    # Document IDs are created by hashing URLs (matches database indexing)
    for search_result in query.search_results:
        doc_url = search_result.get("page_url", "")

        if doc_url:
            # Hash URL to create consistent doc_id
            doc_id = hashlib.md5(doc_url.encode('utf-8')).hexdigest()
        else:
            # Fallback: hash content if no URL
            content = str(search_result.get("page_snippet", "")) + str(search_result.get("page_name", ""))
            doc_id = hashlib.md5(content.encode('utf-8')).hexdigest()

        relevant_ids.add(doc_id)

    return relevant_ids


def run_baseline_experiment(
    k: int,
    queries: List,
    db_path: str,
    embedding_model: EmbeddingModel,
    evaluator: RetrievalEvaluator,
    aggregation: str = "max_score",
    top_chunks: int = None,
    max_queries: int = None
) -> tuple:
    """
    Run baseline experiment for a specific k value.

    Supports both full document and chunked databases (auto-detected).

    Args:
        k: Number of documents to retrieve
        queries: List of CRAG queries
        db_path: Path to vector database
        embedding_model: Embedding model for queries
        evaluator: RetrievalEvaluator instance
        aggregation: Chunk aggregation strategy (for chunked DBs)
        top_chunks: Number of chunks to retrieve (for chunked DBs, default: k*10)
        max_queries: Optional limit on number of queries

    Returns:
        Tuple of (retrieval_results, evaluation_results, aggregate_metrics)
    """
    print(f"\nRunning baseline experiment with k={k}")
    print("-" * 80)

    # Limit queries if requested
    if max_queries and len(queries) > max_queries:
        print(f"⚠ Limiting to {max_queries} queries for testing")
        queries = queries[:max_queries]

    # Create retriever (auto-detects if database is chunked)
    retriever = BaselineRetrieval.from_database_path(
        db_path=db_path,
        k=k,
        embedding_model=embedding_model,
        aggregation=aggregation,
        top_chunks=top_chunks
    )

    # Print diagnostic info about database type
    is_chunked = retriever.chunk_to_doc_mapping is not None
    print(f"  Database type: {'Chunked' if is_chunked else 'Full document'}")
    if is_chunked:
        print(f"  Chunk aggregation: {aggregation}")
        print(f"  Top chunks to retrieve: {retriever.top_chunks}")

    # Process queries
    print(f"Processing {len(queries)} queries...")

    retrieval_results = []
    evaluation_results = []

    # Track whether we've shown debug info
    shown_debug = False

    for query in tqdm(queries, desc=f"k={k}"):
        # Embed query
        query_embedding = embedding_model.embed_query(query.query)

        # Retrieve
        retrieval_output = retriever.retrieve(
            query_id=query.query_id,
            query_embedding=query_embedding
        )
        retrieval_results.append(retrieval_output)

        # Get ground truth
        relevant_ids = extract_ground_truth_ids(query)

        # Show debug info for first query
        if not shown_debug:
            print(f"\n  [DEBUG] First query diagnostics:")
            print(f"    Query ID: {query.query_id}")
            print(f"    Ground truth IDs (first 3): {list(relevant_ids)[:3]}")
            print(f"    Retrieved IDs (first 3): {retrieval_output.retrieved_ids[:3]}")
            matches = len(set(retrieval_output.retrieved_ids) & relevant_ids)
            print(f"    Matches found: {matches}/{len(relevant_ids)}")
            shown_debug = True

        # Evaluate
        eval_result = evaluator.evaluate_single(
            query_id=query.query_id,
            retrieved_ids=retrieval_output.retrieved_ids,
            retrieved_scores=retrieval_output.retrieved_scores,
            relevant_ids=relevant_ids,
            metadata={
                "query_text": query.query,
                "k": k
            }
        )
        evaluation_results.append(eval_result)

    # Aggregate metrics
    aggregate_metrics = evaluator.evaluate_batch(evaluation_results)

    print(f"\n✓ Completed k={k}")
    print(f"  Recall@{k}: {aggregate_metrics.mean_recall_at_k_labeled:.3f}")
    print(f"  Precision@{k}: {aggregate_metrics.mean_precision_at_k_labeled:.3f}")
    print(f"  MRR: {aggregate_metrics.mean_reciprocal_rank_labeled:.3f}")
    print(f"  NDCG@{k}: {aggregate_metrics.mean_ndcg_at_k_labeled:.3f}")

    return retrieval_results, evaluation_results, aggregate_metrics


def save_results(
    k: int,
    retrieval_results: List,
    evaluation_results: List,
    aggregate_metrics,
    output_dir: Path
):
    """
    Save experiment results to disk.

    Args:
        k: k value for this experiment
        retrieval_results: List of RetrievalOutput objects
        evaluation_results: List of RetrievalResult objects
        aggregate_metrics: AggregateMetrics object
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-query results
    per_query_results = []
    for retrieval, evaluation in zip(retrieval_results, evaluation_results):
        per_query_results.append({
            "query_id": retrieval.query_id,
            "k": k,
            "retrieved_ids": retrieval.retrieved_ids,
            "retrieved_scores": retrieval.retrieved_scores,
            "relevant_ids": list(evaluation.relevant_ids),
            "metrics": {
                "recall_at_k": evaluation.recall_at_k,
                "precision_at_k": evaluation.precision_at_k,
                "reciprocal_rank": evaluation.reciprocal_rank,
                "ndcg_at_k": evaluation.ndcg_at_k,
                "hit_rate": evaluation.hit_rate,
                "average_precision": evaluation.average_precision
            }
        })

    per_query_file = output_dir / f"baseline_k{k}_per_query.json"
    with open(per_query_file, "w") as f:
        json.dump(per_query_results, f, indent=2)

    print(f"  Saved per-query results: {per_query_file}")

    # Save aggregate metrics
    aggregate_file = output_dir / f"baseline_k{k}_aggregate.json"
    aggregate_data = {
        "k": k,
        "n_queries": aggregate_metrics.n_queries,
        "n_labeled_queries": aggregate_metrics.n_labeled_queries,
        "n_unlabeled_queries": aggregate_metrics.n_unlabeled_queries,
        "all_queries": {
            "recall_at_k": aggregate_metrics.mean_recall_at_k,
            "precision_at_k": aggregate_metrics.mean_precision_at_k,
            "mrr": aggregate_metrics.mean_reciprocal_rank,
            "ndcg_at_k": aggregate_metrics.mean_ndcg_at_k,
            "hit_rate": aggregate_metrics.mean_hit_rate,
            "map_at_k": aggregate_metrics.mean_average_precision
        },
        "labeled_queries_only": {
            "recall_at_k": aggregate_metrics.mean_recall_at_k_labeled,
            "precision_at_k": aggregate_metrics.mean_precision_at_k_labeled,
            "mrr": aggregate_metrics.mean_reciprocal_rank_labeled,
            "ndcg_at_k": aggregate_metrics.mean_ndcg_at_k_labeled,
            "hit_rate": aggregate_metrics.mean_hit_rate_labeled,
            "map_at_k": aggregate_metrics.mean_average_precision_labeled
        },
        "coverage": {
            "query_coverage": aggregate_metrics.query_coverage,
            "catalog_coverage": aggregate_metrics.catalog_coverage
        },
        "retrieval_stats": {
            "mean_k": aggregate_metrics.mean_k,
            "min_k": aggregate_metrics.min_k,
            "max_k": aggregate_metrics.max_k
        },
        "quality_flags": {
            "queries_with_duplicates": aggregate_metrics.n_queries_with_duplicates
        }
    }

    with open(aggregate_file, "w") as f:
        json.dump(aggregate_data, f, indent=2)

    print(f"  Saved aggregate metrics: {aggregate_file}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline top-k retrieval experiments")
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20],
        help="k values to test (default: 5 10 15 20)"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of queries to test (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/baseline_experiments",
        help="Output directory for results (default: results/baseline_experiments)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="src\database\crag_snippet_chunked_vector_db",
        help="Path to vector database (default: src/database/crag_vector_db)"
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="max_score",
        choices=["max_score", "mean_score", "sum_score"],
        help="Chunk aggregation strategy for chunked DBs (default: max_score)"
    )
    parser.add_argument(
        "--top-chunks",
        type=int,
        default=None,
        help="Number of chunks to retrieve for chunked DBs (default: k*10)"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("BASELINE TOP-K RETRIEVAL EXPERIMENTS")
    print("="*80)
    print(f"k values: {args.k_values}")
    print(f"Max queries: {args.max_queries if args.max_queries else 'All'}")
    print(f"Database: {args.db_path}")
    print(f"Output: {args.output_dir}")
    print("="*80 + "\n")

    # =========================================================================
    # Step 1: Load CRAG Queries
    # =========================================================================
    print("Step 1: Loading CRAG queries...")
    print("-" * 80)

    loader = CRAGLoader(use_full_html=True)
    queries, documents = loader.load_by_tasks(["1_2"])

    print(f"✓ Loaded {len(queries)} queries")
    print(f"✓ Loaded {len(documents)} documents")
    print()

    # =========================================================================
    # Step 2: Initialize Embedding Model
    # =========================================================================
    print("Step 2: Initializing embedding model...")
    print("-" * 80)

    # Use quality model (same as used to build database)
    embedding_model = EmbeddingModel(model_name=EmbeddingModel.BGE_MODEL)

    print(f"✓ Model: {embedding_model.get_model_name()}")
    print(f"✓ Embedding dimension: {embedding_model.get_embedding_dim()}")
    print(f"✓ Device: {embedding_model.device}")
    print()

    # =========================================================================
    # Step 3: Initialize Evaluator
    # =========================================================================
    print("Step 3: Initializing evaluator...")
    print("-" * 80)

    evaluator = RetrievalEvaluator()
    print()

    # =========================================================================
    # Step 4: Run Experiments for Each k
    # =========================================================================
    print("Step 4: Running baseline experiments...")
    print("="*80)

    output_dir = Path(args.output_dir)

    all_aggregate_metrics = {}

    start_time = time.time()

    for k in args.k_values:
        retrieval_results, evaluation_results, aggregate_metrics = run_baseline_experiment(
            k=k,
            queries=queries,
            db_path=args.db_path,
            embedding_model=embedding_model,
            evaluator=evaluator,
            aggregation=args.aggregation,
            top_chunks=args.top_chunks,
            max_queries=args.max_queries
        )

        # Save results
        save_results(
            k=k,
            retrieval_results=retrieval_results,
            evaluation_results=evaluation_results,
            aggregate_metrics=aggregate_metrics,
            output_dir=output_dir
        )

        all_aggregate_metrics[k] = aggregate_metrics

    total_time = time.time() - start_time

    # =========================================================================
    # Step 5: Create Comparison Table
    # =========================================================================
    print("\n" + "="*80)
    print("RESULTS COMPARISON (Labeled Queries Only)")
    print("="*80)

    print(f"\n{'k':<8} {'Recall@k':<12} {'Precision@k':<15} {'MRR':<10} {'NDCG@k':<10} {'MAP@k':<10}")
    print("-" * 80)

    for k in args.k_values:
        metrics = all_aggregate_metrics[k]
        print(
            f"{k:<8} "
            f"{metrics.mean_recall_at_k_labeled:<12.3f} "
            f"{metrics.mean_precision_at_k_labeled:<15.3f} "
            f"{metrics.mean_reciprocal_rank_labeled:<10.3f} "
            f"{metrics.mean_ndcg_at_k_labeled:<10.3f} "
            f"{metrics.mean_average_precision_labeled:<10.3f}"
        )

    # Save comparison table
    comparison_file = output_dir / "baseline_comparison.json"
    comparison_data = {
        "experiment": "baseline_top_k",
        "n_queries": all_aggregate_metrics[args.k_values[0]].n_queries,
        "n_labeled_queries": all_aggregate_metrics[args.k_values[0]].n_labeled_queries,
        "k_values": args.k_values,
        "metrics": {
            str(k): {
                "recall_at_k": metrics.mean_recall_at_k_labeled,
                "precision_at_k": metrics.mean_precision_at_k_labeled,
                "mrr": metrics.mean_reciprocal_rank_labeled,
                "ndcg_at_k": metrics.mean_ndcg_at_k_labeled,
                "hit_rate": metrics.mean_hit_rate_labeled,
                "map_at_k": metrics.mean_average_precision_labeled,
                "query_coverage": metrics.query_coverage,
                "catalog_coverage": metrics.catalog_coverage
            }
            for k, metrics in all_aggregate_metrics.items()
        },
        "runtime_seconds": total_time
    }

    with open(comparison_file, "w") as f:
        json.dump(comparison_data, f, indent=2)

    print(f"\n✓ Saved comparison table: {comparison_file}")

    print("\n" + "="*80)
    print("✓ Baseline experiments completed!")
    print(f"✓ Total time: {total_time:.2f}s")
    print(f"✓ Results saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
