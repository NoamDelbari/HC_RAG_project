"""
Run HC-based Adaptive Retrieval Experiments

Tests HC-based retrieval with different gamma and min_hc values and compares
with baseline top-k retrieval using standard IR metrics.

Usage:
    python src/tests/run_hc_experiments.py                    # Run all queries
    python src/tests/run_hc_experiments.py --max-queries 100  # Test subset
"""

import argparse
import json
from pathlib import Path
from typing import List, Set
import numpy as np
from tqdm import tqdm
import time

from hc_rag.data.crag_loader import CRAGLoader
from hc_rag.embeddings.embedding_model import EmbeddingModel
from hc_rag.embeddings.vector_database import VectorDatabase
from hc_rag.retrieval.hc_retrieval import HCRetrieval
from hc_rag.evaluation.evaluator import RetrievalEvaluator
from hc_rag.hc.null_distribution import QueryNullDistributions


def extract_ground_truth_ids(query) -> Set[str]:
    """
    Extract ground truth relevant document IDs from query.

    Document IDs are created by hashing the page URL to ensure
    consistent identification across queries.

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


def run_hc_experiment(
    gamma: float,
    min_hc: float,
    allow_empty: bool,
    max_candidates: int,
    queries: List,
    db_path: str,
    embedding_model: EmbeddingModel,
    query_null_distributions: QueryNullDistributions,
    evaluator: RetrievalEvaluator,
    aggregation: str = "max_score",
    top_chunks_multiplier: int = 10,
    max_queries: int = None,
    max_k: int = None
) -> tuple:
    """
    Run HC experiment with specific parameters.

    Supports both full document and chunked databases (auto-detected).

    Args:
        gamma: HC gamma parameter
        min_hc: Minimum HC statistic threshold
        allow_empty: Whether to allow empty results
        max_candidates: Maximum candidates to fetch
        queries: List of CRAG queries
        db_path: Path to vector database
        embedding_model: Embedding model for queries
        query_null_distributions: Per-query null distributions
        evaluator: RetrievalEvaluator instance
        aggregation: Chunk aggregation strategy (for chunked DBs)
        top_chunks_multiplier: Multiplier for max_candidates when retrieving chunks
        max_queries: Optional limit on number of queries
        max_k: Maximum number of documents to return (None = unlimited, use HC threshold only)

    Returns:
        Tuple of (retrieval_results, evaluation_results, aggregate_metrics)
    """
    print(f"\nRunning HC experiment:")
    print(f"  gamma={gamma}, min_hc={min_hc}, allow_empty={allow_empty}, max_candidates={max_candidates}, max_k={max_k if max_k else 'unlimited'}")
    print("-" * 80)

    # Limit queries if requested
    if max_queries and len(queries) > max_queries:
        print(f"⚠ Limiting to {max_queries} queries for testing")
        queries = queries[:max_queries]

    # Create retriever (auto-detects if database is chunked)
    retriever = HCRetrieval.from_database_path(
        db_path=db_path,
        query_null_distributions=query_null_distributions,
        gamma=gamma,
        max_candidates=max_candidates,
        min_hc=min_hc,
        allow_empty=allow_empty,
        embedding_model=embedding_model,
        aggregation=aggregation,
        top_chunks_multiplier=top_chunks_multiplier,
        max_k=max_k
    )

    # Process queries
    print(f"Processing {len(queries)} queries...")

    retrieval_results = []
    evaluation_results = []

    # Track empty results
    n_empty = 0

    for query in tqdm(queries, desc=f"HC γ={gamma}, min_hc={min_hc}"):
        # Embed query
        query_embedding = embedding_model.embed_query(query.query)

        # Retrieve
        retrieval_output = retriever.retrieve(
            query_id=query.query_id,
            query_embedding=query_embedding
        )
        retrieval_results.append(retrieval_output)

        # Track empty results
        if retrieval_output.k == 0:
            n_empty += 1

        # Get ground truth
        relevant_ids = extract_ground_truth_ids(query)

        # Evaluate
        eval_result = evaluator.evaluate_single(
            query_id=query.query_id,
            retrieved_ids=retrieval_output.retrieved_ids,
            retrieved_scores=retrieval_output.retrieved_scores,
            relevant_ids=relevant_ids,
            metadata={
                "query_text": query.query,
                "gamma": gamma,
                "min_hc": min_hc,
                "allow_empty": allow_empty,
                "max_candidates": max_candidates,
                "max_k": max_k,
                "threshold": retrieval_output.threshold
            }
        )
        evaluation_results.append(eval_result)

    # Aggregate metrics
    aggregate_metrics = evaluator.evaluate_batch(evaluation_results)

    # Calculate average k
    avg_k = np.mean([r.k for r in retrieval_results])
    median_k = np.median([r.k for r in retrieval_results])

    print(f"\n✓ Completed HC experiment")
    print(f"  Empty results: {n_empty} / {len(queries)} ({100*n_empty/len(queries):.1f}%)")
    print(f"  Avg k: {avg_k:.2f}, Median k: {median_k:.0f}")
    print(f"  Recall@k: {aggregate_metrics.mean_recall_at_k_labeled:.3f}")
    print(f"  Precision@k: {aggregate_metrics.mean_precision_at_k_labeled:.3f}")
    print(f"  MRR: {aggregate_metrics.mean_reciprocal_rank_labeled:.3f}")
    print(f"  NDCG@k: {aggregate_metrics.mean_ndcg_at_k_labeled:.3f}")

    return retrieval_results, evaluation_results, aggregate_metrics


def save_hc_results(
    gamma: float,
    min_hc: float,
    allow_empty: bool,
    max_candidates: int,
    retrieval_results: List,
    evaluation_results: List,
    aggregate_metrics,
    output_dir: Path,
    max_k: int = None
):
    """
    Save HC experiment results to disk.

    Args:
        gamma: Gamma parameter
        min_hc: Minimum HC threshold
        allow_empty: Allow empty flag
        max_candidates: Max candidates
        retrieval_results: List of RetrievalOutput objects
        evaluation_results: List of RetrievalResult objects
        aggregate_metrics: AggregateMetrics object
        output_dir: Directory to save results
        max_k: Maximum documents to return (None = unlimited)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename
    max_k_str = f"_maxK{max_k}" if max_k else ""
    config_str = f"gamma{gamma:.2f}_minHC{min_hc:.1f}_maxCand{max_candidates}{max_k_str}"
    if not allow_empty:
        config_str += "_noEmpty"

    # Save per-query results
    per_query_results = []
    for retrieval, evaluation in zip(retrieval_results, evaluation_results):
        per_query_results.append({
            "query_id": retrieval.query_id,
            "gamma": gamma,
            "min_hc": min_hc,
            "max_candidates": max_candidates,
            "max_k": max_k,
            "allow_empty": allow_empty,
            "k": retrieval.k,
            "threshold": retrieval.threshold,
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

    per_query_file = output_dir / f"hc_{config_str}_per_query.json"
    with open(per_query_file, "w") as f:
        json.dump(per_query_results, f, indent=2)

    print(f"  Saved per-query results: {per_query_file}")

    # Save aggregate metrics
    aggregate_file = output_dir / f"hc_{config_str}_aggregate.json"

    # Calculate k statistics
    k_values = [r.k for r in retrieval_results]

    aggregate_data = {
        "method": "hc",
        "config": {
            "gamma": gamma,
            "min_hc": min_hc,
            "max_candidates": max_candidates,
            "max_k": max_k,
            "allow_empty": allow_empty
        },
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
            "mean_k": float(np.mean(k_values)),
            "median_k": float(np.median(k_values)),
            "min_k": int(np.min(k_values)),
            "max_k": int(np.max(k_values)),
            "std_k": float(np.std(k_values)),
            "n_empty": int(np.sum([k == 0 for k in k_values])),
            "empty_rate": float(np.mean([k == 0 for k in k_values]))
        },
        "quality_flags": {
            "queries_with_duplicates": getattr(aggregate_metrics, "n_queries_with_duplicates", 0)
        }
    }

    with open(aggregate_file, "w") as f:
        json.dump(aggregate_data, f, indent=2)

    print(f"  Saved aggregate metrics: {aggregate_file}")


def main():
    parser = argparse.ArgumentParser(description="Run HC-based adaptive retrieval experiments")
    parser.add_argument(
        "--gamma-values",
        nargs="+",
        type=float,
        default=[0.07, 0.1, 0.4],
        help="Gamma values to test (default: 0.07 0.1 0.4)"
    )
    parser.add_argument(
        "--min-hc-values",
        nargs="+",
        type=float,
        default=[0.0],
        help="Min HC thresholds to test (default: 0.0 for no gating)"
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=50,
        help="Maximum candidates to fetch before HC filtering (default: 50)"
    )
    parser.add_argument(
        "--allow-empty",
        dest="allow_empty",
        action="store_true",
        default=True,
        help="Allow empty results when HC < min_hc (default: True)"
    )
    parser.add_argument(
        "--no-allow-empty",
        dest="allow_empty",
        action="store_false",
        help="Disallow empty results when HC < min_hc"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
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
        default="results/hc_experiments",
        help="Output directory for results (default: results/hc_experiments)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="src/database/crag_snippet_chunked_vector_db",
        help="Path to vector database (default: src/database/crag_snippet_chunked_vector_db)"
    )
    parser.add_argument(
        "--query-null-dist-path",
        type=str,
        default=None,
        help="Path to pre-computed per-query null distributions (REQUIRED)"
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="max_score",
        choices=["max_score", "mean_score", "sum_score"],
        help="Chunk aggregation strategy for chunked DBs (default: max_score)"
    )
    parser.add_argument(
        "--top-chunks-multiplier",
        type=int,
        default=10,
        help="Multiplier for max_candidates when retrieving chunks (default: 10)"
    )
    parser.add_argument(
        "--max-k-values",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20],
        help="Maximum documents to return for each config (default: 5 10 15 20)"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    print("\n" + "="*80)
    print("HC-BASED ADAPTIVE RETRIEVAL EXPERIMENTS")
    print("="*80)
    print(f"Seed: {args.seed}")
    print(f"Per-query null distributions: {args.query_null_dist_path}")
    print(f"Gamma values: {args.gamma_values}")
    print(f"Min HC values: {args.min_hc_values}")
    print(f"Max k values: {args.max_k_values}")
    print(f"Max candidates: {args.max_candidates}")
    print(f"Allow empty: {args.allow_empty}")
    print(f"Max queries: {args.max_queries if args.max_queries else 'All'}")
    print(f"Database: {args.db_path}")
    print(f"Output: {args.output_dir}")
    print("="*80 + "\n")

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load CRAG Queries
    # =========================================================================
    print("Step 1: Loading CRAG queries...")
    print("-" * 80)

    # Use snippets (not full HTML) to match the snippet-based database
    loader = CRAGLoader(use_full_html=False)
    queries, documents = loader.load_by_tasks(["1_2"])

    print(f"✓ Loaded {len(queries)} queries")
    print(f"✓ Loaded {len(documents)} documents")
    print()

    # =========================================================================
    # Step 2: Initialize Embedding Model
    # =========================================================================
    print("Step 2: Initializing embedding model...")
    print("-" * 80)

    # Use BGE model to match the snippet-based chunked database
    embedding_model = EmbeddingModel(model_name=EmbeddingModel.BGE_MODEL)

    print(f"✓ Model: {embedding_model.get_model_name()}")
    print(f"✓ Embedding dimension: {embedding_model.get_embedding_dim()}")
    print(f"✓ Device: {embedding_model.device}")
    print()

    # =========================================================================
    # Step 3: Load Per-Query Null Distributions
    # =========================================================================
    print("Step 3: Loading per-query null distributions...")
    print("-" * 80)

    if not args.query_null_dist_path:
        print("❌ ERROR: --query-null-dist-path is REQUIRED")
        print("  Run: python src/tests/build_query_document_null.py")
        exit(1)

    if not Path(args.query_null_dist_path).exists():
        print(f"❌ ERROR: File not found: {args.query_null_dist_path}")
        exit(1)

    query_null_distributions = QueryNullDistributions.load(args.query_null_dist_path)

    # Validate it's the correct type
    if not isinstance(query_null_distributions, QueryNullDistributions):
        print(f"❌ ERROR: File contains old format (single NullDistribution)")
        print(f"  You need to regenerate using the new per-query architecture:")
        print(f"  python src/tests/build_query_document_null.py --negatives-per-query 1000")
        exit(1)

    print(f"✓ Loaded per-query null distributions")
    print(f"  Path: {args.query_null_dist_path}")
    print(f"  {query_null_distributions}")
    print()

    # =========================================================================
    # Step 4: Initialize Evaluator
    # =========================================================================
    print("Step 4: Initializing evaluator...")
    print("-" * 80)

    evaluator = RetrievalEvaluator()
    print()

    # =========================================================================
    # Step 5: Run HC Experiments
    # =========================================================================
    print("Step 5: Running HC experiments...")
    print("="*80)

    output_dir = Path(args.output_dir)
    all_results = {}

    start_time = time.time()

    for gamma in args.gamma_values:
        for min_hc in args.min_hc_values:
            for max_k in args.max_k_values:
                config_key = f"gamma_{gamma}_minHC_{min_hc}_maxK_{max_k}"

                retrieval_results, evaluation_results, aggregate_metrics = run_hc_experiment(
                    gamma=gamma,
                    min_hc=min_hc,
                    allow_empty=args.allow_empty,
                    max_candidates=args.max_candidates,
                    queries=queries,
                    db_path=args.db_path,
                    embedding_model=embedding_model,
                    query_null_distributions=query_null_distributions,
                    evaluator=evaluator,
                    aggregation=args.aggregation,
                    top_chunks_multiplier=args.top_chunks_multiplier,
                    max_queries=args.max_queries,
                    max_k=max_k
                )

                # Save results
                save_hc_results(
                    gamma=gamma,
                    min_hc=min_hc,
                    allow_empty=args.allow_empty,
                    max_candidates=args.max_candidates,
                    retrieval_results=retrieval_results,
                    evaluation_results=evaluation_results,
                    aggregate_metrics=aggregate_metrics,
                    output_dir=output_dir,
                    max_k=max_k
                )

                all_results[config_key] = {
                    "gamma": gamma,
                    "min_hc": min_hc,
                    "max_k": max_k,
                    "metrics": aggregate_metrics,
                    "retrieval_stats": {
                        "mean_k": float(np.mean([r.k for r in retrieval_results])),
                        "median_k": float(np.median([r.k for r in retrieval_results])),
                        "n_empty": int(np.sum([r.k == 0 for r in retrieval_results]))
                    }
                }

    total_time = time.time() - start_time

    # Guard against empty results
    if not all_results:
        print("\n⚠ Warning: No configurations produced results. Exiting.")
        return

    # =========================================================================
    # Step 6: Create Comparison Table
    # =========================================================================
    print("\n" + "="*80)
    print("HC EXPERIMENTS RESULTS COMPARISON (Labeled Queries Only)")
    print("="*80)

    print(f"\n{'Gamma':<8} {'Min HC':<10} {'Max k':<8} {'Avg k':<10} {'Recall@k':<12} {'Precision@k':<15} {'MRR':<10} {'NDCG@k':<10}")
    print("-" * 90)

    for config_key, result in all_results.items():
        gamma = result["gamma"]
        min_hc = result["min_hc"]
        max_k = result["max_k"]
        metrics = result["metrics"]
        avg_k = result["retrieval_stats"]["mean_k"]

        print(
            f"{gamma:<8.2f} "
            f"{min_hc:<10.1f} "
            f"{max_k:<8} "
            f"{avg_k:<10.2f} "
            f"{metrics.mean_recall_at_k_labeled:<12.3f} "
            f"{metrics.mean_precision_at_k_labeled:<15.3f} "
            f"{metrics.mean_reciprocal_rank_labeled:<10.3f} "
            f"{metrics.mean_ndcg_at_k_labeled:<10.3f}"
        )

    # Save comparison summary
    summary_file = output_dir / "hc_comparison_summary.json"
    summary_data = {
        "experiment": "hc_adaptive",
        "n_queries": all_results[list(all_results.keys())[0]]["metrics"].n_queries,
        "n_labeled_queries": all_results[list(all_results.keys())[0]]["metrics"].n_labeled_queries,
        "configurations": [
            {
                "gamma": result["gamma"],
                "min_hc": result["min_hc"],
                "max_k": result["max_k"],
                "max_candidates": args.max_candidates,
                "allow_empty": args.allow_empty,
                "retrieval_stats": result["retrieval_stats"],
                "metrics": {
                    "recall_at_k": result["metrics"].mean_recall_at_k_labeled,
                    "precision_at_k": result["metrics"].mean_precision_at_k_labeled,
                    "mrr": result["metrics"].mean_reciprocal_rank_labeled,
                    "ndcg_at_k": result["metrics"].mean_ndcg_at_k_labeled,
                    "hit_rate": result["metrics"].mean_hit_rate_labeled,
                    "map_at_k": result["metrics"].mean_average_precision_labeled,
                    "query_coverage": result["metrics"].query_coverage,
                    "catalog_coverage": result["metrics"].catalog_coverage
                }
            }
            for result in all_results.values()
        ],
        "runtime_seconds": total_time
    }

    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n✓ Saved comparison summary: {summary_file}")

    print("\n" + "="*80)
    print("✓ HC experiments completed!")
    print(f"✓ Total time: {total_time:.2f}s")
    print(f"✓ Results saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
