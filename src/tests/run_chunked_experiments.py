"""
Run Chunked Retrieval Experiments

Evaluate chunked retrieval performance on CRAG benchmark.
Compares different chunking strategies and aggregation methods.

Usage:
    python run_chunked_experiments.py                                   # Basic run
    python run_chunked_experiments.py --max-queries 100                # Test on 100 queries
    python run_chunked_experiments.py --db crag_chunked_vector_db      # Custom database
    python run_chunked_experiments.py --aggregation max_score          # Specific aggregation
    python run_chunked_experiments.py --compare-aggregations           # Test all methods
"""

import argparse
import sys
from pathlib import Path
import time
import json
from datetime import datetime
from typing import List, Dict, Set

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.crag_loader import CRAGLoader
from embeddings.embedding_model import EmbeddingModel
from retrieval.chunked_retrieval import ChunkedRetrieval
from evaluation.evaluator import RetrievalEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_ground_truth_ids(query) -> Set[str]:
    """
    Extract ground truth relevant document IDs from query.

    In CRAG, each query has search_results which are the relevant documents.
    Document IDs are created using enumeration index: {query_id}_doc_{idx}

    Args:
        query: CRAGQuery object

    Returns:
        Set of relevant document IDs
    """
    relevant_ids = set()

    # Document IDs are created using enumeration index
    for idx, search_result in enumerate(query.search_results):
        # Document ID format: {query_id}_doc_{index}
        doc_id = f"{query.query_id}_doc_{idx}"
        relevant_ids.add(doc_id)

    return relevant_ids


def run_experiment(
    db_path: str,
    queries,
    embedding_model: EmbeddingModel,
    k: int = 10,
    top_chunks: int = 100,
    aggregation: str = "max_score",
    max_queries: int = None
) -> Dict:
    """
    Run a single chunked retrieval experiment.
    
    Args:
        db_path: Path to chunked vector database
        queries: List of query objects
        embedding_model: Embedding model for queries
        k: Number of documents to retrieve
        top_chunks: Number of chunks to retrieve before aggregation
        aggregation: Aggregation strategy
        max_queries: Limit number of queries (for testing)
    
    Returns:
        Dictionary with results
    """
    print(f"\nExperiment: {aggregation}, k={k}, top_chunks={top_chunks}")
    print("-" * 80)
    
    # Limit queries if requested
    if max_queries:
        queries = queries[:max_queries]
        print(f"⚠ Limited to {max_queries} queries for testing")
    
    # Initialize retrieval system
    print("Loading chunked retrieval system...")
    retriever = ChunkedRetrieval.from_database_path(
        db_path=db_path,
        k=k,
        top_chunks=top_chunks,
        aggregation=aggregation,
        embedding_model=embedding_model
    )
    
    # Embed queries
    print(f"Embedding {len(queries)} queries...")
    query_texts = [q.query for q in queries]
    query_embeddings = embedding_model.embed_documents(query_texts, show_progress=True)
    
    # Run retrieval
    print(f"Running retrieval...")
    start_time = time.time()
    
    query_ids = [q.query_id for q in queries]
    retrieval_outputs = retriever.batch_retrieve(query_ids, query_embeddings)
    
    retrieval_time = time.time() - start_time
    print(f"✓ Retrieved in {retrieval_time:.2f}s ({len(queries)/retrieval_time:.1f} queries/sec)")
    
    # Evaluate
    print("Evaluating results...")
    evaluator = RetrievalEvaluator()
    
    # Build evaluation results
    evaluation_results = []
    
    for query, retrieval_output in zip(queries, retrieval_outputs):
        relevant_ids = extract_ground_truth_ids(query)
        
        eval_result = evaluator.evaluate_single(
            query_id=query.query_id,
            retrieved_ids=retrieval_output.retrieved_ids,
            retrieved_scores=retrieval_output.retrieved_scores,
            relevant_ids=relevant_ids,
            metadata={"query_text": query.query, "k": k}
        )
        evaluation_results.append(eval_result)
    
    # Aggregate metrics
    aggregate_metrics = evaluator.evaluate_batch(evaluation_results)
    
    # Print results
    print(f"\nResults:")
    print(f"  Recall@{k}: {aggregate_metrics.mean_recall_at_k_labeled:.3f}")
    print(f"  Precision@{k}: {aggregate_metrics.mean_precision_at_k_labeled:.3f}")
    print(f"  NDCG@{k}: {aggregate_metrics.mean_ndcg_at_k_labeled:.3f}")
    
    return {
        "aggregation": aggregation,
        "k": k,
        "top_chunks": top_chunks,
        "n_queries": len(queries),
        "retrieval_time": retrieval_time,
        "metrics": {
            "recall": aggregate_metrics.mean_recall_at_k_labeled,
            "precision": aggregate_metrics.mean_precision_at_k_labeled,
            "ndcg": aggregate_metrics.mean_ndcg_at_k_labeled,
            "mrr": aggregate_metrics.mean_reciprocal_rank_labeled,
            "hit_rate": aggregate_metrics.mean_hit_rate_labeled
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Run chunked retrieval experiments")
    parser.add_argument(
        "--db",
        type=str,
        default="crag_chunked_vector_db",
        help="Path to chunked vector database (default: crag_chunked_vector_db)"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["1_2"],
        choices=["1_2", "3"],
        help="Which CRAG tasks to evaluate on (default: 1_2)"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of queries to evaluate (default: all)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of documents to retrieve (default: 10)"
    )
    parser.add_argument(
        "--top-chunks",
        type=int,
        default=100,
        help="Number of chunks to retrieve before aggregation (default: 100)"
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="max_score",
        choices=["max_score", "mean_score", "sum_score"],
        help="Aggregation strategy (default: max_score)"
    )
    parser.add_argument(
        "--compare-aggregations",
        action="store_true",
        help="Compare all aggregation strategies"
    )
    parser.add_argument(
        "--compare-k",
        action="store_true",
        help="Compare different k values (5, 10, 15, 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (default: results/chunked_experiments/chunked_TIMESTAMP.json)"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("CHUNKED RETRIEVAL EXPERIMENTS")
    print("="*80)
    print(f"Database: {args.db}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Max queries: {args.max_queries if args.max_queries else 'All'}")
    if args.compare_aggregations:
        print("Mode: Compare all aggregation strategies")
    elif args.compare_k:
        print("Mode: Compare different k values")
    else:
        print(f"Mode: Single experiment (k={args.k}, aggregation={args.aggregation})")
    print("="*80 + "\n")

    # Load database first to get available document IDs
    print("Loading database to check available documents...")
    print("-" * 80)
    import pickle
    mapping_file = f"{args.db}.chunk_mapping.pkl"
    with open(mapping_file, "rb") as f:
        mappings = pickle.load(f)
    
    available_doc_ids = set(mappings['doc_to_chunks'].keys())
    print(f"✓ Database contains {len(available_doc_ids)} documents")
    print()
    
    # Load queries
    print("Loading CRAG dataset...")
    print("-" * 80)
    loader = CRAGLoader()
    queries, documents = loader.load_by_tasks(args.tasks)
    
    print(f"✓ Loaded {len(queries)} queries")
    
    # Filter queries to only those whose ground truth docs are in the database
    print("\nFiltering queries to match database coverage...")
    print("-" * 80)
    
    filtered_queries = []
    for query in queries:
        ground_truth_ids = extract_ground_truth_ids(query)
        # Keep query if at least one ground truth doc is in the database
        if ground_truth_ids & available_doc_ids:  # Set intersection
            filtered_queries.append(query)
    
    print(f"✓ Kept {len(filtered_queries)}/{len(queries)} queries with ground truth in database")
    print(f"  ({len(filtered_queries)/len(queries)*100:.1f}% coverage)")
    
    queries = filtered_queries
    
    if len(queries) == 0:
        print("\n❌ ERROR: No queries match the documents in the database!")
        print("   Database and dataset might be mismatched.")
        return
    
    print()

    # Initialize embedding model
    print("Initializing embedding model...")
    print("-" * 80)
    
    chunker_config = mappings.get("chunker_config", {})
    print(f"Database chunker config: {chunker_config}")
    
    # Use quality model (same as what should be in the database)
    model = EmbeddingModel(model_name=EmbeddingModel.QUALITY_MODEL)
    print(f"✓ Model: {model.get_model_name()}")
    print()

    # Run experiments
    results = []
    
    if args.compare_aggregations:
        print("="*80)
        print("COMPARING AGGREGATION STRATEGIES")
        print("="*80)
        
        for aggregation in ["max_score", "mean_score", "sum_score"]:
            result = run_experiment(
                db_path=args.db,
                queries=queries,
                embedding_model=model,
                k=args.k,
                top_chunks=args.top_chunks,
                aggregation=aggregation,
                max_queries=args.max_queries
            )
            results.append(result)
        
        # Print comparison
        print("\n" + "="*80)
        print("AGGREGATION COMPARISON")
        print("="*80)
        print(f"{'Strategy':<15} {'Recall@10':<12} {'Precision@10':<15} {'NDCG@10':<12}")
        print("-" * 80)
        for result in results:
            metrics = result["metrics"]
            print(f"{result['aggregation']:<15} {metrics['recall']:<12.3f} {metrics['precision']:<15.3f} {metrics['ndcg']:<12.3f}")
    
    elif args.compare_k:
        print("="*80)
        print("COMPARING K VALUES")
        print("="*80)
        
        for k in [5, 10, 15, 20]:
            result = run_experiment(
                db_path=args.db,
                queries=queries,
                embedding_model=model,
                k=k,
                top_chunks=args.top_chunks,
                aggregation=args.aggregation,
                max_queries=args.max_queries
            )
            results.append(result)
        
        # Print comparison
        print("\n" + "="*80)
        print("K VALUE COMPARISON")
        print("="*80)
        print(f"{'k':<8} {'Recall@k':<12} {'Precision@k':<15} {'NDCG@k':<12}")
        print("-" * 80)
        for result in results:
            metrics = result["metrics"]
            k = result["k"]
            print(f"{k:<8} {metrics['recall']:<12.3f} {metrics['precision']:<15.3f} {metrics['ndcg']:<12.3f}")
    
    else:
        # Single experiment
        result = run_experiment(
            db_path=args.db,
            queries=queries,
            embedding_model=model,
            k=args.k,
            top_chunks=args.top_chunks,
            aggregation=args.aggregation,
            max_queries=args.max_queries
        )
        results.append(result)

    # Save results
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent.parent / "results" / "chunked_experiments"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"chunked_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "db_path": args.db,
                "tasks": args.tasks,
                "max_queries": args.max_queries,
                "chunker_config": chunker_config
            },
            "results": results
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("✓ Experiments completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
