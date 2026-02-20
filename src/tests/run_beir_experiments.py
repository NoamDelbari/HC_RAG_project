"""
Run HC vs Baseline Experiments on BEIR Datasets

Compares HC adaptive retrieval against fixed top-k baseline on BEIR datasets.

Usage:
    # Quick test on FiQA
    python src/tests/run_beir_experiments.py --dataset fiqa --max-queries 100
    
    # Full experiment
    python src/tests/run_beir_experiments.py --dataset fiqa
    
    # Compare different k values for baseline
    python src/tests/run_beir_experiments.py --dataset fiqa --baseline-k 5 10 15 20
"""

import argparse
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.embedding_model import EmbeddingModel, create_embedding_model
from embeddings.vector_database import VectorDatabase
from hc.higher_criticism import HigherCriticism
from hc.null_distribution import NullDistribution, QueryNullDistributions


@dataclass
class ExperimentResult:
    """Results from a single retrieval experiment."""
    method: str
    config: Dict
    n_queries: int
    metrics: Dict[str, float]
    per_query_k: List[int]  # k used per query
    per_query_recall: List[float]
    runtime_seconds: float


def compute_metrics(
    retrieved_doc_ids: List[str],
    relevant_doc_ids: Set[str],
    k: int
) -> Dict[str, float]:
    """
    Compute IR metrics for a single query.
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs (ranked)
        relevant_doc_ids: Set of ground truth relevant document IDs
        k: Number of retrieved documents
        
    Returns:
        Dictionary of metrics including adaptive retrieval metrics
    """
    n_gt = len(relevant_doc_ids)  # Ground truth count
    
    if not relevant_doc_ids:
        return {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "hit_rate": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "k_ratio": 0.0,  # k / n_gt
            "efficiency": 0.0,  # recall / k
        }
    
    # Hit rate: did we retrieve any relevant doc?
    hits = [1 if doc_id in relevant_doc_ids else 0 for doc_id in retrieved_doc_ids]
    hit_rate = 1.0 if sum(hits) > 0 else 0.0
    
    # Precision@k and Recall@k
    n_relevant_retrieved = sum(hits)
    precision = n_relevant_retrieved / k if k > 0 else 0.0
    recall = n_relevant_retrieved / n_gt
    
    # F1 Score: harmonic mean of precision and recall
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    # MRR: reciprocal rank of first relevant document
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, 1):
        if doc_id in relevant_doc_ids:
            mrr = 1.0 / rank
            break
    
    # NDCG@k
    dcg = sum(hits[i] / np.log2(i + 2) for i in range(len(hits)))
    ideal_hits = sorted(hits, reverse=True)
    idcg = sum(ideal_hits[i] / np.log2(i + 2) for i in range(len(ideal_hits)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    # Adaptive retrieval metrics
    # k_ratio: how many docs retrieved vs GT (k/n_gt), ideal=1.0
    k_ratio = k / n_gt if n_gt > 0 else 0.0
    
    # Efficiency: recall per document retrieved (higher = better)
    efficiency = recall / k if k > 0 else 0.0
    
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "ndcg": ndcg,
        "k_ratio": k_ratio,
        "efficiency": efficiency,
    }


def aggregate_metrics(per_query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-query metrics into mean metrics."""
    if not per_query_metrics:
        return {}
    
    aggregated = {}
    for key in per_query_metrics[0].keys():
        values = [m[key] for m in per_query_metrics]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))
    
    return aggregated


def run_baseline_experiment(
    db: VectorDatabase,
    queries: List[Dict],
    query_embeddings: np.ndarray,
    k: int,
    chunk_to_doc: Dict[str, str] = None
) -> ExperimentResult:
    """
    Run baseline top-k retrieval experiment.
    
    Args:
        db: Vector database
        queries: List of query dictionaries with 'relevant_docs'
        query_embeddings: Pre-computed query embeddings
        k: Number of documents to retrieve
        chunk_to_doc: Optional chunk-to-document mapping for chunked DBs
    
    Returns:
        ExperimentResult with metrics
    """
    print(f"\nRunning Baseline (k={k})...")
    
    start_time = datetime.now()
    per_query_metrics = []
    per_query_recall = []
    per_query_k = []
    
    for i, query in enumerate(tqdm(queries, desc=f"Baseline k={k}")):
        query_emb = query_embeddings[i:i+1]
        
        # Search
        results = db.search(query_emb[0], k=k)
        
        # Get retrieved doc IDs
        if chunk_to_doc:
            # Map chunks to parent documents
            retrieved_ids = list(dict.fromkeys(
                chunk_to_doc.get(r.doc_id, r.doc_id) for r in results
            ))[:k]
        else:
            retrieved_ids = [r.doc_id for r in results]
        
        # Compute metrics
        relevant_ids = set(query['relevant_docs'])
        metrics = compute_metrics(retrieved_ids, relevant_ids, len(retrieved_ids))
        
        per_query_metrics.append(metrics)
        per_query_recall.append(metrics['recall'])
        per_query_k.append(k)
    
    runtime = (datetime.now() - start_time).total_seconds()
    
    return ExperimentResult(
        method="baseline",
        config={"k": k},
        n_queries=len(queries),
        metrics=aggregate_metrics(per_query_metrics),
        per_query_k=per_query_k,
        per_query_recall=per_query_recall,
        runtime_seconds=runtime
    )


def run_hc_experiment(
    db: VectorDatabase,
    queries: List[Dict],
    query_embeddings: np.ndarray,
    per_query_nulls: QueryNullDistributions,
    gamma: float = 0.1,
    min_hc: float = 0.0,
    max_candidates: int = 100,
    max_k: int = None,
    chunk_to_doc: Dict[str, str] = None
) -> ExperimentResult:
    """
    Run HC adaptive retrieval experiment with Z-SCORE NORMALIZATION.
    
    Uses z-score normalization within each query's top-K, then computes p-values
    against a shared z-score null distribution. This preserves ranking signal
    while removing query-specific shift/scale.
    
    Note: Position-aware nulls calibrate AWAY the ranking signal and hurt performance.
    The shared z-score null correctly assigns smaller p-values to higher-ranked docs.
    
    Args:
        db: Vector database
        queries: List of query dictionaries
        query_embeddings: Pre-computed query embeddings
        per_query_nulls: Null distributions (uses shared z-score null)
        gamma: HC gamma parameter (fraction of scores to search)
        min_hc: Minimum HC statistic to accept results
        max_candidates: Maximum candidates to fetch
        max_k: Maximum k to return (caps HC)
        chunk_to_doc: Optional chunk-to-document mapping
    
    Returns:
        ExperimentResult with metrics
    """
    config_str = f"γ={gamma}, min_hc={min_hc}, max_k={max_k or 'None'}"
    print(f"\nRunning HC ({config_str})...")
    
    # Create HC instance
    hc = HigherCriticism()
    
    start_time = datetime.now()
    per_query_metrics = []
    per_query_recall = []
    per_query_k = []
    
    for i, query in enumerate(tqdm(queries, desc=f"HC {config_str}")):
        query_emb = query_embeddings[i]
        query_id = query['query_id']
        
        # Get candidates
        results = db.search(query_emb, k=max_candidates)
        
        if not results:
            per_query_metrics.append({
                "recall": 0.0, "precision": 0.0, "hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0
            })
            per_query_recall.append(0.0)
            per_query_k.append(0)
            continue
        
        # Get similarity scores
        similarities = np.array([r.similarity for r in results])
        
        # CRITICAL: ROBUST Z-normalize similarities within this query's top-K
        median_sim = np.median(similarities)
        q25, q75 = np.percentile(similarities, [25, 75])
        iqr = q75 - q25
        scale = iqr * 0.7413 if iqr > 1e-6 else 1.0
        if scale > 1e-6:
            z_scores = (similarities - median_sim) / scale
        else:
            z_scores = np.zeros_like(similarities)
        
        # Use shared z-score null distribution (NOT position-aware)
        # This preserves ranking signal: higher positions get smaller p-values
        query_null = per_query_nulls.get(query_id)
        hc.set_null_distribution(query_null)
        
        # Compute HC threshold using z-scores
        threshold_result = hc.compute_hc_threshold(z_scores, gamma=gamma, min_hc=min_hc, allow_empty=True)
        threshold_k = threshold_result.k
        
        # Apply max_k cap if specified
        if max_k is not None:
            threshold_k = min(threshold_k, max_k)
        
        # Get top-k results based on HC threshold
        selected_results = results[:threshold_k]
        
        # Map to parent docs if chunked
        if chunk_to_doc:
            retrieved_ids = list(dict.fromkeys(
                chunk_to_doc.get(r.doc_id, r.doc_id) for r in selected_results
            ))
        else:
            retrieved_ids = [r.doc_id for r in selected_results]
        
        # Compute metrics
        relevant_ids = set(query['relevant_docs'])
        metrics = compute_metrics(retrieved_ids, relevant_ids, max(1, len(retrieved_ids)))
        
        per_query_metrics.append(metrics)
        per_query_recall.append(metrics['recall'])
        per_query_k.append(threshold_k)
    
    runtime = (datetime.now() - start_time).total_seconds()
    
    # Add k statistics to metrics
    agg_metrics = aggregate_metrics(per_query_metrics)
    agg_metrics.update({
        "k_mean": float(np.mean(per_query_k)),
        "k_std": float(np.std(per_query_k)),
        "k_min": int(np.min(per_query_k)),
        "k_max": int(np.max(per_query_k)),
        "k_zero_pct": float(np.mean([1 if k == 0 else 0 for k in per_query_k])) * 100
    })
    
    return ExperimentResult(
        method="hc",
        config={"gamma": gamma, "min_hc": min_hc, "max_k": max_k, "max_candidates": max_candidates},
        n_queries=len(queries),
        metrics=agg_metrics,
        per_query_k=per_query_k,
        per_query_recall=per_query_recall,
        runtime_seconds=runtime
    )


def load_beir_database(db_path: str) -> Tuple[VectorDatabase, List[Dict], np.ndarray, QueryNullDistributions, Dict]:
    """
    Load BEIR vector database and associated query data.
    
    Returns:
        Tuple of (db, queries, query_embeddings, per_query_nulls, chunk_to_doc)
    """
    db_path = Path(db_path)
    
    # Load database
    db = VectorDatabase.load(str(db_path))
    print(f"Loaded database: {db.get_num_documents()} documents")
    
    # Load query data
    query_data_path = db_path.with_suffix(".query_data.pkl")
    with open(query_data_path, "rb") as f:
        query_data = pickle.load(f)
    
    queries = query_data["queries"]
    query_embeddings = query_data["query_embeddings"]
    
    print(f"Loaded {len(queries)} queries")
    
    # Load per-query null distributions (REQUIRED for valid HC)
    per_query_nulls_path = db_path.with_suffix(".per_query_nulls.pkl")
    if not per_query_nulls_path.exists():
        raise FileNotFoundError(
            f"Per-query null distributions not found at {per_query_nulls_path}. "
            f"Run build_per_query_nulls.py first to generate them. "
            f"Global null distributions produce invalid p-values (fail KS uniformity test)."
        )
    
    per_query_nulls = QueryNullDistributions.load(str(per_query_nulls_path))
    print(f"Loaded per-query null distributions: {len(per_query_nulls.distributions)} queries")
    
    # Load chunk mapping if exists
    chunk_mapping_path = db_path.with_suffix(".chunk_mapping.pkl")
    if chunk_mapping_path.exists():
        with open(chunk_mapping_path, "rb") as f:
            chunk_mapping = pickle.load(f)
        chunk_to_doc = chunk_mapping.get("chunk_to_doc", None)
        print(f"Loaded chunk mapping: {len(chunk_to_doc)} chunks")
    else:
        chunk_to_doc = None
    
    return db, queries, query_embeddings, per_query_nulls, chunk_to_doc


def main():
    parser = argparse.ArgumentParser(
        description="Run HC vs Baseline experiments on BEIR datasets"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="fiqa",
        help="BEIR dataset name (default: fiqa)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to vector database (default: datasets/beir/{dataset}_vector_db)"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit number of queries (for testing)"
    )
    
    # Baseline parameters
    parser.add_argument(
        "--baseline-k",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20],
        help="k values for baseline (default: 5 10 15 20)"
    )
    
    # HC parameters
    parser.add_argument(
        "--gamma-values",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help="HC gamma values to test (default: 0.1 0.2 0.3 0.4 0.5)"
    )
    parser.add_argument(
        "--min-hc-values",
        nargs="+",
        type=float,
        default=[0.0],
        help="HC min_hc values to test (default: 0.0)"
    )
    parser.add_argument(
        "--max-k-values",
        nargs="+",
        type=int,
        default=[0],
        help="HC max_k caps to test (use 0 for None)"
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=100,
        help="Maximum candidates to fetch for HC (default: 100)"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/beir_experiments",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Set default db path
    if args.db_path is None:
        args.db_path = f"datasets/beir/{args.dataset}_vector_db"
    
    # Convert max_k values (0 means None)
    max_k_values = [None if k == 0 else k for k in args.max_k_values]
    
    print("="*80)
    print("BEIR HC vs BASELINE EXPERIMENT")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Database: {args.db_path}")
    print(f"Baseline k values: {args.baseline_k}")
    print(f"HC gamma values: {args.gamma_values}")
    print(f"HC min_hc values: {args.min_hc_values}")
    print(f"HC max_k values: {max_k_values}")
    print("="*80)
    
    # Load database
    print("\nLoading database...")
    db, queries, query_embeddings, per_query_nulls, chunk_to_doc = load_beir_database(args.db_path)
    
    # Limit queries if requested
    if args.max_queries and len(queries) > args.max_queries:
        print(f"\nLimiting to {args.max_queries} queries")
        queries = queries[:args.max_queries]
        query_embeddings = query_embeddings[:args.max_queries]
    
    # Show relevance distribution
    rel_counts = [len(q['relevant_docs']) for q in queries]
    print(f"\nRelevance per query: mean={np.mean(rel_counts):.2f}, "
          f"std={np.std(rel_counts):.2f}, range={min(rel_counts)}-{max(rel_counts)}")
    
    results = []
    
    # Run baseline experiments
    print("\n" + "="*80)
    print("BASELINE EXPERIMENTS")
    print("="*80)
    
    for k in args.baseline_k:
        result = run_baseline_experiment(
            db=db,
            queries=queries,
            query_embeddings=query_embeddings,
            k=k,
            chunk_to_doc=chunk_to_doc
        )
        results.append(result)
        print(f"  Recall@{k}: {result.metrics['recall_mean']:.4f}")
        print(f"  Precision@{k}: {result.metrics['precision_mean']:.4f}")
        print(f"  Hit Rate@{k}: {result.metrics['hit_rate_mean']:.4f}")
    
    # Run HC experiments
    print("\n" + "="*80)
    print("HC EXPERIMENTS")
    print("="*80)
    
    for gamma in args.gamma_values:
        for min_hc in args.min_hc_values:
            for max_k in max_k_values:
                result = run_hc_experiment(
                    db=db,
                    queries=queries,
                    query_embeddings=query_embeddings,
                    per_query_nulls=per_query_nulls,
                    gamma=gamma,
                    min_hc=min_hc,
                    max_candidates=args.max_candidates,
                    max_k=max_k,
                    chunk_to_doc=chunk_to_doc
                )
                results.append(result)
                print(f"  Recall: {result.metrics['recall_mean']:.4f}, "
                      f"k_mean: {result.metrics['k_mean']:.1f}, "
                      f"k_range: {result.metrics['k_min']}-{result.metrics['k_max']}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - ALL METHODS")
    print("="*80)
    print(f"Ground truth avg relevant docs: {np.mean(rel_counts):.2f}")
    print()
    print(f"{'Method':<35} {'Avg K':>7} | {'Recall':>7} {'Prec':>7} {'F1':>7} {'Effic':>7}")
    print("-"*80)
    
    for r in results:
        if r.method == "baseline":
            name = f"Baseline (k={r.config['k']})"
            avg_k = r.config['k']
        else:
            max_k_str = f", max_k={r.config['max_k']}" if r.config['max_k'] else ""
            name = f"HC (γ={r.config['gamma']}, min_hc={r.config['min_hc']}{max_k_str})"
            avg_k = r.metrics['k_mean']
        
        print(f"{name:<35} {avg_k:>7.1f} | {r.metrics['recall_mean']:>7.3f} "
              f"{r.metrics['precision_mean']:>7.3f} {r.metrics['f1_mean']:>7.3f} "
              f"{r.metrics['efficiency_mean']:>7.3f}")
    
    # Fair comparison: Find baseline with similar avg k to best HC
    hc_results = [r for r in results if r.method == "hc"]
    baseline_results = [r for r in results if r.method == "baseline"]
    
    if hc_results and baseline_results:
        # Best HC by F1 (balances precision and recall)
        best_hc_f1 = max(hc_results, key=lambda r: r.metrics['f1_mean'])
        hc_avg_k = best_hc_f1.metrics['k_mean']
        
        # Find closest baseline by k
        closest_baseline = min(baseline_results, key=lambda r: abs(r.config['k'] - hc_avg_k))
        
        print("\n" + "="*80)
        print("FAIR COMPARISON (matched k)")
        print("="*80)
        
        hc_k = best_hc_f1.metrics['k_mean']
        bl_k = closest_baseline.config['k']
        
        print(f"HC avg k: {hc_k:.1f} | Baseline k: {bl_k}")
        print()
        
        # Table header
        print(f"{'Metric':<20} {'HC (best F1)':<15} {'Baseline':<15} {'Δ':>10}")
        print("-"*60)
        
        metrics_to_compare = [
            ('Recall', best_hc_f1.metrics['recall_mean'], closest_baseline.metrics['recall_mean']),
            ('Precision', best_hc_f1.metrics['precision_mean'], closest_baseline.metrics['precision_mean']),
            ('F1 Score', best_hc_f1.metrics['f1_mean'], closest_baseline.metrics['f1_mean']),
            ('Efficiency', best_hc_f1.metrics['efficiency_mean'], closest_baseline.metrics['efficiency_mean']),
            ('Hit Rate', best_hc_f1.metrics['hit_rate_mean'], closest_baseline.metrics['hit_rate_mean']),
        ]
        
        for metric_name, hc_val, bl_val in metrics_to_compare:
            if bl_val > 0:
                delta = (hc_val - bl_val) / bl_val * 100
                delta_str = f"{delta:+.1f}%"
            else:
                delta_str = "N/A"
            print(f"{metric_name:<20} {hc_val:<15.4f} {bl_val:<15.4f} {delta_str:>10}")
        
        print()
        print(f"Best HC config: γ={best_hc_f1.config['gamma']}, min_hc={best_hc_f1.config['min_hc']}")
        print(f"Closest Baseline: k={closest_baseline.config['k']}")
        
        # Key insight
        print("\n" + "-"*60)
        print("KEY INSIGHTS:")
        f1_improvement = (best_hc_f1.metrics['f1_mean'] - closest_baseline.metrics['f1_mean']) / closest_baseline.metrics['f1_mean'] * 100
        eff_improvement = (best_hc_f1.metrics['efficiency_mean'] - closest_baseline.metrics['efficiency_mean']) / closest_baseline.metrics['efficiency_mean'] * 100
        
        print(f"  • F1 Score:   HC is {f1_improvement:+.1f}% vs Baseline at similar k")
        print(f"  • Efficiency: HC retrieves {eff_improvement:+.1f}% more relevant docs per doc retrieved")
        
        if best_hc_f1.metrics['f1_mean'] > closest_baseline.metrics['f1_mean']:
            print(f"\n  ✓ HC outperforms baseline on F1 with adaptive k selection!")
        else:
            print(f"\n  ⚠ Baseline has better F1 - HC may be too conservative")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{args.dataset}_{timestamp}.json"
    
    output_data = {
        "dataset": args.dataset,
        "n_queries": len(queries),
        "relevance_stats": {
            "mean": float(np.mean(rel_counts)),
            "std": float(np.std(rel_counts)),
            "min": int(min(rel_counts)),
            "max": int(max(rel_counts))
        },
        "null_distribution": "per_query (from .per_query_nulls.pkl)",
        "results": [asdict(r) for r in results],
        "timestamp": timestamp
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
