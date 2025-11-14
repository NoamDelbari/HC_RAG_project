"""
Test and Compare Chunking Strategies

Compares different chunking approaches on retrieval quality.

Usage:
    python src/tests/test_chunking.py --max-queries 20 --strategies recursive fixed
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.crag_loader import CRAGLoader, CRAGDocument
from embeddings.embedding_model import EmbeddingModel
from embeddings.chunking import get_chunker, Chunk
from embeddings.vector_database import VectorDatabase
from retrieval.baseline_retrieval import BaselineRetrieval
from evaluation.evaluator import RetrievalEvaluator
from tests.run_baseline_experiments import extract_ground_truth_ids


def chunk_documents(documents: List, chunker, show_progress: bool = True) -> tuple:
    """
    Chunk all documents.
    
    Returns:
        (chunks, chunk_to_doc_mapping)
    """
    all_chunks = []
    chunk_to_doc = {}  # Maps chunk_id to parent doc_id
    
    iterator = tqdm(documents, desc="Chunking") if show_progress else documents
    
    for doc in iterator:
        metadata = {
            "title": doc.title,
            "url": doc.url,
            "original_length": len(doc.text)
        }
        
        doc_chunks = chunker.chunk_document(doc.doc_id, doc.text, metadata)
        all_chunks.extend(doc_chunks)
        
        for chunk in doc_chunks:
            chunk_to_doc[chunk.chunk_id] = doc.doc_id
    
    return all_chunks, chunk_to_doc


def map_chunk_results_to_docs(retrieved_chunk_ids: List[str], chunk_to_doc: dict) -> tuple:
    """
    Map retrieved chunks back to parent documents.
    
    Returns:
        (unique_doc_ids, doc_scores) - best score per document
    """
    doc_scores = {}  # doc_id -> best score
    
    for idx, chunk_id in enumerate(retrieved_chunk_ids):
        parent_doc_id = chunk_to_doc.get(chunk_id, chunk_id)
        
        # Keep best (highest) score for each document
        # Score decreases with rank, so first occurrence is best
        if parent_doc_id not in doc_scores:
            doc_scores[parent_doc_id] = 1.0 / (idx + 1)  # Simple rank-based score
    
    # Sort by score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    doc_ids = [doc_id for doc_id, _ in sorted_docs]
    scores = [score for _, score in sorted_docs]
    
    return doc_ids, scores


def evaluate_chunking_strategy(
    strategy_name: str,
    chunker,
    documents: List,
    queries: List,
    embedding_model: EmbeddingModel,
    k: int = 10,
    max_queries: int = None
) -> dict:
    """Evaluate a specific chunking strategy."""
    
    print(f"\n{'='*80}")
    print(f"EVALUATING STRATEGY: {strategy_name.upper()}")
    print(f"{'='*80}")
    
    if max_queries:
        queries = queries[:max_queries]
    
    # Step 1: Chunk documents
    print(f"\nStep 1: Chunking {len(documents)} documents...")
    start_time = time.time()
    
    chunks, chunk_to_doc = chunk_documents(documents, chunker, show_progress=True)
    
    chunk_time = time.time() - start_time
    
    print(f"✓ Created {len(chunks)} chunks from {len(documents)} documents")
    print(f"✓ Average chunks per doc: {len(chunks)/len(documents):.1f}")
    print(f"✓ Chunking time: {chunk_time:.2f}s")
    
    # Step 2: Embed chunks
    print(f"\nStep 2: Embedding {len(chunks)} chunks...")
    chunk_texts = [chunk.text for chunk in chunks]
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    
    start_time = time.time()
    chunk_embeddings = embedding_model.embed_documents(chunk_texts, show_progress=True)
    embed_time = time.time() - start_time
    
    print(f"✓ Embedded {len(chunk_embeddings)} chunks in {embed_time:.2f}s")
    
    # Step 3: Build vector database
    print(f"\nStep 3: Building vector database...")
    metadata = [chunk.metadata for chunk in chunks]
    
    vector_db = VectorDatabase(embedding_dim=embedding_model.get_embedding_dim())
    vector_db.add_documents(chunk_ids, chunk_embeddings, metadata)
    
    print(f"✓ Built database with {vector_db.get_num_documents()} chunks")
    
    # Step 4: Evaluate retrieval
    print(f"\nStep 4: Evaluating retrieval on {len(queries)} queries...")
    
    evaluator = RetrievalEvaluator()
    retriever = BaselineRetrieval(vector_db=vector_db, k=k*3)  # Retrieve more chunks
    
    evaluation_results = []
    query_times = []
    
    for query in tqdm(queries, desc="Querying"):
        start_time = time.time()
        
        # Embed query
        query_embedding = embedding_model.embed_query(query.query)
        
        # Retrieve chunks
        retrieval_output = retriever.retrieve(
            query_id=query.query_id,
            query_embedding=query_embedding
        )
        
        query_time = time.time() - start_time
        query_times.append(query_time)
        
        # Map chunks back to documents
        doc_ids, doc_scores = map_chunk_results_to_docs(
            retrieval_output.retrieved_ids, 
            chunk_to_doc
        )
        
        # Take top k documents
        doc_ids = doc_ids[:k]
        doc_scores = doc_scores[:k]
        
        # Get ground truth
        relevant_ids = extract_ground_truth_ids(query)
        
        # Evaluate
        eval_result = evaluator.evaluate_single(
            query_id=query.query_id,
            retrieved_ids=doc_ids,
            retrieved_scores=doc_scores,
            relevant_ids=relevant_ids,
            metadata={"query_text": query.query, "k": k}
        )
        evaluation_results.append(eval_result)
    
    # Aggregate metrics
    aggregate_metrics = evaluator.evaluate_batch(evaluation_results)
    
    avg_query_time = np.mean(query_times)
    
    results = {
        "strategy": strategy_name,
        "n_documents": len(documents),
        "n_chunks": len(chunks),
        "avg_chunks_per_doc": len(chunks) / len(documents),
        "chunk_time_seconds": chunk_time,
        "embed_time_seconds": embed_time,
        "total_indexing_time": chunk_time + embed_time,
        "avg_query_time_ms": avg_query_time * 1000,
        "metrics": {
            "recall_at_k": aggregate_metrics.mean_recall_at_k_labeled,
            "precision_at_k": aggregate_metrics.mean_precision_at_k_labeled,
            "mrr": aggregate_metrics.mean_reciprocal_rank_labeled,
            "ndcg_at_k": aggregate_metrics.mean_ndcg_at_k_labeled,
            "map_at_k": aggregate_metrics.mean_average_precision_labeled,
            "hit_rate": aggregate_metrics.mean_hit_rate_labeled,
        }
    }
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"RESULTS: {strategy_name.upper()}")
    print(f"{'='*80}")
    print(f"Chunks: {len(chunks)} ({len(chunks)/len(documents):.1f} per doc)")
    print(f"Indexing time: {chunk_time + embed_time:.1f}s")
    print(f"Query time: {avg_query_time*1000:.1f}ms")
    print(f"\nMetrics:")
    print(f"  Recall@{k}: {results['metrics']['recall_at_k']:.3f}")
    print(f"  Precision@{k}: {results['metrics']['precision_at_k']:.3f}")
    print(f"  NDCG@{k}: {results['metrics']['ndcg_at_k']:.3f}")
    print(f"  MRR: {results['metrics']['mrr']:.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test chunking strategies")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["fixed", "sentence", "recursive", "none"],
        default=["recursive", "none"],
        help="Chunking strategies to test"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size in tokens (default: 256)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in tokens (default: 50)"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=20,
        help="Max queries to test (default: 20)"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Max documents to use (default: all)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of documents to retrieve (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/chunking_comparison",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CHUNKING STRATEGY COMPARISON")
    print("="*80)
    print(f"Strategies: {', '.join(args.strategies)}")
    print(f"Chunk size: {args.chunk_size} tokens")
    print(f"Chunk overlap: {args.chunk_overlap} tokens")
    print(f"Max queries: {args.max_queries}")
    print(f"k: {args.k}")
    print("="*80)
    
    # Load data
    print("\nLoading CRAG dataset...")
    loader = CRAGLoader(use_full_html=True)
    queries, documents = loader.load_by_tasks(["1_2"])
    
    if args.max_docs:
        documents = documents[:args.max_docs]
    
    print(f"✓ Loaded {len(queries)} queries")
    print(f"✓ Loaded {len(documents)} documents")
    
    # Initialize model and tokenizer
    print("\nInitializing embedding model...")
    embedding_model = EmbeddingModel(model_name=EmbeddingModel.QUALITY_MODEL)
    
    print(f"✓ Model: {embedding_model.get_model_name()}")
    print(f"✓ Embedding dim: {embedding_model.get_embedding_dim()}")
    
    # Load tokenizer for chunking
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(embedding_model.get_model_name())
    print(f"✓ Tokenizer max length: {tokenizer.model_max_length}")
    
    # Evaluate each strategy
    all_results = []
    
    for strategy_name in args.strategies:
        if strategy_name == "none":
            # Baseline without chunking
            print(f"\n{'='*80}")
            print("BASELINE: NO CHUNKING")
            print(f"{'='*80}")
            
            # Build database directly
            print(f"\nEmbedding {len(documents)} documents...")
            doc_texts = [doc.text for doc in documents]
            doc_ids = [doc.doc_id for doc in documents]
            
            start_time = time.time()
            doc_embeddings = embedding_model.embed_documents(doc_texts, show_progress=True)
            embed_time = time.time() - start_time
            
            metadata = [{"title": doc.title, "url": doc.url} for doc in documents]
            vector_db = VectorDatabase(embedding_dim=embedding_model.get_embedding_dim())
            vector_db.add_documents(doc_ids, doc_embeddings, metadata)
            
            # Evaluate
            queries_subset = queries[:args.max_queries]
            evaluator = RetrievalEvaluator()
            retriever = BaselineRetrieval(vector_db=vector_db, k=args.k)
            
            evaluation_results = []
            query_times = []
            
            for query in tqdm(queries_subset, desc="Querying"):
                start_time = time.time()
                query_embedding = embedding_model.embed_query(query.query)
                retrieval_output = retriever.retrieve(query.query_id, query_embedding)
                query_time = time.time() - start_time
                query_times.append(query_time)
                
                relevant_ids = extract_ground_truth_ids(query)
                eval_result = evaluator.evaluate_single(
                    query.query_id,
                    retrieval_output.retrieved_ids,
                    retrieval_output.retrieved_scores,
                    relevant_ids,
                    {"query_text": query.query, "k": args.k}
                )
                evaluation_results.append(eval_result)
            
            aggregate_metrics = evaluator.evaluate_batch(evaluation_results)
            
            results = {
                "strategy": "none",
                "n_documents": len(documents),
                "n_chunks": len(documents),
                "avg_chunks_per_doc": 1.0,
                "chunk_time_seconds": 0,
                "embed_time_seconds": embed_time,
                "total_indexing_time": embed_time,
                "avg_query_time_ms": np.mean(query_times) * 1000,
                "metrics": {
                    "recall_at_k": aggregate_metrics.mean_recall_at_k_labeled,
                    "precision_at_k": aggregate_metrics.mean_precision_at_k_labeled,
                    "mrr": aggregate_metrics.mean_reciprocal_rank_labeled,
                    "ndcg_at_k": aggregate_metrics.mean_ndcg_at_k_labeled,
                    "map_at_k": aggregate_metrics.mean_average_precision_labeled,
                    "hit_rate": aggregate_metrics.mean_hit_rate_labeled,
                }
            }
            
            print(f"\nBaseline Results:")
            print(f"  Recall@{args.k}: {results['metrics']['recall_at_k']:.3f}")
            print(f"  NDCG@{args.k}: {results['metrics']['ndcg_at_k']:.3f}")
            
            all_results.append(results)
        else:
            # Test chunking strategy
            chunker = get_chunker(
                strategy=strategy_name,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                tokenizer=tokenizer
            )
            
            results = evaluate_chunking_strategy(
                strategy_name,
                chunker,
                documents,
                queries,
                embedding_model,
                k=args.k,
                max_queries=args.max_queries
            )
            
            all_results.append(results)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"chunking_comparison_{timestamp}.json"
    
    comparison_data = {
        "experiment": "chunking_strategy_comparison",
        "timestamp": timestamp,
        "config": {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "k": args.k,
            "n_queries": args.max_queries,
            "n_documents": len(documents)
        },
        "results": all_results
    }
    
    with open(output_file, "w") as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Strategy':<15} {'Chunks':<8} {'Recall@{args.k}':<12} {'NDCG@{args.k}':<12} {'Query Time':<12}")
    print("-" * 80)
    
    for result in all_results:
        print(
            f"{result['strategy']:<15} "
            f"{result['n_chunks']:<8} "
            f"{result['metrics']['recall_at_k']:<12.3f} "
            f"{result['metrics']['ndcg_at_k']:<12.3f} "
            f"{result['avg_query_time_ms']:<12.1f}ms"
        )
    
    # Find best strategy
    best_result = max(all_results, key=lambda x: x['metrics']['recall_at_k'])
    
    print(f"\n{'='*80}")
    print(f"🏆 BEST STRATEGY: {best_result['strategy'].upper()}")
    print(f"{'='*80}")
    print(f"Recall@{args.k}: {best_result['metrics']['recall_at_k']:.3f}")
    print(f"NDCG@{args.k}: {best_result['metrics']['ndcg_at_k']:.3f}")
    
    if best_result['strategy'] != 'none':
        baseline = next((r for r in all_results if r['strategy'] == 'none'), None)
        if baseline:
            recall_improvement = (best_result['metrics']['recall_at_k'] - baseline['metrics']['recall_at_k']) / baseline['metrics']['recall_at_k'] * 100
            print(f"\n📈 Improvement over baseline: {recall_improvement:+.1f}%")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
