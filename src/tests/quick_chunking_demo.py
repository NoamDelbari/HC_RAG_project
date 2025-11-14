"""
Quick Chunking Demo

Demonstrates chunking improvement on a tiny subset (fast on CPU).

Usage:
    python src/tests/quick_chunking_demo.py
"""

import sys
from pathlib import Path
import time
import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.crag_loader import CRAGLoader
from embeddings.embedding_model import EmbeddingModel
from embeddings.chunking import get_chunker
from embeddings.vector_database import VectorDatabase
from retrieval.baseline_retrieval import BaselineRetrieval
from evaluation.evaluator import RetrievalEvaluator
from tests.run_baseline_experiments import extract_ground_truth_ids

print("\n" + "="*80)
print("QUICK CHUNKING DEMONSTRATION")
print("="*80)
print("This will compare baseline (no chunking) vs chunking on 10 queries")
print("="*80 + "\n")

# Load tiny subset
print("Loading data...")
loader = CRAGLoader(use_full_html=True)
queries, documents = loader.load_by_tasks(["1_2"])

# Limit to make it fast
documents = documents[:500]  # Only 500 docs for speed
queries = queries[:10]  # Only 10 queries

print(f"✓ Using {len(documents)} documents, {len(queries)} queries\n")

# Initialize model
model_name = "sentence-transformers/all-mpnet-base-v2"
embedding_model = EmbeddingModel(model_name=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Model: {model_name}")
print(f"Max sequence length: {384} tokens")
print(f"Device: {embedding_model.device}\n")

# ============================================================================
# BASELINE: No Chunking
# ============================================================================
print("="*80)
print("BASELINE: NO CHUNKING")
print("="*80)

print("\nEmbedding documents (will be truncated)...")
start_time = time.time()

doc_texts = [doc.text for doc in documents]
doc_ids = [doc.doc_id for doc in documents]
doc_embeddings = embedding_model.embed_documents(doc_texts, show_progress=True)

embed_time = time.time() - start_time

print(f"✓ Embedded {len(doc_embeddings)} documents in {embed_time:.1f}s")

# Build database
metadata = [{"title": doc.title, "url": doc.url} for doc in documents]
baseline_db = VectorDatabase(embedding_dim=embedding_model.get_embedding_dim())
baseline_db.add_documents(doc_ids, doc_embeddings, metadata)

# Evaluate
print("\nEvaluating baseline...")
evaluator = RetrievalEvaluator()
retriever = BaselineRetrieval(vector_db=baseline_db, k=10)

baseline_results = []
for query in queries:
    query_embedding = embedding_model.embed_query(query.query)
    retrieval_output = retriever.retrieve(query.query_id, query_embedding)
    relevant_ids = extract_ground_truth_ids(query)
    
    eval_result = evaluator.evaluate_single(
        query.query_id,
        retrieval_output.retrieved_ids,
        retrieval_output.retrieved_scores,
        relevant_ids,
        {}
    )
    baseline_results.append(eval_result)

baseline_metrics = evaluator.evaluate_batch(baseline_results)

print(f"\nBaseline Results:")
print(f"  Recall@10: {baseline_metrics.mean_recall_at_k_labeled:.3f}")
print(f"  Precision@10: {baseline_metrics.mean_precision_at_k_labeled:.3f}")
print(f"  NDCG@10: {baseline_metrics.mean_ndcg_at_k_labeled:.3f}")

# ============================================================================
# WITH CHUNKING
# ============================================================================
print(f"\n{'='*80}")
print("WITH CHUNKING (Recursive, 256 tokens, 50 overlap)")
print("="*80)

# Initialize chunker
chunker = get_chunker(
    strategy="recursive",
    chunk_size=256,
    chunk_overlap=50,
    tokenizer=tokenizer
)

print("\nChunking documents...")
start_time = time.time()

all_chunks = []
chunk_to_doc = {}

for doc in documents:
    chunks = chunker.chunk_document(
        doc.doc_id,
        doc.text,
        {"title": doc.title, "url": doc.url}
    )
    all_chunks.extend(chunks)
    for chunk in chunks:
        chunk_to_doc[chunk.chunk_id] = doc.doc_id

chunk_time = time.time() - start_time

print(f"✓ Created {len(all_chunks)} chunks in {chunk_time:.1f}s")
print(f"✓ Avg chunks per doc: {len(all_chunks)/len(documents):.1f}")

# Embed chunks
print("\nEmbedding chunks...")
start_time = time.time()

chunk_texts = [chunk.text for chunk in all_chunks]
chunk_ids = [chunk.chunk_id for chunk in all_chunks]
chunk_embeddings = embedding_model.embed_documents(chunk_texts, show_progress=True)

embed_time = time.time() - start_time

print(f"✓ Embedded {len(chunk_embeddings)} chunks in {embed_time:.1f}s")

# Build database
chunk_metadata = [chunk.metadata for chunk in all_chunks]
chunk_db = VectorDatabase(embedding_dim=embedding_model.get_embedding_dim())
chunk_db.add_documents(chunk_ids, chunk_embeddings, chunk_metadata)

# Evaluate
print("\nEvaluating with chunking...")
chunk_retriever = BaselineRetrieval(vector_db=chunk_db, k=30)  # Retrieve more chunks

chunk_results = []
for query in queries:
    query_embedding = embedding_model.embed_query(query.query)
    retrieval_output = chunk_retriever.retrieve(query.query_id, query_embedding)
    
    # Map chunks back to documents
    doc_scores = {}
    for chunk_id in retrieval_output.retrieved_ids:
        parent_doc_id = chunk_to_doc.get(chunk_id, chunk_id)
        if parent_doc_id not in doc_scores:
            doc_scores[parent_doc_id] = []
    
    # Get top 10 documents
    doc_ids_ranked = list(doc_scores.keys())[:10]
    doc_scores_ranked = [1.0 / (i + 1) for i in range(len(doc_ids_ranked))]
    
    relevant_ids = extract_ground_truth_ids(query)
    
    eval_result = evaluator.evaluate_single(
        query.query_id,
        doc_ids_ranked,
        doc_scores_ranked,
        relevant_ids,
        {}
    )
    chunk_results.append(eval_result)

chunk_metrics = evaluator.evaluate_batch(chunk_results)

print(f"\nChunking Results:")
print(f"  Recall@10: {chunk_metrics.mean_recall_at_k_labeled:.3f}")
print(f"  Precision@10: {chunk_metrics.mean_precision_at_k_labeled:.3f}")
print(f"  NDCG@10: {chunk_metrics.mean_ndcg_at_k_labeled:.3f}")

# ============================================================================
# COMPARISON
# ============================================================================
print(f"\n{'='*80}")
print("COMPARISON")
print("="*80)

print(f"\n{'Metric':<20} {'Baseline':<12} {'Chunking':<12} {'Improvement':<12}")
print("-" * 60)

metrics_to_compare = [
    ("Recall@10", baseline_metrics.mean_recall_at_k_labeled, chunk_metrics.mean_recall_at_k_labeled),
    ("Precision@10", baseline_metrics.mean_precision_at_k_labeled, chunk_metrics.mean_precision_at_k_labeled),
    ("NDCG@10", baseline_metrics.mean_ndcg_at_k_labeled, chunk_metrics.mean_ndcg_at_k_labeled),
]

for metric_name, baseline_val, chunk_val in metrics_to_compare:
    if baseline_val > 0:
        improvement = ((chunk_val - baseline_val) / baseline_val) * 100
    else:
        improvement = 0.0
    
    print(f"{metric_name:<20} {baseline_val:<12.3f} {chunk_val:<12.3f} {improvement:+.1f}%")

print(f"\n{'='*80}")
print("SUMMARY")
print("="*80)

avg_improvement = np.mean([
    ((chunk_metrics.mean_recall_at_k_labeled - baseline_metrics.mean_recall_at_k_labeled) / baseline_metrics.mean_recall_at_k_labeled) * 100 if baseline_metrics.mean_recall_at_k_labeled > 0 else 0,
    ((chunk_metrics.mean_ndcg_at_k_labeled - baseline_metrics.mean_ndcg_at_k_labeled) / baseline_metrics.mean_ndcg_at_k_labeled) * 100 if baseline_metrics.mean_ndcg_at_k_labeled > 0 else 0
])

if chunk_metrics.mean_recall_at_k_labeled > baseline_metrics.mean_recall_at_k_labeled:
    print(f"\n✅ Chunking IMPROVES retrieval quality!")
    print(f"   Average improvement: {avg_improvement:.1f}%")
    print(f"\n💡 Recommendation: Rebuild vector database with chunking")
    print(f"   Command: python src/tests/test_chunking.py --max-queries 100")
else:
    print(f"\n⚠️  Results inconclusive with this small sample")
    print(f"   Try with more queries: --max-queries 50")

print(f"\n{'='*80}\n")
