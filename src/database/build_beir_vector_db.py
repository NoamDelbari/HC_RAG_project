"""
Build BEIR Vector Database

Builds FAISS vector databases from BEIR benchmark datasets (FiQA, SciFact, etc.)
for use with HC retrieval experiments.

Usage:
    # Build FiQA database with default settings
    python build_beir_vector_db.py --dataset fiqa

    # Build with specific model
    python build_beir_vector_db.py --dataset fiqa --model bge

    # Build chunked database
    python build_beir_vector_db.py --dataset fiqa --chunked --chunk-size 256
    
    # Limit documents for testing
    python build_beir_vector_db.py --dataset fiqa --max-docs 5000
"""

import argparse
import sys
import time
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.beir_loader import BEIRLoader, BEIRDocument
from embeddings.embedding_model import EmbeddingModel, create_embedding_model
from embeddings.vector_database import VectorDatabase
from embeddings.chunking import get_chunker, Chunk
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_beir_db(args):
    """Build vector database from BEIR dataset."""
    
    mode = "CHUNKED" if args.chunked else "FULL DOCUMENT"
    
    print("\n" + "="*80)
    print(f"BEIR VECTOR DATABASE BUILDER - {mode} MODE")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Model: {args.model}")
    print(f"Max documents: {args.max_docs if args.max_docs else 'All'}")
    if args.chunked:
        print(f"Chunking: {args.chunker} (size={args.chunk_size}, overlap={args.chunk_overlap})")
    print(f"Output: {args.output}")
    print("="*80 + "\n")
    
    # Step 1: Load BEIR dataset
    print("Step 1: Loading BEIR dataset...")
    print("-" * 80)
    
    loader = BEIRLoader(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        split=args.split,
        download_if_missing=True
    )
    
    documents, queries = loader.load_all()
    
    print(f"✓ Loaded {len(documents)} documents")
    print(f"✓ Loaded {len(queries)} queries")
    
    # Show stats
    stats = loader.get_stats()
    rel_stats = stats['relevance_per_query']
    print(f"✓ Relevance per query: mean={rel_stats['mean']:.2f}, range={rel_stats['min']}-{rel_stats['max']}")
    
    # Limit documents if requested
    if args.max_docs and len(documents) > args.max_docs:
        print(f"⚠ Limiting to {args.max_docs} documents")
        documents = documents[:args.max_docs]
    
    print()
    
    # Step 2: Initialize embedding model
    print("Step 2: Initializing embedding model...")
    print("-" * 80)
    
    # Map model shorthand to full model name
    model_name = (
        EmbeddingModel.FAST_MODEL if args.model == "fast"
        else EmbeddingModel.BGE_MODEL if args.model == "bge"
        else EmbeddingModel.GEMINI_MODEL if args.model == "gemini"
        else EmbeddingModel.QUALITY_MODEL if args.model == "quality"
        else args.model
    )
    
    # Auto-detect optimal batch size
    if args.batch_size:
        batch_size = args.batch_size
    else:
        if model_name.startswith("models/") or "gemini" in model_name.lower():
            batch_size = 100
        else:
            try:
                import torch
                if torch.cuda.is_available():
                    batch_size = 512
                    print(f"🚀 GPU detected - using large batch size: {batch_size}")
                else:
                    batch_size = 64
            except ImportError:
                batch_size = 64
    
    model = create_embedding_model(model_name=model_name, batch_size=batch_size)
    
    print(f"✓ Model: {model.get_model_name()}")
    print(f"✓ Embedding dimension: {model.get_embedding_dim()}")
    if hasattr(model, 'device'):
        print(f"✓ Device: {model.device}")
    print(f"✓ Batch size: {batch_size}")
    print()
    
    # Step 3: Prepare texts (chunk if needed)
    if args.chunked:
        print("Step 3: Chunking documents...")
        print("-" * 80)
        
        chunker = get_chunker(
            strategy=args.chunker,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            model_name=model.get_model_name()
        )
        
        all_chunks: List[Chunk] = []
        chunk_count_per_doc: Dict[str, int] = {}
        
        start_time = time.time()
        for i, doc in enumerate(documents):
            if i % 1000 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                print(f"  Chunked {i}/{len(documents)} documents ({rate:.1f} docs/sec)")
            
            chunks = chunker.chunk_document(
                doc_id=doc.doc_id,
                text=doc.get_full_text(),
                metadata={"title": doc.title or ""}
            )
            all_chunks.extend(chunks)
            chunk_count_per_doc[doc.doc_id] = len(chunks)
        
        chunking_time = time.time() - start_time
        
        print(f"✓ Chunked {len(documents)} documents into {len(all_chunks)} chunks")
        print(f"✓ Time: {chunking_time:.2f}s")
        print(f"✓ Average chunks per document: {len(all_chunks)/len(documents):.1f}")
        
        # Prepare texts and IDs
        texts = [chunk.text for chunk in all_chunks]
        ids = [chunk.chunk_id for chunk in all_chunks]
        metadata = [
            {
                "parent_doc_id": chunk.parent_doc_id,
                "chunk_index": chunk.chunk_index,
                "title": chunk.metadata.get("title", ""),
                "start_char": chunk.start_char,
                "end_char": chunk.end_char
            }
            for chunk in all_chunks
        ]
        
        # Create chunk-to-doc mapping
        chunk_to_doc_mapping = {
            chunk.chunk_id: chunk.parent_doc_id
            for chunk in all_chunks
        }
        
    else:
        print("Step 3: Preparing documents...")
        print("-" * 80)
        
        texts = [doc.get_full_text() for doc in documents]
        ids = [doc.doc_id for doc in documents]
        metadata = [
            {"title": doc.title or "", "text_length": len(doc.text)}
            for doc in documents
        ]
        chunk_to_doc_mapping = None
        chunk_count_per_doc = None
        
        print(f"✓ Prepared {len(texts)} documents")
    
    print()
    
    # Step 4: Generate embeddings
    print("Step 4: Generating embeddings...")
    print("-" * 80)
    
    print(f"Embedding {len(texts)} {'chunks' if args.chunked else 'documents'}...")
    start_time = time.time()
    
    embeddings = model.embed_documents(texts, show_progress=True)
    
    embedding_time = time.time() - start_time
    
    print(f"✓ Generated {len(embeddings)} embeddings")
    print(f"✓ Time: {embedding_time:.2f}s ({len(texts)/embedding_time:.1f} items/sec)")
    print()
    
    # Step 5: Build vector database
    print("Step 5: Building vector database...")
    print("-" * 80)
    
    db = VectorDatabase(
        embedding_dim=model.get_embedding_dim(),
        embedding_model_name=model.get_model_name()
    )
    db.add_documents(ids, embeddings, metadata)
    
    print(f"✓ Built database with {db.get_num_documents()} entries")
    print()
    
    # Step 6: Build null distributions for queries
    print("Step 6: Building per-query null distributions...")
    print("-" * 80)
    
    # Embed all queries
    query_texts = [q.text for q in queries]
    print(f"Embedding {len(query_texts)} queries...")
    query_embeddings = model.embed_documents(query_texts, show_progress=True)
    query_embeddings = np.array(query_embeddings)
    
    # Build PER-QUERY null distributions (critical for HC correctness!)
    # Each query has its own similarity baseline that differs from the global average
    n_null_samples_per_query = 500  # Samples per query for null distribution
    n_docs = db.get_num_documents()
    all_doc_embeddings = db.index.reconstruct_n(0, n_docs)
    
    print(f"Building per-query null distributions ({n_null_samples_per_query} samples each)...")
    
    # Get doc_id to index mapping for excluding relevant docs
    doc_ids = db.doc_ids
    doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    
    per_query_null = {}
    np.random.seed(42)
    
    for qi, query in enumerate(queries):
        if (qi + 1) % 100 == 0:
            print(f"  Progress: {qi+1}/{len(queries)} queries")
        
        # Get indices of relevant docs (to exclude)
        relevant_indices = set()
        for rel_doc_id in query.relevant_docs:
            if rel_doc_id in doc_id_to_idx:
                relevant_indices.add(doc_id_to_idx[rel_doc_id])
        
        # Get all non-relevant indices 
        all_indices = set(range(n_docs))
        non_relevant_indices = list(all_indices - relevant_indices)
        
        # Sample from non-relevant docs
        if len(non_relevant_indices) >= n_null_samples_per_query:
            sample_indices = np.random.choice(non_relevant_indices, size=n_null_samples_per_query, replace=False)
        else:
            sample_indices = np.random.choice(non_relevant_indices, size=n_null_samples_per_query, replace=True)
        
        # Compute similarities to sampled docs
        q_emb = query_embeddings[qi]
        sampled_doc_embs = all_doc_embeddings[sample_indices]
        null_sims = np.dot(sampled_doc_embs, q_emb)
        
        per_query_null[query.query_id] = {
            'similarities': null_sims,
            'mean': float(np.mean(null_sims)),
            'std': float(np.std(null_sims)),
            'n_samples': len(null_sims)
        }
    
    # Also compute global null for reference
    all_null_sims = np.concatenate([v['similarities'] for v in per_query_null.values()])
    
    print(f"✓ Built {len(per_query_null)} per-query null distributions")
    print(f"✓ Samples per query: {n_null_samples_per_query}")
    print(f"✓ Global null (reference):")
    print(f"  Mean: {all_null_sims.mean():.4f}")
    print(f"  Std: {all_null_sims.std():.4f}")
    print(f"  Min/Max: {all_null_sims.min():.4f} / {all_null_sims.max():.4f}")
    
    # Show per-query variation
    means = [v['mean'] for v in per_query_null.values()]
    stds = [v['std'] for v in per_query_null.values()]
    print(f"✓ Per-query null mean range: [{min(means):.4f}, {max(means):.4f}]")
    print(f"✓ Per-query null std range: [{min(stds):.4f}, {max(stds):.4f}]")
    print()
    
    # Step 7: Save database and metadata
    print("Step 7: Saving database and metadata...")
    print("-" * 80)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    db.save(str(output_path))
    
    # Save query data
    query_data_path = output_path.with_suffix(".query_data.pkl")
    query_data = {
        "queries": [
            {
                "query_id": q.query_id,
                "text": q.text,
                "relevant_docs": q.relevant_docs,
                "relevance_scores": q.relevance_scores
            }
            for q in queries
        ],
        "query_embeddings": np.array(query_embeddings),
        "per_query_null": per_query_null,  # Per-query null distributions
        "global_null": {  # Keep global for reference
            "mean": float(all_null_sims.mean()),
            "std": float(all_null_sims.std()),
            "samples": all_null_sims
        }
    }
    with open(query_data_path, "wb") as f:
        pickle.dump(query_data, f)
    
    # Save dataset info
    info_path = output_path.with_suffix(".info.pkl")
    info = {
        "dataset": args.dataset,
        "split": args.split,
        "model_name": model.get_model_name(),
        "embedding_dim": model.get_embedding_dim(),
        "n_documents": len(documents),
        "n_queries": len(queries),
        "chunked": args.chunked,
        "stats": stats
    }
    
    if args.chunked:
        info["chunker_config"] = {
            "strategy": args.chunker,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap
        }
        info["n_chunks"] = len(all_chunks)
        
        # Save chunk mapping
        mapping_path = output_path.with_suffix(".chunk_mapping.pkl")
        with open(mapping_path, "wb") as f:
            pickle.dump({
                "chunk_to_doc": chunk_to_doc_mapping,
                "chunk_count_per_doc": chunk_count_per_doc
            }, f)
        print(f"  - {mapping_path} (chunk mappings)")
    
    with open(info_path, "wb") as f:
        pickle.dump(info, f)
    
    print(f"✓ Saved to: {output_path}")
    print(f"  - {output_path}.faiss (FAISS index)")
    print(f"  - {output_path}.metadata.pkl (document metadata)")
    print(f"  - {query_data_path} (query data & embeddings)")
    print(f"  - {info_path} (dataset info)")
    print()
    
    # Step 8: Test retrieval
    print("Step 8: Testing retrieval...")
    print("-" * 80)
    
    if queries:
        test_query = queries[0]
        print(f"Query: {test_query.text[:100]}...")
        print(f"Relevant docs: {len(test_query.relevant_docs)}")
        
        query_embedding = model.embed_query(test_query.text)
        results = db.search(query_embedding, k=10)
        
        # Check hits
        retrieved_ids = [r.doc_id for r in results]
        if args.chunked:
            # Map to parent docs
            retrieved_docs = set(chunk_to_doc_mapping.get(cid, cid) for cid in retrieved_ids)
        else:
            retrieved_docs = set(retrieved_ids)
        
        hits = len(set(test_query.relevant_docs) & retrieved_docs)
        
        print(f"\nTop 5 results:")
        for result in results[:5]:
            doc_id = result.doc_id if not args.chunked else chunk_to_doc_mapping.get(result.doc_id, result.doc_id)
            is_relevant = "✓" if doc_id in test_query.relevant_docs else " "
            print(f"  [{is_relevant}] Score: {result.similarity:.4f} - {doc_id[:40]}...")
        
        print(f"\nHits@10: {hits}/{len(test_query.relevant_docs)}")
    
    print()
    print("="*80)
    print("✓ BEIR vector database built successfully!")
    print(f"✓ Dataset: {args.dataset}")
    print(f"✓ Documents: {len(documents)}")
    if args.chunked:
        print(f"✓ Chunks: {len(all_chunks)}")
    print(f"✓ Queries: {len(queries)}")
    print(f"✓ Ready for HC retrieval experiments")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build BEIR vector database for HC retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build FiQA database (recommended for HC testing)
  python build_beir_vector_db.py --dataset fiqa
  
  # Build with BGE model (higher quality)
  python build_beir_vector_db.py --dataset fiqa --model bge
  
  # Build chunked database  
  python build_beir_vector_db.py --dataset fiqa --chunked --chunk-size 256
  
  # Test with subset
  python build_beir_vector_db.py --dataset fiqa --max-docs 5000
        """
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="BEIR dataset name (e.g., fiqa, scifact, nfcorpus)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split to use (default: test)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets/beir",
        help="Directory for BEIR datasets (default: datasets/beir)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="bge",
        help="Embedding model: 'fast', 'bge', 'quality', 'gemini', or full name (default: bge)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding batch size (default: auto)"
    )
    
    # Chunking arguments
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Enable chunked document mode"
    )
    parser.add_argument(
        "--chunker",
        type=str,
        default="recursive",
        choices=["fixed", "sentence", "recursive"],
        help="Chunking strategy (default: recursive)"
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
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: datasets/beir/{dataset}_vector_db)"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit number of documents (for testing)"
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        suffix = "_chunked" if args.chunked else ""
        args.output = f"datasets/beir/{args.dataset}_vector_db{suffix}"
    
    build_beir_db(args)


if __name__ == "__main__":
    main()
