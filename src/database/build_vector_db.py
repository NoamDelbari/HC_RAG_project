"""
Build CRAG Vector Database

Unified script to build FAISS vector databases from CRAG documents.
Supports both full-document and chunked document embedding modes.

Usage:
    # Full document mode (simple)
    python build_vector_db.py --mode full
    python build_vector_db.py --mode full --model quality

    # Chunked document mode (advanced)
    python build_vector_db.py --mode chunked
    python build_vector_db.py --mode chunked --chunker recursive --chunk-size 512
    python build_vector_db.py --mode chunked --save-chunks-only

    # Common options
    python build_vector_db.py --mode full --max-docs 5000
    python build_vector_db.py --mode chunked --tasks 1_2 3
"""

import argparse
import sys
from pathlib import Path
import time
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.crag_loader import CRAGLoader
from embeddings.embedding_model import EmbeddingModel
from embeddings.vector_database import VectorDatabase
from embeddings.chunking import get_chunker, Chunk
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_full_db(args):
    """Build vector database from full documents (no chunking)"""

    print("\n" + "="*80)
    print("CRAG VECTOR DATABASE BUILDER - FULL DOCUMENT MODE")
    print("="*80)
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Model: {args.model}")
    print(f"Max documents: {args.max_docs if args.max_docs else 'All'}")
    print(f"Use full HTML: {args.use_full_html}")
    print(f"Output: {args.output}")
    print("="*80 + "\n")

    # Step 1: Load CRAG data
    print("Step 1: Loading CRAG dataset...")
    print("-" * 80)
    loader = CRAGLoader(use_full_html=args.use_full_html)
    queries, documents = loader.load_by_tasks(args.tasks)

    print(f"✓ Loaded {len(queries)} queries")
    print(f"✓ Loaded {len(documents)} documents")

    # Limit documents if requested
    if args.max_docs and len(documents) > args.max_docs:
        print(f"⚠ Limiting to {args.max_docs} documents")
        documents = documents[:args.max_docs]

    print()

    # Step 2: Initialize embedding model
    print("Step 2: Initializing embedding model...")
    print("-" * 80)

    model_name = (
        EmbeddingModel.FAST_MODEL if args.model == "fast"
        else EmbeddingModel.BGE_MODEL if args.model == "bge"
        else EmbeddingModel.QUALITY_MODEL
    )

    # Auto-detect optimal batch size for GPU if batch_size specified
    if args.batch_size:
        batch_size = args.batch_size
    else:
        import torch
        if torch.cuda.is_available():
            batch_size = 512
            print(f"🚀 GPU detected - using large batch size: {batch_size}")
        else:
            batch_size = 32

    model = EmbeddingModel(model_name=model_name, batch_size=batch_size)

    print(f"✓ Model: {model.get_model_name()}")
    print(f"✓ Embedding dimension: {model.get_embedding_dim()}")
    print(f"✓ Device: {model.device}")
    print(f"✓ Batch size: {batch_size}")
    print()

    # Step 3: Generate embeddings
    print("Step 3: Generating embeddings...")
    print("-" * 80)

    doc_texts = [doc.text for doc in documents]
    doc_ids = [doc.doc_id for doc in documents]

    print(f"Embedding {len(doc_texts)} documents...")
    start_time = time.time()

    doc_embeddings = model.embed_documents(doc_texts, show_progress=True)

    embedding_time = time.time() - start_time

    print(f"✓ Generated {len(doc_embeddings)} embeddings")
    print(f"✓ Time: {embedding_time:.2f}s ({len(doc_texts)/embedding_time:.1f} docs/sec)")
    print()

    # Step 4: Build vector database
    print("Step 4: Building vector database...")
    print("-" * 80)

    # Prepare metadata
    metadata = [
        {
            "title": doc.title,
            "url": doc.url,
            "text_length": len(doc.text)
        }
        for doc in documents
    ]

    db = VectorDatabase(embedding_dim=model.get_embedding_dim())
    db.add_documents(doc_ids, doc_embeddings, metadata)

    print(f"✓ Built database with {db.get_num_documents()} documents")
    print()

    # Step 5: Sample null distribution
    print("Step 5: Sampling null distribution for HC statistics...")
    print("-" * 80)

    null_similarities = db.get_similarity_distribution(n_samples=10000, seed=42)

    print(f"✓ Null distribution statistics:")
    print(f"  Mean: {null_similarities.mean():.4f}")
    print(f"  Std: {null_similarities.std():.4f}")
    print(f"  Min/Max: {null_similarities.min():.4f} / {null_similarities.max():.4f}")
    print()

    # Step 6: Save database
    print("Step 6: Saving database...")
    print("-" * 80)

    db.save(args.output)

    print(f"✓ Database saved to: {args.output}")
    print(f"  - {args.output}.faiss (FAISS index)")
    print(f"  - {args.output}.metadata.pkl (document metadata)")
    print()

    # Step 7: Test query
    print("Step 7: Testing with sample query...")
    print("-" * 80)

    if queries:
        test_query = queries[0]
        print(f"Query: {test_query.query}")

        query_embedding = model.embed_query(test_query.query)
        results = db.search(query_embedding, k=3)

        print(f"\nTop 3 results:")
        for result in results:
            print(f"  [Rank {result.rank}] Score: {result.similarity:.4f}")
            print(f"    {result.metadata.get('title', 'N/A')[:60]}...")

    print()
    print("="*80)
    print("✓ Vector database built successfully!")
    print(f"✓ Total documents: {db.get_num_documents()}")
    print(f"✓ Ready for HC-RAG retrieval experiments")
    print("="*80 + "\n")


def build_chunked_db(args):
    """Build vector database from chunked documents"""

    print("\n" + "="*80)
    print("CRAG VECTOR DATABASE BUILDER - CHUNKED DOCUMENT MODE")
    print("="*80)
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Model: {args.model}")
    print(f"Chunking: {args.chunker} (size={args.chunk_size}, overlap={args.chunk_overlap})")
    print(f"Max documents: {args.max_docs if args.max_docs else 'All'}")
    print(f"Use full HTML: {args.use_full_html}")
    print(f"Batch size: {args.batch_size if args.batch_size else 'Auto'}")
    print(f"Output: {args.output}")
    if args.save_chunks_only:
        print("Mode: CHUNK ONLY (no embedding)")
    elif args.load_chunks:
        print(f"Mode: EMBED ONLY (loading chunks from {args.load_chunks})")
    print("="*80 + "\n")

    # Step 1: Load CRAG data
    print("Step 1: Loading CRAG dataset...")
    print("-" * 80)

    if args.load_chunks:
        # Skip loading if we're only embedding pre-chunked data
        print("Skipping dataset load (using pre-chunked data)")
        queries = []
        documents = []
    else:
        loader = CRAGLoader(use_full_html=args.use_full_html)
        queries, documents = loader.load_by_tasks(args.tasks)

        print(f"✓ Loaded {len(queries)} queries")
        print(f"✓ Loaded {len(documents)} documents")

        # Limit documents if requested
        if args.max_docs and len(documents) > args.max_docs:
            print(f"⚠ Limiting to {args.max_docs} documents")
            documents = documents[:args.max_docs]

    print()

    # Step 2: Initialize embedding model
    print("Step 2: Initializing embedding model...")
    print("-" * 80)

    model_name = (
        EmbeddingModel.FAST_MODEL if args.model == "fast"
        else EmbeddingModel.BGE_MODEL if args.model == "bge"
        else EmbeddingModel.QUALITY_MODEL
    )

    # Auto-detect optimal batch size for GPU
    if args.batch_size:
        batch_size = args.batch_size
    else:
        import torch
        if torch.cuda.is_available():
            # GPU: Use large batches
            batch_size = 512
            print(f"🚀 GPU detected - using large batch size: {batch_size}")
        else:
            # CPU: Use smaller batches
            batch_size = 32

    model = EmbeddingModel(model_name=model_name, batch_size=batch_size)

    print(f"✓ Model: {model.get_model_name()}")
    print(f"✓ Embedding dimension: {model.get_embedding_dim()}")
    print(f"✓ Device: {model.device}")
    print(f"✓ Batch size: {batch_size}")
    print()

    # Step 3: Initialize chunker
    print("Step 3: Initializing chunker...")
    print("-" * 80)

    if args.load_chunks:
        print("Skipping chunker initialization (using pre-chunked data)")
        chunker = None
    else:
        chunker = get_chunker(
            strategy=args.chunker,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            model_name=model.get_model_name()
        )

        print(f"✓ Chunker: {args.chunker}")
        print(f"✓ Chunk size: {args.chunk_size} tokens")
        print(f"✓ Chunk overlap: {args.chunk_overlap} tokens")
    print()

    # Step 4: Chunk documents
    print("Step 4: Chunking documents...")
    print("-" * 80)

    if args.load_chunks:
        # Load pre-chunked data
        print(f"Loading chunks from {args.load_chunks}...")
        with open(args.load_chunks, "rb") as f:
            chunk_data = pickle.load(f)

        all_chunks = chunk_data["chunks"]
        chunk_count_per_doc = chunk_data["chunk_count_per_doc"]

        print(f"✓ Loaded {len(all_chunks)} pre-chunked chunks")
        print(f"✓ From {len(chunk_count_per_doc)} documents")
        print(f"✓ Average chunks per document: {len(all_chunks)/len(chunk_count_per_doc):.1f}")
    else:
        # Chunk documents
        all_chunks: List[Chunk] = []
        chunk_count_per_doc: Dict[str, int] = {}

        start_time = time.time()

        for i, doc in enumerate(documents):
            if i % 100 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = len(documents) - i
                eta = remaining / rate
                print(f"  Chunked {i}/{len(documents)} documents ({rate:.1f} docs/sec, ETA: {eta:.0f}s)")

            chunks = chunker.chunk_document(
                doc_id=doc.doc_id,
                text=doc.text,
                metadata={
                    "title": doc.title,
                    "url": doc.url
                }
            )
            all_chunks.extend(chunks)
            chunk_count_per_doc[doc.doc_id] = len(chunks)

        chunking_time = time.time() - start_time

        print(f"✓ Chunked {len(documents)} documents into {len(all_chunks)} chunks")
        print(f"✓ Time: {chunking_time:.2f}s ({len(documents)/chunking_time:.1f} docs/sec)")
        print(f"✓ Average chunks per document: {len(all_chunks)/len(documents):.1f}")
        print(f"✓ Min/Max chunks: {min(chunk_count_per_doc.values())}/{max(chunk_count_per_doc.values())}")

        # Save chunks immediately if requested
        if args.save_chunks_only:
            chunks_file = f"{args.output}.chunks.pkl"
            print(f"\n💾 Saving chunks to {chunks_file}...")
            with open(chunks_file, "wb") as f:
                pickle.dump({
                    "chunks": all_chunks,
                    "chunk_count_per_doc": chunk_count_per_doc,
                    "chunker_config": {
                        "strategy": args.chunker,
                        "chunk_size": args.chunk_size,
                        "chunk_overlap": args.chunk_overlap
                    }
                }, f)
            print(f"✓ Chunks saved! Use --load-chunks {chunks_file} to resume embedding")
            print("\n" + "="*80)
            print("✓ Chunking complete! Run with --load-chunks to continue.")
            print("="*80 + "\n")
            return

    print()

    # Step 5: Generate embeddings for chunks
    print("Step 5: Generating embeddings for chunks...")
    print("-" * 80)

    chunk_texts = [chunk.text for chunk in all_chunks]
    chunk_ids = [chunk.chunk_id for chunk in all_chunks]

    print(f"Embedding {len(chunk_texts)} chunks...")
    start_time = time.time()

    chunk_embeddings = model.embed_documents(chunk_texts, show_progress=True)

    embedding_time = time.time() - start_time

    print(f"✓ Generated {len(chunk_embeddings)} embeddings")
    print(f"✓ Time: {embedding_time:.2f}s ({len(chunk_texts)/embedding_time:.1f} chunks/sec)")
    print()

    # Step 6: Build vector database
    print("Step 6: Building vector database...")
    print("-" * 80)

    # Prepare metadata with parent document mapping
    metadata = [
        {
            "parent_doc_id": chunk.parent_doc_id,
            "chunk_index": chunk.chunk_index,
            "title": chunk.metadata.get("title", ""),
            "url": chunk.metadata.get("url", ""),
            "start_char": chunk.start_char,
            "end_char": chunk.end_char
        }
        for chunk in all_chunks
    ]

    db = VectorDatabase(embedding_dim=model.get_embedding_dim())
    db.add_documents(chunk_ids, chunk_embeddings, metadata)

    print(f"✓ Built database with {db.get_num_documents()} chunks")
    print()

    # Step 7: Create chunk-to-document mapping
    print("Step 7: Creating chunk-to-document mapping...")
    print("-" * 80)

    chunk_to_doc_mapping = {
        chunk.chunk_id: chunk.parent_doc_id
        for chunk in all_chunks
    }

    doc_to_chunks_mapping = {}
    for chunk in all_chunks:
        if chunk.parent_doc_id not in doc_to_chunks_mapping:
            doc_to_chunks_mapping[chunk.parent_doc_id] = []
        doc_to_chunks_mapping[chunk.parent_doc_id].append(chunk.chunk_id)

    print(f"✓ Created mappings for {len(chunk_to_doc_mapping)} chunks")
    print(f"✓ Mapped to {len(doc_to_chunks_mapping)} parent documents")
    print()

    # Step 8: Sample null distribution
    print("Step 8: Sampling null distribution for HC statistics...")
    print("-" * 80)

    null_similarities = db.get_similarity_distribution(n_samples=10000, seed=42)

    print(f"✓ Null distribution statistics:")
    print(f"  Mean: {null_similarities.mean():.4f}")
    print(f"  Std: {null_similarities.std():.4f}")
    print(f"  Min/Max: {null_similarities.min():.4f} / {null_similarities.max():.4f}")
    print()

    # Step 9: Save database and mappings
    print("Step 9: Saving database and mappings...")
    print("-" * 80)

    db.save(args.output)

    # Save chunk mappings
    mapping_file = f"{args.output}.chunk_mapping.pkl"
    with open(mapping_file, "wb") as f:
        pickle.dump({
            "chunk_to_doc": chunk_to_doc_mapping,
            "doc_to_chunks": doc_to_chunks_mapping,
            "chunk_count_per_doc": chunk_count_per_doc,
            "model_name": model.get_model_name(),
            "chunker_config": {
                "strategy": args.chunker,
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap
            }
        }, f)

    print(f"✓ Database saved to: {args.output}")
    print(f"  - {args.output}.faiss (FAISS index)")
    print(f"  - {args.output}.metadata.pkl (chunk metadata)")
    print(f"  - {args.output}.chunk_mapping.pkl (chunk-document mappings)")
    print()

    # Step 10: Test query
    print("Step 10: Testing with sample query...")
    print("-" * 80)

    if queries:
        test_query = queries[0]
        print(f"Query: {test_query.query}")

        query_embedding = model.embed_query(test_query.query)
        results = db.search(query_embedding, k=5)

        print(f"\nTop 5 chunk results:")
        for result in results:
            parent_doc = result.metadata.get('parent_doc_id', 'N/A')
            chunk_idx = result.metadata.get('chunk_index', 'N/A')
            title = result.metadata.get('title', 'N/A')
            print(f"  [Rank {result.rank}] Score: {result.similarity:.4f}")
            print(f"    Parent: {parent_doc} (chunk {chunk_idx})")
            print(f"    Title: {title[:60]}...")

    print()
    print("="*80)
    print("✓ Chunked vector database built successfully!")
    print(f"✓ Total chunks: {db.get_num_documents()}")
    print(f"✓ Parent documents: {len(documents) if documents else len(doc_to_chunks_mapping)}")
    print(f"✓ Ready for chunked retrieval experiments")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build CRAG vector database (full or chunked mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full document mode
  python build_vector_db.py --mode full
  python build_vector_db.py --mode full --model quality --max-docs 5000

  # Chunked document mode
  python build_vector_db.py --mode chunked
  python build_vector_db.py --mode chunked --chunker recursive --chunk-size 512
  python build_vector_db.py --mode chunked --save-chunks-only
        """
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["full", "chunked"],
        help="Database building mode: 'full' for full documents, 'chunked' for chunked documents"
    )

    # Common arguments
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["1_2"],
        choices=["1_2", "3"],
        help="Which CRAG tasks to load (default: 1_2)"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to index (default: all)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["fast", "quality", "bge"],
        help="Embedding model to use (default: 'fast' for full mode, 'quality' for chunked mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for database (default: auto-set based on mode)"
    )
    parser.add_argument(
        "--use-full-html",
        action="store_true",
        help="Use full HTML instead of snippets"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding batch size (default: auto-detect based on device)"
    )

    # Chunking-specific arguments (only used in chunked mode)
    parser.add_argument(
        "--chunker",
        type=str,
        default="recursive",
        choices=["fixed", "sentence", "recursive"],
        help="[CHUNKED MODE] Chunking strategy (default: recursive)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=384,
        help="[CHUNKED MODE] Target chunk size in tokens (default: 384)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="[CHUNKED MODE] Overlap between chunks in tokens (default: 50)"
    )
    parser.add_argument(
        "--save-chunks-only",
        action="store_true",
        help="[CHUNKED MODE] Only chunk and save, skip embedding (for resuming later)"
    )
    parser.add_argument(
        "--load-chunks",
        type=str,
        default=None,
        help="[CHUNKED MODE] Load pre-chunked data from file and only do embedding"
    )

    args = parser.parse_args()

    # Set defaults based on mode
    if args.model is None:
        args.model = "fast" if args.mode == "full" else "quality"

    if args.output is None:
        args.output = "crag_vector_db" if args.mode == "full" else "crag_chunked_vector_db"

    # Route to appropriate builder
    if args.mode == "full":
        build_full_db(args)
    else:  # chunked
        build_chunked_db(args)


if __name__ == "__main__":
    main()
