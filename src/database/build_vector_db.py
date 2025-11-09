"""
Build CRAG Vector Database

Simple script to build a FAISS vector database from CRAG documents.
This is the first step before implementing HC-RAG retrieval.

Usage:
    python build_vector_db.py                    # Build from Tasks 1&2
    python build_vector_db.py --tasks 1_2 3      # Build from all tasks
    python build_vector_db.py --max-docs 5000    # Limit documents
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.crag_loader import CRAGLoader
from embeddings.embedding_model import EmbeddingModel
from embeddings.vector_database import VectorDatabase
import time


def main():
    parser = argparse.ArgumentParser(description="Build CRAG vector database")
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
        default="fast",
        choices=["fast", "quality"],
        help="Embedding model to use (default: fast)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="crag_vector_db",
        help="Output path for database (default: crag_vector_db)"
    )
    parser.add_argument(
        "--use-full-html",
        action="store_true",
        help="Use full HTML instead of snippets"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("CRAG VECTOR DATABASE BUILDER")
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
        else EmbeddingModel.QUALITY_MODEL
    )
    model = EmbeddingModel(model_name=model_name)

    print(f"✓ Model: {model.get_model_name()}")
    print(f"✓ Embedding dimension: {model.get_embedding_dim()}")
    print(f"✓ Device: {model.device}")
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


if __name__ == "__main__":
    main()
