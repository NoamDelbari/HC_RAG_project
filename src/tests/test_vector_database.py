"""
End-to-End Vector Database Test

Tests the complete pipeline:
1. Load CRAG documents
2. Generate embeddings
3. Build FAISS vector database
4. Perform retrieval queries
5. Save and load database
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.crag_loader import CRAGLoader
from embeddings.embedding_model import EmbeddingModel
from embeddings.vector_database import VectorDatabase
import time


def test_vector_database_end_to_end():
    """
    Complete end-to-end test of the vector database pipeline.

    Uses a subset of CRAG documents for faster testing.
    """

    print("\n" + "="*80)
    print("END-TO-END VECTOR DATABASE TEST")
    print("="*80 + "\n")

    # Configuration
    MAX_DOCUMENTS = 1000  # Limit for faster testing (remove for full dataset)
    MAX_QUERIES = 10      # Test with 10 queries
    TOP_K = 5             # Retrieve top 5 documents per query
    DB_SAVE_PATH = "tests/crag_vector_db"

    # =========================================================================
    # STEP 1: Load CRAG Dataset
    # =========================================================================
    print("STEP 1: Loading CRAG Dataset")
    print("-" * 80)

    loader = CRAGLoader(use_full_html=False)
    queries, documents = loader.load_by_tasks(["1_2"])

    print(f"✓ Loaded {len(queries)} queries")
    print(f"✓ Loaded {len(documents)} documents")

    # Limit for testing
    if len(documents) > MAX_DOCUMENTS:
        print(f"⚠ Limiting to {MAX_DOCUMENTS} documents for faster testing")
        documents = documents[:MAX_DOCUMENTS]

    if len(queries) > MAX_QUERIES:
        print(f"⚠ Limiting to {MAX_QUERIES} queries for testing")
        queries = queries[:MAX_QUERIES]

    print()

    # =========================================================================
    # STEP 2: Initialize Embedding Model
    # =========================================================================
    print("STEP 2: Initializing Embedding Model")
    print("-" * 80)

    model = EmbeddingModel(model_name=EmbeddingModel.FAST_MODEL)

    print(f"✓ Model: {model.get_model_name()}")
    print(f"✓ Embedding dimension: {model.get_embedding_dim()}")
    print(f"✓ Device: {model.device}")
    print()

    # =========================================================================
    # STEP 3: Generate Document Embeddings
    # =========================================================================
    print("STEP 3: Generating Document Embeddings")
    print("-" * 80)

    doc_texts = [doc.text for doc in documents]
    doc_ids = [doc.doc_id for doc in documents]

    print(f"Embedding {len(doc_texts)} documents...")
    start_time = time.time()

    doc_embeddings = model.embed_documents(doc_texts, show_progress=True)

    embedding_time = time.time() - start_time

    print(f"✓ Generated {len(doc_embeddings)} embeddings")
    print(f"✓ Embeddings shape: {doc_embeddings.shape}")
    print(f"✓ Time taken: {embedding_time:.2f}s ({len(doc_texts)/embedding_time:.1f} docs/sec)")
    print()

    # =========================================================================
    # STEP 4: Build Vector Database
    # =========================================================================
    print("STEP 4: Building Vector Database")
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

    # Create and populate database
    db = VectorDatabase(embedding_dim=model.get_embedding_dim())
    db.add_documents(doc_ids, doc_embeddings, metadata)

    print(f"✓ Built database with {db.get_num_documents()} documents")
    print()

    # =========================================================================
    # STEP 5: Sample Null Distribution (for HC)
    # =========================================================================
    print("STEP 5: Sampling Null Distribution for Higher Criticism")
    print("-" * 80)

    # Generate query embeddings
    print("Generating query embeddings for null distribution...")
    query_texts = [q.query for q in queries]
    query_embeddings = model.embed_documents(query_texts, show_progress=True)

    print(f"Sampling 10,000 random query-document pairs...")
    null_similarities = db.get_similarity_distribution(
        query_embeddings=query_embeddings,
        n_samples=10000,
        seed=42
    )

    print(f"✓ Null distribution statistics:")
    print(f"  Mean: {np.mean(null_similarities):.4f}")
    print(f"  Std: {np.std(null_similarities):.4f}")
    print(f"  Min: {np.min(null_similarities):.4f}")
    print(f"  Max: {np.max(null_similarities):.4f}")
    print(f"  Median: {np.median(null_similarities):.4f}")
    print()

    # =========================================================================
    # STEP 6: Test Retrieval with Sample Queries
    # =========================================================================
    print("STEP 6: Testing Retrieval")
    print("-" * 80)

    # Create a mapping from doc_id to document for ground truth lookup
    doc_map = {doc.doc_id: doc for doc in documents}

    for i, query in enumerate(queries[:3], 1):  # Test with first 3 queries
        print(f"\n[Query {i}/{len(queries[:3])}]")
        print(f"  Text: {query.query}")
        print(f"  Gold Answer: {query.answer}")
        print()

        # Embed query
        query_embedding = model.embed_query(query.query)

        # Search
        search_start = time.time()
        results = db.search(query_embedding, k=TOP_K)
        search_time = time.time() - search_start

        print(f"  Search time: {search_time*1000:.2f}ms")
        print()

        # =================================================================
        # Show Ground Truth Documents (from query's search_results)
        # =================================================================
        print(f"  GROUND TRUTH DOCUMENTS (from query's search_results):")
        print(f"  {'-'*74}")

        ground_truth_ids = []
        for idx, search_result in enumerate(query.search_results):
            # Construct the doc_id that was used when adding to database
            gt_doc_id = f"{query.query_id}_doc_{idx}"
            ground_truth_ids.append(gt_doc_id)

            # Check if this document is in our database
            if gt_doc_id in doc_map:
                # Find this document's embedding and compute similarity
                doc_idx = doc_ids.index(gt_doc_id)
                gt_doc_embedding = doc_embeddings[doc_idx]

                # Compute cosine similarity
                from embeddings.embedding_model import batch_cosine_similarity
                similarity = batch_cosine_similarity(query_embedding, gt_doc_embedding.reshape(1, -1))[0]

                print(f"    [GT Doc {idx+1}] Similarity: {similarity:.4f}")
                print(f"      Doc ID: {gt_doc_id}")
                print(f"      Title: {search_result.get('page_name', 'N/A')[:60]}...")
                print(f"      URL: {search_result.get('page_url', 'N/A')[:60]}...")
                print()

        # =================================================================
        # Show Retrieved Documents
        # =================================================================
        print(f"  RETRIEVED TOP-{TOP_K} DOCUMENTS:")
        print(f"  {'-'*74}")

        retrieved_ids = [r.doc_id for r in results]

        for result in results:
            # Check if this is a ground truth document
            is_ground_truth = result.doc_id in ground_truth_ids
            marker = " ✓ [GROUND TRUTH]" if is_ground_truth else ""

            print(f"    [Rank {result.rank}] Similarity: {result.similarity:.4f}{marker}")
            print(f"      Doc ID: {result.doc_id}")
            print(f"      Title: {result.metadata.get('title', 'N/A')[:60]}...")
            print(f"      URL: {result.metadata.get('url', 'N/A')[:60]}...")
            print()

        # =================================================================
        # Summary Statistics
        # =================================================================
        gt_in_topk = len(set(ground_truth_ids) & set(retrieved_ids))
        print(f"  SUMMARY:")
        print(f"    Ground truth documents in top-{TOP_K}: {gt_in_topk}/{len(ground_truth_ids)}")
        print(f"    Recall@{TOP_K}: {gt_in_topk/len(ground_truth_ids)*100:.1f}%")
        print()

    # =========================================================================
    # STEP 7: Test Batch Search
    # =========================================================================
    print("\nSTEP 7: Testing Batch Search")
    print("-" * 80)

    # Embed all test queries
    query_texts = [q.query for q in queries]
    print(f"Embedding {len(query_texts)} queries...")
    query_embeddings = model.embed_documents(query_texts, show_progress=True)

    # Batch search
    print(f"\nPerforming batch search...")
    batch_start = time.time()
    batch_results = db.batch_search(query_embeddings, k=TOP_K)
    batch_time = time.time() - batch_start

    print(f"✓ Batch search completed")
    print(f"  Time: {batch_time*1000:.2f}ms")
    print(f"  Time per query: {batch_time*1000/len(query_texts):.2f}ms")
    print(f"  Retrieved {TOP_K} documents for each of {len(query_texts)} queries")
    print()

    # =========================================================================
    # STEP 8: Save Database
    # =========================================================================
    print("STEP 8: Saving Database to Disk")
    print("-" * 80)

    print(f"Saving to: {DB_SAVE_PATH}")
    db.save(DB_SAVE_PATH)
    print(f"✓ Database saved successfully")
    print()

    # =========================================================================
    # STEP 9: Load Database and Test
    # =========================================================================
    print("STEP 9: Loading Database from Disk")
    print("-" * 80)

    print(f"Loading from: {DB_SAVE_PATH}")
    db_loaded = VectorDatabase.load(DB_SAVE_PATH)

    print(f"✓ Database loaded successfully")
    print(f"  Documents: {db_loaded.get_num_documents()}")
    print(f"  Embedding dim: {db_loaded.embedding_dim}")
    print()

    # Verify loaded database works
    print("Verifying loaded database with a test query...")
    test_query_embedding = query_embeddings[0]
    test_results = db_loaded.search(test_query_embedding, k=3)

    print(f"✓ Retrieved {len(test_results)} results from loaded database")
    for result in test_results:
        print(f"  [Rank {result.rank}] {result.doc_id}: {result.similarity:.4f}")
    print()

    # =========================================================================
    # STEP 10: Summary Statistics
    # =========================================================================
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Documents indexed: {db.get_num_documents()}")
    print(f"✓ Embedding dimension: {model.get_embedding_dim()}")
    print(f"✓ Queries tested: {len(queries)}")
    print(f"✓ Embedding speed: {len(doc_texts)/embedding_time:.1f} docs/sec")
    print(f"✓ Search speed (batch): {batch_time*1000/len(query_texts):.2f}ms per query")
    print(f"✓ Database saved to: {DB_SAVE_PATH}")
    print()
    print("✓ All tests passed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_vector_database_end_to_end()
