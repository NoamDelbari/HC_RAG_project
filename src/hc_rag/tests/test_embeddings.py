"""
Test Embedding Model with CRAG Dataset

Tests the embedding model using a real query and documents from CRAG Task 1/2.
"""

from pathlib import Path

from hc_rag.data.crag_loader import CRAGLoader
from hc_rag.embeddings.embedding_model import EmbeddingModel, batch_cosine_similarity
import numpy as np


def test_embeddings_with_crag():
    """Test embedding model with a single CRAG query and its documents."""

    print("\n" + "="*80)
    print("Testing Embedding Model with CRAG Dataset")
    print("="*80 + "\n")

    # Step 1: Load CRAG data
    print("Step 1: Loading CRAG Task 1/2 dataset...")
    print("-" * 80)
    loader = CRAGLoader(use_full_html=False)
    queries, all_documents = loader.load_by_tasks(["1_2"])

    print(f"✓ Loaded {len(queries)} queries")
    print(f"✓ Loaded {len(all_documents)} documents total\n")

    # Step 2: Select a single query for testing
    print("Step 2: Selecting a test query...")
    print("-" * 80)

    # Find a query with good search results
    test_query = None
    for q in queries:
        if q.search_results and len(q.search_results) >= 3:
            test_query = q
            break

    if not test_query:
        print("❌ No suitable query found with search results")
        return

    print(f"Selected Query:")
    print(f"  ID: {test_query.query_id}")
    print(f"  Query: {test_query.query}")
    print(f"  Gold Answer: {test_query.answer}")
    print(f"  Domain: {test_query.task}")
    print(f"  Number of documents: {len(test_query.search_results)}\n")

    # Step 3: Extract documents for this query
    print("Step 3: Extracting documents for this query...")
    print("-" * 80)

    from hc_rag.data.crag_loader import CRAGDocument
    test_documents = []
    for idx, result in enumerate(test_query.search_results):
        doc = CRAGDocument.from_dict(result, doc_id=f"{test_query.query_id}_doc_{idx}")
        if doc.text.strip():  # Only non-empty documents
            test_documents.append(doc)

    print(f"✓ Extracted {len(test_documents)} documents\n")

    # Display documents
    for i, doc in enumerate(test_documents):
        print(f"Document {i+1}:")
        print(f"  Title: {doc.title}")
        print(f"  URL: {doc.url}")
        print(f"  Text length: {len(doc.text)} chars")
        print(f"  Preview: {doc.text[:100]}...")
        print()

    # Step 4: Initialize embedding model
    print("\nStep 4: Initializing embedding model...")
    print("-" * 80)
    model = EmbeddingModel(model_name=EmbeddingModel.FAST_MODEL)
    print(f"✓ Model: {model.get_model_name()}")
    print(f"✓ Embedding dimension: {model.get_embedding_dim()}")
    print(f"✓ Device: {model.device}\n")

    # Step 5: Generate embeddings
    print("Step 5: Generating embeddings...")
    print("-" * 80)

    # Embed query
    print("Embedding query...")
    query_embedding = model.embed_query(test_query.query)
    print(f"✓ Query embedding shape: {query_embedding.shape}")

    # Embed documents
    print("Embedding documents...")
    doc_texts = [doc.text for doc in test_documents]
    doc_embeddings = model.embed_documents(doc_texts, show_progress=True)
    print(f"✓ Document embeddings shape: {doc_embeddings.shape}\n")

    # Step 6: Compute similarities
    print("Step 6: Computing similarity scores...")
    print("-" * 80)
    similarities = batch_cosine_similarity(query_embedding, doc_embeddings)
    print(f"✓ Computed {len(similarities)} similarity scores\n")

    # Step 7: Display results
    print("\n" + "="*80)
    print("SIMILARITY RESULTS")
    print("="*80)
    print(f"\nQuery: {test_query.query}")
    print(f"Gold Answer: {test_query.answer}\n")
    print("-" * 80)

    # Sort by similarity
    ranked_indices = np.argsort(similarities)[::-1]

    for rank, idx in enumerate(ranked_indices, 1):
        doc = test_documents[idx]
        score = similarities[idx]

        print(f"\n[Rank {rank}] Similarity Score: {score:.4f}")
        print(f"Document {idx+1}:")
        print(f"  Title: {doc.title}")
        print(f"  URL: {doc.url}")
        print(f"  Text preview: {doc.text[:150]}...")
        print()

    # Step 8: Summary statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Number of documents: {len(test_documents)}")
    print(f"Highest similarity: {np.max(similarities):.4f}")
    print(f"Lowest similarity: {np.min(similarities):.4f}")
    print(f"Mean similarity: {np.mean(similarities):.4f}")
    print(f"Std similarity: {np.std(similarities):.4f}")
    print("\n" + "="*80)
    print("✓ Test completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_embeddings_with_crag()
