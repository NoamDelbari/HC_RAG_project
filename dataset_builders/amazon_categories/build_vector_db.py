"""
Build Vector Database for Amazon Categories Dataset

Embeds all documents using BGE model and builds FAISS IndexFlatIP.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embeddings.embedding_model import EmbeddingModel
from embeddings.vector_database import VectorDatabase


def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / "datasets" / "amazon_categories"
    db_path = output_dir / "amazon_categories_vector_db"

    print("=" * 70)
    print("Build Vector DB for Amazon Categories Dataset")
    print("=" * 70)

    # Load corpus
    print("\nLoading corpus...")
    corpus_df = pd.read_csv(output_dir / "corpus.csv")
    documents = corpus_df.to_dict("records")

    doc_ids = [d["doc_id"] for d in documents]
    doc_texts = [d["text"] for d in documents]
    print(f"  Loaded {len(documents):,} documents")

    # Initialize embedding model (BGE)
    print("\nInitializing BGE embedding model...")
    model = EmbeddingModel(
        model_name=EmbeddingModel.BGE_MODEL,
        normalize_embeddings=True,
    )
    print(f"  Model: {model.get_model_name()}")
    print(f"  Embedding dim: {model.get_embedding_dim()}")

    # Embed all documents
    print("\nEmbedding documents...")
    embeddings = model.embed_documents(doc_texts, show_progress=True)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Build vector database
    print("\nBuilding FAISS index...")
    vector_db = VectorDatabase(
        embedding_dim=model.get_embedding_dim(),
        embedding_model_name=model.get_model_name(),
    )

    # Add metadata for each document
    metadata = []
    for doc in documents:
        meta = {"type": doc.get("type", "unknown")}
        if doc.get("subcategory"):
            meta["subcategory"] = doc["subcategory"]
        if doc.get("k_bucket"):
            meta["k_bucket"] = doc["k_bucket"]
        metadata.append(meta)

    vector_db.add_documents(doc_ids, embeddings, metadata)

    # Save
    print(f"\nSaving vector database to {db_path}...")
    vector_db.save(str(db_path))

    print(f"\nDone! Vector DB has {vector_db.get_num_documents():,} documents")
    print(f"  FAISS index: {db_path}.faiss")
    print(f"  Metadata: {db_path}.metadata.pkl")


if __name__ == "__main__":
    main()
