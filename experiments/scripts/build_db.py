"""Build FAISS vector database from corpus CSV.

Usage: python -m experiments.scripts.build_db --config experiments/configs/amazon_compound.yaml
"""

import argparse
from pathlib import Path

from hc_rag.embeddings.embedding_model import create_embedding_model
from hc_rag.embeddings.vector_database import VectorDatabase

from experiments.lib.config import load_config
from experiments.lib.registry import get_adapter


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector database")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = parser.parse_args()

    config = load_config(args.config)
    import experiments.datasets  # noqa: F401
    adapter = get_adapter(config.dataset.name)

    artifacts_dir = Path(config.dataset.artifacts_dir)
    db_path = artifacts_dir / "vector_db"
    if (Path(str(db_path) + ".faiss")).exists() and not args.force:
        print(f"Vector DB already exists at {db_path}. Use --force to rebuild.")
        return

    print(f"Loading corpus from {config.dataset.data_dir}...")
    corpus = adapter.load_corpus(config.dataset.data_dir)
    print(f"  Loaded {len(corpus):,} documents")

    doc_ids = list(corpus.keys())
    doc_texts = [str(corpus[did]["text"]) for did in doc_ids]

    print(f"\nInitializing embedding model: {config.embedding.model}")
    model = create_embedding_model(
        config.embedding.model,
        normalize_embeddings=config.embedding.normalize,
    )
    print(f"  Embedding dim: {model.get_embedding_dim()}")

    print("\nEmbedding documents...")
    embeddings = model.embed_documents(doc_texts, show_progress=True)
    print(f"  Embeddings shape: {embeddings.shape}")

    print("\nBuilding FAISS index...")
    vector_db = VectorDatabase(
        embedding_dim=model.get_embedding_dim(),
        embedding_model_name=model.get_model_name(),
    )

    metadata = []
    for did in doc_ids:
        doc = corpus[did]
        meta = {k: v for k, v in doc.items() if k not in ("doc_id", "text")}
        metadata.append(meta)

    vector_db.add_documents(doc_ids, embeddings, metadata)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving vector database to {db_path}...")
    vector_db.save(str(db_path))

    print(f"\nDone! Vector DB has {vector_db.get_num_documents():,} documents")


if __name__ == "__main__":
    main()
