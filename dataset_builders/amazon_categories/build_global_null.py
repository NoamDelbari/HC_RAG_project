"""
Build Global Null Distribution for Amazon Categories Dataset (Z-Score)

For each query, samples 5000 non-relevant docs (corpus is ~100K),
z-score standardizes the cosine similarities (z = (sim - mean_q) / std_q),
then pools all z-scores across queries into a single null distribution.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embeddings.embedding_model import EmbeddingModel
from embeddings.vector_database import VectorDatabase
from hc.null_distribution import NullDistribution

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_amazon_dataset


def build_global_null(queries, qrels, vector_db, embedding_model, sample_size=5000):
    """
    Build a single global null distribution from z-score standardized
    non-relevant similarities across all queries.

    For each query, samples up to `sample_size` non-relevant docs,
    computes z = (sim - mean_q) / std_q, then pools all z-scores.

    Args:
        queries: List of AmazonQuery objects
        qrels: Dict mapping query_id -> Set of relevant doc_ids
        vector_db: VectorDatabase with indexed documents
        embedding_model: EmbeddingModel for query encoding
        sample_size: Number of non-relevant docs to sample per query

    Returns:
        NullDistribution with pooled z-scores
    """
    doc_ids_set = set(vector_db.doc_ids)
    doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(vector_db.doc_ids)}
    doc_embeddings = vector_db.index.reconstruct_n(0, len(vector_db.doc_ids))

    all_z_scores = []
    skipped = 0
    rng = np.random.RandomState(42)

    print(f"Building global z-score null distribution...")
    print(f"  Queries: {len(queries)}")
    print(f"  Total docs in DB: {len(doc_ids_set)}")
    print(f"  Sampling {sample_size} non-relevant docs per query")

    # Precompute relevant mask per query for efficiency
    all_doc_ids = vector_db.doc_ids

    for i, query in enumerate(queries):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(queries)} queries")

        # Get relevant docs from qrels
        relevant = qrels.get(query.query_id, set())

        # Compute similarities against ALL docs (vectorized)
        query_embedding = embedding_model.embed([query.text], show_progress=False)[0]
        all_sims = np.dot(doc_embeddings, query_embedding)

        # Get non-relevant similarities (ALL of them for accurate mu/sigma)
        is_relevant = np.array([did in relevant for did in all_doc_ids])
        nonrel_sims = all_sims[~is_relevant]

        # Z-score standardize using statistics from ALL non-relevant docs
        mean_q = np.mean(nonrel_sims)
        std_q = np.std(nonrel_sims)
        if std_q < 1e-12:
            skipped += 1
            continue

        z_all = (nonrel_sims - mean_q) / std_q

        # Sample z-scores to keep memory manageable
        if len(z_all) > sample_size:
            sampled_idx = rng.choice(len(z_all), size=sample_size, replace=False)
            z_scores = z_all[sampled_idx]
        else:
            z_scores = z_all
        all_z_scores.append(z_scores)

    if skipped > 0:
        print(f"  Skipped {skipped} queries with zero std")

    # Pool all z-scores into one array
    pooled = np.concatenate(all_z_scores)

    print(f"  Total pooled z-score pairs: {len(pooled)}")

    # Build single NullDistribution
    null_dist = NullDistribution(
        similarities=pooled,
        mean=float(np.mean(pooled)),
        std=float(np.std(pooled)),
        min_val=float(np.min(pooled)),
        max_val=float(np.max(pooled)),
        n_samples=len(pooled),
    )

    return null_dist


def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / "datasets" / "amazon_categories"
    db_path = str(output_dir / "amazon_categories_vector_db")
    null_dir = output_dir / "null_distributions"
    null_dir.mkdir(parents=True, exist_ok=True)
    null_path = str(null_dir / "amazon_global_null")

    print("=" * 70)
    print("Build Global Z-Score Null Distribution for Amazon Categories Dataset")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    queries, qrels = load_amazon_dataset(str(output_dir))

    # Load vector DB
    print("\nLoading vector database...")
    vector_db = VectorDatabase.load(db_path)
    print(f"  Documents: {vector_db.get_num_documents()}")

    # Load embedding model (must match what was used to build DB)
    print("\nLoading embedding model...")
    model = EmbeddingModel(
        model_name=EmbeddingModel.BGE_MODEL,
        normalize_embeddings=True,
    )

    # Build global z-score null distribution
    print("\nBuilding global z-score null distribution...")
    null_dist = build_global_null(queries, qrels, vector_db, model)

    # Save
    print(f"\nSaving global z-score null distribution to {null_path}...")
    null_dist.save(null_path)

    # Print summary
    print("\nGlobal z-score null distribution summary:")
    print(f"  Total pooled z-score pairs: {null_dist.n_samples}")
    print(f"  Mean z-score: {null_dist.mean:.4f}")
    print(f"  Std z-score: {null_dist.std:.4f}")
    print(f"  Min z-score: {null_dist.min_val:.4f}")
    print(f"  Max z-score: {null_dist.max_val:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
