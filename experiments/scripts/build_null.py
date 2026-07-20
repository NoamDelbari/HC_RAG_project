"""Build global z-score null distribution from vector DB.

Usage: python -m experiments.scripts.build_null --config experiments/configs/amazon_compound.yaml
"""

import argparse
import numpy as np
from pathlib import Path

from hc_rag.embeddings.embedding_model import create_embedding_model
from hc_rag.embeddings.vector_database import VectorDatabase
from hc_rag.hc.null_distribution import NullDistribution

from experiments.lib.config import load_config
from experiments.lib.registry import get_adapter


def build_global_null(queries, qrels, vector_db, embedding_model, sample_size=5000):
    """Build global null from z-score standardized non-relevant similarities."""
    doc_embeddings = vector_db.index.reconstruct_n(0, len(vector_db.doc_ids))
    all_doc_ids = vector_db.doc_ids
    rng = np.random.RandomState(42)
    all_z_scores = []
    skipped = 0

    print(f"Building global z-score null distribution...")
    print(f"  Queries: {len(queries)}")
    print(f"  Docs in DB: {len(all_doc_ids)}")
    print(f"  Sampling {sample_size} non-relevant docs per query")

    for i, query in enumerate(queries):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(queries)} queries")

        relevant = qrels.get(query.query_id, set())
        query_emb = embedding_model.embed([query.text], show_progress=False)[0]
        all_sims = np.dot(doc_embeddings, query_emb)

        is_relevant = np.array([did in relevant for did in all_doc_ids])
        nonrel_sims = all_sims[~is_relevant]

        mean_q = np.mean(nonrel_sims)
        std_q = np.std(nonrel_sims)
        if std_q < 1e-12:
            skipped += 1
            continue

        z_all = (nonrel_sims - mean_q) / std_q

        if len(z_all) > sample_size:
            idx = rng.choice(len(z_all), size=sample_size, replace=False)
            z_scores = z_all[idx]
        else:
            z_scores = z_all

        all_z_scores.append(z_scores)

    if skipped > 0:
        print(f"  Skipped {skipped} queries with zero std")

    pooled = np.concatenate(all_z_scores)
    print(f"  Total pooled z-scores: {len(pooled):,}")

    return NullDistribution(
        similarities=pooled,
        mean=float(np.mean(pooled)),
        std=float(np.std(pooled)),
        min_val=float(np.min(pooled)),
        max_val=float(np.max(pooled)),
        n_samples=len(pooled),
    )


def main():
    parser = argparse.ArgumentParser(description="Build global null distribution")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = parser.parse_args()

    config = load_config(args.config)
    # Note: config.null.z_score_fraction is an inference-time parameter used by
    # HCRetrieval to estimate mu/sigma from the bottom fraction of candidates.
    # During null building, we use ground-truth qrels to identify non-relevant docs,
    # so the fraction parameter does not apply here.
    import experiments.datasets  # noqa: F401
    adapter = get_adapter(config.dataset.name)

    artifacts_dir = Path(config.dataset.artifacts_dir)
    null_path = artifacts_dir / "global_null"
    if Path(str(null_path) + ".pkl").exists() and not args.force:
        print(f"Global null already exists at {null_path}. Use --force to rebuild.")
        return

    queries, qrels, _ = adapter.load_dataset(config.dataset.data_dir)

    db_path = str(artifacts_dir / "vector_db")
    print(f"Loading vector database from {db_path}...")
    vector_db = VectorDatabase.load(db_path)
    print(f"  Documents: {vector_db.get_num_documents():,}")

    print(f"\nLoading embedding model: {config.embedding.model}")
    model = create_embedding_model(
        config.embedding.model,
        normalize_embeddings=config.embedding.normalize,
    )

    null_dist = build_global_null(queries, qrels, vector_db, model)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    null_dist.save(str(null_path))

    print(f"\nGlobal null saved to {null_path}")
    print(f"  Mean: {null_dist.mean:.4f}, Std: {null_dist.std:.4f}")
    print(f"  Samples: {null_dist.n_samples:,}")


if __name__ == "__main__":
    main()
