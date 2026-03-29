"""Build pool-matched global null distribution.

Fixes the calibration mismatch between the original null (built from the full
corpus) and HC inference (which sees only the top-K FAISS candidates).  This
script builds the null using the exact same procedure HC uses at inference:
FAISS top-K retrieval followed by z-scoring from the bottom fraction of the
candidate pool.

Usage:
    python -m experiments.scripts.pool_null.build_pool_null \
        --config experiments/configs/amazon_compound.yaml [--force]
"""

import argparse
import numpy as np
from pathlib import Path

from hc_rag.embeddings.embedding_model import create_embedding_model
from hc_rag.embeddings.vector_database import VectorDatabase
from hc_rag.hc.null_distribution import NullDistribution

from experiments.lib.config import load_config
from experiments.lib.registry import get_adapter


def build_pool_matched_null(queries, qrels, vector_db, embedding_model,
                            pool_size, z_score_fraction):
    """Build global null from pool-matched z-score standardized non-relevant similarities.

    For each query the procedure mirrors what HC does at inference:
      1. Embed the query, FAISS search for top ``pool_size`` candidates.
      2. Sort similarities descending; take the bottom ``z_score_fraction``
         to estimate mu_est and sigma_est (relevant docs may be included --
         this matches inference where ground truth is unavailable).
      3. Z-score ALL candidates using mu_est / sigma_est.
      4. Identify non-relevant candidates via ground-truth qrels.
      5. Add ONLY non-relevant z-scores to the global pool.

    The concatenated pool of non-relevant z-scores is returned as a
    ``NullDistribution``.
    """
    all_z_scores = []
    skipped = 0

    print("Building pool-matched null distribution...")
    print(f"  Queries: {len(queries)}")
    print(f"  Pool size (FAISS top-K): {pool_size}")
    print(f"  Z-score fraction: {z_score_fraction}")
    print(f"  Docs in DB: {vector_db.get_num_documents()}")

    for i, query in enumerate(queries):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(queries)} queries")

        # Step 1: embed query and retrieve top-K from FAISS
        query_emb = embedding_model.embed([query.text], show_progress=False)[0]
        candidates = vector_db.search(query_emb, k=pool_size)

        if len(candidates) == 0:
            skipped += 1
            continue

        # Step 2: sort similarities descending, estimate null params from bottom fraction
        sims = np.array([c.similarity for c in candidates])
        sorted_desc = np.sort(sims)[::-1]
        n = len(sorted_desc)
        n_null_est = max(int(n * z_score_fraction), 10)

        bottom = sorted_desc[-n_null_est:]  # lowest similarities in pool
        mu_est = np.mean(bottom)
        sigma_est = np.std(bottom)

        if sigma_est < 1e-12:
            skipped += 1
            continue

        # Step 3: z-score ALL candidates (includes relevant -- matches inference)
        z_scores = (sims - mu_est) / sigma_est

        # Step 4: identify non-relevant candidates using qrels
        relevant = qrels.get(query.query_id, set())
        candidate_ids = [c.doc_id for c in candidates]
        nonrel_mask = np.array([cid not in relevant for cid in candidate_ids])

        # Step 5: collect only non-relevant z-scores
        nonrel_z = z_scores[nonrel_mask]
        if len(nonrel_z) > 0:
            all_z_scores.append(nonrel_z)

    if skipped > 0:
        print(f"  Skipped {skipped} queries (empty results or zero std)")

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
    parser = argparse.ArgumentParser(
        description="Build pool-matched global null distribution"
    )
    parser.add_argument("--config", required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if output exists")
    args = parser.parse_args()

    config = load_config(args.config)
    import experiments.datasets  # noqa: F401  -- registers dataset adapters
    adapter = get_adapter(config.dataset.name)

    artifacts_dir = Path(config.dataset.artifacts_dir)
    null_dir = artifacts_dir / "pool_null"
    null_path = null_dir / "global_pool_null"
    if Path(str(null_path) + ".pkl").exists() and not args.force:
        print(f"Pool-matched null already exists at {null_path}.pkl. "
              "Use --force to rebuild.")
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

    null_dist = build_pool_matched_null(
        queries=queries,
        qrels=qrels,
        vector_db=vector_db,
        embedding_model=model,
        pool_size=config.hc.pool_size,
        z_score_fraction=config.null.z_score_fraction,
    )

    null_dir.mkdir(parents=True, exist_ok=True)
    null_dist.save(str(null_path))

    print(f"\nPool-matched null saved to {null_path}.pkl")
    print(f"  Mean: {null_dist.mean:.4f}, Std: {null_dist.std:.4f}")
    print(f"  Samples: {null_dist.n_samples:,}")


if __name__ == "__main__":
    main()
