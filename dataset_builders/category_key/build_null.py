"""
Build Null Distributions for Category-Key Dataset

For each query, sample non-relevant documents, compute similarities,
and build per-query null distributions using qrels-based ground truth.
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Set, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embeddings.embedding_model import EmbeddingModel
from embeddings.vector_database import VectorDatabase
from hc.null_distribution import (
    NullDistribution, QueryNullDistributions, NegativePairingNull
)
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_category_key_dataset


class CategoryKeyNegativePairingNull(NegativePairingNull):
    """
    Null distribution builder that uses qrels dict for ground truth
    instead of CRAGQuery.search_results.
    """

    def __init__(self, vector_db, embedding_model, qrels: Dict[str, Set[str]], seed: int = 42):
        """
        Args:
            vector_db: Vector database with indexed documents
            embedding_model: Embedding model for query encoding
            qrels: Dict mapping query_id -> Set of relevant doc_ids
            seed: Random seed
        """
        # Initialize parent (sets up doc_ids_set, doc_to_idx, doc_embeddings)
        super().__init__(vector_db, embedding_model, seed)
        self.qrels = qrels

    def build(self, queries, negatives_per_query: int = 0) -> QueryNullDistributions:
        """
        Build per-query null distributions using qrels for relevance.

        Args:
            queries: List of CategoryKeyQuery objects (need .query_id and .text)
            negatives_per_query: Number of negative samples per query.
                                 0 = use ALL non-relevant docs (recommended for small corpus).

        Returns:
            QueryNullDistributions
        """
        print(f"Building per-query null distributions (CategoryKey)...")
        print(f"  Queries: {len(queries)}")
        print(f"  Negatives per query: {'ALL' if negatives_per_query == 0 else negatives_per_query}")
        print(f"  Total docs in DB: {len(self.doc_ids_set)}")

        distributions = {}

        for i, query in enumerate(queries):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(queries)} queries")

            # Get relevant docs from qrels
            relevant = self.qrels.get(query.query_id, set())

            # Get negatives: all non-relevant or sampled subset
            non_relevant = list(self.doc_ids_set - relevant)
            if negatives_per_query == 0 or negatives_per_query >= len(non_relevant):
                negative_docs = non_relevant
            else:
                negative_docs = list(np.random.choice(
                    non_relevant, size=negatives_per_query, replace=False
                ))

            # Compute similarities between query and negative docs
            query_embedding = self.embedding_model.embed([query.text], show_progress=False)[0]
            similarities = np.empty(len(negative_docs), dtype=np.float32)
            for j, doc_id in enumerate(negative_docs):
                d_idx = self.doc_to_idx[doc_id]
                similarities[j] = np.dot(query_embedding, self.doc_embeddings[d_idx])

            # Fit null distribution
            distributions[query.query_id] = self._fit(similarities)

        print(f"  Complete: {len(distributions)} per-query null distributions")
        return QueryNullDistributions(distributions)


def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / "datasets" / "category_key"
    db_path = str(output_dir / "category_key_vector_db")
    null_path = str(output_dir / "null_distributions" / "category_key_per_query_null")

    print("=" * 70)
    print("Build Null Distributions for Category-Key Dataset")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    queries, qrels = load_category_key_dataset(str(output_dir))

    # Load vector DB
    print("\nLoading vector database...")
    vector_db = VectorDatabase.load(db_path)
    print(f"  Documents: {vector_db.get_num_documents()}")

    # Load embedding model (must match what was used to build DB)
    print("\nLoading embedding model...")
    model = EmbeddingModel(
        model_name=EmbeddingModel.BGE_MODEL,
        normalize_embeddings=True
    )

    # Build null distributions
    print("\nBuilding null distributions...")
    null_builder = CategoryKeyNegativePairingNull(
        vector_db=vector_db,
        embedding_model=model,
        qrels=qrels,
        seed=42
    )

    # Use ALL non-relevant docs (0 = all) for best null quality on small corpus
    null_dists = null_builder.build(queries, negatives_per_query=0)

    # Save
    print(f"\nSaving null distributions to {null_path}...")
    null_dists.save(null_path)

    # Print summary
    print("\nNull distribution summary:")
    means = [d.mean for d in null_dists.distributions.values()]
    stds = [d.std for d in null_dists.distributions.values()]
    print(f"  Queries: {len(null_dists.distributions)}")
    print(f"  Mean similarity: {np.mean(means):.4f} +/- {np.std(means):.4f}")
    print(f"  Mean std: {np.mean(stds):.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
