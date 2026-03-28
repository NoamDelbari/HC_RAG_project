"""Null Distribution for Higher Criticism."""

import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Set


@dataclass
class NullDistribution:
    """Container for null distribution."""
    similarities: np.ndarray
    mean: float
    std: float
    min_val: float
    max_val: float
    n_samples: int
    # GPD tail fit parameters (optional, for composite p-values)
    gpd_shape: Optional[float] = None      # GPD shape parameter (xi)
    gpd_scale: Optional[float] = None      # GPD scale parameter (sigma)
    gpd_threshold: Optional[float] = None  # Threshold above which GPD applies
    gpd_rate_above: Optional[float] = None # Fraction of null samples above threshold

    def __repr__(self):
        return (f"NullDistribution(n={self.n_samples}, "
                f"mean={self.mean:.4f}, std={self.std:.4f})")

    def save(self, path: str):
        """Save global null distribution to disk."""
        path = Path(path).with_suffix('.pkl')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "NullDistribution":
        """Load global null distribution from disk."""
        path = Path(path).with_suffix('.pkl')
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        # Backward compat: old pickles lack GPD fields
        for attr in ('gpd_shape', 'gpd_scale', 'gpd_threshold', 'gpd_rate_above'):
            if not hasattr(obj, attr):
                object.__setattr__(obj, attr, None)
        return obj


@dataclass
class QueryNullDistributions:
    """Container for per-query null distributions."""
    distributions: Dict[str, NullDistribution]

    def __repr__(self):
        return f"QueryNullDistributions(n_queries={len(self.distributions)})"

    def get(self, query_id: str) -> NullDistribution:
        """Get null distribution for a specific query."""
        return self.distributions[query_id]

    def save(self, path: str):
        """Save to disk."""
        path = Path(path).with_suffix('.pkl')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "QueryNullDistributions":
        """Load from disk."""
        path = Path(path).with_suffix('.pkl')
        with open(path, 'rb') as f:
            return pickle.load(f)


class NegativePairingNull:
    """Build per-query null distributions from verified negative pairs."""

    def __init__(self, vector_db, embedding_model, seed: int = 42):
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.seed = seed
        np.random.seed(seed)
        self.doc_ids_set = set(vector_db.doc_ids)
        self.doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(vector_db.doc_ids)}
        self.doc_embeddings = vector_db.index.reconstruct_n(0, len(vector_db.doc_ids))

    def build(self, queries: List, negatives_per_query: int = 50) -> QueryNullDistributions:
        """Build per-query null distributions."""
        print(f"Building per-query null distributions...")
        print(f"  Queries: {len(queries)}")
        print(f"  Negatives per query: {negatives_per_query}")

        distributions = {}

        for i, query in enumerate(queries):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(queries)} queries")

            # Get relevance for this query
            relevant = self._get_relevant_docs(query)

            # Generate negative pairs for this query
            negative_docs = self._sample_negatives(query, relevant, negatives_per_query)

            # Compute similarities for this query's negatives
            similarities = self._compute_similarities(query.query, negative_docs)

            # Fit null distribution for this query
            distributions[query.query_id] = self._fit(similarities)

        print(f"  Complete: {len(distributions)} per-query null distributions")
        return QueryNullDistributions(distributions)

    def _get_relevant_docs(self, query) -> Set[str]:
        """Get relevant doc IDs for this query."""
        relevant = set()
        for idx, _ in enumerate(query.search_results):
            doc_id = f"{query.query_id}_doc_{idx}"
            if doc_id in self.doc_ids_set:
                relevant.add(doc_id)
        return relevant

    def _sample_negatives(self, query, relevant: Set[str], n: int) -> List[str]:
        """Sample negative (non-relevant) documents for this query."""
        non_relevant = list(self.doc_ids_set - relevant)

        if len(non_relevant) < n:
            return list(np.random.choice(non_relevant, size=n, replace=True))
        else:
            return list(np.random.choice(non_relevant, size=n, replace=False))

    def _compute_similarities(self, query_text: str, doc_ids: List[str]) -> np.ndarray:
        """Compute similarities between query and negative documents."""
        # Embed query
        query_embedding = self.embedding_model.embed([query_text], show_progress=False)[0]

        # Compute similarities
        similarities = np.empty(len(doc_ids), dtype=np.float32)
        for i, doc_id in enumerate(doc_ids):
            d_idx = self.doc_to_idx[doc_id]
            similarities[i] = np.dot(query_embedding, self.doc_embeddings[d_idx])

        return similarities

    def _fit(self, similarities: np.ndarray) -> NullDistribution:
        """Create NullDistribution from similarity scores."""
        return NullDistribution(
            similarities=similarities,
            mean=float(np.mean(similarities)),
            std=float(np.std(similarities)),
            min_val=float(np.min(similarities)),
            max_val=float(np.max(similarities)),
            n_samples=len(similarities)
        )


class RandomVectorNull:
    """Generate null distribution from random vectors (for testing only)."""

    def __init__(self, embedding_dim: int, seed: int = 42):
        self.embedding_dim = embedding_dim
        self.seed = seed
        np.random.seed(seed)

    def build(self, n_samples: int = 10000) -> NullDistribution:
        """Generate null distribution."""
        vectors = np.random.randn(n_samples, self.embedding_dim).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        idx1 = np.random.randint(0, n_samples, size=n_samples)
        idx2 = np.random.randint(0, n_samples, size=n_samples)
        similarities = np.sum(vectors[idx1] * vectors[idx2], axis=1)

        return self._fit(similarities)

    def _fit(self, similarities: np.ndarray) -> NullDistribution:
        return NullDistribution(
            similarities=similarities,
            mean=float(np.mean(similarities)),
            std=float(np.std(similarities)),
            min_val=float(np.min(similarities)),
            max_val=float(np.max(similarities)),
            n_samples=len(similarities)
        )


class GaussianNull:
    """Generate null distribution using Gaussian approximation (for testing only)."""

    def __init__(self, embedding_dim: int, seed: int = 42):
        self.embedding_dim = embedding_dim
        self.seed = seed
        np.random.seed(seed)

    def build(self, n_samples: int = 10000) -> NullDistribution:
        """Generate null distribution."""
        std = 1.0 / np.sqrt(self.embedding_dim)
        similarities = np.random.normal(0, std, n_samples)
        similarities = np.clip(similarities, -1, 1)

        return self._fit(similarities)

    def _fit(self, similarities: np.ndarray) -> NullDistribution:
        return NullDistribution(
            similarities=similarities,
            mean=float(np.mean(similarities)),
            std=float(np.std(similarities)),
            min_val=float(np.min(similarities)),
            max_val=float(np.max(similarities)),
            n_samples=len(similarities)
        )
