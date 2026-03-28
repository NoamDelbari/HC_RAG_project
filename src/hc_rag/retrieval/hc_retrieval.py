"""
HC-based Adaptive Retrieval

Adaptive retrieval strategy using Higher Criticism statistics to determine
the optimal number of documents to retrieve for each query.

Supports both full document and chunked document retrieval with automatic aggregation.
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.vector_database import VectorDatabase, SearchResult
from embeddings.embedding_model import EmbeddingModel
from hc.higher_criticism import HigherCriticism
from hc.null_distribution import NullDistribution, QueryNullDistributions
from retrieval.baseline_retrieval import RetrievalOutput
from retrieval.chunked_retrieval import ChunkedRetrievalMixin

logger = logging.getLogger(__name__)


class HCRetrieval(ChunkedRetrievalMixin):
    """
    Higher Criticism based adaptive retrieval.

    Uses HC statistics to adaptively determine the number of relevant documents
    for each query, rather than using a fixed k.

    Supports chunked retrieval: if chunk_to_doc_mapping is provided,
    retrieves chunks and aggregates them to parent documents after HC filtering.
    """

    def __init__(
        self,
        vector_db: VectorDatabase,
        query_null_distributions: Optional[QueryNullDistributions] = None,
        gamma: float = 0.1,
        max_candidates: int = 50,
        min_hc: float = 0.0,
        allow_empty: bool = True,
        embedding_model: Optional[EmbeddingModel] = None,
        chunk_to_doc_mapping: Optional[Dict[str, str]] = None,
        aggregation: str = "max_score",
        top_chunks_multiplier: int = 10,
        max_k: Optional[int] = None,
        global_null_distribution: Optional[NullDistribution] = None,
        use_zscore: bool = False,
        parametric: bool = False,
        z_cap: Optional[float] = None,
        method: str = "hc",
        gpd_composite: bool = False
    ):
        """
        Initialize HC-based retriever.

        Args:
            vector_db: Vector database containing indexed documents or chunks
            query_null_distributions: Per-query null distributions (used when global_null_distribution is not set)
            gamma: Fraction of top scores to search for HC maximum (0 < gamma <= 1)
            max_candidates: Maximum number of candidates to fetch before HC filtering
            min_hc: Minimum HC statistic to accept any documents (0 = disabled)
            allow_empty: If True, can return empty set when HC < min_hc
            embedding_model: Optional embedding model for query text → embedding
            chunk_to_doc_mapping: Optional dict mapping chunk_id -> parent_doc_id
                                 If provided, enables chunked retrieval mode
            aggregation: Chunk score aggregation strategy: "max_score", "mean_score", or "sum_score"
            top_chunks_multiplier: Multiplier for max_candidates when retrieving chunks
                                  (retrieves max_candidates * multiplier chunks)
            max_k: Maximum number of documents to return (None = no limit, use HC threshold only)
                   Returns min(HC_threshold_k, max_k) documents
            global_null_distribution: A single NullDistribution used for ALL queries.
                                     If set, overrides per-query null distributions.
            use_zscore: If True, use z-score standardized p-values for HC computation.
                       Defaults to True when global_null_distribution is set.
            parametric: If True and use_zscore=True, use Gaussian CDF instead of empirical
                       CDF for p-value computation. Eliminates p-value floor saturation.
            z_cap: If not None, clip z-scores at this upper bound before computing p-values.
                   Only used when use_zscore=True.
            method: Statistical method for thresholding. "hc" for Higher Criticism (default),
                    "berk_jones" for Berk-Jones statistic. BJ avoids the p-value floor
                    pathology of HC by using KL divergence instead of variance normalization.
            gpd_composite: If True, use GPD composite p-values (empirical CDF + GPD tail).
                          Requires null_distribution to have GPD parameters fitted.
        """
        self.vector_db = vector_db
        self.query_null_distributions = query_null_distributions
        self.gamma = gamma
        self.max_candidates = max_candidates
        self.min_hc = min_hc
        self.allow_empty = allow_empty
        self.embedding_model = embedding_model
        self.chunk_to_doc_mapping = chunk_to_doc_mapping
        self.aggregation = aggregation
        self.top_chunks_multiplier = top_chunks_multiplier
        self.max_k = max_k
        self.global_null_distribution = global_null_distribution
        self.use_zscore = use_zscore
        self.parametric = parametric
        self.z_cap = z_cap
        self.method = method
        self.gpd_composite = gpd_composite

        # Validate: must have at least one null distribution source
        if query_null_distributions is None and global_null_distribution is None:
            raise ValueError(
                "Must provide either query_null_distributions or global_null_distribution (or both)"
            )

        # Validate parameters
        if not (0 < gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {max_candidates}")
        if min_hc < 0:
            raise ValueError(f"min_hc must be >= 0, got {min_hc}")
        if chunk_to_doc_mapping is not None and aggregation not in ["max_score", "mean_score", "sum_score"]:
            raise ValueError(f"Invalid aggregation: {aggregation}. Must be 'max_score', 'mean_score', or 'sum_score'")
        if max_k is not None and max_k < 1:
            raise ValueError(f"max_k must be >= 1 or None, got {max_k}")
        if method not in ("hc", "berk_jones"):
            raise ValueError(f"method must be 'hc' or 'berk_jones', got '{method}'")

        logger.info(f"HCRetrieval initialized:")
        logger.info(f"  gamma={gamma}")
        logger.info(f"  max_candidates={max_candidates}")
        logger.info(f"  min_hc={min_hc}")
        logger.info(f"  allow_empty={allow_empty}")
        logger.info(f"  max_k={max_k if max_k else 'unlimited'}")
        logger.info(f"  method={self.method}")
        logger.info(f"  use_zscore={self.use_zscore}")
        logger.info(f"  parametric={self.parametric}")
        if global_null_distribution is not None:
            logger.info(f"  Global null distribution: {global_null_distribution}")
        if query_null_distributions is not None:
            logger.info(f"  Per-query null distributions: {len(query_null_distributions.distributions)} queries")
        logger.info(f"  Vector DB contains {vector_db.get_num_documents()} {'chunks' if chunk_to_doc_mapping else 'documents'}")
        if chunk_to_doc_mapping:
            logger.info(f"  Chunked mode: top_chunks={max_candidates * top_chunks_multiplier}, aggregation={aggregation}")

    @classmethod
    def from_database_path(
        cls,
        db_path: str,
        query_null_distributions: Optional[QueryNullDistributions] = None,
        gamma: float = 0.1,
        max_candidates: int = 50,
        min_hc: float = 0.0,
        allow_empty: bool = True,
        embedding_model: Optional[EmbeddingModel] = None,
        aggregation: str = "max_score",
        top_chunks_multiplier: int = 10,
        max_k: Optional[int] = None,
        global_null_distribution: Optional[NullDistribution] = None,
        use_zscore: bool = False,
        parametric: bool = False,
        z_cap: Optional[float] = None,
        method: str = "hc",
        gpd_composite: bool = False
    ) -> "HCRetrieval":
        """
        Load HC retrieval system from database path.

        Automatically detects if the database is chunked (has .chunk_mapping.pkl file)
        and enables chunked retrieval mode accordingly.

        Args:
            db_path: Path to database (without extension)
            query_null_distributions: Per-query null distributions (used when global_null_distribution is not set)
            gamma: Fraction of top scores to search for HC maximum
            max_candidates: Maximum number of candidates before HC filtering
            min_hc: Minimum HC statistic threshold
            allow_empty: Allow empty result sets
            embedding_model: Optional embedding model
            aggregation: Aggregation strategy for chunked mode
            top_chunks_multiplier: Multiplier for retrieving chunks
            max_k: Maximum number of documents to return (None = unlimited)
            global_null_distribution: A single NullDistribution used for ALL queries.
                                     If set, overrides per-query null distributions.
            use_zscore: If True, use z-score standardized p-values for HC computation.
            parametric: If True and use_zscore=True, use Gaussian CDF instead of
                       empirical CDF for p-value computation.
            z_cap: If not None, clip z-scores at this upper bound.
            method: Statistical method for thresholding. "hc" or "berk_jones".
            gpd_composite: If True, use GPD composite p-values.

        Returns:
            HCRetrieval instance
        """
        # Load vector database
        vector_db = VectorDatabase.load(db_path)

        # Load chunk mapping (if exists)
        chunk_to_doc_mapping = cls._load_chunk_mapping(db_path)

        if chunk_to_doc_mapping is not None:
            # Chunked database detected
            logger.info(f"Loaded chunked database from {db_path}")
            logger.info(f"  Chunks: {vector_db.get_num_documents()}")
        else:
            # Full document database
            logger.info(f"Loaded full document database from {db_path}")
            logger.info(f"  Documents: {vector_db.get_num_documents()}")

        return cls(
            vector_db=vector_db,
            query_null_distributions=query_null_distributions,
            gamma=gamma,
            max_candidates=max_candidates,
            min_hc=min_hc,
            allow_empty=allow_empty,
            embedding_model=embedding_model,
            chunk_to_doc_mapping=chunk_to_doc_mapping,
            aggregation=aggregation,
            top_chunks_multiplier=top_chunks_multiplier,
            max_k=max_k,
            global_null_distribution=global_null_distribution,
            use_zscore=use_zscore,
            parametric=parametric,
            z_cap=z_cap,
            method=method,
            gpd_composite=gpd_composite
        )

    def retrieve(
        self,
        query_id: str,
        query_embedding: np.ndarray
    ) -> RetrievalOutput:
        """
        Retrieve documents using HC-based adaptive thresholding.

        If chunk_to_doc_mapping is provided, retrieves chunks, applies HC filtering,
        then aggregates to documents. Otherwise, retrieves documents directly.

        Args:
            query_id: Unique query identifier
            query_embedding: Query embedding vector

        Returns:
            RetrievalOutput with adaptively selected documents and HC metadata
        """
        # Determine number of candidates to retrieve
        if self.chunk_to_doc_mapping is not None:
            # Chunked mode: retrieve more chunks
            num_candidates = self.max_candidates * self.top_chunks_multiplier
        else:
            # Full document mode
            num_candidates = self.max_candidates

        # Step 1: Fetch candidates from vector DB
        candidates = self.vector_db.search(query_embedding, k=num_candidates)

        # Handle case where DB has fewer candidates
        if len(candidates) == 0:
            logger.debug(f"Query {query_id}: No candidates found")
            return RetrievalOutput(
                query_id=query_id,
                retrieved_ids=[],
                retrieved_scores=[],
                k=0,
                method="hc",
                threshold=np.inf
            )

        # Extract similarities
        candidate_scores = np.array([c.similarity for c in candidates])

        # Step 2: Get null distribution (global overrides per-query)
        if self.global_null_distribution is not None:
            null_dist = self.global_null_distribution
        else:
            null_dist = self.query_null_distributions.get(query_id)
        hc_module = HigherCriticism(null_distribution=null_dist)

        # Step 3: Compute threshold using query-specific null distribution
        hc_result = hc_module.compute_hc_threshold(
            similarities=candidate_scores,
            gamma=self.gamma,
            min_hc=self.min_hc,
            allow_empty=self.allow_empty,
            use_zscore=self.use_zscore,
            parametric=self.parametric,
            z_cap=self.z_cap,
            method=self.method,
            gpd_composite=self.gpd_composite
        )

        # Step 4: Filter candidates by HC threshold
        if hc_result.k == 0 or np.isinf(hc_result.threshold):
            # Empty result (pure noise detected)
            logger.debug(
                f"Query {query_id}: HC returned empty set "
                f"(HC={hc_result.hc_statistic:.3f}, threshold={hc_result.threshold})"
            )
            retrieved_ids = []
            retrieved_scores = []
            method = f"hc_{self.aggregation}" if self.chunk_to_doc_mapping else "hc"
        else:
            # Apply max_k cap: return min(hc_k, max_k) candidates
            effective_k = hc_result.k
            if self.max_k is not None:
                effective_k = min(hc_result.k, self.max_k)

            # Select top k candidates (already sorted by similarity)
            selected_candidates = candidates[:effective_k]

            if self.chunk_to_doc_mapping is not None:
                # Chunked mode: aggregate chunks to documents
                # Use max_k as limit if set, otherwise use max_candidates
                doc_limit = self.max_k if self.max_k else self.max_candidates
                retrieved_ids, retrieved_scores = self._aggregate_chunks_to_documents(
                    selected_candidates, doc_limit
                )
                method = f"hc_{self.aggregation}"

                logger.debug(
                    f"Query {query_id}: Retrieved {hc_result.k} chunks (capped to {effective_k}) -> {len(retrieved_ids)} docs "
                    f"(HC={hc_result.hc_statistic:.3f}, threshold={hc_result.threshold:.3f}, max_k={self.max_k})"
                )
            else:
                # Full document mode: use filtered candidates directly
                retrieved_ids = [c.doc_id for c in selected_candidates]
                retrieved_scores = [c.similarity for c in selected_candidates]
                method = "hc"

                logger.debug(
                    f"Query {query_id}: Retrieved {effective_k} docs (HC selected {hc_result.k}) "
                    f"(HC={hc_result.hc_statistic:.3f}, threshold={hc_result.threshold:.3f}, max_k={self.max_k})"
                )

        # Final cap on returned documents (applies to both chunked and non-chunked)
        if self.max_k is not None and len(retrieved_ids) > self.max_k:
            retrieved_ids = retrieved_ids[:self.max_k]
            retrieved_scores = retrieved_scores[:self.max_k]

        return RetrievalOutput(
            query_id=query_id,
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            k=len(retrieved_ids),
            method=method,
            threshold=float(hc_result.threshold)
        )

    def retrieve_from_text(
        self,
        query_id: str,
        query_text: str
    ) -> RetrievalOutput:
        """
        Retrieve documents from query text.

        Requires embedding_model to be set during initialization.

        Args:
            query_id: Unique query identifier
            query_text: Query text to embed and search

        Returns:
            RetrievalOutput with retrieved documents and scores
        """
        if self.embedding_model is None:
            raise ValueError("embedding_model must be provided to use retrieve_from_text()")

        # Embed query text
        query_embedding = self.embedding_model.embed_query(query_text)

        # Retrieve
        return self.retrieve(query_id, query_embedding)

    def batch_retrieve(
        self,
        query_ids: List[str],
        query_embeddings: np.ndarray
    ) -> List[RetrievalOutput]:
        """
        Retrieve for multiple queries at once.

        Note: HC is computed independently for each query, so batch processing
        is just a convenience wrapper around individual retrieves.

        Args:
            query_ids: List of query identifiers
            query_embeddings: Query embeddings, shape (n_queries, embedding_dim)

        Returns:
            List of RetrievalOutput objects
        """
        if len(query_ids) != len(query_embeddings):
            raise ValueError(
                f"Length mismatch: {len(query_ids)} query_ids vs "
                f"{len(query_embeddings)} query_embeddings"
            )

        outputs = []
        for query_id, query_embedding in zip(query_ids, query_embeddings):
            output = self.retrieve(query_id, query_embedding)
            outputs.append(output)

        return outputs

    def get_config(self) -> dict:
        """Get retriever configuration."""
        return {
            'method': self.method,
            'gamma': self.gamma,
            'max_candidates': self.max_candidates,
            'min_hc': self.min_hc,
            'allow_empty': self.allow_empty,
            'max_k': self.max_k,
            'use_zscore': self.use_zscore,
            'parametric': self.parametric,
            'z_cap': self.z_cap,
            'gpd_composite': self.gpd_composite
        }


def main():
    """Example usage of HCRetrieval."""
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Set seed for reproducibility
    np.random.seed(42)

    print("\n" + "="*70)
    print("HC-based Adaptive Retrieval Demo")
    print("="*70 + "\n")

    # =========================================================================
    # Step 1: Create Sample Vector Database
    # =========================================================================
    print("Step 1: Creating sample vector database...")
    print("-" * 70)

    embedding_dim = 384
    n_docs = 100

    doc_ids = [f"doc_{i}" for i in range(n_docs)]
    embeddings = np.random.randn(n_docs, embedding_dim).astype(np.float32)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Create database
    vector_db = VectorDatabase(embedding_dim=embedding_dim)
    vector_db.add_documents(doc_ids, embeddings)

    print(f"✓ Created database with {n_docs} documents")
    print()

    # =========================================================================
    # Step 2: Fit Null Distribution
    # =========================================================================
    print("Step 2: Generating null distribution...")
    print("-" * 70)

    builder = GaussianNull(embedding_dim=embedding_dim, seed=42)
    null_dist = builder.build(n_samples=10000)
    print(f"  {null_dist}")

    hc = HigherCriticism(null_distribution=null_dist)
    print()

    # =========================================================================
    # Step 3: Create HC Retriever
    # =========================================================================
    print("Step 3: Creating HC retriever...")
    print("-" * 70)

    retriever = HCRetrieval(
        vector_db=vector_db,
        hc_module=hc,
        gamma=0.1,
        max_candidates=20,
        min_hc=0.0,  # No gate for demo
        allow_empty=True
    )
    print(f"✓ Retriever configured: {retriever.get_config()}")
    print()

    # =========================================================================
    # Step 4: Test Retrieval with Signal
    # =========================================================================
    print("Step 4: Testing retrieval with signal (relevant documents)...")
    print("-" * 70)

    # Create query with high similarity to some docs (signal)
    query_with_signal = embeddings[0] + np.random.randn(embedding_dim) * 0.1
    query_with_signal = query_with_signal / np.linalg.norm(query_with_signal)

    result_signal = retriever.retrieve(
        query_id="q_signal",
        query_embedding=query_with_signal
    )

    print(f"Query ID: {result_signal.query_id}")
    print(f"Method: {result_signal.method}")
    print(f"Threshold: {result_signal.threshold:.4f}")
    print(f"Retrieved k={result_signal.k} documents")
    if result_signal.k > 0:
        print(f"Top 3 docs: {result_signal.retrieved_ids[:3]}")
        print(f"Top 3 scores: {[f'{s:.3f}' for s in result_signal.retrieved_scores[:3]]}")
    print()

    # =========================================================================
    # Step 5: Test Retrieval with Pure Noise
    # =========================================================================
    print("Step 5: Testing retrieval with pure noise (no signal)...")
    print("-" * 70)

    # Create random query (pure noise)
    query_noise = np.random.randn(embedding_dim).astype(np.float32)
    query_noise = query_noise / np.linalg.norm(query_noise)

    # Use HC gate to detect noise
    retriever_with_gate = HCRetrieval(
        vector_db=vector_db,
        hc_module=hc,
        gamma=0.1,
        max_candidates=20,
        min_hc=2.0,  # Require HC > 2.0 to return results
        allow_empty=True
    )

    result_noise = retriever_with_gate.retrieve(
        query_id="q_noise",
        query_embedding=query_noise
    )

    print(f"Query ID: {result_noise.query_id}")
    print(f"Method: {result_noise.method}")
    print(f"Threshold: {result_noise.threshold}")
    print(f"Retrieved k={result_noise.k} documents")
    if result_noise.k == 0:
        print("✓ Empty set returned (noise detected)")
    print()

    # =========================================================================
    # Step 6: Batch Retrieval
    # =========================================================================
    print("Step 6: Testing batch retrieval...")
    print("-" * 70)

    n_queries = 3
    query_ids = [f"q{i}" for i in range(1, n_queries + 1)]
    query_embeddings = np.random.randn(n_queries, embedding_dim).astype(np.float32)

    # Normalize
    norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_embeddings = query_embeddings / norms

    batch_results = retriever.batch_retrieve(query_ids, query_embeddings)

    for result in batch_results:
        print(f"{result.query_id}: k={result.k}, threshold={result.threshold:.3f}")

    print("\n" + "="*70)
    print("✓ HC retrieval demo completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
