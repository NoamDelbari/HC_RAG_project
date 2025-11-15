"""
HC-based Adaptive Retrieval

Adaptive retrieval strategy using Higher Criticism statistics to determine
the optimal number of documents to retrieve for each query.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.vector_database import VectorDatabase, SearchResult
from embeddings.embedding_model import EmbeddingModel
from hc.higher_criticism import HigherCriticism
from hc.null_distribution import QueryNullDistributions
from retrieval.baseline_retrieval import RetrievalOutput

logger = logging.getLogger(__name__)


class HCRetrieval:
    """
    Higher Criticism based adaptive retrieval.

    Uses HC statistics to adaptively determine the number of relevant documents
    for each query, rather than using a fixed k.
    """

    def __init__(
        self,
        vector_db: VectorDatabase,
        query_null_distributions: QueryNullDistributions,
        gamma: float = 0.1,
        max_candidates: int = 50,
        min_hc: float = 0.0,
        allow_empty: bool = True,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        """
        Initialize HC-based retriever.

        Args:
            vector_db: Vector database containing indexed documents
            query_null_distributions: Per-query null distributions
            gamma: Fraction of top scores to search for HC maximum (0 < gamma <= 1)
            max_candidates: Maximum number of candidates to fetch before HC filtering
            min_hc: Minimum HC statistic to accept any documents (0 = disabled)
            allow_empty: If True, can return empty set when HC < min_hc
            embedding_model: Optional embedding model for query text → embedding
        """
        self.vector_db = vector_db
        self.query_null_distributions = query_null_distributions
        self.gamma = gamma
        self.max_candidates = max_candidates
        self.min_hc = min_hc
        self.allow_empty = allow_empty
        self.embedding_model = embedding_model

        # Validate parameters
        if not (0 < gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {max_candidates}")
        if min_hc < 0:
            raise ValueError(f"min_hc must be >= 0, got {min_hc}")

        logger.info(f"HCRetrieval initialized:")
        logger.info(f"  gamma={gamma}")
        logger.info(f"  max_candidates={max_candidates}")
        logger.info(f"  min_hc={min_hc}")
        logger.info(f"  allow_empty={allow_empty}")
        logger.info(f"  Per-query null distributions: {len(query_null_distributions.distributions)} queries")
        logger.info(f"  Vector DB contains {vector_db.get_num_documents()} documents")

    def retrieve(
        self,
        query_id: str,
        query_embedding: np.ndarray
    ) -> RetrievalOutput:
        """
        Retrieve documents using HC-based adaptive thresholding.

        Args:
            query_id: Unique query identifier
            query_embedding: Query embedding vector

        Returns:
            RetrievalOutput with adaptively selected documents and HC metadata
        """
        # Step 1: Fetch max_candidates from vector DB
        candidates = self.vector_db.search(query_embedding, k=self.max_candidates)

        # Handle case where DB has fewer than max_candidates documents
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

        # Step 2: Get query-specific null distribution and create HC module
        null_dist = self.query_null_distributions.get(query_id)
        hc_module = HigherCriticism(null_distribution=null_dist)

        # Step 3: Compute HC threshold using query-specific null distribution
        result = hc_module.compute_hc_threshold(
            similarities=candidate_scores,
            gamma=self.gamma,
            min_hc=self.min_hc,
            allow_empty=self.allow_empty
        )

        # Step 3: Filter candidates by threshold
        if result.k == 0 or np.isinf(result.threshold):
            # Empty result (pure noise detected)
            logger.debug(
                f"Query {query_id}: HC returned empty set "
                f"(HC={result.hc_statistic:.3f}, threshold={result.threshold})"
            )
            retrieved_ids = []
            retrieved_scores = []
        else:
            # Select top k candidates (already sorted by similarity)
            selected_candidates = candidates[:result.k]
            retrieved_ids = [c.doc_id for c in selected_candidates]
            retrieved_scores = [c.similarity for c in selected_candidates]

            logger.debug(
                f"Query {query_id}: Retrieved {result.k} docs "
                f"(HC={result.hc_statistic:.3f}, threshold={result.threshold:.3f})"
            )

        return RetrievalOutput(
            query_id=query_id,
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            k=len(retrieved_ids),
            method="hc",
            threshold=float(result.threshold)
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
            'method': 'hc',
            'gamma': self.gamma,
            'max_candidates': self.max_candidates,
            'min_hc': self.min_hc,
            'allow_empty': self.allow_empty
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
