"""
Baseline Top-k Retrieval

Simple top-k retrieval strategy that returns a fixed number of most similar documents.
Serves as the baseline for comparison with HC-based adaptive retrieval.
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

logger = logging.getLogger(__name__)


@dataclass
class RetrievalOutput:
    """
    Unified retrieval output format.

    Used by both BaselineRetrieval and HCRetrieval for consistent API.
    """
    query_id: str
    retrieved_ids: List[str]  # Document IDs in rank order
    retrieved_scores: List[float]  # Similarity scores
    k: int  # Number of documents retrieved
    method: str  # "baseline" or "hc"

    # Method-specific metadata
    threshold: Optional[float] = None  # For HC: the adaptive threshold used


class BaselineRetrieval:
    """
    Baseline top-k retrieval strategy.

    Returns a fixed number (k) of most similar documents for each query.
    Simple and deterministic - serves as baseline for HC comparison.
    """

    def __init__(
        self,
        vector_db: VectorDatabase,
        k: int = 10,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        """
        Initialize baseline retriever.

        Args:
            vector_db: Vector database containing indexed documents
            k: Number of documents to retrieve per query
            embedding_model: Optional embedding model for query text → embedding
        """
        self.vector_db = vector_db
        self.k = k
        self.embedding_model = embedding_model

        logger.info(f"BaselineRetrieval initialized with k={k}")
        logger.info(f"  Vector DB contains {vector_db.get_num_documents()} documents")

    def retrieve(
        self,
        query_id: str,
        query_embedding: np.ndarray
    ) -> RetrievalOutput:
        """
        Retrieve top-k most similar documents for a query.

        Args:
            query_id: Unique query identifier
            query_embedding: Query embedding vector

        Returns:
            RetrievalOutput with retrieved documents and scores
        """
        # Search using vector database
        results = self.vector_db.search(query_embedding, k=self.k)

        # Convert to unified format
        retrieved_ids = [r.doc_id for r in results]
        retrieved_scores = [r.similarity for r in results]

        return RetrievalOutput(
            query_id=query_id,
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            k=len(retrieved_ids),
            method="baseline",
            threshold=None  # Baseline doesn't use threshold
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

        # Batch search
        batch_results = self.vector_db.batch_search(query_embeddings, k=self.k)

        # Convert to unified format
        outputs = []
        for query_id, results in zip(query_ids, batch_results):
            retrieved_ids = [r.doc_id for r in results]
            retrieved_scores = [r.similarity for r in results]

            output = RetrievalOutput(
                query_id=query_id,
                retrieved_ids=retrieved_ids,
                retrieved_scores=retrieved_scores,
                k=len(retrieved_ids),
                method="baseline",
                threshold=None
            )
            outputs.append(output)

        return outputs

    def get_k(self) -> int:
        """Get the k value used by this retriever."""
        return self.k


def main():
    """Example usage of BaselineRetrieval."""
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("Baseline Top-k Retrieval Demo")
    print("="*70 + "\n")

    # Create sample vector database
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

    # Create baseline retriever
    print("Step 2: Creating baseline retriever...")
    print("-" * 70)

    retriever = BaselineRetrieval(vector_db=vector_db, k=5)
    print(f"✓ Retriever configured with k={retriever.get_k()}")
    print()

    # Single query retrieval
    print("Step 3: Single query retrieval...")
    print("-" * 70)

    query_embedding = np.random.randn(embedding_dim).astype(np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    result = retriever.retrieve(
        query_id="q1",
        query_embedding=query_embedding
    )

    print(f"Query ID: {result.query_id}")
    print(f"Method: {result.method}")
    print(f"Retrieved k={result.k} documents:")
    for i, (doc_id, score) in enumerate(zip(result.retrieved_ids, result.retrieved_scores), 1):
        print(f"  [{i}] {doc_id}: {score:.4f}")
    print()

    # Batch retrieval
    print("Step 4: Batch retrieval (3 queries)...")
    print("-" * 70)

    n_queries = 3
    query_ids = [f"q{i}" for i in range(1, n_queries + 1)]
    query_embeddings = np.random.randn(n_queries, embedding_dim).astype(np.float32)

    # Normalize
    norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_embeddings = query_embeddings / norms

    batch_results = retriever.batch_retrieve(query_ids, query_embeddings)

    for result in batch_results:
        print(f"\n{result.query_id}: Retrieved {result.k} documents")
        print(f"  Top 3: {result.retrieved_ids[:3]}")
        print(f"  Scores: {[f'{s:.3f}' for s in result.retrieved_scores[:3]]}")

    print("\n" + "="*70)
    print("✓ Baseline retrieval demo completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
