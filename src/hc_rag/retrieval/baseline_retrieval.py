"""
Baseline Top-k Retrieval

Simple top-k retrieval strategy that returns a fixed number of most similar documents.
Serves as the baseline for comparison with HC-based adaptive retrieval.

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
from retrieval.chunked_retrieval import ChunkedRetrievalMixin

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


class BaselineRetrieval(ChunkedRetrievalMixin):
    """
    Baseline top-k retrieval strategy.

    Returns a fixed number (k) of most similar documents for each query.
    Simple and deterministic - serves as baseline for HC comparison.

    Supports chunked retrieval: if chunk_to_doc_mapping is provided,
    retrieves chunks and aggregates them to parent documents.
    """

    def __init__(
        self,
        vector_db: VectorDatabase,
        k: int = 10,
        embedding_model: Optional[EmbeddingModel] = None,
        chunk_to_doc_mapping: Optional[Dict[str, str]] = None,
        aggregation: str = "max_score",
        top_chunks: Optional[int] = None
    ):
        """
        Initialize baseline retriever.

        Args:
            vector_db: Vector database containing indexed documents or chunks
            k: Number of documents to retrieve per query
            embedding_model: Optional embedding model for query text → embedding
            chunk_to_doc_mapping: Optional dict mapping chunk_id -> parent_doc_id
                                 If provided, enables chunked retrieval mode
            aggregation: Chunk score aggregation strategy: "max_score", "mean_score", or "sum_score"
                        Only used when chunk_to_doc_mapping is provided
            top_chunks: Number of chunks to retrieve before aggregation (default: k * 10)
                       Only used when chunk_to_doc_mapping is provided
        """
        self.vector_db = vector_db
        self.k = k
        self.embedding_model = embedding_model
        self.chunk_to_doc_mapping = chunk_to_doc_mapping
        self.aggregation = aggregation

        # Set top_chunks default based on k
        if chunk_to_doc_mapping is not None:
            self.top_chunks = top_chunks if top_chunks is not None else k * 10
        else:
            self.top_chunks = None

        # Validate aggregation strategy
        if chunk_to_doc_mapping is not None and aggregation not in ["max_score", "mean_score", "sum_score"]:
            raise ValueError(f"Invalid aggregation: {aggregation}. Must be 'max_score', 'mean_score', or 'sum_score'")

        logger.info(f"BaselineRetrieval initialized with k={k}")
        logger.info(f"  Vector DB contains {vector_db.get_num_documents()} {'chunks' if chunk_to_doc_mapping else 'documents'}")
        if chunk_to_doc_mapping:
            logger.info(f"  Chunked mode: top_chunks={self.top_chunks}, aggregation={aggregation}")

    @classmethod
    def from_database_path(
        cls,
        db_path: str,
        k: int = 10,
        embedding_model: Optional[EmbeddingModel] = None,
        aggregation: str = "max_score",
        top_chunks: Optional[int] = None
    ) -> "BaselineRetrieval":
        """
        Load retrieval system from database path.

        Automatically detects if the database is chunked (has .chunk_mapping.pkl file)
        and enables chunked retrieval mode accordingly.

        Args:
            db_path: Path to database (without extension)
            k: Number of documents to retrieve
            embedding_model: Optional embedding model
            aggregation: Aggregation strategy for chunked mode
            top_chunks: Number of chunks to retrieve in chunked mode

        Returns:
            BaselineRetrieval instance
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
            k=k,
            embedding_model=embedding_model,
            chunk_to_doc_mapping=chunk_to_doc_mapping,
            aggregation=aggregation,
            top_chunks=top_chunks
        )

    def retrieve(
        self,
        query_id: str,
        query_embedding: np.ndarray
    ) -> RetrievalOutput:
        """
        Retrieve top-k most similar documents for a query.

        If chunk_to_doc_mapping is provided, retrieves chunks and aggregates to documents.
        Otherwise, retrieves documents directly.

        Args:
            query_id: Unique query identifier
            query_embedding: Query embedding vector

        Returns:
            RetrievalOutput with retrieved documents and scores
        """
        # Determine retrieval mode
        if self.chunk_to_doc_mapping is not None:
            # Chunked retrieval mode
            # Step 1: Retrieve top chunks
            chunk_results = self.vector_db.search(query_embedding, k=self.top_chunks)

            # Step 2: Aggregate chunks to documents
            retrieved_ids, retrieved_scores = self._aggregate_chunks_to_documents(
                chunk_results, self.k
            )

            method = f"baseline_{self.aggregation}"
        else:
            # Full document retrieval mode
            results = self.vector_db.search(query_embedding, k=self.k)

            # Convert to unified format
            retrieved_ids = [r.doc_id for r in results]
            retrieved_scores = [r.similarity for r in results]

            method = "baseline"

        return RetrievalOutput(
            query_id=query_id,
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            k=len(retrieved_ids),
            method=method,
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
