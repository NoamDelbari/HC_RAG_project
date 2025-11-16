"""
Chunked Document Retrieval

Retrieval strategy that works with chunked documents.
Retrieves relevant chunks and aggregates them back to parent documents.

Supports multiple aggregation strategies:
- max_score: Use the highest chunk score as document score
- mean_score: Use average of chunk scores as document score
- sum_score: Sum all chunk scores (biases toward longer documents)
"""

import numpy as np
from typing import List, Optional, Dict, Set
from dataclasses import dataclass
import logging
import pickle

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.vector_database import VectorDatabase, SearchResult
from embeddings.embedding_model import EmbeddingModel
from retrieval.baseline_retrieval import RetrievalOutput

logger = logging.getLogger(__name__)


class ChunkedRetrieval:
    """
    Retrieval strategy for chunked documents.
    
    Retrieves chunks, then aggregates to parent documents.
    """

    def __init__(
        self,
        vector_db: VectorDatabase,
        chunk_to_doc_mapping: Dict[str, str],
        k: int = 10,
        top_chunks: int = 100,
        aggregation: str = "max_score",
        embedding_model: Optional[EmbeddingModel] = None
    ):
        """
        Initialize chunked retriever.

        Args:
            vector_db: Vector database containing chunk embeddings
            chunk_to_doc_mapping: Dict mapping chunk_id -> parent_doc_id
            k: Number of documents to retrieve per query
            top_chunks: Number of chunks to retrieve before aggregation
            aggregation: Strategy for aggregating chunk scores ("max_score", "mean_score", "sum_score")
            embedding_model: Optional embedding model for query text → embedding
        """
        self.vector_db = vector_db
        self.chunk_to_doc_mapping = chunk_to_doc_mapping
        self.k = k
        self.top_chunks = top_chunks
        self.aggregation = aggregation
        self.embedding_model = embedding_model

        if aggregation not in ["max_score", "mean_score", "sum_score"]:
            raise ValueError(f"Invalid aggregation: {aggregation}. Must be 'max_score', 'mean_score', or 'sum_score'")

        logger.info(f"ChunkedRetrieval initialized")
        logger.info(f"  k={k}, top_chunks={top_chunks}, aggregation={aggregation}")
        logger.info(f"  Vector DB contains {vector_db.get_num_documents()} chunks")
        logger.info(f"  Chunk-to-doc mapping has {len(chunk_to_doc_mapping)} entries")

    @classmethod
    def from_database_path(
        cls,
        db_path: str,
        k: int = 10,
        top_chunks: int = 100,
        aggregation: str = "max_score",
        embedding_model: Optional[EmbeddingModel] = None
    ) -> "ChunkedRetrieval":
        """
        Load chunked retrieval system from database path.

        Args:
            db_path: Path to database (without extension)
            k: Number of documents to retrieve
            top_chunks: Number of chunks to retrieve before aggregation
            aggregation: Aggregation strategy
            embedding_model: Optional embedding model

        Returns:
            ChunkedRetrieval instance
        """
        # Load vector database
        vector_db = VectorDatabase.load(db_path)

        # Load chunk mappings
        mapping_file = f"{db_path}.chunk_mapping.pkl"
        with open(mapping_file, "rb") as f:
            mappings = pickle.load(f)
        
        chunk_to_doc = mappings["chunk_to_doc"]

        logger.info(f"Loaded chunked database from {db_path}")
        logger.info(f"  Chunks: {vector_db.get_num_documents()}")
        logger.info(f"  Chunker config: {mappings.get('chunker_config', 'N/A')}")

        return cls(
            vector_db=vector_db,
            chunk_to_doc_mapping=chunk_to_doc,
            k=k,
            top_chunks=top_chunks,
            aggregation=aggregation,
            embedding_model=embedding_model
        )

    def retrieve(
        self,
        query_id: str,
        query_embedding: np.ndarray
    ) -> RetrievalOutput:
        """
        Retrieve top-k documents for a query.

        Args:
            query_id: Unique query identifier
            query_embedding: Query embedding vector

        Returns:
            RetrievalOutput with retrieved documents and scores
        """
        # Step 1: Retrieve top chunks
        chunk_results = self.vector_db.search(query_embedding, k=self.top_chunks)

        # Step 2: Aggregate chunks to documents
        doc_chunks: Dict[str, List[SearchResult]] = {}
        
        for chunk_result in chunk_results:
            parent_doc_id = self.chunk_to_doc_mapping.get(chunk_result.doc_id)
            
            if parent_doc_id is None:
                logger.warning(f"Chunk {chunk_result.doc_id} has no parent mapping")
                continue
            
            if parent_doc_id not in doc_chunks:
                doc_chunks[parent_doc_id] = []
            doc_chunks[parent_doc_id].append(chunk_result)

        # Step 3: Calculate document scores
        doc_scores: Dict[str, float] = {}
        
        for doc_id, chunks in doc_chunks.items():
            chunk_scores = [c.similarity for c in chunks]
            
            if self.aggregation == "max_score":
                doc_scores[doc_id] = max(chunk_scores)
            elif self.aggregation == "mean_score":
                doc_scores[doc_id] = np.mean(chunk_scores)
            elif self.aggregation == "sum_score":
                doc_scores[doc_id] = sum(chunk_scores)

        # Step 4: Sort and select top-k documents
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_docs = sorted_docs[:self.k]

        retrieved_ids = [doc_id for doc_id, _ in top_k_docs]
        retrieved_scores = [score for _, score in top_k_docs]

        return RetrievalOutput(
            query_id=query_id,
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            k=len(retrieved_ids),
            method=f"chunked_{self.aggregation}",
            threshold=None
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

        # Retrieve for each query
        outputs = []
        for query_id, query_embedding in zip(query_ids, query_embeddings):
            output = self.retrieve(query_id, query_embedding)
            outputs.append(output)

        return outputs

    def get_k(self) -> int:
        """Get the k value used by this retriever."""
        return self.k

    def get_aggregation(self) -> str:
        """Get the aggregation strategy."""
        return self.aggregation


def main():
    """Example usage of ChunkedRetrieval."""
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("Chunked Retrieval Demo")
    print("="*70 + "\n")

    # Create sample vector database with chunks
    print("Step 1: Creating sample chunked database...")
    print("-" * 70)

    embedding_dim = 384
    n_docs = 20
    chunks_per_doc = 5
    n_chunks = n_docs * chunks_per_doc

    # Create chunks
    chunk_ids = []
    chunk_to_doc = {}
    for doc_idx in range(n_docs):
        doc_id = f"doc_{doc_idx}"
        for chunk_idx in range(chunks_per_doc):
            chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
            chunk_ids.append(chunk_id)
            chunk_to_doc[chunk_id] = doc_id

    # Create embeddings
    embeddings = np.random.randn(n_chunks, embedding_dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Create database
    vector_db = VectorDatabase(embedding_dim=embedding_dim)
    vector_db.add_documents(chunk_ids, embeddings)

    print(f"✓ Created database with {n_chunks} chunks")
    print(f"✓ From {n_docs} documents ({chunks_per_doc} chunks each)")
    print()

    # Create chunked retriever
    print("Step 2: Creating chunked retriever...")
    print("-" * 70)

    retriever = ChunkedRetrieval(
        vector_db=vector_db,
        chunk_to_doc_mapping=chunk_to_doc,
        k=5,
        top_chunks=20,
        aggregation="max_score"
    )
    print(f"✓ Retriever configured")
    print(f"  k={retriever.get_k()}")
    print(f"  top_chunks=20")
    print(f"  aggregation={retriever.get_aggregation()}")
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

    # Test different aggregation strategies
    print("Step 4: Comparing aggregation strategies...")
    print("-" * 70)

    for agg in ["max_score", "mean_score", "sum_score"]:
        retriever_agg = ChunkedRetrieval(
            vector_db=vector_db,
            chunk_to_doc_mapping=chunk_to_doc,
            k=3,
            top_chunks=20,
            aggregation=agg
        )
        
        result = retriever_agg.retrieve("q1", query_embedding)
        
        print(f"\n{agg}:")
        print(f"  Top 3: {result.retrieved_ids[:3]}")
        print(f"  Scores: {[f'{s:.3f}' for s in result.retrieved_scores[:3]]}")

    print("\n" + "="*70)
    print("✓ Chunked retrieval demo completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
