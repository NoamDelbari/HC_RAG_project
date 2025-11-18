"""
FAISS Vector Database

High-performance vector database for document retrieval using FAISS.
Supports fast similarity search and persistence.
"""

# Workaround for faiss-cpu 1.12.0 bug with GPU classes
import os
os.environ['FAISS_NO_AVX2'] = '1'

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result."""
    doc_id: str
    similarity: float
    rank: int
    metadata: Optional[Dict[str, Any]] = None


class VectorDatabase:
    """
    FAISS-based vector database for document embeddings.

    Supports fast k-nearest neighbor search using cosine similarity.
    """

    def __init__(self, embedding_dim: int, index_type: str = "flat", embedding_model_name: Optional[str] = None):
        """
        Initialize vector database.

        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ("flat" for exact search, "ivf" for approximate)
            embedding_model_name: Name of embedding model used (e.g., "all-MiniLM-L6-v2")
                                 Stored for consistency checking when building null distributions
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.embedding_model_name = embedding_model_name

        # Initialize FAISS index
        # IndexFlatIP = Inner Product (for normalized vectors, IP = cosine similarity)
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # Store document metadata
        self.doc_ids: List[str] = []
        self.doc_metadata: List[Dict[str, Any]] = []

        logger.info(f"Initialized VectorDatabase with embedding_dim={embedding_dim}, index_type={index_type}")
        if embedding_model_name:
            logger.info(f"  Embedding model: {embedding_model_name}")

    def add_documents(
        self,
        doc_ids: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add documents to the vector database.

        Args:
            doc_ids: List of document IDs
            embeddings: Document embeddings, shape (n_docs, embedding_dim)
            metadata: Optional list of metadata dictionaries for each document
        """
        if len(doc_ids) != len(embeddings):
            raise ValueError(f"Mismatch: {len(doc_ids)} doc_ids but {len(embeddings)} embeddings")

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}")

        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype(np.float32)

        # Normalize embeddings for cosine similarity (if not already normalized)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store metadata
        self.doc_ids.extend(doc_ids)

        if metadata is None:
            metadata = [{} for _ in doc_ids]
        self.doc_metadata.extend(metadata)

        logger.info(f"Added {len(doc_ids)} documents to index. Total: {len(self.doc_ids)}")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        return_embeddings: bool = False
    ) -> List[SearchResult]:
        """
        Search for top-k most similar documents.

        Args:
            query_embedding: Query embedding vector, shape (embedding_dim,)
            k: Number of results to return
            return_embeddings: Whether to include embeddings in results

        Returns:
            List of SearchResult objects, sorted by similarity (highest first)
        """
        if len(self.doc_ids) == 0:
            logger.warning("Index is empty, no documents to search")
            return []

        # Ensure query is the right shape and type
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        # Normalize query
        norm = np.linalg.norm(query_embedding)
        query_embedding = query_embedding / (norm + 1e-8)

        # Limit k to number of documents in index
        k = min(k, len(self.doc_ids))

        # Search
        similarities, indices = self.index.search(query_embedding, k)

        # Convert to SearchResult objects
        results = []
        for rank, (idx, sim) in enumerate(zip(indices[0], similarities[0]), 1):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            result = SearchResult(
                doc_id=self.doc_ids[idx],
                similarity=float(sim),
                rank=rank,
                metadata=self.doc_metadata[idx].copy() if self.doc_metadata[idx] else None
            )
            results.append(result)

        return results

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10
    ) -> List[List[SearchResult]]:
        """
        Search for multiple queries at once.

        Args:
            query_embeddings: Query embeddings, shape (n_queries, embedding_dim)
            k: Number of results per query

        Returns:
            List of result lists, one per query
        """
        if len(self.doc_ids) == 0:
            logger.warning("Index is empty, no documents to search")
            return [[] for _ in range(len(query_embeddings))]

        # Ensure correct shape and type
        query_embeddings = query_embeddings.astype(np.float32)

        # Normalize queries
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_embeddings = query_embeddings / (norms + 1e-8)

        # Limit k
        k = min(k, len(self.doc_ids))

        # Search
        similarities, indices = self.index.search(query_embeddings, k)

        # Convert to SearchResult objects
        all_results = []
        for query_idx in range(len(query_embeddings)):
            results = []
            for rank, (idx, sim) in enumerate(zip(indices[query_idx], similarities[query_idx]), 1):
                if idx == -1:
                    continue

                result = SearchResult(
                    doc_id=self.doc_ids[idx],
                    similarity=float(sim),
                    rank=rank,
                    metadata=self.doc_metadata[idx].copy() if self.doc_metadata[idx] else None
                )
                results.append(result)
            all_results.append(results)

        return all_results

    def get_similarity_distribution(
        self,
        query_embeddings: np.ndarray,
        n_samples: int = 10000,
        seed: int = 42
    ) -> np.ndarray:
        """
        Sample query-document similarity scores for null distribution (HC statistics).

        Randomly samples (query, document) pairs and computes their similarity.
        This represents the null hypothesis: query and document are unrelated.

        CRITICAL: Uses query-document pairs (NOT document-document pairs).
        This is essential for proper HC statistics in retrieval systems.

        Args:
            query_embeddings: Query embeddings, shape (n_queries, embedding_dim)
            n_samples: Number of random query-document pairs to sample
            seed: Random seed for reproducibility

        Returns:
            Array of similarity scores representing null distribution

        Raises:
            ValueError: If query_embeddings is empty or has wrong dimensions
        """
        # Validate inputs
        if len(self.doc_ids) == 0:
            logger.warning("No documents in database")
            return np.array([])

        if query_embeddings is None or len(query_embeddings) == 0:
            raise ValueError("query_embeddings cannot be empty")

        # Ensure query_embeddings is 2D
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        if query_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {query_embeddings.shape[1]}"
            )

        # Convert to float32 and normalize
        query_embeddings = query_embeddings.astype(np.float32)
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_embeddings = query_embeddings / (norms + 1e-8)

        np.random.seed(seed)

        # Get all document embeddings from the index
        doc_embeddings = self.index.reconstruct_n(0, len(self.doc_ids))

        # Sample random query-document pairs
        n_queries = len(query_embeddings)
        n_docs = len(self.doc_ids)
        similarities = []

        logger.info(f"Sampling {n_samples} query-document pairs for null distribution...")
        logger.info(f"  Available: {n_queries} queries, {n_docs} documents")

        for _ in range(n_samples):
            # Sample random query and random document
            query_idx = np.random.randint(0, n_queries)
            doc_idx = np.random.randint(0, n_docs)

            # Compute cosine similarity (dot product of normalized vectors)
            sim = np.dot(query_embeddings[query_idx], doc_embeddings[doc_idx])
            similarities.append(sim)

        similarities = np.array(similarities)
        logger.info(f"✓ Sampled {n_samples} query-document similarity scores")
        logger.info(f"  Min: {np.min(similarities):.4f}, Max: {np.max(similarities):.4f}, Mean: {np.mean(similarities):.4f}, Std: {np.std(similarities):.4f}")

        return similarities

    def save(self, path: str) -> None:
        """
        Save the vector database to disk.

        Args:
            path: Path to save (without extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = path.with_suffix(".faiss")
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        metadata_path = path.with_suffix(".metadata.pkl")
        metadata = {
            "doc_ids": self.doc_ids,
            "doc_metadata": self.doc_metadata,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type
        }

        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved vector database to {path}")
        logger.info(f"  Index: {index_path}")
        logger.info(f"  Metadata: {metadata_path}")

    @classmethod
    def load(cls, path: str) -> "VectorDatabase":
        """
        Load a vector database from disk.

        Args:
            path: Path to load (without extension)

        Returns:
            Loaded VectorDatabase instance
        """
        path = Path(path)

        # Load metadata
        metadata_path = path.with_suffix(".metadata.pkl")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # Create instance
        db = cls(
            embedding_dim=metadata["embedding_dim"],
            index_type=metadata["index_type"]
        )

        # Load FAISS index
        index_path = path.with_suffix(".faiss")
        db.index = faiss.read_index(str(index_path))

        # Restore metadata
        db.doc_ids = metadata["doc_ids"]
        db.doc_metadata = metadata["doc_metadata"]

        logger.info(f"Loaded vector database from {path}")
        logger.info(f"  Documents: {len(db.doc_ids)}")
        logger.info(f"  Embedding dim: {db.embedding_dim}")

        return db

    def get_num_documents(self) -> int:
        """Get the number of documents in the database."""
        return len(self.doc_ids)

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document metadata dictionary, or None if not found
        """
        try:
            idx = self.doc_ids.index(doc_id)
            return self.doc_metadata[idx].copy() if self.doc_metadata[idx] else None
        except ValueError:
            return None


def main():
    """Example usage of VectorDatabase."""
    print("\n" + "="*60)
    print("FAISS Vector Database Demo")
    print("="*60 + "\n")

    # Create sample embeddings
    embedding_dim = 384
    n_docs = 100

    print(f"Creating sample data ({n_docs} documents, {embedding_dim} dims)...")
    doc_ids = [f"doc_{i}" for i in range(n_docs)]
    embeddings = np.random.randn(n_docs, embedding_dim).astype(np.float32)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Add metadata
    metadata = [{"title": f"Document {i}", "category": f"cat_{i % 5}"} for i in range(n_docs)]

    # Create database
    print("\nBuilding vector database...")
    db = VectorDatabase(embedding_dim=embedding_dim)
    db.add_documents(doc_ids, embeddings, metadata)

    # Search
    print("\nPerforming search...")
    query = np.random.randn(embedding_dim).astype(np.float32)
    query = query / np.linalg.norm(query)

    results = db.search(query, k=5)

    print("\nTop 5 Results:")
    print("-" * 60)
    for result in results:
        print(f"[Rank {result.rank}] {result.doc_id}")
        print(f"  Similarity: {result.similarity:.4f}")
        print(f"  Metadata: {result.metadata}")
        print()

    # Sample null distribution
    print("\nSampling null distribution...")
    # Create sample query embeddings
    n_queries = 10
    query_embeddings = np.random.randn(n_queries, embedding_dim).astype(np.float32)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    null_similarities = db.get_similarity_distribution(
        query_embeddings=query_embeddings,
        n_samples=1000
    )
    print(f"Null distribution stats:")
    print(f"  Mean: {np.mean(null_similarities):.4f}")
    print(f"  Std: {np.std(null_similarities):.4f}")
    print(f"  Min: {np.min(null_similarities):.4f}")
    print(f"  Max: {np.max(null_similarities):.4f}")

    # Save and load
    print("\nTesting save/load...")
    db.save("test_db")
    db_loaded = VectorDatabase.load("test_db")
    print(f"✓ Loaded database with {db_loaded.get_num_documents()} documents")

    print("\n" + "="*60)
    print("✓ Demo completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
