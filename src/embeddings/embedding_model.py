"""
Embedding Model Wrapper

Provides a unified interface for generating embeddings using sentence transformers.
Supports batched processing and GPU acceleration.
"""

import numpy as np
import torch
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for sentence transformer models.

    Handles text embedding generation with batching and device management.
    """

    # Recommended models from phase1.MD
    FAST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dim, fast
    QUALITY_MODEL = "sentence-transformers/all-mpnet-base-v2"  # 768 dim, better quality
    BGE_MODEL = "BAAI/bge-base-en-v1.5"  # 768 dim, SOTA for retrieval (MTEB #1)

    def __init__(
        self,
        model_name: str = FAST_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True
    ):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to L2-normalize embeddings (recommended for cosine similarity)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                # Log GPU information
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
                logger.info(f"🚀 GPU detected: {gpu_name}")
                logger.info(f"   GPU memory: {gpu_memory:.1f} GB")
            else:
                self.device = "cpu"
                logger.warning("⚠️  No GPU detected - using CPU (will be slower)")
                logger.warning("   For faster processing, install CUDA and PyTorch with GPU support")
        else:
            self.device = device
            if device == "cuda" and not torch.cuda.is_available():
                logger.error("❌ CUDA device requested but not available! Falling back to CPU")
                self.device = "cpu"

        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Device: {self.device.upper()}")

        # Load model
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Embedding dimension: {self.embedding_dim}")
        logger.info(f"  Batch size: {self.batch_size}")

    def embed(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        convert_to_numpy: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for input texts.

        Args:
            texts: Single text string or list of text strings
            show_progress: Whether to show progress bar
            convert_to_numpy: Whether to convert to numpy array

        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            logger.warning("Empty text list provided")
            return np.array([])

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=self.normalize_embeddings,
            device=self.device
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        Args:
            query: Query text

        Returns:
            Numpy array of shape (embedding_dim,)
        """
        embedding = self.embed([query], show_progress=False)
        return embedding[0]

    def embed_documents(
        self,
        documents: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of document texts
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings with shape (n_documents, embedding_dim)
        """
        return self.embed(documents, show_progress=show_progress)

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension of the model."""
        return self.embedding_dim

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name

    def to(self, device: str) -> "EmbeddingModel":
        """
        Move model to specified device.

        Args:
            device: Target device ('cuda' or 'cpu')

        Returns:
            Self for method chaining
        """
        self.device = device
        self.model = self.model.to(device)
        logger.info(f"Model moved to device: {device}")
        return self


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def batch_cosine_similarity(query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query and multiple documents.

    Optimized for batch computation using matrix operations.

    Args:
        query_embedding: Query embedding vector of shape (embedding_dim,)
        document_embeddings: Document embeddings of shape (n_docs, embedding_dim)

    Returns:
        Array of similarity scores of shape (n_docs,)
    """
    # If embeddings are already normalized, dot product = cosine similarity
    if len(document_embeddings.shape) == 1:
        document_embeddings = document_embeddings.reshape(1, -1)

    return np.dot(document_embeddings, query_embedding)


def main():
    """Example usage of EmbeddingModel."""
    print("\n" + "="*60)
    print("Embedding Model Demo")
    print("="*60 + "\n")

    # Initialize model
    print("Initializing embedding model...")
    model = EmbeddingModel(model_name=EmbeddingModel.FAST_MODEL)

    # Example texts
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital and most populous city of France.",
        "London is the capital of England and the United Kingdom.",
        "The Eiffel Tower is located in Paris, France.",
        "France is a country in Western Europe.",
        "Berlin is the capital of Germany."
    ]

    # Generate embeddings
    print("\nGenerating embeddings...")
    query_emb = model.embed_query(query)
    doc_embs = model.embed_documents(documents, show_progress=False)

    print(f"Query embedding shape: {query_emb.shape}")
    print(f"Document embeddings shape: {doc_embs.shape}")

    # Compute similarities
    print("\n" + "="*60)
    print("Similarity Scores")
    print("="*60)
    print(f"Query: {query}\n")

    similarities = batch_cosine_similarity(query_emb, doc_embs)

    for i, (doc, score) in enumerate(zip(documents, similarities)):
        print(f"[Doc {i+1}] Score: {score:.4f}")
        print(f"  Text: {doc}")
        print()

    # Rank by similarity
    ranked_indices = np.argsort(similarities)[::-1]
    print("="*60)
    print("Ranked Results (Most Relevant First)")
    print("="*60)
    for rank, idx in enumerate(ranked_indices, 1):
        print(f"{rank}. [Doc {idx+1}] Score: {similarities[idx]:.4f}")
        print(f"   {documents[idx][:60]}...")
        print()


if __name__ == "__main__":
    main()
