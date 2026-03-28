"""
Embedding Model Wrapper

Provides a unified interface for generating embeddings using:
- Local models (sentence transformers) with GPU acceleration
- API-based models (Google Gemini, OpenAI) via API calls

Supports batched processing and handles API rate limiting.
"""

import numpy as np
import torch
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
import logging
import os
from dotenv import load_dotenv
import time
from abc import ABC, abstractmethod

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for sentence transformer models.

    Handles text embedding generation with batching and device management.
    """

    # Recommended local models
    FAST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dim, fast
    QUALITY_MODEL = "sentence-transformers/all-mpnet-base-v2"  # 768 dim, better quality
    BGE_MODEL = "BAAI/bge-base-en-v1.5"  # 768 dim, SOTA for retrieval (MTEB #1)

    # API-based models
    GEMINI_MODEL = "models/text-embedding-004"  # 768 dim, Google Gemini

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


class GeminiEmbeddingModel:
    """
    Google Gemini API-based embedding model.

    Uses Google's Generative AI API for generating embeddings.
    Handles API rate limiting and batching automatically.
    """

    # Gemini embedding models
    GEMINI_EMBEDDING_004 = "models/text-embedding-004"  # 768 dim, latest model
    GEMINI_EMBEDDING_003 = "models/embedding-001"  # 768 dim (deprecated, use 004)

    # Embedding dimensions for each model
    EMBEDDING_DIMS = {
        "models/text-embedding-004": 768,
        "models/embedding-001": 768,
    }

    def __init__(
        self,
        model_name: str = GEMINI_EMBEDDING_004,
        api_key: Optional[str] = None,
        batch_size: int = 100,  # Gemini supports up to 100 texts per request
        rate_limit_delay: float = 0.1,  # Delay between API calls (seconds)
        task_type: str = "retrieval_document",  # or "retrieval_query"
        normalize_embeddings: bool = True
    ):
        """
        Initialize Gemini embedding model.

        Args:
            model_name: Gemini model identifier
            api_key: Google API key (if None, reads from GEMINI_API_KEY env var)
            batch_size: Number of texts to send per API request (max 100)
            rate_limit_delay: Delay between API calls to avoid rate limiting
            task_type: Type of embedding task ("retrieval_document" or "retrieval_query")
            normalize_embeddings: Whether to L2-normalize embeddings
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not found. "
                "Install it with: pip install google-generativeai"
            )

        self.model_name = model_name
        self.batch_size = min(batch_size, 100)  # Gemini max is 100
        self.rate_limit_delay = rate_limit_delay
        self.task_type = task_type
        self.normalize_embeddings = normalize_embeddings

        # Get API key from parameter or environment variable
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY not found. Please set it in your .env file or pass it as a parameter."
                )

        # Configure the API
        genai.configure(api_key=api_key)

        # Get embedding dimension
        self.embedding_dim = self.EMBEDDING_DIMS.get(model_name, 768)

        logger.info(f"✓ Gemini API configured")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Embedding dimension: {self.embedding_dim}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Task type: {task_type}")

    def _embed_batch(self, texts: List[str], task_type: Optional[str] = None) -> np.ndarray:
        """
        Embed a batch of texts using Gemini API.

        Args:
            texts: List of texts to embed (max 100)
            task_type: Override default task type for this batch

        Returns:
            Numpy array of embeddings
        """
        import google.generativeai as genai

        if task_type is None:
            task_type = self.task_type

        try:
            # Call Gemini API
            result = genai.embed_content(
                model=self.model_name,
                content=texts,
                task_type=task_type
            )

            # Extract embeddings
            embeddings = np.array(result['embedding'])

            # Handle single text case (API returns 1D array)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            # Normalize if requested
            if self.normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-8)

            return embeddings

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise

    def embed(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        task_type: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embeddings for input texts.

        Args:
            texts: Single text string or list of text strings
            show_progress: Whether to show progress bar
            task_type: Override default task type ("retrieval_document" or "retrieval_query")

        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            logger.warning("Empty text list provided")
            return np.array([])

        # Process in batches
        all_embeddings = []

        if show_progress:
            from tqdm import tqdm
            batches = tqdm(
                range(0, len(texts), self.batch_size),
                desc="Embedding batches"
            )
        else:
            batches = range(0, len(texts), self.batch_size)

        for i in batches:
            batch_texts = texts[i:i + self.batch_size]

            # Call API
            batch_embeddings = self._embed_batch(batch_texts, task_type)
            all_embeddings.append(batch_embeddings)

            # Rate limiting delay (except for last batch)
            if i + self.batch_size < len(texts):
                time.sleep(self.rate_limit_delay)

        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        Args:
            query: Query text

        Returns:
            Numpy array of shape (embedding_dim,)
        """
        embedding = self.embed([query], show_progress=False, task_type="retrieval_query")
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
        return self.embed(documents, show_progress=show_progress, task_type="retrieval_document")

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension of the model."""
        return self.embedding_dim

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name


class OpenAIEmbeddingModel:
    """
    OpenAI API-based embedding model.

    Uses OpenAI's embedding API for generating embeddings.
    Handles API rate limiting and batching automatically.
    """

    # OpenAI embedding models
    SMALL_MODEL = "text-embedding-3-small"   # 1536 dim
    LARGE_MODEL = "text-embedding-3-large"   # 3072 dim
    ADA_MODEL = "text-embedding-ada-002"     # 1536 dim (legacy)

    EMBEDDING_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = SMALL_MODEL,
        api_key: Optional[str] = None,
        batch_size: int = 500,
        rate_limit_delay: float = 0.05,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize OpenAI embedding model.

        Args:
            model_name: OpenAI model identifier
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            batch_size: Number of texts per API request (max 2048)
            rate_limit_delay: Delay between API calls in seconds
            normalize_embeddings: Whether to L2-normalize embeddings
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not found. Install it with: pip install openai"
            )

        self.model_name = model_name
        self.batch_size = min(batch_size, 2048)
        self.rate_limit_delay = rate_limit_delay
        self.normalize_embeddings = normalize_embeddings

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found. Set it in your .env file or pass it as a parameter."
                )

        self.client = OpenAI(api_key=api_key)
        self.embedding_dim = self.EMBEDDING_DIMS.get(model_name, 1536)

        logger.info(f"OpenAI embedding model configured")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Embedding dimension: {self.embedding_dim}")
        logger.info(f"  Batch size: {self.batch_size}")

    def _truncate_text(self, text: str, max_chars: int = 20000) -> str:
        """Rough truncation to stay under 8192 token limit (~3 chars/token conservative)."""
        if len(text) > max_chars:
            return text[:max_chars]
        return text

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts using OpenAI API. Auto-splits on token limit errors."""
        texts = [self._truncate_text(t) for t in texts]
        try:
            response = self.client.embeddings.create(input=texts, model=self.model_name)
        except Exception as e:
            err_msg = str(e)
            if "max_tokens" in err_msg or "maximum input length" in err_msg:
                if len(texts) > 1:
                    mid = len(texts) // 2
                    left = self._embed_batch(texts[:mid])
                    time.sleep(self.rate_limit_delay)
                    right = self._embed_batch(texts[mid:])
                    return np.vstack([left, right])
                else:
                    # Single text too long — aggressively truncate
                    texts = [t[:10000] for t in texts]
                    response = self.client.embeddings.create(input=texts, model=self.model_name)
            else:
                raise

        embeddings = np.array(
            [item.embedding for item in response.data], dtype=np.float32
        )

        if self.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        return embeddings

    def embed(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for input texts.

        Args:
            texts: Single text string or list of text strings
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            logger.warning("Empty text list provided")
            return np.array([])

        all_embeddings = []

        if show_progress:
            from tqdm import tqdm
            batches = tqdm(
                range(0, len(texts), self.batch_size),
                desc="Embedding batches",
            )
        else:
            batches = range(0, len(texts), self.batch_size)

        for i in batches:
            batch_texts = texts[i : i + self.batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            all_embeddings.append(batch_embeddings)

            if i + self.batch_size < len(texts):
                time.sleep(self.rate_limit_delay)

        return np.vstack(all_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        return self.embed([query], show_progress=False)[0]

    def embed_documents(
        self, documents: List[str], show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for a list of documents."""
        return self.embed(documents, show_progress=show_progress)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def get_model_name(self) -> str:
        return self.model_name


def create_embedding_model(
    model_name: str,
    batch_size: Optional[int] = None,
    **kwargs
) -> Union[EmbeddingModel, GeminiEmbeddingModel]:
    """
    Factory function to create the appropriate embedding model.

    Args:
        model_name: Model identifier (local model name or Gemini model)
        batch_size: Batch size for encoding
        **kwargs: Additional arguments passed to model constructor

    Returns:
        EmbeddingModel or GeminiEmbeddingModel instance
    """
    # Check if it's a Gemini model
    if model_name.startswith("models/") or "gemini" in model_name.lower():
        logger.info("Creating Gemini API-based embedding model")
        if batch_size is None:
            batch_size = 100  # Gemini default
        return GeminiEmbeddingModel(model_name=model_name, batch_size=batch_size, **kwargs)
    # Check if it's an OpenAI model
    elif "text-embedding" in model_name or model_name in OpenAIEmbeddingModel.EMBEDDING_DIMS:
        logger.info("Creating OpenAI API-based embedding model")
        if batch_size is None:
            batch_size = 500
        return OpenAIEmbeddingModel(model_name=model_name, batch_size=batch_size, **kwargs)
    else:
        logger.info("Creating local embedding model")
        if batch_size is None:
            # Auto-detect based on GPU availability
            if torch.cuda.is_available():
                batch_size = 512
            else:
                batch_size = 32
        return EmbeddingModel(model_name=model_name, batch_size=batch_size, **kwargs)


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
