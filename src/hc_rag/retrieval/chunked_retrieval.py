"""
Chunked Retrieval Base Class

Provides shared functionality for retrievers that work with chunked documents.
Handles chunk-to-document aggregation and database loading logic.

Supports multiple aggregation strategies:
- max_score: Use the highest chunk score as document score
- mean_score: Use average of chunk scores as document score
- sum_score: Sum all chunk scores (biases toward longer documents)
"""

import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

from hc_rag.embeddings.vector_database import VectorDatabase, SearchResult

logger = logging.getLogger(__name__)


class ChunkedRetrievalMixin:
    """
    Mixin class providing shared chunked retrieval functionality.

    Subclasses must set:
    - self.chunk_to_doc_mapping (Optional[Dict[str, str]])
    - self.aggregation (str)
    """

    @staticmethod
    def _load_chunk_mapping(db_path: str) -> Optional[Dict[str, str]]:
        """
        Load chunk-to-document mapping from database path.

        Args:
            db_path: Path to database (without extension)

        Returns:
            Dict mapping chunk_id -> parent_doc_id, or None if not a chunked database
        """
        mapping_file = Path(f"{db_path}.chunk_mapping.pkl")

        if not mapping_file.exists():
            # Not a chunked database
            return None

        # Load chunk mappings
        with open(mapping_file, "rb") as f:
            mappings = pickle.load(f)

        chunk_to_doc = mappings["chunk_to_doc"]

        logger.info(f"Loaded chunk mapping from {mapping_file}")
        logger.info(f"  Mapping entries: {len(chunk_to_doc)}")
        logger.info(f"  Chunker config: {mappings.get('chunker_config', 'N/A')}")

        return chunk_to_doc

    def _aggregate_chunks_to_documents(
        self,
        chunk_results: List[SearchResult],
        k: int
    ) -> Tuple[List[str], List[float]]:
        """
        Aggregate chunk results to parent documents.

        Uses self.chunk_to_doc_mapping and self.aggregation.

        Args:
            chunk_results: List of SearchResult objects from chunk retrieval
            k: Number of documents to return

        Returns:
            Tuple of (document_ids, document_scores)
        """
        if self.chunk_to_doc_mapping is None:
            raise ValueError("chunk_to_doc_mapping is None. Cannot aggregate chunks.")

        # Step 1: Group chunks by parent document
        doc_chunks: Dict[str, List[SearchResult]] = {}

        for chunk_result in chunk_results:
            parent_doc_id = self.chunk_to_doc_mapping.get(chunk_result.doc_id)

            if parent_doc_id is None:
                logger.warning(f"Chunk {chunk_result.doc_id} has no parent mapping")
                continue

            if parent_doc_id not in doc_chunks:
                doc_chunks[parent_doc_id] = []
            doc_chunks[parent_doc_id].append(chunk_result)

        # Step 2: Calculate document scores based on aggregation strategy
        doc_scores: Dict[str, float] = {}

        for doc_id, chunks in doc_chunks.items():
            chunk_scores = [c.similarity for c in chunks]

            if self.aggregation == "max_score":
                doc_scores[doc_id] = max(chunk_scores)
            elif self.aggregation == "mean_score":
                doc_scores[doc_id] = np.mean(chunk_scores)
            elif self.aggregation == "sum_score":
                doc_scores[doc_id] = sum(chunk_scores)
            else:
                raise ValueError(f"Unknown aggregation strategy: {self.aggregation}")

        # Step 3: Sort and select top-k documents
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_docs = sorted_docs[:k]

        retrieved_ids = [doc_id for doc_id, _ in top_k_docs]
        retrieved_scores = [score for _, score in top_k_docs]

        return retrieved_ids, retrieved_scores
