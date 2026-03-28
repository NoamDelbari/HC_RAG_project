"""
Retrieval module for HC-RAG.

Contains baseline and HC-based retrieval implementations.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RetrievalOutput:
    """
    Unified retrieval output format.
    Used by both BaselineRetrieval and HCRetrieval for consistent API.
    """
    query_id: str
    retrieved_ids: List[str]
    retrieved_scores: List[float]
    k: int
    method: str
    threshold: Optional[float] = None


from .baseline_retrieval import BaselineRetrieval
from .hc_retrieval import HCRetrieval

__all__ = ["BaselineRetrieval", "HCRetrieval", "RetrievalOutput"]
