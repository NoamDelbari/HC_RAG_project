"""
Retrieval module for HC-RAG.

Contains baseline and HC-based retrieval implementations.
"""

from .baseline_retrieval import BaselineRetrieval, RetrievalOutput
from .hc_retrieval import HCRetrieval

__all__ = ["BaselineRetrieval", "HCRetrieval", "RetrievalOutput"]
