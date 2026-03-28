import re
from typing import List, Dict, Optional
from dataclasses import dataclass

# ── V2 Constants ──────────────────────────────────────────────
TITLE_TRUNCATE = 100  # V2: increased from 50


@dataclass
class LLMRAGInput:
    """
    Standardized input format for LLM RAG systems.
    Represents a query along with its retrieved documents.
    """
    query: str
    documents: List[str]  # Retrieved documents as text
    metadata: Optional[dict] = None  # Optional additional info (e.g., source, date)


def format_docs(llm_rag_inputs: LLMRAGInput, max_tokens: int) -> str:
    """Format the retrieved documents into a single string within token limits."""
    formatted_docs = ""
    current_tokens = 0

    for doc in llm_rag_inputs.documents:
        doc_tokens = len(doc.split())  # Simple token count by word count
        if current_tokens + doc_tokens <= max_tokens:
            formatted_docs += doc + "\n\n"
            current_tokens += doc_tokens
        else:
            break

    return formatted_docs.strip()


def clean_output(raw_output: str) -> str:
    return raw_output.strip()


def format_docs_v2(documents: list, max_tokens: int = 100000) -> str:
    """Format documents numbered and separated, full text preserved."""
    formatted = []
    current_tokens = 0

    for i, doc_text in enumerate(documents, 1):
        entry = f"--- Document {i} ---\n{doc_text}"

        doc_tokens = len(entry.split())
        if current_tokens + doc_tokens > max_tokens:
            break
        formatted.append(entry)
        current_tokens += doc_tokens

    return "\n\n".join(formatted)


def sanitize_text(text: str) -> str:
    """Remove control characters from text."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)


def retrieval_output_to_llm_input(
    retrieval_output,
    query_text: str,
    corpus: Dict[str, dict],
    text_field: str = "text",
) -> LLMRAGInput:
    """
    Convert a RetrievalOutput to LLMRAGInput by looking up doc texts from corpus.

    Args:
        retrieval_output: RetrievalOutput from retrieval pipeline
        query_text: The query text string
        corpus: Dict mapping doc_id -> document dict
        text_field: Key in document dict containing the text content

    Returns:
        LLMRAGInput ready for LLM consumption
    """
    documents = []
    for doc_id in retrieval_output.retrieved_ids:
        doc = corpus.get(doc_id)
        if doc is not None and text_field in doc:
            documents.append(str(doc[text_field]))
        else:
            documents.append(f"[Document {doc_id} not found]")

    return LLMRAGInput(
        query=query_text,
        documents=documents,
        metadata={
            "query_id": retrieval_output.query_id,
            "method": retrieval_output.method,
            "k": retrieval_output.k,
            "retrieved_ids": retrieval_output.retrieved_ids,
        },
    )
