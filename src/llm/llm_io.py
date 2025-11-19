from typing import List, Optional
from dataclasses import dataclass

# TODO: Convert RetrivalOutput to LLMRAGInput (missing query and text docs)
@dataclass
class LLMRAGInput:
    """
    Standardized input format for LLM RAG systems.
    Represents a query along with its retrieved documents.
    """
    query: str
    documents: List[str]  # Retrieved documents as text
    metadata: Optional[dict] = None  # Optional additional info (e.g., source, date)

# TODO: Improve token counting mechanism, 
# TODO: Use chunking (embeddings' chunker can be reused here)
# TODO: Consider the dataset's document structure: CRAG has HTML content
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

# TODO: Implement better output cleaning
def clean_output(raw_output: str) -> str:
    return raw_output.strip()
