"""
Document Chunking Strategies

Implements various chunking strategies for handling long documents
that exceed embedding model's maximum sequence length.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging
import warnings

# Suppress transformers warnings about sequence length
import transformers
transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a chunk of a document."""
    chunk_id: str  # Format: {doc_id}_chunk_{index}
    parent_doc_id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentChunker:
    """
    Base class for document chunking strategies.
    """
    
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 50):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
    
    def chunk_document(self, doc_id: str, text: str, metadata: Dict = None) -> List[Chunk]:
        """
        Chunk a single document.
        
        Args:
            doc_id: Document ID
            text: Document text
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Chunk objects
        """
        raise NotImplementedError


class FixedSizeChunker(DocumentChunker):
    """
    Chunks documents into fixed-size chunks with overlap.
    
    Simple and fast, but may break sentences/words.
    """
    
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 50, tokenizer=None):
        """
        Initialize fixed-size chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Number of overlapping tokens between chunks
            tokenizer: HuggingFace tokenizer (if None, uses character-based approximation)
        """
        super().__init__(chunk_size, chunk_overlap)
        self.tokenizer = tokenizer
        
        # If no tokenizer, approximate tokens as ~4 chars
        if tokenizer is None:
            logger.warning("No tokenizer provided, using character-based approximation")
            self.chars_per_token = 4
    
    def chunk_document(self, doc_id: str, text: str, metadata: Dict = None) -> List[Chunk]:
        """Chunk document into fixed-size chunks."""
        if not text.strip():
            return []
        
        chunks = []
        
        if self.tokenizer:
            # Token-based chunking
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False, max_length=None)
            except Exception:
                # If tokenization fails, fall back to character-based
                return self._chunk_by_chars(doc_id, text, metadata)
            
            stride = self.chunk_size - self.chunk_overlap
            for i in range(0, len(tokens), stride):
                chunk_tokens = tokens[i:i + self.chunk_size]
                
                if not chunk_tokens:
                    break
                
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                
                chunk = Chunk(
                    chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                    parent_doc_id=doc_id,
                    text=chunk_text,
                    chunk_index=len(chunks),
                    start_char=-1,  # Not tracking char positions with tokenizer
                    end_char=-1,
                    metadata=metadata or {}
                )
                chunks.append(chunk)
                
                # Stop if we've processed all tokens
                if i + self.chunk_size >= len(tokens):
                    break
        else:
            chunks = self._chunk_by_chars(doc_id, text, metadata)
        
        logger.debug(f"Chunked {doc_id} into {len(chunks)} chunks")
        return chunks
    
    def _chunk_by_chars(self, doc_id: str, text: str, metadata: Dict) -> List[Chunk]:
        """Fallback character-based chunking."""
        chunks = []
        approx_chars_per_chunk = self.chunk_size * self.chars_per_token
        approx_overlap_chars = self.chunk_overlap * self.chars_per_token
        stride = approx_chars_per_chunk - approx_overlap_chars
        
        for i in range(0, len(text), stride):
            end_pos = min(i + approx_chars_per_chunk, len(text))
            chunk_text = text[i:end_pos]
            
            if not chunk_text.strip():
                break
            
            chunk = Chunk(
                chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                parent_doc_id=doc_id,
                text=chunk_text,
                chunk_index=len(chunks),
                start_char=i,
                end_char=end_pos,
                metadata=metadata or {}
            )
            chunks.append(chunk)
            
            if end_pos >= len(text):
                break
        
        return chunks


class SentenceChunker(DocumentChunker):
    """
    Chunks documents by sentences, respecting sentence boundaries.
    
    Better than fixed-size for semantic coherence, but slower.
    """
    
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 50, tokenizer=None):
        """
        Initialize sentence-based chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Number of overlapping sentences between chunks
            tokenizer: HuggingFace tokenizer for token counting
        """
        super().__init__(chunk_size, chunk_overlap)
        self.tokenizer = tokenizer
        
        if tokenizer is None:
            logger.warning("No tokenizer provided, using character-based approximation")
            self.chars_per_token = 4
    
    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter."""
        import re
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_token_count(self, text: str) -> int:
        """Get token count for text."""
        if self.tokenizer:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return len(self.tokenizer.encode(text, add_special_tokens=False, truncation=False, max_length=None))
            except Exception:
                return len(text) // self.chars_per_token
        else:
            return len(text) // self.chars_per_token
    
    def chunk_document(self, doc_id: str, text: str, metadata: Dict = None) -> List[Chunk]:
        """Chunk document by sentences."""
        if not text.strip():
            return []
        
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        
        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = 0
        char_position = 0
        
        for sentence in sentences:
            sentence_tokens = self._get_token_count(sentence)
            
            # If adding this sentence exceeds chunk size, finalize current chunk
            if current_chunk_sentences and current_chunk_tokens + sentence_tokens > self.chunk_size:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk_sentences)
                chunk = Chunk(
                    chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                    parent_doc_id=doc_id,
                    text=chunk_text,
                    chunk_index=len(chunks),
                    start_char=char_position,
                    end_char=char_position + len(chunk_text),
                    metadata=metadata or {}
                )
                chunks.append(chunk)
                
                # Calculate overlap: keep last N sentences
                overlap_sentences = []
                overlap_tokens = 0
                for sent in reversed(current_chunk_sentences):
                    sent_tokens = self._get_token_count(sent)
                    if overlap_tokens + sent_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk_sentences = overlap_sentences
                current_chunk_tokens = overlap_tokens
                char_position += len(chunk_text)
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunk = Chunk(
                chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                parent_doc_id=doc_id,
                text=chunk_text,
                chunk_index=len(chunks),
                start_char=char_position,
                end_char=char_position + len(chunk_text),
                metadata=metadata or {}
            )
            chunks.append(chunk)
        
        logger.debug(f"Chunked {doc_id} into {len(chunks)} sentence-based chunks")
        return chunks


class RecursiveChunker(DocumentChunker):
    """
    Recursively chunks text by trying to split on different separators.
    
    Hierarchy: paragraphs -> sentences -> words
    Best balance of semantic coherence and efficiency.
    """
    
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 50, tokenizer=None):
        super().__init__(chunk_size, chunk_overlap)
        self.tokenizer = tokenizer
        
        if tokenizer is None:
            self.chars_per_token = 4
        
        # Separators in order of preference
        self.separators = ["\n\n", "\n", ". ", " ", ""]
    
    def _get_token_count(self, text: str) -> int:
        """Get token count for text."""
        if self.tokenizer:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return len(self.tokenizer.encode(text, add_special_tokens=False, truncation=False, max_length=None))
            except Exception:
                return len(text) // self.chars_per_token
        else:
            return len(text) // self.chars_per_token
    
    def _split_text(self, text: str, separator: str) -> List[str]:
        """Split text by separator."""
        if separator == "":
            return list(text)
        return text.split(separator)
    
    def chunk_document(self, doc_id: str, text: str, metadata: Dict = None) -> List[Chunk]:
        """Recursively chunk document."""
        if not text.strip():
            return []
        
        return self._recursive_chunk(doc_id, text, metadata or {})
    
    def _recursive_chunk(
        self, 
        doc_id: str, 
        text: str, 
        metadata: Dict,
        separator_idx: int = 0
    ) -> List[Chunk]:
        """Recursively split text."""
        chunks = []
        
        if separator_idx >= len(self.separators):
            # Base case: no more separators, create single chunk
            if text.strip():
                chunk = Chunk(
                    chunk_id=f"{doc_id}_chunk_0",
                    parent_doc_id=doc_id,
                    text=text,
                    chunk_index=0,
                    start_char=0,
                    end_char=len(text),
                    metadata=metadata
                )
                return [chunk]
            return []
        
        separator = self.separators[separator_idx]
        splits = self._split_text(text, separator)
        
        current_chunk_parts = []
        current_tokens = 0
        
        for split in splits:
            if not split:
                continue
            
            split_tokens = self._get_token_count(split)
            
            # If single split is too large, recurse with next separator
            if split_tokens > self.chunk_size:
                # First, finalize current chunk if it exists
                if current_chunk_parts:
                    chunk_text = separator.join(current_chunk_parts)
                    chunk = Chunk(
                        chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                        parent_doc_id=doc_id,
                        text=chunk_text,
                        chunk_index=len(chunks),
                        start_char=0,
                        end_char=len(chunk_text),
                        metadata=metadata
                    )
                    chunks.append(chunk)
                    current_chunk_parts = []
                    current_tokens = 0
                
                # Recursively chunk the oversized split
                sub_chunks = self._recursive_chunk(
                    f"{doc_id}_sub{len(chunks)}", 
                    split, 
                    metadata, 
                    separator_idx + 1
                )
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_id = f"{doc_id}_chunk_{len(chunks)}"
                    sub_chunk.parent_doc_id = doc_id  # Fix: Use original parent doc_id
                    sub_chunk.chunk_index = len(chunks)
                    chunks.append(sub_chunk)
                continue
            
            # Check if adding this split exceeds chunk size
            if current_chunk_parts and current_tokens + split_tokens > self.chunk_size:
                # Finalize current chunk
                chunk_text = separator.join(current_chunk_parts)
                chunk = Chunk(
                    chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                    parent_doc_id=doc_id,
                    text=chunk_text,
                    chunk_index=len(chunks),
                    start_char=0,
                    end_char=len(chunk_text),
                    metadata=metadata
                )
                chunks.append(chunk)
                
                # Handle overlap (keep last part if within overlap size)
                if current_chunk_parts:
                    last_part_tokens = self._get_token_count(current_chunk_parts[-1])
                    if last_part_tokens <= self.chunk_overlap:
                        current_chunk_parts = [current_chunk_parts[-1]]
                        current_tokens = last_part_tokens
                    else:
                        current_chunk_parts = []
                        current_tokens = 0
            
            # Add split to current chunk
            current_chunk_parts.append(split)
            current_tokens += split_tokens
        
        # Finalize last chunk
        if current_chunk_parts:
            chunk_text = separator.join(current_chunk_parts)
            chunk = Chunk(
                chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                parent_doc_id=doc_id,
                text=chunk_text,
                chunk_index=len(chunks),
                start_char=0,
                end_char=len(chunk_text),
                metadata=metadata
            )
            chunks.append(chunk)
        
        return chunks


def get_chunker(strategy: str = "recursive", model_name: str = None, **kwargs) -> DocumentChunker:
    """
    Factory function to get a chunker.
    
    Args:
        strategy: Chunking strategy ("fixed", "sentence", "recursive")
        model_name: Name of the model to load tokenizer from (optional)
        **kwargs: Arguments to pass to chunker (chunk_size, chunk_overlap)
        
    Returns:
        DocumentChunker instance
    """
    # Load tokenizer if model_name provided
    tokenizer = None
    if model_name:
        from sentence_transformers import SentenceTransformer
        try:
            model = SentenceTransformer(model_name)
            tokenizer = model.tokenizer
        except Exception as e:
            logger.warning(f"Could not load tokenizer from {model_name}: {e}")
    
    strategies = {
        "fixed": FixedSizeChunker,
        "sentence": SentenceChunker,
        "recursive": RecursiveChunker,
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")
    
    # Add tokenizer to kwargs
    kwargs['tokenizer'] = tokenizer
    
    return strategies[strategy](**kwargs)
