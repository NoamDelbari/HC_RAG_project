"""
BEIR Dataset Loader

Handles loading and processing of BEIR benchmark datasets (FiQA, SciFact, etc.)
for use with HC retrieval experiments.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BEIRDocument:
    """Represents a document in a BEIR corpus."""
    doc_id: str
    text: str
    title: Optional[str] = None
    
    def get_full_text(self) -> str:
        """Get combined title and text for embedding."""
        if self.title:
            return f"{self.title}\n{self.text}"
        return self.text


@dataclass
class BEIRQuery:
    """Represents a query in a BEIR dataset."""
    query_id: str
    text: str
    relevant_docs: List[str] = None  # List of relevant doc_ids
    relevance_scores: Dict[str, int] = None  # doc_id -> relevance score
    
    def __post_init__(self):
        if self.relevant_docs is None:
            self.relevant_docs = []
        if self.relevance_scores is None:
            self.relevance_scores = {}


class BEIRLoader:
    """
    Loader for BEIR benchmark datasets.
    
    Supports FiQA, SciFact, SCIDOCS, NFCorpus, and other BEIR datasets.
    Handles automatic downloading and caching.
    """
    
    # Datasets that are publicly available via BEIR
    AVAILABLE_DATASETS = [
        "fiqa", "scifact", "scidocs", "nfcorpus", "arguana",
        "quora", "fever", "climate-fever", "hotpotqa", "nq",
        "trec-covid", "webis-touche2020", "dbpedia-entity", "msmarco"
    ]
    
    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "datasets/beir",
        split: str = "test",
        download_if_missing: bool = True
    ):
        """
        Initialize BEIR loader.
        
        Args:
            dataset_name: Name of the BEIR dataset (e.g., "fiqa", "scifact")
            data_dir: Directory to store/load datasets
            split: Data split to load ("train", "dev", "test")
            download_if_missing: Whether to download if not found locally
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = Path(data_dir)
        self.split = split
        self.download_if_missing = download_if_missing
        
        if self.dataset_name not in self.AVAILABLE_DATASETS:
            logger.warning(
                f"Dataset '{dataset_name}' not in known list. "
                f"Available: {self.AVAILABLE_DATASETS}"
            )
        
        self.data_path = self.data_dir / self.dataset_name
        
        # Lazy-loaded data
        self._corpus: Optional[Dict[str, dict]] = None
        self._queries: Optional[Dict[str, str]] = None
        self._qrels: Optional[Dict[str, Dict[str, int]]] = None
    
    def _ensure_downloaded(self) -> None:
        """Download dataset if not present."""
        if self.data_path.exists() and (self.data_path / "qrels").exists():
            return
        
        if not self.download_if_missing:
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                f"Set download_if_missing=True to download automatically."
            )
        
        logger.info(f"Downloading {self.dataset_name}...")
        try:
            from beir import util
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
            util.download_and_unzip(url, str(self.data_dir))
        except ImportError:
            raise ImportError(
                "beir package required for downloading. Install with: pip install beir"
            )
    
    def _load_data(self) -> None:
        """Load corpus, queries, and qrels."""
        if self._corpus is not None:
            return  # Already loaded
        
        self._ensure_downloaded()
        
        try:
            from beir.datasets.data_loader import GenericDataLoader
            self._corpus, self._queries, self._qrels = GenericDataLoader(
                str(self.data_path)
            ).load(split=self.split)
            
            logger.info(f"Loaded {self.dataset_name}:")
            logger.info(f"  Corpus: {len(self._corpus)} documents")
            logger.info(f"  Queries: {len(self._queries)} queries")
            logger.info(f"  QRels: {len(self._qrels)} query-doc judgments")
            
        except Exception as e:
            # Try alternative split
            try:
                alt_split = "dev" if self.split == "test" else "test"
                logger.warning(f"Failed to load {self.split} split, trying {alt_split}")
                from beir.datasets.data_loader import GenericDataLoader
                self._corpus, self._queries, self._qrels = GenericDataLoader(
                    str(self.data_path)
                ).load(split=alt_split)
            except:
                raise RuntimeError(f"Failed to load dataset: {e}")
    
    def load_documents(self) -> List[BEIRDocument]:
        """
        Load all documents from the corpus.
        
        Returns:
            List of BEIRDocument objects
        """
        self._load_data()
        
        documents = []
        for doc_id, doc_data in self._corpus.items():
            doc = BEIRDocument(
                doc_id=doc_id,
                text=doc_data.get("text", ""),
                title=doc_data.get("title", None)
            )
            documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def load_queries(self) -> List[BEIRQuery]:
        """
        Load all queries with their relevance judgments.
        
        Returns:
            List of BEIRQuery objects with relevant_docs populated
        """
        self._load_data()
        
        queries = []
        for query_id, query_text in self._queries.items():
            # Get relevant documents for this query
            relevant_docs = []
            relevance_scores = {}
            
            if query_id in self._qrels:
                for doc_id, score in self._qrels[query_id].items():
                    if score > 0:  # Only include positively relevant docs
                        relevant_docs.append(doc_id)
                        relevance_scores[doc_id] = score
            
            query = BEIRQuery(
                query_id=query_id,
                text=query_text,
                relevant_docs=relevant_docs,
                relevance_scores=relevance_scores
            )
            queries.append(query)
        
        logger.info(f"Loaded {len(queries)} queries")
        return queries
    
    def load_all(self) -> Tuple[List[BEIRDocument], List[BEIRQuery]]:
        """
        Load both documents and queries.
        
        Returns:
            Tuple of (documents, queries)
        """
        documents = self.load_documents()
        queries = self.load_queries()
        return documents, queries
    
    def get_corpus_dict(self) -> Dict[str, dict]:
        """Get raw corpus dictionary (doc_id -> {text, title})."""
        self._load_data()
        return self._corpus
    
    def get_qrels(self) -> Dict[str, Dict[str, int]]:
        """Get raw qrels dictionary (query_id -> {doc_id -> score})."""
        self._load_data()
        return self._qrels
    
    def iterate_documents(self, batch_size: int = 1000) -> Iterator[List[BEIRDocument]]:
        """
        Iterate over documents in batches.
        
        Useful for memory-efficient processing of large corpora.
        
        Args:
            batch_size: Number of documents per batch
            
        Yields:
            Lists of BEIRDocument objects
        """
        self._load_data()
        
        batch = []
        for doc_id, doc_data in self._corpus.items():
            doc = BEIRDocument(
                doc_id=doc_id,
                text=doc_data.get("text", ""),
                title=doc_data.get("title", None)
            )
            batch.append(doc)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch
    
    def get_stats(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        self._load_data()
        
        # Compute relevance distribution
        rel_counts = []
        for qid, docs in self._qrels.items():
            n_rel = sum(1 for score in docs.values() if score > 0)
            rel_counts.append(n_rel)
        
        import numpy as np
        rel_counts = np.array(rel_counts)
        
        return {
            "dataset": self.dataset_name,
            "split": self.split,
            "n_documents": len(self._corpus),
            "n_queries": len(self._queries),
            "n_qrels": len(self._qrels),
            "relevance_per_query": {
                "mean": float(np.mean(rel_counts)),
                "std": float(np.std(rel_counts)),
                "min": int(np.min(rel_counts)),
                "max": int(np.max(rel_counts)),
                "median": float(np.median(rel_counts))
            }
        }


def main():
    """Example usage of BEIRLoader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and inspect BEIR dataset")
    parser.add_argument("--dataset", type=str, default="fiqa", help="Dataset name")
    parser.add_argument("--split", type=str, default="test", help="Data split")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Loading BEIR Dataset: {args.dataset}")
    print(f"{'='*60}\n")
    
    loader = BEIRLoader(
        dataset_name=args.dataset,
        split=args.split,
        download_if_missing=True
    )
    
    # Load data
    documents, queries = loader.load_all()
    
    # Print stats
    stats = loader.get_stats()
    print(f"\nDataset Statistics:")
    print(f"  Documents: {stats['n_documents']:,}")
    print(f"  Queries: {stats['n_queries']}")
    print(f"  Relevant docs per query:")
    rel_stats = stats['relevance_per_query']
    print(f"    Mean: {rel_stats['mean']:.2f}")
    print(f"    Std: {rel_stats['std']:.2f}")
    print(f"    Range: {rel_stats['min']} - {rel_stats['max']}")
    
    # Show samples
    print(f"\n{'='*60}")
    print("Sample Documents:")
    print(f"{'='*60}")
    for doc in documents[:3]:
        print(f"\nID: {doc.doc_id}")
        print(f"Title: {doc.title}")
        print(f"Text: {doc.text[:200]}...")
    
    print(f"\n{'='*60}")
    print("Sample Queries:")
    print(f"{'='*60}")
    for query in queries[:5]:
        print(f"\nQ: {query.text}")
        print(f"   Relevant docs: {len(query.relevant_docs)}")


if __name__ == "__main__":
    main()
