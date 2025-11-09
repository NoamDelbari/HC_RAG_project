"""
CRAG Dataset Loader

Handles extraction and loading of CRAG benchmark dataset files.
Supports both .jsonl.bz2 and .tar.bz2 formats.
"""

import bz2
import json
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CRAGQuery:
    """Represents a single query in the CRAG dataset."""
    query_id: str
    query: str
    answer: Optional[str] = None
    search_results: Optional[List[Dict]] = None
    task: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "CRAGQuery":
        """Create CRAGQuery from dictionary."""
        return cls(
            query_id=data.get("interaction_id", ""),
            query=data.get("query", ""),
            answer=data.get("answer"),
            search_results=data.get("search_results", []),
            task=data.get("domain", "")
        )


@dataclass
class CRAGDocument:
    """Represents a document in the CRAG corpus."""
    doc_id: str
    text: str
    title: Optional[str] = None
    url: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict, doc_id: str = None, use_full_html: bool = False) -> "CRAGDocument":
        """
        Create CRAGDocument from dictionary.

        Args:
            data: Dictionary from search_results containing page_* fields
            doc_id: Optional document ID to use
            use_full_html: If True, use page_result (full HTML) instead of page_snippet
        """
        # Choose between full HTML or snippet
        if use_full_html:
            text = data.get("page_result", data.get("page_snippet", ""))
        else:
            text = data.get("page_snippet", "")

        return cls(
            doc_id=doc_id or data.get("id", ""),
            text=text,
            title=data.get("page_name", ""),
            url=data.get("page_url", "")
        )


class CRAGLoader:
    """
    Loader for CRAG benchmark dataset.

    Handles extraction of compressed files and parsing of JSONL format.
    """

    def __init__(self, dataset_dir: str = "datasets/crag", use_full_html: bool = False):
        """
        Initialize CRAG loader.

        Args:
            dataset_dir: Path to directory containing CRAG dataset files
            use_full_html: If True, use full HTML (page_result) instead of snippets (page_snippet)
        """
        self.dataset_dir = Path(dataset_dir)
        self.use_full_html = use_full_html
        self._validate_dataset_dir()

    def _validate_dataset_dir(self) -> None:
        """Validate that dataset directory exists."""
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.dataset_dir}")

    def extract_bz2_jsonl(self, filename: str) -> Iterator[Dict]:
        """
        Extract and parse .jsonl.bz2 file.

        Args:
            filename: Name of the .jsonl.bz2 file

        Yields:
            Parsed JSON objects from the file
        """
        filepath = self.dataset_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info(f"Extracting {filename}...")

        with bz2.open(filepath, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue

    def extract_tar_bz2(self, filename: str) -> Path:
        """
        Extract .tar.bz2 file directly to dataset directory.

        Args:
            filename: Name of the .tar.bz2 file

        Returns:
            Path to the dataset directory where files were extracted
        """
        filepath = self.dataset_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Check if already extracted by looking for JSONL files
        existing_jsonl = list(self.dataset_dir.rglob(f"{filepath.stem.replace('.tar', '')}*.jsonl"))
        if existing_jsonl:
            logger.info(f"Task 3 JSONL files already extracted ({len(existing_jsonl)} files found)")
            return self.dataset_dir

        logger.info(f"Extracting {filename} to {self.dataset_dir}...")

        with tarfile.open(filepath, 'r:bz2') as tar:
            tar.extractall(path=self.dataset_dir)

        logger.info(f"Extraction complete: {self.dataset_dir}")
        return self.dataset_dir

    def load_queries(self, filename: str = "crag_task_1_and_2_dev_v4.jsonl.bz2") -> List[CRAGQuery]:
        """
        Load queries from CRAG dataset.

        Args:
            filename: Name of the JSONL file containing queries

        Returns:
            List of CRAGQuery objects
        """
        queries = []

        for data in self.extract_bz2_jsonl(filename):
            query = CRAGQuery.from_dict(data)
            queries.append(query)

        logger.info(f"Loaded {len(queries)} queries from {filename}")
        return queries

    def load_documents_from_queries(self, queries: List[CRAGQuery]) -> List[CRAGDocument]:
        """
        Extract all documents from query search results.

        Args:
            queries: List of CRAGQuery objects containing search results

        Returns:
            List of unique CRAGDocument objects
        """
        documents = []
        doc_ids_seen = set()

        for query in queries:
            if not query.search_results:
                continue

            for idx, result in enumerate(query.search_results):
                # Create unique document ID
                doc_id = f"{query.query_id}_doc_{idx}"

                if doc_id in doc_ids_seen:
                    continue

                doc = CRAGDocument.from_dict(result, doc_id=doc_id, use_full_html=self.use_full_html)

                # Only add documents with non-empty text
                if doc.text.strip():
                    documents.append(doc)
                    doc_ids_seen.add(doc_id)

        logger.info(f"Extracted {len(documents)} unique documents from queries")
        return documents

    def load_from_jsonl(self, jsonl_path: Path) -> List[CRAGQuery]:
        """
        Load queries from an extracted JSONL file.

        Args:
            jsonl_path: Path to the JSONL file

        Returns:
            List of CRAGQuery objects
        """
        queries = []

        logger.info(f"Loading from {jsonl_path}...")

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    query = CRAGQuery.from_dict(data)
                    queries.append(query)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(queries)} queries from {jsonl_path}")
        return queries

    def load_task_3(self, tar_filename: str = "crag_task_3_dev_v4.tar.bz2") -> tuple[List[CRAGQuery], List[CRAGDocument]]:
        """
        Load Task 3 data (up to 50 pages per query with full HTML).

        Args:
            tar_filename: Name of the Task 3 .tar.bz2 file

        Returns:
            Tuple of (queries, documents)
        """
        # Extract the tar.bz2 archive
        self.extract_tar_bz2(tar_filename)

        # Find all JSONL files recursively (handles nested directory structure)
        task_prefix = tar_filename.replace(".tar.bz2", "")
        jsonl_files = list(self.dataset_dir.rglob(f"{task_prefix}*.jsonl"))

        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files found for Task 3 in {self.dataset_dir}")

        logger.info(f"Found {len(jsonl_files)} JSONL file(s) in Task 3 archive")

        # Load queries from all JSONL files
        all_queries = []
        for jsonl_file in jsonl_files:
            queries = self.load_from_jsonl(jsonl_file)
            all_queries.extend(queries)

        logger.info(f"Loaded total {len(all_queries)} queries from Task 3")

        # Extract documents
        documents = self.load_documents_from_queries(all_queries)

        return all_queries, documents

    def load_by_tasks(self, tasks: List[str] = ["1_2"]) -> tuple[List[CRAGQuery], List[CRAGDocument]]:
        """
        Load data from selected tasks.

        Args:
            tasks: List of task identifiers. Options: "1_2" (Tasks 1&2), "3" (Task 3), or both

        Returns:
            Tuple of (queries, documents) from all selected tasks

        Examples:
            >>> loader.load_by_tasks(["1_2"])  # Load only Tasks 1 & 2
            >>> loader.load_by_tasks(["3"])    # Load only Task 3
            >>> loader.load_by_tasks(["1_2", "3"])  # Load both
        """
        all_queries = []
        all_documents = []

        if "1_2" in tasks:
            logger.info("Loading Tasks 1 & 2...")
            queries, docs = self.load_all(filename="crag_task_1_and_2_dev_v4.jsonl.bz2")
            all_queries.extend(queries)
            all_documents.extend(docs)

        if "3" in tasks:
            logger.info("Loading Task 3...")
            queries, docs = self.load_task_3(tar_filename="crag_task_3_dev_v4.tar.bz2")
            all_queries.extend(queries)
            all_documents.extend(docs)

        logger.info(f"Total loaded: {len(all_queries)} queries, {len(all_documents)} documents")
        return all_queries, all_documents

    def load_all(self, filename: str = "crag_task_1_and_2_dev_v4.jsonl.bz2") -> tuple[List[CRAGQuery], List[CRAGDocument]]:
        """
        Load both queries and documents from CRAG dataset.

        Args:
            filename: Name of the JSONL file

        Returns:
            Tuple of (queries, documents)
        """
        queries = self.load_queries(filename)
        documents = self.load_documents_from_queries(queries)

        return queries, documents


def main():
    """Example usage of CRAGLoader."""
    import sys

    # Default: Load Tasks 1 & 2 only
    task_selection = ["1_2"]

    # Check command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "task3":
            task_selection = ["3"]
        elif arg == "both":
            task_selection = ["1_2", "3"]
        elif arg == "task12":
            task_selection = ["1_2"]

    print(f"\n{'='*60}")
    print(f"CRAG Dataset Loader")
    print(f"{'='*60}")
    print(f"Loading tasks: {', '.join(task_selection)}")
    print(f"Use full HTML: False (using snippets)")
    print(f"{'='*60}\n")

    # Initialize loader
    loader = CRAGLoader(use_full_html=False)

    # Load selected tasks
    queries, documents = loader.load_by_tasks(tasks=task_selection)

    # Print summary
    print(f"\n{'='*60}")
    print(f"CRAG Dataset Summary")
    print(f"{'='*60}")
    print(f"Total queries: {len(queries)}")
    print(f"Total documents: {len(documents)}")

    if queries:
        print(f"\nFirst query example:")
        print(f"  ID: {queries[0].query_id}")
        print(f"  Query: {queries[0].query[:100]}...")
        print(f"  Answer: {queries[0].answer[:100] if queries[0].answer else 'N/A'}...")
        print(f"  Search results: {len(queries[0].search_results) if queries[0].search_results else 0}")

    if documents:
        print(f"\nFirst document example:")
        print(f"  ID: {documents[0].doc_id}")
        print(f"  Title: {documents[0].title}")
        print(f"  Text length: {len(documents[0].text)} chars")
        print(f"  Text preview: {documents[0].text[:100]}...")

    print(f"\n{'='*60}")
    print(f"Usage examples:")
    print(f"  python src/data/crag_loader.py          # Load Tasks 1 & 2")
    print(f"  python src/data/crag_loader.py task3    # Load Task 3 only")
    print(f"  python src/data/crag_loader.py both     # Load all tasks")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
