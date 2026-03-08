"""
Data Loader for Category-Key Dataset

Provides CategoryKeyQuery dataclass and loader function compatible
with the existing retrieval/evaluation pipeline.
"""

import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass
class CategoryKeyQuery:
    """A single query from the category-key dataset."""
    query_id: str
    text: str
    country: str
    size_bucket: str  # "small", "medium", "large"


def load_category_key_dataset(
    output_dir: str = None
) -> Tuple[List[CategoryKeyQuery], Dict[str, Set[str]]]:
    """
    Load the category-key dataset.

    Args:
        output_dir: Path to output directory (default: datasets/category_key)

    Returns:
        (queries, qrels) where:
            queries: List of CategoryKeyQuery objects
            qrels: Dict mapping query_id -> Set of relevant doc_ids
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "datasets" / "category_key"
    else:
        output_dir = Path(output_dir)

    # Load queries
    queries_df = pd.read_csv(output_dir / "queries.csv")
    queries = [
        CategoryKeyQuery(
            query_id=row["query_id"],
            text=row["text"],
            country=row["country"],
            size_bucket=row["size_bucket"],
        )
        for _, row in queries_df.iterrows()
    ]

    # Load qrels
    qrels_df = pd.read_csv(output_dir / "qrels.csv")
    qrels: Dict[str, Set[str]] = {}
    for _, row in qrels_df.iterrows():
        if row["relevance"] > 0:
            qrels.setdefault(row["query_id"], set()).add(row["doc_id"])

    print(f"Loaded category-key dataset:")
    print(f"  Queries: {len(queries)}")
    print(f"  Queries with relevance labels: {len(qrels)}")
    if qrels:
        k_values = [len(v) for v in qrels.values()]
        print(f"  K range: {min(k_values)} - {max(k_values)}")

    return queries, qrels


def load_corpus(output_dir: str = None) -> Dict[str, dict]:
    """
    Load the corpus as a dict mapping doc_id -> document.

    Args:
        output_dir: Path to output directory

    Returns:
        Dict mapping doc_id -> document dict
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "datasets" / "category_key"
    else:
        output_dir = Path(output_dir)

    corpus_df = pd.read_csv(output_dir / "corpus.csv")
    corpus = {
        row["doc_id"]: row.to_dict()
        for _, row in corpus_df.iterrows()
    }

    return corpus


if __name__ == "__main__":
    queries, qrels = load_category_key_dataset()
    for q in queries[:3]:
        k = len(qrels.get(q.query_id, set()))
        print(f"  {q.query_id}: '{q.text}' (K={k}, bucket={q.size_bucket})")
