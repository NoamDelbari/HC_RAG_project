"""
Data Loader for Amazon Compound Query Dataset

Provides CompoundQuery dataclass and loader functions compatible
with the existing retrieval/evaluation pipeline.
"""

import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass
class CompoundQuery:
    """A single query from the Amazon compound dataset."""
    query_id: str
    text: str
    category: str
    price_direction: str      # "above" or "below"
    price_threshold: float
    k_bucket: str             # "small", "medium", "large"


def load_compound_dataset(
    output_dir: str = None,
) -> Tuple[List[CompoundQuery], Dict[str, Set[str]]]:
    """
    Load the Amazon compound query dataset.

    Args:
        output_dir: Path to output directory (default: datasets/amazon_compound)

    Returns:
        (queries, qrels) where:
            queries: List of CompoundQuery objects
            qrels: Dict mapping query_id -> Set of relevant doc_ids
    """
    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "datasets"
            / "amazon_compound"
        )
    else:
        output_dir = Path(output_dir)

    # Load queries
    queries_df = pd.read_csv(output_dir / "queries.csv")
    queries = [
        CompoundQuery(
            query_id=row["query_id"],
            text=row["text"],
            category=row["category"],
            price_direction=row["price_direction"],
            price_threshold=float(row["price_threshold"]),
            k_bucket=row["k_bucket"],
        )
        for _, row in queries_df.iterrows()
    ]

    # Load qrels
    qrels_df = pd.read_csv(output_dir / "qrels.csv")
    qrels: Dict[str, Set[str]] = {}
    for _, row in qrels_df.iterrows():
        if row["relevance"] > 0:
            qrels.setdefault(row["query_id"], set()).add(row["doc_id"])

    print(f"Loaded Amazon compound dataset:")
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
        output_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "datasets"
            / "amazon_compound"
        )
    else:
        output_dir = Path(output_dir)

    corpus_df = pd.read_csv(output_dir / "corpus.csv")
    corpus = {row["doc_id"]: row.to_dict() for _, row in corpus_df.iterrows()}

    return corpus


def generate_gold_answers(
    output_dir: str = None,
) -> Dict[str, str]:
    """
    Generate gold answers for each query based on relevant document titles.

    For compound queries, the gold answer lists titles of products that
    match both the category and price constraint.

    Args:
        output_dir: Path to dataset directory

    Returns:
        Dict mapping query_id -> gold answer string
    """
    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "datasets"
            / "amazon_compound"
        )
    else:
        output_dir = Path(output_dir)

    queries, qrels = load_compound_dataset(str(output_dir))
    corpus = load_corpus(str(output_dir))

    gold_answers = {}
    for query in queries:
        relevant_ids = qrels.get(query.query_id, set())
        if not relevant_ids:
            gold_answers[query.query_id] = "No relevant documents found."
            continue

        titles = []
        for doc_id in sorted(relevant_ids):
            doc = corpus.get(doc_id)
            if doc and "text" in doc:
                full_text = str(doc["text"])
                dot_pos = full_text.find(". ")
                title = full_text[:dot_pos] if dot_pos > 0 else full_text
                title = title[:50].strip()
                titles.append(title)

        gold_answers[query.query_id] = ", ".join(titles) if titles else "No titles available."

    return gold_answers


if __name__ == "__main__":
    queries, qrels = load_compound_dataset()
    for q in queries[:5]:
        k = len(qrels.get(q.query_id, set()))
        print(f"  {q.query_id}: '{q.text}' (K={k}, bucket={q.k_bucket}, "
              f"dir={q.price_direction}, threshold=${q.price_threshold:.2f})")
