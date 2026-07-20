"""
Generic CSV data loader for experiments.

Loads queries, qrels, and corpus from standard CSV files.
Extra columns beyond required fields go into Query.metadata dict.
"""

import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Query:
    """A generic query with metadata."""
    query_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def load_queries_from_csv(data_dir: str) -> list[Query]:
    """Load queries.csv. Columns beyond query_id and text go into metadata."""
    df = pd.read_csv(Path(data_dir) / "queries.csv")
    queries = []
    for _, row in df.iterrows():
        metadata = {
            col: row[col]
            for col in df.columns
            if col not in ("query_id", "text")
        }
        queries.append(Query(
            query_id=str(row["query_id"]),
            text=str(row["text"]),
            metadata=metadata,
        ))
    return queries


def load_qrels_from_csv(data_dir: str) -> dict[str, set[str]]:
    """Load qrels.csv. Filter relevance > 0. Returns {query_id: {doc_ids}}."""
    df = pd.read_csv(Path(data_dir) / "qrels.csv")
    qrels: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        if int(row["relevance"]) > 0:
            qrels.setdefault(str(row["query_id"]), set()).add(str(row["doc_id"]))
    return qrels


def load_corpus_from_csv(data_dir: str) -> dict[str, dict]:
    """Load corpus.csv. Returns {doc_id: {text: ..., ...all columns}}."""
    df = pd.read_csv(Path(data_dir) / "corpus.csv")
    corpus = {}
    for _, row in df.iterrows():
        doc_id = str(row["doc_id"])
        corpus[doc_id] = {col: row[col] for col in df.columns}
    return corpus
