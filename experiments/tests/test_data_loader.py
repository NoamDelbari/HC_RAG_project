"""Tests for the generic data loader."""

import pytest
import csv
from experiments.lib.data_loader import (
    Query,
    load_queries_from_csv,
    load_qrels_from_csv,
    load_corpus_from_csv,
)


@pytest.fixture
def sample_data_dir(tmp_path):
    with open(tmp_path / "queries.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "text", "subcategory", "k_bucket"])
        writer.writeheader()
        writer.writerow({"query_id": "q1", "text": "test query 1", "subcategory": "cat_a", "k_bucket": "small"})
        writer.writerow({"query_id": "q2", "text": "test query 2", "subcategory": "cat_b", "k_bucket": "large"})

    with open(tmp_path / "qrels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "doc_id", "relevance"])
        writer.writeheader()
        writer.writerow({"query_id": "q1", "doc_id": "d1", "relevance": 1})
        writer.writerow({"query_id": "q1", "doc_id": "d2", "relevance": 1})
        writer.writerow({"query_id": "q1", "doc_id": "d3", "relevance": 0})
        writer.writerow({"query_id": "q2", "doc_id": "d4", "relevance": 1})

    with open(tmp_path / "corpus.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "text", "type"])
        writer.writeheader()
        writer.writerow({"doc_id": "d1", "text": "Title One. Description one.", "type": "relevant"})
        writer.writerow({"doc_id": "d2", "text": "Title Two. Description two.", "type": "relevant"})
        writer.writerow({"doc_id": "d3", "text": "Title Three. Description three.", "type": "distractor"})
        writer.writerow({"doc_id": "d4", "text": "Title Four. Description four.", "type": "relevant"})
    return tmp_path


def test_load_queries_returns_query_objects(sample_data_dir):
    queries = load_queries_from_csv(str(sample_data_dir))
    assert len(queries) == 2
    assert isinstance(queries[0], Query)
    assert queries[0].query_id == "q1"
    assert queries[0].text == "test query 1"


def test_load_queries_extra_columns_in_metadata(sample_data_dir):
    queries = load_queries_from_csv(str(sample_data_dir))
    assert queries[0].metadata["subcategory"] == "cat_a"
    assert queries[0].metadata["k_bucket"] == "small"
    assert "query_id" not in queries[0].metadata
    assert "text" not in queries[0].metadata


def test_load_qrels_filters_relevance(sample_data_dir):
    qrels = load_qrels_from_csv(str(sample_data_dir))
    assert "q1" in qrels
    assert qrels["q1"] == {"d1", "d2"}
    assert qrels["q2"] == {"d4"}


def test_load_corpus_returns_dict(sample_data_dir):
    corpus = load_corpus_from_csv(str(sample_data_dir))
    assert len(corpus) == 4
    assert corpus["d1"]["text"] == "Title One. Description one."
    assert corpus["d1"]["type"] == "relevant"
