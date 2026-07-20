"""Tests for the dataset adapter registry."""

import pytest
from experiments.lib.data_loader import Query
from experiments.lib.registry import DatasetAdapter, register, get_adapter, _REGISTRY


@pytest.fixture(autouse=True)
def clear_registry():
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()


class FakeAdapter(DatasetAdapter):
    name = "fake_dataset"

    def get_result_metadata(self, query) -> dict:
        return {"custom_field": query.metadata.get("custom_field")}


def test_register_and_get_adapter():
    adapter = FakeAdapter()
    register(adapter)
    retrieved = get_adapter("fake_dataset")
    assert retrieved is adapter


def test_get_unregistered_adapter_raises():
    with pytest.raises(KeyError):
        get_adapter("nonexistent")


def test_default_get_bucket_field():
    adapter = FakeAdapter()
    assert adapter.get_bucket_field() == "k_bucket"


def test_default_get_result_metadata_override():
    adapter = FakeAdapter()
    query = Query(query_id="q1", text="test", metadata={"custom_field": "value"})
    result = adapter.get_result_metadata(query)
    assert result == {"custom_field": "value"}


def test_default_get_gold_answers():
    adapter = FakeAdapter()
    query = Query(query_id="q1", text="test", metadata={})
    corpus = {
        "d1": {"text": "Title One. Description here."},
        "d2": {"text": "Title Two. Another description."},
    }
    qrels = {"q1": {"d1", "d2"}}
    gold = adapter.get_gold_answers(query, corpus, qrels)
    assert len(gold) == 2
    assert "Title One" in gold
    assert "Title Two" in gold
