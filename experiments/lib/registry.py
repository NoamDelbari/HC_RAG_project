"""
Dataset adapter registry.

Each dataset provides a small adapter class that defines dataset-specific
behavior. The generic pipeline scripts use the adapter via the registry.
"""

from experiments.lib.data_loader import (
    Query,
    load_queries_from_csv,
    load_qrels_from_csv,
    load_corpus_from_csv,
)


class DatasetAdapter:
    """Base class for dataset-specific behavior."""

    name: str = ""

    def load_queries(self, data_dir: str) -> list[Query]:
        return load_queries_from_csv(data_dir)

    def load_qrels(self, data_dir: str) -> dict[str, set[str]]:
        return load_qrels_from_csv(data_dir)

    def load_corpus(self, data_dir: str) -> dict[str, dict]:
        return load_corpus_from_csv(data_dir)

    def load_dataset(self, data_dir: str) -> tuple[list[Query], dict[str, set[str]], dict[str, dict]]:
        return self.load_queries(data_dir), self.load_qrels(data_dir), self.load_corpus(data_dir)

    def get_gold_answers(self, query: Query, corpus: dict, qrels: dict) -> list[str]:
        """Default: relevant doc titles (first sentence, truncated to 100 chars)."""
        relevant_ids = qrels.get(query.query_id, set())
        return [
            corpus[did]["text"].split(". ")[0][:100]
            for did in relevant_ids
            if did in corpus
        ]

    def get_result_metadata(self, query: Query) -> dict:
        return {}

    def get_bucket_field(self) -> str:
        return "k_bucket"


_REGISTRY: dict[str, DatasetAdapter] = {}


def register(adapter: DatasetAdapter):
    _REGISTRY[adapter.name] = adapter


def get_adapter(name: str) -> DatasetAdapter:
    if name not in _REGISTRY:
        raise KeyError(
            f"Dataset adapter '{name}' not registered. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]
