"""Adapter for Amazon Compound dataset (135 queries, ~80K docs, category + price)."""
from experiments.lib.registry import DatasetAdapter, register
from experiments.lib.data_loader import Query


class AmazonCompoundAdapter(DatasetAdapter):
    name = "amazon_compound"

    def get_result_metadata(self, query: Query) -> dict:
        return {
            "category": query.metadata.get("category"),
            "price_direction": query.metadata.get("price_direction"),
            "price_threshold": query.metadata.get("price_threshold"),
            "k_bucket": query.metadata.get("k_bucket"),
        }

    def get_gold_answers(self, query: Query, corpus: dict, qrels: dict) -> list[str]:
        relevant_ids = qrels.get(query.query_id, set())
        titles = []
        for doc_id in sorted(relevant_ids):
            doc = corpus.get(doc_id)
            if doc and "text" in doc:
                full_text = str(doc["text"])
                dot_pos = full_text.find(". ")
                title = full_text[:dot_pos] if dot_pos > 0 else full_text
                titles.append(title[:100].strip())
        return titles


register(AmazonCompoundAdapter())
