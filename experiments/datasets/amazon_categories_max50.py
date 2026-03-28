"""Adapter for Amazon Categories Max50 dataset (92 queries, true_k <= 50)."""
from experiments.lib.registry import DatasetAdapter, register
from experiments.lib.data_loader import Query


class AmazonCategoriesMax50Adapter(DatasetAdapter):
    name = "amazon_categories_max50"

    def get_result_metadata(self, query: Query) -> dict:
        return {
            "subcategory": query.metadata.get("subcategory"),
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


register(AmazonCategoriesMax50Adapter())
