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


register(AmazonCategoriesMax50Adapter())
