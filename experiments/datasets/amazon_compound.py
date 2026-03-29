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


register(AmazonCompoundAdapter())
