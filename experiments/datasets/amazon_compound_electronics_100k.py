"""Adapter for Amazon Compound Electronics 100K dataset (44 queries, ~100K docs)."""
from experiments.lib.registry import DatasetAdapter, register
from experiments.lib.data_loader import Query


class AmazonCompoundElectronics100kAdapter(DatasetAdapter):
    name = "amazon_compound_electronics_100k"

    def get_result_metadata(self, query: Query) -> dict:
        return {
            "category": query.metadata.get("category"),
            "price_direction": query.metadata.get("price_direction"),
            "price_threshold": query.metadata.get("price_threshold"),
            "k_bucket": query.metadata.get("k_bucket"),
        }


register(AmazonCompoundElectronics100kAdapter())
