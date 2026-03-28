"""Results serialization and analysis utilities."""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def save_results_json(data: dict, path: str):
    """Save results dict to JSON with numpy type handling."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def load_results_json(path: str) -> dict:
    """Load results dict from JSON."""
    with open(path) as f:
        return json.load(f)


def analyze_by_bucket(
    results: list[dict],
    bucket_field: str = "k_bucket",
) -> dict[str, dict]:
    """Group per-query results by k-bucket and compute mean metrics."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        bucket = r.get("metadata", {}).get(bucket_field, "unknown")
        buckets[str(bucket)].append(r)

    summary = {}
    metric_keys = [
        "recall", "precision", "mrr", "ndcg", "map", "hit_rate", "hc_k",
    ]
    for bucket_name, bucket_results in sorted(buckets.items()):
        bucket_summary = {"count": len(bucket_results)}
        for key in metric_keys:
            values = [r[key] for r in bucket_results if key in r]
            if values:
                bucket_summary[f"mean_{key}"] = float(np.mean(values))
                bucket_summary[f"median_{key}"] = float(np.median(values))
        summary[bucket_name] = bucket_summary

    return summary
