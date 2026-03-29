"""
Generate Amazon Compound Query Dataset (Category + Price Constraints)

Loads Amazon metadata from HuggingFace, selects subcategories where a price
threshold produces K=5-50 relevant products, and outputs corpus.csv,
queries.csv, qrels.csv, metadata.json.

Key design:
  - Query: "Find {category} products priced above/below ${threshold}"
  - Relevance = category match AND price constraint satisfied
  - Hard negatives = same category, wrong price (included in corpus)
  - Easy negatives = random products from other categories

Usage:
  python generate_dataset.py                              # all 3 configs
  python generate_dataset.py --configs Clothing_Shoes_and_Jewelry Electronics
  python generate_dataset.py --max-queries 50             # limit queries
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_CONFIGS = [
    "raw_meta_Clothing_Shoes_and_Jewelry",
    "raw_meta_Electronics",
    "raw_meta_Home_and_Kitchen",
]

K_BUCKETS = {
    "small": (5, 15),
    "medium": (16, 30),
    "large": (31, 50),
}

TARGET_PER_BUCKET = (50, 80)
TARGET_CORPUS_SIZE = (80_000, 120_000)
CONTENT_MIN_WORDS = 30
SEED = 42


def parse_price(price_raw) -> float | None:
    """Parse Amazon price string to float."""
    if price_raw is None:
        return None
    s = str(price_raw).strip()
    if not s:
        return None
    m = re.search(r"\$?([\d,]+(?:\.\d{1,2})?)", s)
    if not m:
        return None
    try:
        val = float(m.group(1).replace(",", ""))
        if val <= 0 or val > 100_000:
            return None
        return val
    except (ValueError, OverflowError):
        return None


def make_doc_id(parent_asin: str) -> str:
    """Create deterministic doc ID from parent_asin."""
    return hashlib.md5(parent_asin.encode()).hexdigest()[:12]


def make_query_id(category_path: str, direction: str) -> str:
    """Create deterministic query ID from category path + direction."""
    sanitized = category_path.lower().replace(" > ", "_").replace(" ", "_")
    sanitized = "".join(c for c in sanitized if c.isalnum() or c == "_")
    return f"q_{sanitized[:70]}_{direction}"


def build_doc_text(item: dict, price: float) -> str:
    """Build document text with price suffix."""
    title = item.get("title") or ""
    description = item.get("description") or []
    features = item.get("features") or []

    if isinstance(description, list):
        description = " ".join(str(d) for d in description)
    if isinstance(features, list):
        features = " ".join(str(f) for f in features)

    text = title
    if description:
        text += ". " + description
    if features:
        text += " " + features

    # Append price as structured suffix
    text += f" Price: ${price:.2f}."
    return text.strip()


def extract_category_path(categories: list, level: int = 2) -> str | None:
    """Extract category path up to the given level (inclusive)."""
    if not categories or not isinstance(categories, list):
        return None
    if isinstance(categories[0], list):
        path = categories[0]
    elif isinstance(categories[0], str):
        path = categories
    else:
        return None
    if len(path) <= level:
        return None
    return " > ".join(str(s) for s in path[: level + 1])


def passes_content_filter(item: dict) -> bool:
    """Check if product has title and >= 30 words combined text."""
    title = item.get("title") or ""
    if not title.strip():
        return False
    desc = item.get("description") or ""
    feats = item.get("features") or ""
    if isinstance(desc, list):
        desc = " ".join(str(d) for d in desc)
    if isinstance(feats, list):
        feats = " ".join(str(f) for f in feats)
    combined = f"{title} {desc} {feats}"
    return len(combined.split()) >= CONTENT_MIN_WORDS


def find_threshold_for_k(
    prices: list[float], k_lo: int, k_hi: int
) -> tuple | None:
    """
    Find a price threshold producing K in [k_lo, k_hi].
    Returns (direction, threshold, k) or None.
    """
    if len(prices) < k_lo:
        return None

    sorted_prices = sorted(prices)
    n = len(sorted_prices)

    # Try both directions; prefer the one giving K closest to middle of range
    best = None
    target_mid = (k_lo + k_hi) / 2

    for direction in ["above", "below"]:
        if direction == "above":
            for k in range(k_lo, min(k_hi + 1, n)):
                idx = n - k
                if idx <= 0:
                    continue
                threshold = (sorted_prices[idx - 1] + sorted_prices[idx]) / 2.0
                actual_k = sum(1 for p in prices if p > threshold)
                if k_lo <= actual_k <= k_hi:
                    dist = abs(actual_k - target_mid)
                    if best is None or dist < best[3]:
                        best = (direction, round(threshold, 2), actual_k, dist)
        else:
            for k in range(k_lo, min(k_hi + 1, n)):
                if k >= n:
                    continue
                threshold = (sorted_prices[k - 1] + sorted_prices[k]) / 2.0
                actual_k = sum(1 for p in prices if p < threshold)
                if k_lo <= actual_k <= k_hi:
                    dist = abs(actual_k - target_mid)
                    if best is None or dist < best[3]:
                        best = (direction, round(threshold, 2), actual_k, dist)

    if best is None:
        return None
    return best[0], best[1], best[2]


def load_and_filter_data(configs: list[str], max_items_per_config: int = 0):
    """Load metadata from HuggingFace configs and filter products."""
    import os
    import time
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    # category -> list of product dicts (with price)
    cat_products: Dict[str, List[dict]] = defaultdict(list)
    # All products for distractor pool (with price and content)
    all_products: List[dict] = []
    seen_asins: Set[str] = set()

    total_all = 0
    total_content = 0
    total_priced = 0

    for config_name in configs:
        print(f"\nLoading {config_name}...")
        if max_items_per_config:
            print(f"  (capped at {max_items_per_config:,} items)")

        max_retries = 5
        items_processed = 0

        for attempt in range(max_retries):
            try:
                ds = load_dataset(
                    "McAuley-Lab/Amazon-Reviews-2023",
                    config_name,
                    split="full",
                    streaming=True,
                    trust_remote_code=True,
                )

                for i, item in enumerate(ds):
                    if i < items_processed:
                        continue
                    if max_items_per_config and items_processed >= max_items_per_config:
                        break
                    items_processed += 1
                    total_all += 1
                    if items_processed % 200_000 == 0:
                        print(f"  Processed {items_processed:,}...")

                    asin = item.get("parent_asin")
                    if not asin or asin in seen_asins:
                        continue

                    if not passes_content_filter(item):
                        continue
                    total_content += 1

                    price = parse_price(item.get("price"))
                    if price is None:
                        continue
                    total_priced += 1

                    seen_asins.add(asin)

                    cat_path = extract_category_path(item.get("categories", []), level=2)

                    product = {
                        "parent_asin": asin,
                        "title": item.get("title", ""),
                        "description": item.get("description", []),
                        "features": item.get("features", []),
                        "categories": item.get("categories", []),
                        "price": price,
                        "category_path": cat_path,
                        "config": config_name,
                    }

                    all_products.append(product)
                    if cat_path:
                        cat_products[cat_path].append(product)

                break  # completed successfully
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  Connection error at item {items_processed}: {e}")
                    print(f"  Retrying in {wait}s (attempt {attempt+2}/{max_retries})...")
                    time.sleep(wait)
                else:
                    print(f"  FATAL: Failed after {max_retries} attempts")
                    raise

        print(f"  {config_name}: {items_processed:,} items streamed, "
              f"{total_priced:,} with content+price so far")

    print(f"\nTotal: {total_all:,} items streamed")
    print(f"  With content: {total_content:,}")
    print(f"  With content + price: {total_priced:,}")
    print(f"  Unique subcategories: {len(cat_products)}")

    return cat_products, all_products


def select_queries(
    cat_products: Dict[str, List[dict]],
    seed: int = SEED,
    max_queries: int = 0,
) -> List[dict]:
    """
    Select subcategories and find price thresholds producing K=5-50.
    Returns list of query specs.
    """
    random.seed(seed)

    # Find viable categories — for each category, try ALL bucket ranges
    viable_by_bucket: Dict[str, List[dict]] = {b: [] for b in K_BUCKETS}
    for cat_path, products in cat_products.items():
        prices = [p["price"] for p in products]
        for bname, (blo, bhi) in K_BUCKETS.items():
            result = find_threshold_for_k(prices, blo, bhi)
            if result is None:
                continue
            direction, threshold, k = result
            viable_by_bucket[bname].append({
                "category_path": cat_path,
                "direction": direction,
                "threshold": threshold,
                "k": k,
                "bucket": bname,
                "n_total_in_cat": len(products),
            })

    # Greedily assign categories to buckets, prioritizing scarce buckets
    viable = []
    used_cats = set()
    bucket_order = sorted(viable_by_bucket.keys(),
                          key=lambda b: len(viable_by_bucket[b]))
    for bname in bucket_order:
        candidates = [v for v in viable_by_bucket[bname]
                       if v["category_path"] not in used_cats]
        random.shuffle(candidates)
        for v in candidates:
            used_cats.add(v["category_path"])
            viable.append(v)

    print(f"\nViable subcategories: {len(viable)}")
    for bname in K_BUCKETS:
        count = sum(1 for v in viable if v["bucket"] == bname)
        print(f"  {bname}: {count} categories")

    # Balance across K buckets
    bucketed: Dict[str, List[dict]] = {b: [] for b in K_BUCKETS}
    for v in viable:
        bucketed[v["bucket"]].append(v)

    print("Before selection:")
    for bname, items in bucketed.items():
        print(f"  {bname}: {len(items)}")

    # Select balanced set
    selected = []
    target_lo, target_hi = TARGET_PER_BUCKET
    for bname, items in bucketed.items():
        random.shuffle(items)
        n = min(len(items), target_hi)
        if n < target_lo:
            print(f"  WARNING: {bname} has only {len(items)} (target {target_lo}-{target_hi})")
        selected.extend(items[:n])

    if max_queries and len(selected) > max_queries:
        random.shuffle(selected)
        selected = selected[:max_queries]

    # Re-count buckets
    print(f"\nSelected queries: {len(selected)}")
    for bname in K_BUCKETS:
        count = sum(1 for s in selected if s["bucket"] == bname)
        print(f"  {bname}: {count}")

    return selected


def build_corpus(
    query_specs: List[dict],
    cat_products: Dict[str, List[dict]],
    all_products: List[dict],
    seed: int = SEED,
) -> Tuple[List[dict], Dict[str, List[str]], List[dict]]:
    """
    Build corpus with relevant, hard-negative, and easy-negative documents.

    Returns (documents, qrels, queries)
    """
    random.seed(seed)

    documents = []
    qrels: Dict[str, List[str]] = {}
    queries = []
    seen_asins: Set[str] = set()

    # Track which categories are selected
    selected_cats = {qs["category_path"] for qs in query_specs}

    print("\nBuilding corpus...")

    # For each query: add relevant docs AND hard negatives (same cat, wrong price)
    for qs in query_specs:
        cat_path = qs["category_path"]
        direction = qs["direction"]
        threshold = qs["threshold"]
        bucket = qs["bucket"]

        query_id = make_query_id(cat_path, direction)
        leaf_cat = cat_path.split(" > ")[-1]
        if direction == "above":
            query_text = f"Find {leaf_cat} products priced above ${threshold:.2f}"
        else:
            query_text = f"Find {leaf_cat} products priced below ${threshold:.2f}"

        queries.append({
            "query_id": query_id,
            "text": query_text,
            "category": cat_path,
            "price_direction": direction,
            "price_threshold": threshold,
            "k_bucket": bucket,
        })

        qrels[query_id] = []

        products_in_cat = cat_products.get(cat_path, [])

        # Split into relevant and hard-negative candidates
        relevant_prods = []
        hard_neg_prods = []
        for product in products_in_cat:
            price = product["price"]
            is_rel = (direction == "above" and price > threshold) or \
                     (direction == "below" and price < threshold)
            if is_rel:
                relevant_prods.append(product)
            else:
                hard_neg_prods.append(product)

        # Cap hard negatives: keep at most 5x the relevant count (min 50)
        max_hard_neg = max(len(relevant_prods) * 5, 50)
        if len(hard_neg_prods) > max_hard_neg:
            random.shuffle(hard_neg_prods)
            hard_neg_prods = hard_neg_prods[:max_hard_neg]

        # Add all products (relevant first, then hard negatives)
        for product in relevant_prods + hard_neg_prods:
            asin = product["parent_asin"]
            price = product["price"]
            is_rel = (direction == "above" and price > threshold) or \
                     (direction == "below" and price < threshold)

            if asin in seen_asins:
                if is_rel:
                    doc_id = make_doc_id(asin)
                    qrels[query_id].append(doc_id)
                continue

            seen_asins.add(asin)
            doc_id = make_doc_id(asin)
            text = build_doc_text(product, price)

            if is_rel:
                doc_type = "relevant"
                qrels[query_id].append(doc_id)
            else:
                doc_type = "hard_negative"

            documents.append({
                "doc_id": doc_id,
                "text": text,
                "type": doc_type,
                "subcategory": cat_path,
                "k_bucket": bucket if doc_type == "relevant" else None,
                "price": price,
            })

    n_relevant = sum(1 for d in documents if d["type"] == "relevant")
    n_hard_neg = sum(1 for d in documents if d["type"] == "hard_negative")
    print(f"  Relevant documents: {n_relevant:,}")
    print(f"  Hard negatives (same cat, wrong price): {n_hard_neg:,}")

    # Add easy negatives from non-selected categories
    print("  Building easy-negative pool...")
    distractor_pool = []
    for product in all_products:
        asin = product["parent_asin"]
        cat = product.get("category_path")
        if asin in seen_asins:
            continue
        if cat in selected_cats:
            continue
        distractor_pool.append(product)

    random.shuffle(distractor_pool)

    target_lo, target_hi = TARGET_CORPUS_SIZE
    current = len(documents)
    if current >= target_hi:
        needed = 0
    else:
        needed = max(target_lo - current, 0)
        needed = min(needed, target_hi - current, len(distractor_pool))

    print(f"  Easy-negative pool: {len(distractor_pool):,}")
    print(f"  Sampling {needed:,} easy negatives...")

    for product in distractor_pool[:needed]:
        asin = product["parent_asin"]
        if asin in seen_asins:
            continue
        seen_asins.add(asin)

        doc_id = make_doc_id(asin)
        price = product["price"]
        text = build_doc_text(product, price)
        documents.append({
            "doc_id": doc_id,
            "text": text,
            "type": "easy_negative",
            "subcategory": None,
            "k_bucket": None,
            "price": price,
        })

    n_easy_neg = sum(1 for d in documents if d["type"] == "easy_negative")
    print(f"  Easy negatives added: {n_easy_neg:,}")
    print(f"  Total corpus: {len(documents):,}")

    pct_rel = n_relevant / len(documents) * 100
    pct_hard = n_hard_neg / len(documents) * 100
    pct_easy = n_easy_neg / len(documents) * 100
    print(f"  Breakdown: relevant={pct_rel:.1f}%, hard_neg={pct_hard:.1f}%, easy_neg={pct_easy:.1f}%")

    # Shuffle corpus
    random.shuffle(documents)

    return documents, qrels, queries


def save_dataset(
    output_dir: Path,
    documents: List[dict],
    queries: List[dict],
    qrels: Dict[str, List[str]],
    metadata: dict,
):
    """Save dataset to output directory as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save corpus
    corpus_df = pd.DataFrame(documents)
    corpus_df = corpus_df[["doc_id", "text", "type", "subcategory", "k_bucket", "price"]]
    corpus_df.to_csv(output_dir / "corpus.csv", index=False)

    # Save queries
    queries_df = pd.DataFrame(queries)
    queries_df = queries_df[["query_id", "text", "category", "price_direction", "price_threshold", "k_bucket"]]
    queries_df.to_csv(output_dir / "queries.csv", index=False)

    # Save qrels
    qrels_rows = []
    for query_id, doc_ids in sorted(qrels.items()):
        for doc_id in doc_ids:
            qrels_rows.append({"query_id": query_id, "doc_id": doc_id, "relevance": 1})
    qrels_df = pd.DataFrame(qrels_rows)
    qrels_df.to_csv(output_dir / "qrels.csv", index=False)

    # Save metadata
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset saved to {output_dir}")
    print(f"  Corpus: corpus.csv ({len(documents):,} documents)")
    print(f"  Queries: queries.csv ({len(queries)} queries)")
    print(f"  Qrels: qrels.csv ({len(qrels_rows):,} relevance pairs)")
    print(f"  Metadata: metadata.json")


def main():
    parser = argparse.ArgumentParser(description="Generate Amazon Compound Query Dataset")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="HuggingFace config short names (e.g., Clothing_Shoes_and_Jewelry)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Max queries to generate (0 = no limit)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--max-items",
        type=int,
        default=500_000,
        help="Max items to stream per config (default: 500000)",
    )
    args = parser.parse_args()

    if args.configs:
        configs = [f"raw_meta_{c}" if not c.startswith("raw_meta_") else c for c in args.configs]
    else:
        configs = DEFAULT_CONFIGS

    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / "datasets" / "amazon_compound"

    print("=" * 70)
    print("Amazon Compound Query Dataset Generator")
    print("=" * 70)
    print(f"Configs: {configs}")
    print(f"Max items per config: {args.max_items:,}")
    print(f"Seed: {args.seed}")

    # Load and filter data
    cat_products, all_products = load_and_filter_data(configs, max_items_per_config=args.max_items)

    # Select queries
    query_specs = select_queries(cat_products, seed=args.seed, max_queries=args.max_queries)

    # Build corpus
    documents, qrels, queries = build_corpus(query_specs, cat_products, all_products, seed=args.seed)

    # Compute statistics
    k_values = [len(doc_ids) for doc_ids in qrels.values()]
    size_buckets = {}
    for q in queries:
        bucket = q["k_bucket"]
        k = len(qrels[q["query_id"]])
        size_buckets.setdefault(bucket, []).append(k)

    n_relevant = sum(1 for d in documents if d["type"] == "relevant")
    n_hard_neg = sum(1 for d in documents if d["type"] == "hard_negative")
    n_easy_neg = sum(1 for d in documents if d["type"] == "easy_negative")
    sparsity = sum(k_values) / (len(queries) * len(documents)) * 100 if queries else 0

    metadata = {
        "dataset": "amazon_compound",
        "source": f"McAuley-Lab/Amazon-Reviews-2023 ({', '.join(configs)})",
        "hierarchy_level": 2,
        "n_documents": len(documents),
        "n_relevant_documents": n_relevant,
        "n_hard_negative_documents": n_hard_neg,
        "n_easy_negative_documents": n_easy_neg,
        "n_queries": len(queries),
        "k_min": min(k_values) if k_values else 0,
        "k_max": max(k_values) if k_values else 0,
        "k_mean": round(sum(k_values) / len(k_values), 1) if k_values else 0,
        "k_total_relevant": sum(k_values),
        "sparsity_percent": round(sparsity, 4),
        "size_buckets": {
            bucket: {
                "n_queries": len(ks),
                "k_min": min(ks),
                "k_max": max(ks),
                "k_mean": round(sum(ks) / len(ks), 1),
            }
            for bucket, ks in size_buckets.items()
        },
        "seed": args.seed,
    }

    print(f"\nDataset statistics:")
    print(f"  K range: {metadata['k_min']} - {metadata['k_max']}")
    print(f"  K mean: {metadata['k_mean']}")
    print(f"  Sparsity: {sparsity:.4f}%")
    print(f"  Hard negatives: {n_hard_neg:,}")
    for bucket, stats in metadata["size_buckets"].items():
        print(
            f"  {bucket}: {stats['n_queries']} queries, "
            f"K={stats['k_min']}-{stats['k_max']} (mean {stats['k_mean']})"
        )

    save_dataset(output_dir, documents, queries, qrels, metadata)

    print("\nDone!")


if __name__ == "__main__":
    main()
