"""
Generate expanded Electronics-only compound query dataset (~100K documents).

Reuses the 44 existing Electronics queries but expands the corpus:
  - Uncapped hard negatives (same subcategory, wrong price)
  - Department negatives (non-query Electronics subcategories)
  - Easy negatives (Clothing + Home & Kitchen products)

Streams from HuggingFace McAuley-Lab/Amazon-Reviews-2023.

Usage:
  python generate_dataset.py
  python generate_dataset.py --max-items 200000     # smaller test run
  python generate_dataset.py --target-size 80000    # custom corpus size
"""

import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXISTING_QUERIES = PROJECT_ROOT / "datasets" / "amazon_compound_electronics" / "queries.csv"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "amazon_compound_electronics_100k"

ELECTRONICS_CONFIG = "raw_meta_Electronics"
EASY_NEG_CONFIGS = [
    "raw_meta_Clothing_Shoes_and_Jewelry",
    "raw_meta_Home_and_Kitchen",
]

TARGET_CORPUS_SIZE = 100_000
CONTENT_MIN_WORDS = 30
SEED = 42


def parse_price(price_raw) -> float | None:
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
    return hashlib.md5(parent_asin.encode()).hexdigest()[:12]


def passes_content_filter(item: dict) -> bool:
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


def build_doc_text(item: dict, price: float) -> str:
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
    text += f" Price: ${price:.2f}."
    return text.strip()


def extract_category_path(categories: list, level: int = 2) -> str | None:
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


def load_queries() -> pd.DataFrame:
    """Load existing 44 Electronics queries."""
    queries = pd.read_csv(EXISTING_QUERIES)
    print(f"Loaded {len(queries)} existing queries from {EXISTING_QUERIES}")
    return queries


def stream_products(config_name: str, max_items: int = 0) -> List[dict]:
    """Stream products from a HuggingFace config, applying content + price filters."""
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    products = []
    seen_asins = set()
    items_processed = 0
    max_retries = 5

    print(f"\nStreaming {config_name}...")
    if max_items:
        print(f"  (capped at {max_items:,} items)")

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
                if max_items and items_processed >= max_items:
                    break
                items_processed += 1
                if items_processed % 100_000 == 0:
                    print(f"  Processed {items_processed:,} ({len(products):,} passed filters)...")

                asin = item.get("parent_asin")
                if not asin or asin in seen_asins:
                    continue

                if not passes_content_filter(item):
                    continue

                price = parse_price(item.get("price"))
                if price is None:
                    continue

                seen_asins.add(asin)
                cat_path = extract_category_path(item.get("categories", []), level=2)

                products.append({
                    "parent_asin": asin,
                    "title": item.get("title", ""),
                    "description": item.get("description", []),
                    "features": item.get("features", []),
                    "price": price,
                    "category_path": cat_path,
                    "config": config_name,
                })

            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Error at item {items_processed}: {e}")
                print(f"  Retrying in {wait}s (attempt {attempt+2}/{max_retries})...")
                time.sleep(wait)
            else:
                raise

    print(f"  {config_name}: {items_processed:,} streamed, {len(products):,} passed filters")
    return products


def build_corpus(
    queries_df: pd.DataFrame,
    electronics_products: List[dict],
    easy_neg_products: List[dict],
    target_size: int = TARGET_CORPUS_SIZE,
    seed: int = SEED,
) -> Tuple[List[dict], Dict[str, List[str]]]:
    """Build expanded corpus with 4-tier document hierarchy."""
    random.seed(seed)

    # Parse query specs
    query_cats = {}  # category_path -> query info
    for _, row in queries_df.iterrows():
        query_cats[row["category"]] = {
            "query_id": row["query_id"],
            "direction": row["price_direction"],
            "threshold": row["price_threshold"],
            "k_bucket": row["k_bucket"],
        }

    # Index electronics products by category
    cat_products: Dict[str, List[dict]] = defaultdict(list)
    uncategorized = []
    for prod in electronics_products:
        cat = prod.get("category_path")
        if cat:
            cat_products[cat].append(prod)
        else:
            uncategorized.append(prod)

    # Report subcategory stats
    all_elec_cats = set(cat_products.keys())
    query_cat_set = set(query_cats.keys())
    non_query_cats = all_elec_cats - query_cat_set
    missing_cats = query_cat_set - all_elec_cats

    print(f"\nElectronics subcategories found: {len(all_elec_cats)}")
    print(f"  Query subcategories: {len(query_cat_set)} ({len(query_cat_set - missing_cats)} found)")
    print(f"  Non-query subcategories: {len(non_query_cats)}")
    if missing_cats:
        print(f"  WARNING: {len(missing_cats)} query subcats not found in stream!")
        for mc in sorted(missing_cats):
            print(f"    - {mc}")

    documents = []
    qrels: Dict[str, List[str]] = {}
    seen_asins: Set[str] = set()

    # === Tier 1 & 2: Relevant + Hard negatives (query subcategories, UNCAPPED) ===
    print("\nTier 1+2: Relevant + Hard negatives (uncapped)...")
    for cat_path, qinfo in query_cats.items():
        query_id = qinfo["query_id"]
        direction = qinfo["direction"]
        threshold = qinfo["threshold"]
        bucket = qinfo["k_bucket"]
        qrels[query_id] = []

        products_in_cat = cat_products.get(cat_path, [])
        for product in products_in_cat:
            asin = product["parent_asin"]
            if asin in seen_asins:
                continue
            seen_asins.add(asin)

            price = product["price"]
            is_rel = (direction == "above" and price > threshold) or \
                     (direction == "below" and price < threshold)

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
                "k_bucket": bucket if is_rel else None,
                "price": price,
            })

    n_relevant = sum(1 for d in documents if d["type"] == "relevant")
    n_hard_neg = sum(1 for d in documents if d["type"] == "hard_negative")
    print(f"  Relevant: {n_relevant:,}")
    print(f"  Hard negatives: {n_hard_neg:,}")

    # Report K values
    k_values = [len(ids) for ids in qrels.values()]
    print(f"  K range: {min(k_values)}-{max(k_values)}, mean: {np.mean(k_values):.1f}")

    # === Tier 3: Department negatives (non-query Electronics subcategories) ===
    print("\nTier 3: Department negatives (non-query Electronics subcats)...")
    dept_neg_pool = []
    for cat_path in non_query_cats:
        for product in cat_products[cat_path]:
            if product["parent_asin"] not in seen_asins:
                dept_neg_pool.append(product)

    # Also add uncategorized electronics products
    for product in uncategorized:
        if product["parent_asin"] not in seen_asins:
            dept_neg_pool.append(product)

    random.shuffle(dept_neg_pool)

    # Calculate how many dept negatives we need
    current = len(documents)
    remaining = target_size - current
    # Split remaining ~60/40 between dept negatives and easy negatives
    n_dept_neg_target = min(int(remaining * 0.6), len(dept_neg_pool))
    n_dept_neg_target = max(n_dept_neg_target, 0)

    print(f"  Available dept negative pool: {len(dept_neg_pool):,}")
    print(f"  Target dept negatives: {n_dept_neg_target:,}")

    for product in dept_neg_pool[:n_dept_neg_target]:
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
            "type": "dept_negative",
            "subcategory": product.get("category_path"),
            "k_bucket": None,
            "price": price,
        })

    n_dept_neg = sum(1 for d in documents if d["type"] == "dept_negative")
    print(f"  Added dept negatives: {n_dept_neg:,}")

    # === Tier 4: Easy negatives (Clothing + Home & Kitchen) ===
    print("\nTier 4: Easy negatives (other departments)...")
    easy_pool = [p for p in easy_neg_products if p["parent_asin"] not in seen_asins]
    random.shuffle(easy_pool)

    current = len(documents)
    n_easy_target = min(target_size - current, len(easy_pool))
    n_easy_target = max(n_easy_target, 0)

    print(f"  Available easy negative pool: {len(easy_pool):,}")
    print(f"  Target easy negatives: {n_easy_target:,}")

    for product in easy_pool[:n_easy_target]:
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
    print(f"  Added easy negatives: {n_easy_neg:,}")

    # Summary
    print(f"\n=== Corpus Summary ===")
    print(f"  Relevant:        {n_relevant:>8,}  ({n_relevant/len(documents)*100:.1f}%)")
    print(f"  Hard negatives:  {n_hard_neg:>8,}  ({n_hard_neg/len(documents)*100:.1f}%)")
    print(f"  Dept negatives:  {n_dept_neg:>8,}  ({n_dept_neg/len(documents)*100:.1f}%)")
    print(f"  Easy negatives:  {n_easy_neg:>8,}  ({n_easy_neg/len(documents)*100:.1f}%)")
    print(f"  Total:           {len(documents):>8,}")

    # Shuffle corpus
    random.shuffle(documents)

    return documents, qrels


def save_dataset(
    output_dir: Path,
    documents: List[dict],
    queries_df: pd.DataFrame,
    qrels: Dict[str, List[str]],
    metadata: dict,
):
    """Save dataset to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Corpus
    corpus_df = pd.DataFrame(documents)
    corpus_df = corpus_df[["doc_id", "text", "type", "subcategory", "k_bucket", "price"]]
    corpus_df.to_csv(output_dir / "corpus.csv", index=False)

    # Queries (copy from existing)
    queries_df.to_csv(output_dir / "queries.csv", index=False)

    # Qrels
    qrels_rows = []
    for query_id, doc_ids in sorted(qrels.items()):
        for doc_id in doc_ids:
            qrels_rows.append({"query_id": query_id, "doc_id": doc_id, "relevance": 1})
    qrels_df = pd.DataFrame(qrels_rows)
    qrels_df.to_csv(output_dir / "qrels.csv", index=False)

    # Metadata
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset saved to {output_dir}")
    print(f"  corpus.csv:   {len(documents):,} documents")
    print(f"  queries.csv:  {len(queries_df)} queries")
    print(f"  qrels.csv:    {len(qrels_rows):,} relevance pairs")
    print(f"  metadata.json")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate expanded Electronics 100K dataset")
    parser.add_argument("--max-items", type=int, default=500_000,
                        help="Max items to stream per config (default: 500000)")
    parser.add_argument("--target-size", type=int, default=TARGET_CORPUS_SIZE,
                        help="Target corpus size (default: 100000)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print("=" * 70)
    print("Electronics 100K Dataset Generator")
    print("=" * 70)
    print(f"Target corpus: {args.target_size:,}")
    print(f"Max items per config: {args.max_items:,}")
    print(f"Seed: {args.seed}")

    # Load existing queries
    queries_df = load_queries()

    # Stream Electronics products
    electronics_products = stream_products(ELECTRONICS_CONFIG, max_items=args.max_items)

    # Stream easy negative products from other departments
    easy_neg_products = []
    for config in EASY_NEG_CONFIGS:
        prods = stream_products(config, max_items=args.max_items)
        easy_neg_products.extend(prods)

    # Build corpus
    documents, qrels = build_corpus(
        queries_df, electronics_products, easy_neg_products,
        target_size=args.target_size, seed=args.seed,
    )

    # Compute statistics
    k_values = [len(doc_ids) for doc_ids in qrels.values()]
    size_buckets = {}
    for _, row in queries_df.iterrows():
        bucket = row["k_bucket"]
        k = len(qrels.get(row["query_id"], []))
        size_buckets.setdefault(bucket, []).append(k)

    n_relevant = sum(1 for d in documents if d["type"] == "relevant")
    n_hard_neg = sum(1 for d in documents if d["type"] == "hard_negative")
    n_dept_neg = sum(1 for d in documents if d["type"] == "dept_negative")
    n_easy_neg = sum(1 for d in documents if d["type"] == "easy_negative")

    metadata = {
        "dataset": "amazon_compound_electronics_100k",
        "source": "McAuley-Lab/Amazon-Reviews-2023 (Electronics + Clothing + Home)",
        "hierarchy_level": 2,
        "n_documents": len(documents),
        "n_relevant_documents": n_relevant,
        "n_hard_negative_documents": n_hard_neg,
        "n_dept_negative_documents": n_dept_neg,
        "n_easy_negative_documents": n_easy_neg,
        "n_queries": len(queries_df),
        "k_min": min(k_values) if k_values else 0,
        "k_max": max(k_values) if k_values else 0,
        "k_mean": round(np.mean(k_values), 1) if k_values else 0,
        "k_total_relevant": sum(k_values),
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
        "max_items_per_config": args.max_items,
    }

    save_dataset(OUTPUT_DIR, documents, queries_df, qrels, metadata)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
