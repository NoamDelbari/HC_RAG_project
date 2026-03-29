"""
Explore Amazon metadata for compound (category + price) query dataset.

Streams multiple HuggingFace configs (Clothing, Electronics, Home_and_Kitchen)
and profiles price coverage, subcategory diversity, and K-producibility at
various price thresholds.

Output: compound_analysis.json
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CONFIGS = [
    "raw_meta_Clothing_Shoes_and_Jewelry",
    "raw_meta_Electronics",
    "raw_meta_Home_and_Kitchen",
]

TARGET_K_RANGE = (5, 50)
CONTENT_MIN_WORDS = 30


def parse_price(price_raw) -> float | None:
    """Parse Amazon price string to float. Returns None if unparseable."""
    if price_raw is None:
        return None
    s = str(price_raw).strip()
    if not s:
        return None
    # Take first price in ranges like "$15.99 - $24.99"
    # Match first dollar amount
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


def find_threshold_for_k(prices: list[float], k_lo: int, k_hi: int):
    """
    Find a price threshold that produces K relevant docs in [k_lo, k_hi].

    Tries both "above" and "below" directions.
    Returns (direction, threshold, k) or None.
    """
    if len(prices) < k_lo:
        return None

    sorted_prices = sorted(prices)
    n = len(sorted_prices)

    # Try "above" direction: products priced above threshold
    # K = number of products with price > threshold
    for direction in ["above", "below"]:
        if direction == "above":
            # threshold = sorted_prices[n - k] means k products are >= threshold
            # We want k products strictly above threshold
            for k in range(k_lo, min(k_hi + 1, n)):
                idx = n - k  # k products at or above this index
                if idx <= 0:
                    continue
                threshold = (sorted_prices[idx - 1] + sorted_prices[idx]) / 2.0
                actual_k = sum(1 for p in prices if p > threshold)
                if k_lo <= actual_k <= k_hi:
                    return direction, round(threshold, 2), actual_k
        else:
            # "below": products priced below threshold
            for k in range(k_lo, min(k_hi + 1, n)):
                if k >= n:
                    continue
                threshold = (sorted_prices[k - 1] + sorted_prices[k]) / 2.0
                actual_k = sum(1 for p in prices if p < threshold)
                if k_lo <= actual_k <= k_hi:
                    return direction, round(threshold, 2), actual_k

    return None


def analyze_config(config_name: str, max_items: int = 0):
    """Stream a single HuggingFace config and profile price data."""
    import os
    import time
    from datasets import load_dataset

    # Increase HuggingFace download timeout (default 10s is too short)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    print(f"\n{'=' * 60}")
    print(f"Analyzing config: {config_name}")
    print(f"{'=' * 60}")

    # Per-category data
    cat_prices: dict[str, list[float]] = defaultdict(list)
    cat_total: Counter = Counter()
    total = 0
    has_content = 0
    has_price = 0
    has_both = 0

    # Retry-aware streaming: if connection drops mid-stream, re-open and skip
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
                    continue  # skip already-processed items on retry
                if max_items and items_processed >= max_items:
                    break
                items_processed += 1
                total += 1
                if total % 200_000 == 0:
                    print(f"  Processed {total:,}...")

                content_ok = passes_content_filter(item)
                if content_ok:
                    has_content += 1

                price = parse_price(item.get("price"))
                if price is not None:
                    has_price += 1

                if not content_ok:
                    continue

                cat_path = extract_category_path(item.get("categories", []), level=2)
                if cat_path is None:
                    continue

                cat_total[cat_path] += 1

                if price is not None:
                    has_both += 1
                    cat_prices[cat_path].append(price)

            break  # completed successfully
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Connection error at item {items_processed}: {e}")
                print(f"  Retrying in {wait}s (attempt {attempt+2}/{max_retries})...")
                time.sleep(wait)
            else:
                print(f"  FATAL: Failed after {max_retries} attempts at item {items_processed}")
                raise

    # Analyze which categories can produce K in target range
    viable_cats = []
    for cat, prices in cat_prices.items():
        if len(prices) < TARGET_K_RANGE[0]:
            continue
        result = find_threshold_for_k(prices, *TARGET_K_RANGE)
        if result:
            direction, threshold, k = result
            n_total = cat_total[cat]
            n_priced = len(prices)
            price_arr = np.array(prices)
            viable_cats.append({
                "category": cat,
                "n_total": n_total,
                "n_priced": n_priced,
                "price_coverage": round(n_priced / n_total, 3) if n_total else 0,
                "price_min": round(float(price_arr.min()), 2),
                "price_max": round(float(price_arr.max()), 2),
                "price_median": round(float(np.median(price_arr)), 2),
                "price_q25": round(float(np.percentile(price_arr, 25)), 2),
                "price_q75": round(float(np.percentile(price_arr, 75)), 2),
                "best_direction": direction,
                "best_threshold": threshold,
                "best_k": k,
            })

    # K bucket distribution
    k_buckets = {"small_5_15": 0, "medium_16_30": 0, "large_31_50": 0}
    for vc in viable_cats:
        k = vc["best_k"]
        if 5 <= k <= 15:
            k_buckets["small_5_15"] += 1
        elif 16 <= k <= 30:
            k_buckets["medium_16_30"] += 1
        elif 31 <= k <= 50:
            k_buckets["large_31_50"] += 1

    summary = {
        "config": config_name,
        "total_products": total,
        "products_with_content": has_content,
        "products_with_price": has_price,
        "products_with_both": has_both,
        "price_coverage_pct": round(has_price / total * 100, 1) if total else 0,
        "content_coverage_pct": round(has_content / total * 100, 1) if total else 0,
        "n_subcategories_total": len(cat_total),
        "n_subcategories_with_price": len(cat_prices),
        "n_viable_subcategories": len(viable_cats),
        "k_bucket_distribution": k_buckets,
        "top_20_viable": sorted(viable_cats, key=lambda x: -x["n_priced"])[:20],
    }

    print(f"\n  Total products: {total:,}")
    print(f"  With content (>= {CONTENT_MIN_WORDS} words): {has_content:,} ({summary['content_coverage_pct']}%)")
    print(f"  With price: {has_price:,} ({summary['price_coverage_pct']}%)")
    print(f"  With both: {has_both:,}")
    print(f"  Subcategories total: {len(cat_total)}")
    print(f"  Subcategories with price data: {len(cat_prices)}")
    print(f"  Viable subcategories (can produce K={TARGET_K_RANGE[0]}-{TARGET_K_RANGE[1]}): {len(viable_cats)}")
    print(f"  K buckets: {k_buckets}")

    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=0,
                        help="Max items per config (0 = no limit)")
    args = parser.parse_args()

    output_dir = Path("E:/HC_RAG_project/dataset_builders/amazon_compound")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Amazon Compound Query Dataset - Price Data Exploration")
    print("=" * 70)
    print(f"Configs to analyze: {CONFIGS}")
    print(f"Target K range: {TARGET_K_RANGE}")
    if args.max_items:
        print(f"Max items per config: {args.max_items:,}")

    results = {}
    for config in CONFIGS:
        results[config] = analyze_config(config, max_items=args.max_items)

    # Decision: recommend best config(s)
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    ranked = sorted(results.items(), key=lambda x: -x[1]["n_viable_subcategories"])
    for config, r in ranked:
        print(f"  {config}: {r['n_viable_subcategories']} viable subcategories, "
              f"price coverage {r['price_coverage_pct']}%")

    best_config = ranked[0][0]
    best = ranked[0][1]
    recommendation = (
        f"Best config: {best_config} with {best['n_viable_subcategories']} "
        f"viable subcategories and {best['price_coverage_pct']}% price coverage."
    )

    # Check if we need to combine configs
    if best["n_viable_subcategories"] < 150:
        combined = sum(r["n_viable_subcategories"] for _, r in ranked)
        recommendation += (
            f" Single config has < 150 viable subcategories. "
            f"Combining all configs gives ~{combined} (with possible overlap)."
        )

    print(f"\n  {recommendation}")

    output = {
        "recommendation": recommendation,
        "target_k_range": list(TARGET_K_RANGE),
        "per_config": results,
    }

    out_path = output_dir / "compound_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
