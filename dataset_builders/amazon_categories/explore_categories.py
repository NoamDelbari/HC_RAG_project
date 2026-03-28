"""
Explore Amazon Books metadata category hierarchy for dataset building.

Downloads Amazon Reviews 2023 Books metadata from HuggingFace and analyzes
the category structure to find the right hierarchy level and K distribution.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Phase 1: Stream a small sample to understand data structure
# ---------------------------------------------------------------------------

def stream_sample(n=20):
    """Stream a small sample to inspect the data structure."""
    from datasets import load_dataset

    print(f"=== Streaming {n} samples to inspect data structure ===")
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Books",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    samples = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        samples.append(item)
        if i < 3:
            print(f"\n--- Sample {i} ---")
            print(f"  title: {item.get('title', '')!r}")
            print(f"  categories (type={type(item.get('categories')).__name__}): {item.get('categories')!r}")
            print(f"  description (type={type(item.get('description')).__name__}): {str(item.get('description', ''))[:200]!r}")
            print(f"  features (type={type(item.get('features')).__name__}): {str(item.get('features', ''))[:200]!r}")
            print(f"  fields: {list(item.keys())}")

    return samples


# ---------------------------------------------------------------------------
# Phase 2: Full streaming analysis of categories
# ---------------------------------------------------------------------------

def analyze_categories_streaming():
    """Stream through the entire dataset to collect category statistics."""
    from datasets import load_dataset

    print("\n=== Streaming full dataset for category analysis ===")
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Books",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    # Counters for each hierarchy level
    level_counters = defaultdict(Counter)  # level -> {category_path: count}
    total = 0
    filtered = 0
    max_depth = 0
    multi_label_count = 0  # products with multiple category paths

    for i, item in enumerate(ds):
        total += 1
        if total % 200_000 == 0:
            print(f"  Processed {total:,} products...")

        # Parse categories
        cats = item.get("categories", None)
        if not cats:
            continue

        # categories is a flat list of strings representing a single hierarchy path
        # e.g. ['Books', 'Literature & Fiction', 'History & Criticism']
        if isinstance(cats, list) and len(cats) > 0:
            if isinstance(cats[0], list):
                paths = cats
            elif isinstance(cats[0], str):
                paths = [cats]
            else:
                continue
        else:
            continue

        if len(paths) > 1:
            multi_label_count += 1

        # Check content filter: title + description/features >= 30 words
        title = item.get("title", "") or ""
        desc = item.get("description", None) or ""
        feats = item.get("features", None) or ""

        # Handle list types
        if isinstance(desc, list):
            desc = " ".join(str(d) for d in desc)
        if isinstance(feats, list):
            feats = " ".join(str(f) for f in feats)

        combined_text = f"{title} {desc} {feats}"
        word_count = len(combined_text.split())
        has_content = bool(title.strip()) and word_count >= 30

        if has_content:
            filtered += 1

        # Count categories at each level
        for path in paths:
            if not isinstance(path, (list, tuple)):
                continue
            depth = len(path)
            if depth > max_depth:
                max_depth = depth
            for level in range(depth):
                # Use full path up to this level as the category key
                cat_key = " > ".join(str(s) for s in path[: level + 1])
                level_counters[level][cat_key] += 1

    print(f"\n=== Summary ===")
    print(f"Total products: {total:,}")
    print(f"Products with content (title + >=30 words): {filtered:,}")
    print(f"Multi-label products: {multi_label_count:,}")
    print(f"Max category depth: {max_depth}")

    return {
        "total": total,
        "filtered": filtered,
        "multi_label_count": multi_label_count,
        "max_depth": max_depth,
        "level_counters": {int(k): dict(v) for k, v in level_counters.items()},
    }


def compute_level_stats(level_counters):
    """Compute statistics per hierarchy level."""
    stats = {}
    for level in sorted(level_counters.keys()):
        counts = level_counters[level]
        sizes = sorted(counts.values(), reverse=True)
        n_cats = len(sizes)

        # Bucket into size ranges
        small = sum(1 for s in sizes if 5 <= s <= 15)
        medium = sum(1 for s in sizes if 20 <= s <= 40)
        large = sum(1 for s in sizes if 50 <= s <= 100)
        tiny = sum(1 for s in sizes if s < 5)
        huge = sum(1 for s in sizes if s > 100)

        # Categories in the 5-100 range
        in_range = sum(1 for s in sizes if 5 <= s <= 100)

        stats[level] = {
            "n_categories": n_cats,
            "min_size": min(sizes) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
            "median_size": sizes[len(sizes) // 2] if sizes else 0,
            "mean_size": sum(sizes) / len(sizes) if sizes else 0,
            "tiny_lt5": tiny,
            "small_5_15": small,
            "medium_20_40": medium,
            "large_50_100": large,
            "huge_gt100": huge,
            "in_range_5_100": in_range,
            "top_10": dict(
                sorted(counts.items(), key=lambda x: -x[1])[:10]
            ),
            "bottom_10": dict(
                sorted(counts.items(), key=lambda x: x[1])[:10]
            ),
        }

        print(f"\nLevel {level}: {n_cats} categories")
        print(f"  Size range: {min(sizes)}-{max(sizes)}, median={sizes[len(sizes)//2]}, mean={sum(sizes)/len(sizes):.1f}")
        print(f"  tiny(<5)={tiny}, small(5-15)={small}, med(20-40)={medium}, large(50-100)={large}, huge(>100)={huge}")
        print(f"  In target range (5-100): {in_range}")

    return stats


def recommend_level(level_stats):
    """Pick the best hierarchy level for our dataset needs."""
    best_level = None
    best_score = -1

    for level, st in level_stats.items():
        n = st["n_categories"]
        in_range = st["in_range_5_100"]

        # We want 200-500 categories with 5-100 products each
        # Score: maximize in_range count, penalize if total categories far from target
        if n < 10:
            continue

        # Fraction of categories in the desired size range
        frac_in_range = in_range / n if n > 0 else 0

        # Prefer levels where total category count is 200-500
        if 200 <= n <= 500:
            count_bonus = 1.0
        elif 100 <= n <= 1000:
            count_bonus = 0.7
        else:
            count_bonus = 0.3

        # Check distribution across small/medium/large
        has_small = st["small_5_15"] >= 5
        has_medium = st["medium_20_40"] >= 5
        has_large = st["large_50_100"] >= 5
        diversity_bonus = (has_small + has_medium + has_large) / 3.0

        score = in_range * frac_in_range * count_bonus * (0.5 + 0.5 * diversity_bonus)

        print(f"  Level {level}: score={score:.1f} (in_range={in_range}, frac={frac_in_range:.2f}, count_bonus={count_bonus}, diversity={diversity_bonus:.2f})")

        if score > best_score:
            best_score = score
            best_level = level

    return best_level


def main():
    output_dir = Path("E:/HC_RAG_project/dataset_builders/amazon_categories")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Inspect structure
    samples = stream_sample(20)

    # Phase 2: Full analysis
    result = analyze_categories_streaming()

    # Phase 3: Compute stats
    print("\n=== Per-level statistics ===")
    level_stats = compute_level_stats(result["level_counters"])

    # Phase 4: Recommend level
    print("\n=== Level recommendation ===")
    rec_level = recommend_level(level_stats)
    print(f"\nRecommended level: {rec_level}")

    if rec_level is not None:
        rec_st = level_stats[rec_level]
        justification = (
            f"Level {rec_level} has {rec_st['n_categories']} categories with "
            f"{rec_st['in_range_5_100']} in the target 5-100 product range. "
            f"Distribution: small(5-15)={rec_st['small_5_15']}, "
            f"medium(20-40)={rec_st['medium_20_40']}, "
            f"large(50-100)={rec_st['large_50_100']}."
        )
    else:
        justification = "No suitable level found."

    # Phase 5: Build K distribution histogram at recommended level
    k_distribution = {}
    if rec_level is not None:
        counts = result["level_counters"][rec_level]
        # Filter to 5-100 range
        filtered_cats = {k: v for k, v in counts.items() if 5 <= v <= 100}
        # Histogram in buckets
        buckets = {"5-10": 0, "11-15": 0, "16-20": 0, "21-30": 0, "31-40": 0, "41-50": 0, "51-75": 0, "76-100": 0}
        for cat, cnt in filtered_cats.items():
            if cnt <= 10:
                buckets["5-10"] += 1
            elif cnt <= 15:
                buckets["11-15"] += 1
            elif cnt <= 20:
                buckets["16-20"] += 1
            elif cnt <= 30:
                buckets["21-30"] += 1
            elif cnt <= 40:
                buckets["31-40"] += 1
            elif cnt <= 50:
                buckets["41-50"] += 1
            elif cnt <= 75:
                buckets["51-75"] += 1
            else:
                buckets["76-100"] += 1
        k_distribution = buckets

    # Save results
    output = {
        "total_products": result["total"],
        "filtered_products": result["filtered"],
        "multi_label_count": result["multi_label_count"],
        "max_depth": result["max_depth"],
        "recommended_level": rec_level,
        "justification": justification,
        "k_distribution_at_recommended_level": k_distribution,
        "per_level_summary": {
            int(level): {
                "n_categories": st["n_categories"],
                "in_range_5_100": st["in_range_5_100"],
                "min_size": st["min_size"],
                "max_size": st["max_size"],
                "median_size": st["median_size"],
                "mean_size": round(st["mean_size"], 1),
                "tiny_lt5": st["tiny_lt5"],
                "small_5_15": st["small_5_15"],
                "medium_20_40": st["medium_20_40"],
                "large_50_100": st["large_50_100"],
                "huge_gt100": st["huge_gt100"],
                "top_10_categories": st["top_10"],
            }
            for level, st in level_stats.items()
        },
    }

    out_path = output_dir / "category_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {out_path}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
