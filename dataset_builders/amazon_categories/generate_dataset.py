"""
Generate Amazon Categories Dataset

Loads Amazon Books metadata from HuggingFace, extracts level-2 category
hierarchy, selects categories with K=5-100 products, balances across
K buckets, and outputs corpus.csv, queries.csv, qrels.csv, metadata.json.
"""

import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from datasets import load_dataset


QUERY_TEMPLATES = [
    "What are {cat} products?",
    "Find {cat} products",
    "Show me {cat} products",
    "List {cat} products",
]

K_BUCKETS = {
    "small": (5, 15),
    "medium": (16, 40),
    "large": (41, 100),
}

TARGET_PER_BUCKET = (150, 200)
TARGET_CORPUS_SIZE = (100_000, 150_000)
SEED = 42


def make_doc_id(parent_asin: str) -> str:
    """Create deterministic doc ID from parent_asin."""
    return hashlib.md5(parent_asin.encode()).hexdigest()[:12]


def make_query_id(category_path: str) -> str:
    """Create deterministic query ID from category path."""
    sanitized = category_path.lower().replace(" > ", "_").replace(" ", "_")
    sanitized = "".join(c for c in sanitized if c.isalnum() or c == "_")
    return f"q_{sanitized[:80]}"


def build_doc_text(item: dict) -> str:
    """Build document text from product fields."""
    title = item.get("title") or ""
    description = item.get("description") or []
    features = item.get("features") or []

    if isinstance(description, list):
        description = " ".join(description)
    if isinstance(features, list):
        features = " ".join(features)

    text = title
    if description:
        text += ". " + description
    if features:
        text += " " + features

    return text.strip()


def extract_category_path(categories: list, level: int = 2) -> str:
    """Extract category path up to the given level (inclusive)."""
    if not categories or len(categories) <= level:
        return None
    return " > ".join(categories[: level + 1])


def passes_content_filter(item: dict) -> bool:
    """Check if product has title and >= 30 words combined text."""
    title = item.get("title") or ""
    if not title.strip():
        return False
    text = build_doc_text(item)
    return len(text.split()) >= 30


def load_and_filter_data():
    """Load Amazon Books metadata and filter products."""
    print("Loading Amazon Books metadata from HuggingFace...")
    print("  (This may take several minutes on first run)")

    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Books",
        split="full",
        trust_remote_code=True,
    )
    print(f"  Loaded {len(ds)} products total")

    # Build category -> products mapping while filtering
    print("\nFiltering products and extracting categories...")
    category_products: Dict[str, List[dict]] = {}
    n_filtered = 0
    n_no_category = 0

    for i, item in enumerate(ds):
        if (i + 1) % 500_000 == 0:
            print(f"  Processed {i + 1:,} / {len(ds):,} products...")

        if not passes_content_filter(item):
            continue
        n_filtered += 1

        categories = item.get("categories") or []
        cat_path = extract_category_path(categories, level=2)
        if cat_path is None:
            n_no_category += 1
            continue

        if cat_path not in category_products:
            category_products[cat_path] = []
        category_products[cat_path].append({
            "parent_asin": item["parent_asin"],
            "title": item.get("title", ""),
            "description": item.get("description", []),
            "features": item.get("features", []),
            "categories": categories,
        })

    print(f"  Filtered products (>= 30 words): {n_filtered:,}")
    print(f"  Products without level-2 category: {n_no_category:,}")
    print(f"  Unique level-2 categories: {len(category_products):,}")

    return category_products


def select_categories(
    category_products: Dict[str, List[dict]],
    seed: int = SEED,
) -> Tuple[Dict[str, List[dict]], Dict[str, str]]:
    """
    Select categories in K=5-100 range, balanced across buckets.

    Returns:
        (selected_categories, category_bucket_map) where:
            selected_categories: category_path -> list of products
            category_bucket_map: category_path -> bucket name
    """
    random.seed(seed)

    # Group categories by bucket
    bucketed: Dict[str, List[str]] = {b: [] for b in K_BUCKETS}
    for cat_path, products in category_products.items():
        k = len(products)
        for bucket_name, (lo, hi) in K_BUCKETS.items():
            if lo <= k <= hi:
                bucketed[bucket_name].append(cat_path)
                break

    print("\nCategory counts per bucket (before selection):")
    for bucket_name, cats in bucketed.items():
        lo, hi = K_BUCKETS[bucket_name]
        print(f"  {bucket_name} (K={lo}-{hi}): {len(cats)} categories")

    # Select target number per bucket
    selected: Dict[str, List[dict]] = {}
    category_bucket_map: Dict[str, str] = {}
    target_lo, target_hi = TARGET_PER_BUCKET

    for bucket_name, cats in bucketed.items():
        random.shuffle(cats)
        n_select = min(len(cats), target_hi)
        if n_select < target_lo:
            print(f"  WARNING: {bucket_name} has only {len(cats)} categories "
                  f"(target {target_lo}-{target_hi})")
        chosen = cats[:n_select]
        for cat_path in chosen:
            selected[cat_path] = category_products[cat_path]
            category_bucket_map[cat_path] = bucket_name

    print(f"\nSelected categories per bucket:")
    for bucket_name in K_BUCKETS:
        count = sum(1 for b in category_bucket_map.values() if b == bucket_name)
        print(f"  {bucket_name}: {count}")
    print(f"  Total selected: {len(selected)}")

    return selected, category_bucket_map


def build_corpus(
    selected_categories: Dict[str, List[dict]],
    category_bucket_map: Dict[str, str],
    all_category_products: Dict[str, List[dict]],
    seed: int = SEED,
) -> Tuple[List[dict], Dict[str, List[str]], List[dict]]:
    """
    Build corpus with relevant products and distractors.

    Returns:
        (documents, qrels, queries)
    """
    random.seed(seed)

    documents = []
    qrels: Dict[str, List[str]] = {}
    queries = []
    seen_asins: Set[str] = set()

    # Add relevant products
    print("\nBuilding relevant product documents...")
    template_idx = 0
    for cat_path, products in sorted(selected_categories.items()):
        bucket = category_bucket_map[cat_path]
        query_id = make_query_id(cat_path)

        # Create query
        template = QUERY_TEMPLATES[template_idx % len(QUERY_TEMPLATES)]
        # Use the leaf category name for query text
        leaf_cat = cat_path.split(" > ")[-1]
        query_text = template.format(cat=leaf_cat)
        queries.append({
            "query_id": query_id,
            "text": query_text,
            "subcategory": cat_path,
            "k_bucket": bucket,
        })
        template_idx += 1

        qrels[query_id] = []
        for product in products:
            asin = product["parent_asin"]
            if asin in seen_asins:
                continue
            seen_asins.add(asin)

            doc_id = make_doc_id(asin)
            text = build_doc_text(product)
            documents.append({
                "doc_id": doc_id,
                "text": text,
                "type": "product",
                "subcategory": cat_path,
                "k_bucket": bucket,
            })
            qrels[query_id].append(doc_id)

    n_relevant = len(documents)
    print(f"  Relevant documents: {n_relevant:,}")

    # Add distractors from non-selected categories
    print("Building distractor pool...")
    distractor_pool = []
    for cat_path, products in all_category_products.items():
        if cat_path in selected_categories:
            continue
        for product in products:
            asin = product["parent_asin"]
            if asin not in seen_asins:
                distractor_pool.append(product)

    random.shuffle(distractor_pool)

    # Target: 60-70% distractors
    target_total_lo, target_total_hi = TARGET_CORPUS_SIZE
    target_distractors = max(
        target_total_lo - n_relevant,
        int(n_relevant * 1.8),  # ~65% distractors
    )
    target_distractors = min(target_distractors, target_total_hi - n_relevant)
    target_distractors = min(target_distractors, len(distractor_pool))

    print(f"  Distractor pool size: {len(distractor_pool):,}")
    print(f"  Sampling {target_distractors:,} distractors...")

    for product in distractor_pool[:target_distractors]:
        asin = product["parent_asin"]
        if asin in seen_asins:
            continue
        seen_asins.add(asin)

        doc_id = make_doc_id(asin)
        text = build_doc_text(product)
        documents.append({
            "doc_id": doc_id,
            "text": text,
            "type": "distractor",
            "subcategory": None,
            "k_bucket": None,
        })

    n_distractors = len(documents) - n_relevant
    print(f"  Total distractors: {n_distractors:,}")
    print(f"  Total corpus: {len(documents):,}")
    pct_relevant = n_relevant / len(documents) * 100
    print(f"  Relevant: {pct_relevant:.1f}%, Distractors: {100 - pct_relevant:.1f}%")

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
    corpus_df = corpus_df[["doc_id", "text", "type", "subcategory", "k_bucket"]]
    corpus_df.to_csv(output_dir / "corpus.csv", index=False)

    # Save queries
    queries_df = pd.DataFrame(queries)
    queries_df = queries_df[["query_id", "text", "subcategory", "k_bucket"]]
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
    """Generate the Amazon categories dataset."""
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / "datasets" / "amazon_categories"

    print("=" * 70)
    print("Amazon Categories Dataset Generator")
    print("=" * 70)

    # Load and filter data
    all_category_products = load_and_filter_data()

    # Select categories balanced across K buckets
    selected_categories, category_bucket_map = select_categories(all_category_products)

    # Build corpus with relevant + distractors
    documents, qrels, queries = build_corpus(
        selected_categories, category_bucket_map, all_category_products
    )

    # Compute statistics
    k_values = [len(doc_ids) for doc_ids in qrels.values()]
    size_buckets = {}
    for q in queries:
        bucket = q["k_bucket"]
        k = len(qrels[q["query_id"]])
        size_buckets.setdefault(bucket, []).append(k)

    sparsity = sum(k_values) / (len(queries) * len(documents)) * 100

    metadata = {
        "dataset": "amazon_categories",
        "source": "McAuley-Lab/Amazon-Reviews-2023 (raw_meta_Books)",
        "hierarchy_level": 2,
        "n_documents": len(documents),
        "n_relevant_documents": sum(1 for d in documents if d["type"] == "product"),
        "n_distractor_documents": sum(1 for d in documents if d["type"] == "distractor"),
        "n_queries": len(queries),
        "n_categories": len(selected_categories),
        "k_min": min(k_values),
        "k_max": max(k_values),
        "k_mean": round(sum(k_values) / len(k_values), 1),
        "k_total_relevant": sum(k_values),
        "sparsity_percent": round(sparsity, 4),
        "size_buckets": {
            bucket: {
                "n_categories": len(ks),
                "k_min": min(ks),
                "k_max": max(ks),
                "k_mean": round(sum(ks) / len(ks), 1),
            }
            for bucket, ks in size_buckets.items()
        },
        "seed": SEED,
    }

    print(f"\nDataset statistics:")
    print(f"  K range: {min(k_values)} - {max(k_values)}")
    print(f"  K mean: {metadata['k_mean']}")
    print(f"  Sparsity: {sparsity:.4f}%")
    for bucket, stats in metadata["size_buckets"].items():
        print(
            f"  {bucket}: {stats['n_categories']} categories, "
            f"K={stats['k_min']}-{stats['k_max']} (mean {stats['k_mean']})"
        )

    # Save
    save_dataset(output_dir, documents, queries, qrels, metadata)

    print("\nDone!")


if __name__ == "__main__":
    main()
