"""
Generate Category-Key Synthetic Dataset

Produces corpus.csv, queries.csv, qrels.csv, and metadata.json
from country/city config and templates. No API calls needed.
"""

import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def load_config(config_dir: Path) -> Tuple[dict, dict]:
    """Load countries_cities and templates configs."""
    with open(config_dir / "countries_cities.json", "r", encoding="utf-8") as f:
        countries_config = json.load(f)
    with open(config_dir / "templates.json", "r", encoding="utf-8") as f:
        templates_config = json.load(f)
    return countries_config, templates_config


def make_doc_id(country: str, city: str) -> str:
    """Create deterministic doc ID from country+city."""
    raw = f"{country}::{city}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def make_distractor_id(category: str, index: int) -> str:
    """Create deterministic doc ID for distractor."""
    raw = f"distractor::{category}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def make_query_id(country: str) -> str:
    """Create deterministic query ID from country."""
    return f"q_{country.lower().replace(' ', '_').replace('.', '')}"


def generate_city_documents(
    countries_config: dict,
    templates_config: dict,
    seed: int = 42
) -> Tuple[List[dict], Dict[str, List[str]]]:
    """
    Generate city documents and ground truth.

    Returns:
        (documents, qrels) where qrels maps query_id -> list of doc_ids
    """
    random.seed(seed)

    doc_templates = templates_config["document_templates"]
    capital_suffixes = templates_config["capital_suffix_templates"]
    large_suffixes = templates_config["large_city_suffix_templates"]

    documents = []
    qrels: Dict[str, List[str]] = {}

    for size_bucket, bucket_data in countries_config["categories"].items():
        for country, cities in bucket_data["countries"].items():
            query_id = make_query_id(country)
            qrels[query_id] = []

            for idx, city in enumerate(cities):
                doc_id = make_doc_id(country, city)

                # Pick template (cycle through to avoid repetition)
                template = doc_templates[(idx + hash(country)) % len(doc_templates)]
                text = template.format(city=city, country=country)

                # Add suffix for capitals (first city) and large cities
                if idx == 0:
                    text += random.choice(capital_suffixes)
                elif idx <= 2 and size_bucket == "large":
                    text += random.choice(large_suffixes)

                documents.append({
                    "doc_id": doc_id,
                    "text": text,
                    "country": country,
                    "city": city,
                    "type": "city",
                    "size_bucket": size_bucket
                })
                qrels[query_id].append(doc_id)

    return documents, qrels


def generate_distractor_documents(countries_config: dict) -> List[dict]:
    """Generate distractor documents from config."""
    documents = []

    for category, texts in countries_config["distractor_categories"].items():
        for idx, text in enumerate(texts):
            doc_id = make_distractor_id(category, idx)
            documents.append({
                "doc_id": doc_id,
                "text": text,
                "type": "distractor",
                "distractor_category": category
            })

    return documents


def generate_queries(
    countries_config: dict,
    templates_config: dict,
    seed: int = 42
) -> List[dict]:
    """Generate one query per country."""
    random.seed(seed)
    query_templates = templates_config["query_templates"]
    queries = []

    for size_bucket, bucket_data in countries_config["categories"].items():
        for country in bucket_data["countries"]:
            query_id = make_query_id(country)
            template = query_templates[hash(country) % len(query_templates)]
            text = template.format(country=country)

            queries.append({
                "query_id": query_id,
                "text": text,
                "country": country,
                "size_bucket": size_bucket
            })

    return queries


def save_dataset(
    output_dir: Path,
    documents: List[dict],
    queries: List[dict],
    qrels: Dict[str, List[str]],
    metadata: dict
):
    """Save dataset to output directory as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save corpus
    corpus_df = pd.DataFrame(documents)
    # Ensure nullable columns exist even for distractors / city docs
    for col in ("country", "city", "size_bucket", "distractor_category"):
        if col not in corpus_df.columns:
            corpus_df[col] = None
    corpus_df = corpus_df[["doc_id", "text", "type", "country", "city",
                            "size_bucket", "distractor_category"]]
    corpus_df.to_csv(output_dir / "corpus.csv", index=False)

    # Save queries
    queries_df = pd.DataFrame(queries)
    queries_df = queries_df[["query_id", "text", "country", "size_bucket"]]
    queries_df.to_csv(output_dir / "queries.csv", index=False)

    # Save qrels
    qrels_rows = []
    for query_id, doc_ids in sorted(qrels.items()):
        for doc_id in doc_ids:
            qrels_rows.append({"query_id": query_id, "doc_id": doc_id,
                               "relevance": 1})
    qrels_df = pd.DataFrame(qrels_rows)
    qrels_df.to_csv(output_dir / "qrels.csv", index=False)

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset saved to {output_dir}")
    print(f"  Corpus: corpus.csv ({len(documents)} documents)")
    print(f"  Queries: queries.csv ({len(queries)} queries)")
    print(f"  Qrels: qrels.csv ({len(qrels_rows)} relevance pairs)")
    print(f"  Metadata: {metadata_path}")


def main():
    """Generate the category-key dataset."""
    script_dir = Path(__file__).parent
    config_dir = script_dir / "config"
    output_dir = script_dir.parent.parent / "datasets" / "category_key"

    print("=" * 70)
    print("Category-Key Synthetic Dataset Generator")
    print("=" * 70)

    # Load configs
    countries_config, templates_config = load_config(config_dir)

    # Generate city documents + ground truth
    city_docs, qrels = generate_city_documents(countries_config, templates_config)
    print(f"\nGenerated {len(city_docs)} city documents")

    # Generate distractor documents
    distractor_docs = generate_distractor_documents(countries_config)
    print(f"Generated {len(distractor_docs)} distractor documents")

    # Combine all documents
    all_docs = city_docs + distractor_docs
    random.seed(42)
    random.shuffle(all_docs)
    print(f"Total corpus: {len(all_docs)} documents")

    # Generate queries
    queries = generate_queries(countries_config, templates_config)
    print(f"Generated {len(queries)} queries")

    # Compute statistics
    k_values = [len(doc_ids) for doc_ids in qrels.values()]
    size_buckets = {}
    for q in queries:
        bucket = q["size_bucket"]
        k = len(qrels[q["query_id"]])
        if bucket not in size_buckets:
            size_buckets[bucket] = []
        size_buckets[bucket].append(k)

    sparsity = sum(k_values) / (len(queries) * len(all_docs)) * 100

    metadata = {
        "n_documents": len(all_docs),
        "n_city_documents": len(city_docs),
        "n_distractor_documents": len(distractor_docs),
        "n_queries": len(queries),
        "n_countries": len(queries),
        "k_min": min(k_values),
        "k_max": max(k_values),
        "k_mean": sum(k_values) / len(k_values),
        "k_total_relevant": sum(k_values),
        "sparsity_percent": round(sparsity, 4),
        "size_buckets": {
            bucket: {
                "n_countries": len(ks),
                "k_min": min(ks),
                "k_max": max(ks),
                "k_mean": round(sum(ks) / len(ks), 1)
            }
            for bucket, ks in size_buckets.items()
        }
    }

    print(f"\nDataset statistics:")
    print(f"  K range: {min(k_values)} - {max(k_values)}")
    print(f"  K mean: {sum(k_values) / len(k_values):.1f}")
    print(f"  Sparsity: {sparsity:.2f}%")
    for bucket, stats in metadata["size_buckets"].items():
        print(f"  {bucket}: {stats['n_countries']} countries, K={stats['k_min']}-{stats['k_max']} (mean {stats['k_mean']})")

    # Save
    save_dataset(output_dir, all_docs, queries, qrels, metadata)

    print("\nDone!")


if __name__ == "__main__":
    main()
