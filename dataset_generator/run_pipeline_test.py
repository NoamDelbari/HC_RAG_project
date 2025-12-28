"""
End-to-End Pipeline Test

Runs a small-scale test of the complete cross-entity query generation pipeline
to validate quality of generated queries.

Usage:
    python run_pipeline_test.py [--queries N] [--provider openai|anthropic]
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env.example in the same directory as this script
env_path = Path(__file__).parent / ".env.example"
load_dotenv(env_path)

sys.path.insert(0, str(Path(__file__).parent))


def create_test_clusters():
    """Create test clusters with real Wikidata entities."""
    return [
        {
            "cluster_id": "creative_inception",
            "cluster_type": "creative_ensemble",
            "binding_description": "Main cast members of the film Inception (2010)",
            "anchor_qid": "Q25188",
            "entities": [
                {
                    "qid": "Q134773",
                    "label": "Leonardo DiCaprio",
                    "description": "American actor and film producer",
                    "key_facts": "Born 1974, Oscar winner for The Revenant, starred in Titanic"
                },
                {
                    "qid": "Q188018",
                    "label": "Tom Hardy",
                    "description": "English actor",
                    "key_facts": "Born 1977, known for Bane in Dark Knight Rises, Mad Max: Fury Road"
                },
                {
                    "qid": "Q232163",
                    "label": "Joseph Gordon-Levitt",
                    "description": "American actor and filmmaker",
                    "key_facts": "Born 1981, former child actor, founded hitRECord"
                },
                {
                    "qid": "Q174843",
                    "label": "Marion Cotillard",
                    "description": "French actress",
                    "key_facts": "Born 1975, Oscar winner for La Vie en Rose"
                },
                {
                    "qid": "Q180004",
                    "label": "Elliot Page",
                    "description": "Canadian actor",
                    "key_facts": "Born 1987, Oscar nominee for Juno"
                }
            ]
        },
        {
            "cluster_id": "award_nobel_physics_2020",
            "cluster_type": "award_recognition",
            "binding_description": "Nobel Prize in Physics 2020 laureates (black hole research)",
            "award_qid": "Q38104",
            "year": 2020,
            "entities": [
                {
                    "qid": "Q188856",
                    "label": "Roger Penrose",
                    "description": "British mathematician and physicist",
                    "key_facts": "Born 1931, Oxford professor, Penrose tiles, worked with Hawking"
                },
                {
                    "qid": "Q92600",
                    "label": "Reinhard Genzel",
                    "description": "German astrophysicist",
                    "key_facts": "Born 1952, Max Planck Institute director, studied galactic center"
                },
                {
                    "qid": "Q6128036",
                    "label": "Andrea Ghez",
                    "description": "American astronomer",
                    "key_facts": "Born 1965, UCLA professor, studied supermassive black hole at Milky Way center"
                }
            ]
        },
        {
            "cluster_id": "geo_nordic_capitals",
            "cluster_type": "geographic_spatial",
            "binding_description": "Capital cities of Nordic countries",
            "region_qid": "Q21195",
            "entities": [
                {
                    "qid": "Q1757",
                    "label": "Helsinki",
                    "description": "Capital city of Finland",
                    "key_facts": "Population ~650,000, founded 1550, nicknamed 'Daughter of the Baltic'"
                },
                {
                    "qid": "Q585",
                    "label": "Oslo",
                    "description": "Capital city of Norway",
                    "key_facts": "Population ~700,000, founded around 1040, Nobel Peace Prize awarded here"
                },
                {
                    "qid": "Q1754",
                    "label": "Stockholm",
                    "description": "Capital city of Sweden",
                    "key_facts": "Population ~1 million, founded 1252, built on 14 islands"
                },
                {
                    "qid": "Q1748",
                    "label": "Copenhagen",
                    "description": "Capital city of Denmark",
                    "key_facts": "Population ~800,000, founded 10th century, home of Tivoli Gardens"
                }
            ]
        }
    ]


def run_pipeline_test(num_queries: int = 6, provider: str = None):
    """
    Run end-to-end pipeline test.

    Args:
        num_queries: Number of queries to generate (2 per difficulty level)
        provider: LLM provider ('openai' or 'anthropic')
    """
    print("=" * 70)
    print("CROSS-ENTITY QUERY GENERATION PIPELINE TEST")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Target queries: {num_queries}")

    # Check for API key
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not anthropic_key and not openai_key:
        print("\n[ERROR] No API key found!")
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
        return False

    # Determine provider
    if provider:
        use_provider = provider
    elif anthropic_key:
        use_provider = "anthropic"
    else:
        use_provider = "openai"

    print(f"Using provider: {use_provider}")

    # Import modules
    from query_generator import QueryGenerator, GenerationConfig
    from generation_planner import GenerationPlanner
    from generation_processor import GenerationProcessor
    from quality_filter import QualityFilterPipeline

    # Create test clusters
    clusters = create_test_clusters()
    print(f"\nTest clusters: {len(clusters)}")
    for c in clusters:
        print(f"  - {c['cluster_id']}: {len(c['entities'])} entities ({c['cluster_type']})")

    # Initialize components - get model from environment variable based on provider
    if use_provider == "anthropic":
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    else:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    config = GenerationConfig(
        provider=use_provider,
        model=model,
        temperature=0.7
    )

    generator = QueryGenerator(config=config, config_dir="config")

    if generator.client is None:
        print(f"\n[ERROR] Failed to initialize {use_provider} client")
        return False

    print(f"\nLLM client initialized: {config.model}")

    # Generate queries
    print("\n" + "-" * 70)
    print("GENERATING QUERIES")
    print("-" * 70)

    results = []
    queries_per_difficulty = max(1, num_queries // 3)

    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        print(f"\nCluster: {cluster_id}")

        previous_queries = []

        for difficulty in ["easy", "medium", "hard"]:
            if len(results) >= num_queries:
                break

            print(f"  Generating {difficulty} query...")

            try:
                output = generator.generate_query(
                    cluster_data=cluster,
                    difficulty=difficulty,
                    target_k=len(cluster["entities"]),
                    previous_queries=previous_queries
                )

                result = {
                    "query_id": f"{cluster_id}_{difficulty}",
                    "query_text": output.query,
                    "cluster_id": cluster_id,
                    "cluster_type": cluster["cluster_type"],
                    "difficulty": difficulty,
                    "target_k": len(cluster["entities"]),
                    "actual_k": len(cluster["entities"]),  # Assume match for test
                    "expected_answer_type": output.expected_answer_type,
                    "entity_coverage": output.entity_coverage,
                    "difficulty_justification": output.difficulty_justification,
                    "is_valid": output.is_valid,
                    "validation_errors": output.validation_errors,
                    "entity_coverage_rate": len(output.entity_coverage) / len(cluster["entities"]) if cluster["entities"] else 0,
                    "passage_count": len(cluster["entities"])  # Simulated
                }

                results.append(result)

                if output.is_valid:
                    previous_queries.append(output.query)
                    print(f"    [OK] {output.query[:60]}...")
                else:
                    print(f"    [FAIL] {output.validation_errors}")

            except Exception as e:
                print(f"    [ERROR] {str(e)[:50]}")

        if len(results) >= num_queries:
            break

    # Process results
    print("\n" + "-" * 70)
    print("PROCESSING RESULTS")
    print("-" * 70)

    processor = GenerationProcessor()
    metrics = processor.compute_metrics(results)
    print(processor.format_report(metrics))

    # Apply quality filters
    print("\n" + "-" * 70)
    print("APPLYING QUALITY FILTERS")
    print("-" * 70)

    pipeline = QualityFilterPipeline()
    filter_report = pipeline.filter_queries(results)
    print(pipeline.format_report(filter_report))

    # Display generated queries
    print("\n" + "-" * 70)
    print("GENERATED QUERIES")
    print("-" * 70)

    for r in results:
        status = "VALID" if r["is_valid"] else "INVALID"
        print(f"\n[{r['difficulty'].upper()}] [{status}] {r['cluster_type']}")
        print(f"  Query: {r['query_text']}")
        if r.get("difficulty_justification"):
            just = r["difficulty_justification"]
            print(f"  Justification:")
            print(f"    - Lexical strategy: {just.get('lexical_overlap_strategy', 'N/A')[:50]}...")
            print(f"    - Reasoning hops: {just.get('reasoning_hops', 'N/A')}")
            print(f"    - Indirection: {just.get('indirection_explanation', 'N/A')[:50]}...")

    # Save results
    output_dir = Path("pipeline_test_output")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "provider": use_provider,
            "model": config.model,
            "num_queries": len(results),
            "metrics": {
                "total": metrics.total_count,
                "valid": metrics.valid_count,
                "success_rate": metrics.success_rate,
                "by_difficulty": metrics.by_difficulty
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total generated: {len(results)}")
    print(f"Valid queries: {metrics.valid_count}")
    print(f"Success rate: {metrics.success_rate:.1%}")
    print(f"Passed all filters: {filter_report.total_passed}")

    if metrics.issues:
        print("\nIssues detected:")
        for issue in metrics.issues:
            print(f"  - [{issue['severity']}] {issue['message']}")

    success = metrics.success_rate >= 0.8
    print(f"\nOverall: {'PASS' if success else 'NEEDS REVIEW'}")

    return success


def main():
    parser = argparse.ArgumentParser(description="Run cross-entity pipeline test")
    parser.add_argument("--queries", type=int, default=6, help="Number of queries to generate")
    parser.add_argument("--provider", choices=["openai", "anthropic"], help="LLM provider")

    args = parser.parse_args()

    success = run_pipeline_test(num_queries=args.queries, provider=args.provider)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
