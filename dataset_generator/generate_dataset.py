"""
Full Dataset Generation Script

Generates the complete cross-entity QA dataset from enumerated clusters
and mapped passages.

Usage:
    # Test mode (small scale)
    python generate_dataset.py --test --queries 6

    # Full generation
    python generate_dataset.py --provider openai

    # Custom settings
    python generate_dataset.py --provider anthropic --target 240 --output dataset_v1
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env.example")
load_dotenv(Path(__file__).parent / ".env", override=True)

sys.path.insert(0, str(Path(__file__).parent))

from generation_planner import GenerationPlanner
from generation_executor import GenerationExecutor
from generation_processor import GenerationProcessor
from quality_filter import QualityFilterPipeline
from dataset_balancer import DatasetBalancer
from dataset_assembler import CorpusAssembler, MetadataGenerator, DatasetWriter, AssembledDataset
from query_generator import QueryGenerator, GenerationConfig


# Target K-ranges for sub-sampling
K_RANGES = [
    (2, 5),    # 25% of queries
    (6, 10),   # 31% of queries
    (11, 15),  # 25% of queries
    (16, 20),  # 19% of queries
]


class DatasetGenerator:
    """
    Complete dataset generation pipeline.

    Orchestrates:
    1. Loading clusters and passages
    2. Creating sub-clusters with target K values
    3. Query generation with LLM
    4. Quality filtering
    5. Dataset balancing
    6. Corpus assembly
    7. Output generation
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        data_dir: str = "data",
        output_dir: str = "output",
        target_queries: int = 240,
        target_sparsity: float = 0.02,
        seed: int = 42
    ):
        self.provider = provider
        self.model = model
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.target_queries = target_queries
        self.target_sparsity = target_sparsity
        self.seed = seed

        random.seed(seed)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.planner = GenerationPlanner()
        self.processor = GenerationProcessor()
        self.filter_pipeline = QualityFilterPipeline()
        self.balancer = DatasetBalancer(total_target=target_queries)
        self.corpus_assembler = CorpusAssembler(target_sparsity=target_sparsity)
        self.metadata_gen = MetadataGenerator(dataset_name="CrossEntityQA")

        # Data containers
        self.clusters = []
        self.passages = {}  # cluster_id -> list of passages
        self.passages_by_entity = {}  # cluster_id -> {qid -> [passages]}
        self.all_passages = []  # flat list of all passages

    def load_data(self):
        """Load clusters and passages from data directory."""
        print("\n" + "=" * 70)
        print("LOADING DATA")
        print("=" * 70)

        clusters_dir = self.data_dir / "clusters"
        passages_dir = self.data_dir / "passages"

        # Load mapping summary
        summary_path = passages_dir / "mapping_summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            print(f"Mapping summary: {summary['clusters_processed']} clusters, "
                  f"{summary['total_entities']} entities, {summary['total_passages']} passages")

        # Load clusters
        cluster_files = list(clusters_dir.glob("*.json"))
        print(f"\nFound {len(cluster_files)} cluster files")

        valid_clusters = 0
        for cluster_file in cluster_files:
            with open(cluster_file, "r", encoding="utf-8") as f:
                cluster = json.load(f)

            cluster_id = cluster.get("cluster_id")

            # Check for corresponding passages
            passage_file = passages_dir / f"{cluster_id}_passages.json"
            if not passage_file.exists():
                print(f"  [SKIP] {cluster_id}: No passage file")
                continue

            # Load passages
            with open(passage_file, "r", encoding="utf-8") as f:
                passage_data = json.load(f)

            # Extract passages and index by entity
            cluster_passages = []
            entity_passages = {}  # qid -> [passages]

            for article in passage_data.get("articles", []):
                qid = article.get("qid")
                entity_passages[qid] = []

                for passage in article.get("passages", []):
                    passage["entity_qid"] = qid
                    passage["entity_label"] = article.get("title", "")
                    passage["cluster_id"] = cluster_id
                    cluster_passages.append(passage)
                    entity_passages[qid].append(passage)

            if not cluster_passages:
                print(f"  [SKIP] {cluster_id}: No passages extracted")
                continue

            # Filter entities with Wikipedia articles
            entities_with_passages = set(p["entity_qid"] for p in cluster_passages)
            cluster["entities"] = [
                e for e in cluster.get("entities", [])
                if e.get("qid") in entities_with_passages
            ]

            if len(cluster["entities"]) < 2:
                print(f"  [SKIP] {cluster_id}: Fewer than 2 entities with passages")
                continue

            self.clusters.append(cluster)
            self.passages[cluster_id] = cluster_passages
            self.passages_by_entity[cluster_id] = entity_passages
            self.all_passages.extend(cluster_passages)
            valid_clusters += 1

            print(f"  [OK] {cluster_id}: {len(cluster['entities'])} entities, "
                  f"{len(cluster_passages)} passages")

        print(f"\nLoaded {valid_clusters} valid clusters with {len(self.all_passages)} total passages")

        return valid_clusters > 0

    def create_sub_cluster(self, cluster: dict, target_k: int) -> dict:
        """
        Create a sub-cluster by sampling target_k entities from the cluster.

        Args:
            cluster: Original cluster with all entities
            target_k: Number of entities to sample (2-20)

        Returns:
            New cluster dict with sampled entities
        """
        entities = cluster.get("entities", [])

        if len(entities) <= target_k:
            # Cluster is small enough, use as-is
            return cluster.copy()

        # Sample entities
        sampled_entities = random.sample(entities, target_k)

        # Create sub-cluster
        sub_cluster = cluster.copy()
        sub_cluster["entities"] = sampled_entities
        sub_cluster["original_entity_count"] = len(entities)
        sub_cluster["is_sub_cluster"] = True

        return sub_cluster

    def create_constraint_sub_cluster(self, cluster: dict, target_k: int) -> tuple:
        """
        Create a sub-cluster using temporal/categorical constraints.

        Instead of random sampling, find entities that share a natural constraint
        (e.g., same year, same decade) that results in approximately target_k entities.

        Args:
            cluster: Original cluster with all entities
            target_k: Target number of entities (2-20)

        Returns:
            Tuple of (sub_cluster dict, constraint description string)
        """
        entities = cluster.get("entities", [])
        cluster_type = cluster.get("cluster_type", "")

        # Determine the year property based on cluster type
        year_property = None
        if "award" in cluster.get("cluster_id", "").lower():
            year_property = "award_year"
        elif "filmography" in cluster.get("cluster_id", "").lower():
            year_property = "release_year"
        elif "position" in cluster.get("cluster_id", "").lower():
            year_property = "start_year"

        # Group entities by year
        entities_by_year = {}
        entities_without_year = []

        for entity in entities:
            props = entity.get("properties", {})
            year = props.get(year_property, "") if year_property else ""

            if year and str(year).isdigit():
                year = int(year)
                if year not in entities_by_year:
                    entities_by_year[year] = []
                entities_by_year[year].append(entity)
            else:
                entities_without_year.append(entity)

        # If no year data, fall back to random sampling
        if not entities_by_year:
            sub_cluster = self.create_sub_cluster(cluster, target_k)
            return sub_cluster, "subset of entities"

        # Find year range that gives approximately target_k entities
        sorted_years = sorted(entities_by_year.keys())

        # Strategy: Expand year range until we have >= target_k entities
        best_range = None
        best_entities = []
        best_diff = float('inf')

        for start_idx in range(len(sorted_years)):
            current_entities = []
            for end_idx in range(start_idx, len(sorted_years)):
                year_range = sorted_years[start_idx:end_idx + 1]
                current_entities = []
                for year in year_range:
                    current_entities.extend(entities_by_year[year])

                diff = abs(len(current_entities) - target_k)

                if diff < best_diff:
                    best_diff = diff
                    best_range = (sorted_years[start_idx], sorted_years[end_idx])
                    best_entities = current_entities.copy()

                # Stop if we have enough entities
                if len(current_entities) >= target_k:
                    break

            # Early termination if perfect match
            if best_diff == 0:
                break

        # If best match is too far off, adjust
        if len(best_entities) > target_k:
            # Randomly sample down to target
            best_entities = random.sample(best_entities, target_k)
        elif len(best_entities) < target_k and entities_without_year:
            # Add some entities without year data
            needed = target_k - len(best_entities)
            additional = random.sample(entities_without_year, min(needed, len(entities_without_year)))
            best_entities.extend(additional)

        # Create constraint description
        if best_range:
            start_year, end_year = best_range
            if start_year == end_year:
                constraint = f"in {start_year}"
            else:
                constraint = f"from {start_year} to {end_year}"
        else:
            constraint = "subset of entities"

        # Create sub-cluster
        sub_cluster = cluster.copy()
        sub_cluster["entities"] = best_entities
        sub_cluster["original_entity_count"] = len(entities)
        sub_cluster["is_sub_cluster"] = True
        sub_cluster["constraint"] = constraint
        sub_cluster["year_range"] = best_range

        return sub_cluster, constraint

    def get_target_k_for_range(self, k_range: tuple) -> int:
        """Get a random K value within the given range."""
        return random.randint(k_range[0], k_range[1])

    def generate_queries(
        self,
        test_mode: bool = False,
        test_queries: int = 6
    ) -> list:
        """Generate queries using LLM with entity sub-sampling."""
        print("\n" + "=" * 70)
        print("GENERATING QUERIES")
        print("=" * 70)

        # Initialize query generator
        config = GenerationConfig(
            provider=self.provider,
            model=self.model or ("claude-sonnet-4-20250514" if self.provider == "anthropic" else "gpt-4o-mini"),
            temperature=0.7
        )

        generator = QueryGenerator(config=config, config_dir="config")

        if generator.client is None:
            print(f"[ERROR] Failed to initialize {self.provider} client")
            print("Check your API key configuration in .env file")
            return []

        print(f"Using provider: {self.provider}")
        print(f"Model: {config.model}")

        if test_mode:
            print(f"Test mode: generating {test_queries} queries")

        results = []

        # Calculate queries per K-range
        if test_mode:
            # In test mode, distribute evenly across K-ranges
            queries_per_range = max(1, test_queries // len(K_RANGES))
            max_total = test_queries
        else:
            # Full mode: follow target distribution (25/31/25/19%)
            range_weights = [0.25, 0.31, 0.25, 0.19]
            max_total = self.target_queries * 2  # Over-generate

        query_count = 0

        # Track which clusters have been used (to ensure variety)
        used_clusters = set()

        for k_range in K_RANGES:
            if query_count >= max_total:
                break

            if test_mode:
                range_target = queries_per_range
            else:
                range_idx = K_RANGES.index(k_range)
                range_target = int(max_total * [0.25, 0.31, 0.25, 0.19][range_idx])

            print(f"\n--- K-range {k_range[0]}-{k_range[1]} (target: {range_target} queries) ---")

            # Shuffle all clusters for variety
            shuffled_clusters = self.clusters.copy()
            random.shuffle(shuffled_clusters)

            range_count = 0

            for cluster in shuffled_clusters:
                if range_count >= range_target:
                    break
                if query_count >= max_total:
                    break

                cluster_id = cluster.get("cluster_id")
                cluster_type = cluster.get("cluster_type", "categorical")
                entity_count = len(cluster.get("entities", []))

                # Determine target K within range
                target_k = self.get_target_k_for_range(k_range)
                target_k = min(target_k, entity_count)  # Can't exceed available entities

                # Create constraint-based sub-cluster (uses temporal filtering when possible)
                sub_cluster, constraint = self.create_constraint_sub_cluster(cluster, target_k)
                actual_k = len(sub_cluster.get("entities", []))

                # Add constraint info to sub-cluster for prompt generation
                sub_cluster["temporal_constraint"] = constraint

                print(f"\nCluster: {cluster_id} ({constraint}, {actual_k} entities)")

                # Generate query for this sub-cluster
                # Difficulty distribution: 90% easy, 7% medium, 3% hard
                difficulty_roll = random.random()
                if difficulty_roll < 0.90:
                    difficulty = "easy"
                elif difficulty_roll < 0.97:
                    difficulty = "medium"
                else:
                    difficulty = "hard"

                try:
                    print(f"  Generating {difficulty} query (K={actual_k})...", end=" ")

                    output = generator.generate_query(
                        cluster_data=sub_cluster,
                        difficulty=difficulty,
                        target_k=actual_k,
                        previous_queries=[r.get("query_text", "") for r in results[-5:]]  # Last 5 for diversity
                    )

                    # Get entity QIDs from sub-cluster
                    sub_cluster_qids = [e.get("qid") for e in sub_cluster.get("entities", [])]

                    result = {
                        "query_id": f"{cluster_id}_{difficulty}_{len(results)}",
                        "query_text": output.query,
                        "cluster_id": cluster_id,
                        "cluster_type": cluster_type,
                        "difficulty": difficulty,
                        "target_k": actual_k,
                        "actual_k": actual_k,
                        "k_range": f"{k_range[0]}-{k_range[1]}",
                        "expected_answer_type": output.expected_answer_type,
                        "entity_coverage": sub_cluster_qids,  # Use sub-cluster entities
                        "difficulty_justification": output.difficulty_justification,
                        "is_valid": output.is_valid,
                        "validation_errors": output.validation_errors,
                        "entity_coverage_rate": 1.0,  # All sub-cluster entities are relevant
                        "sub_cluster_entities": sub_cluster_qids,
                        "original_cluster_size": entity_count,
                        "temporal_constraint": constraint
                    }

                    results.append(result)
                    query_count += 1
                    range_count += 1

                    if output.is_valid:
                        print(f"[OK] {output.query[:50]}...")
                    else:
                        print(f"[INVALID] {output.validation_errors}")

                    # Rate limiting
                    time.sleep(0.5)

                except Exception as e:
                    print(f"[ERROR] {str(e)[:50]}")
                    time.sleep(1)

        # Print distribution summary
        print(f"\n\nGenerated {len(results)} queries total")
        print("\nDistribution by K-range:")
        for k_range in K_RANGES:
            range_label = f"{k_range[0]}-{k_range[1]}"
            count = sum(1 for r in results if r.get("k_range") == range_label)
            print(f"  {range_label}: {count} queries")

        return results

    def process_results(self, results: list) -> tuple:
        """Process and filter generated queries."""
        print("\n" + "=" * 70)
        print("PROCESSING RESULTS")
        print("=" * 70)

        # Compute metrics
        metrics = self.processor.compute_metrics(results)
        print(self.processor.format_report(metrics))

        # Apply quality filters
        print("\n" + "-" * 70)
        print("APPLYING QUALITY FILTERS")
        print("-" * 70)

        filter_report = self.filter_pipeline.filter_queries(results)
        print(self.filter_pipeline.format_report(filter_report))

        return metrics, filter_report

    def balance_dataset(self, passed_queries: list) -> list:
        """Balance the dataset to target distribution."""
        print("\n" + "=" * 70)
        print("BALANCING DATASET")
        print("=" * 70)

        # Add quality score if missing
        for q in passed_queries:
            if "total_score" not in q:
                q["total_score"] = q.get("entity_coverage_rate", 0.5)

        result = self.balancer.select_queries(passed_queries)
        print(self.balancer.format_result(result))

        return result.selected_queries

    def create_passage_mappings(self, selected_queries: list) -> dict:
        """
        Create passage mappings for selected queries.

        Maps exactly 1 passage per entity (the lead/first passage).
        """
        print("\n" + "=" * 70)
        print("CREATING PASSAGE MAPPINGS")
        print("=" * 70)

        mappings = {}

        for query in selected_queries:
            cluster_id = query.get("cluster_id")
            query_id = query.get("query_id")

            # Get entities for this query (from sub-cluster)
            covered_entities = set(query.get("sub_cluster_entities", []) or query.get("entity_coverage", []))

            # Get entity->passages mapping for this cluster
            entity_passages = self.passages_by_entity.get(cluster_id, {})

            # Map exactly 1 passage per entity (first/lead passage)
            matches = []
            for entity_qid in covered_entities:
                entity_psg = entity_passages.get(entity_qid, [])
                if entity_psg:
                    # Take the first (lead) passage for this entity
                    first_passage = entity_psg[0]
                    matches.append({
                        "passage_id": first_passage.get("passage_id"),
                        "entity_qid": entity_qid,
                        "entity_label": first_passage.get("entity_label", "")
                    })

            mappings[query_id] = {"matches": matches}

            # Verify passage count matches target_k
            expected_k = query.get("target_k", 0)
            actual_passages = len(matches)
            if actual_passages != expected_k:
                print(f"  [WARN] {query_id}: Expected {expected_k} passages, got {actual_passages}")

        print(f"Created mappings for {len(mappings)} queries")

        # Summary statistics
        passage_counts = [len(m["matches"]) for m in mappings.values()]
        if passage_counts:
            print(f"  Passages per query: min={min(passage_counts)}, max={max(passage_counts)}, "
                  f"avg={sum(passage_counts)/len(passage_counts):.1f}")

        return mappings

    def assemble_dataset(self, selected_queries: list, mappings: dict) -> AssembledDataset:
        """Assemble the final dataset."""
        print("\n" + "=" * 70)
        print("ASSEMBLING DATASET")
        print("=" * 70)

        # Assemble corpus
        corpus, qrels, stats = self.corpus_assembler.assemble_corpus(
            selected_queries,
            mappings,
            self.all_passages
        )

        print(f"\nCorpus Stats:")
        print(f"  Total passages: {stats.total_passages}")
        print(f"  Relevant: {stats.relevant_passages}")
        print(f"  Hard negatives: {stats.hard_negatives}")
        print(f"  Random negatives: {stats.random_negatives}")
        print(f"  Sparsity: {stats.sparsity:.1%}")
        print(f"\nPassages Per Query:")
        print(f"  Min: {stats.min_passages_per_query}")
        print(f"  Max: {stats.max_passages_per_query}")
        print(f"  Avg: {stats.avg_passages_per_query:.1f}")
        print(f"  Median: {stats.median_passages_per_query:.1f}")
        print(f"\nDistribution by K-range:")
        for k_range, count in stats.passages_per_query_histogram.items():
            print(f"  {k_range}: {count} queries")

        # No train/dev/test splits - single dataset for RAG evaluation
        splits = {}

        # Generate metadata
        metadata = self.metadata_gen.generate_metadata(
            selected_queries, corpus, qrels, splits, stats
        )

        return AssembledDataset(
            corpus=corpus,
            queries=selected_queries,
            qrels=qrels,
            splits=splits,
            metadata=metadata,
            stats=vars(stats)
        )

    def save_dataset(self, dataset: AssembledDataset, name: str = "CrossEntityQA"):
        """Save the dataset to disk."""
        print("\n" + "=" * 70)
        print("SAVING DATASET")
        print("=" * 70)

        dataset_dir = self.output_dir / name
        writer = DatasetWriter(str(dataset_dir))
        writer.write_dataset(dataset)

        print(f"\nDataset saved to: {dataset_dir}")

    def export_all_passages(self, output_path: Optional[str] = None) -> bool:
        """
        Export all passages to a single corpus.jsonl file for RAG indexing.

        This exports ALL passages from data/passages/, not just the ones
        selected for query generation. Useful for building a complete
        vector index.

        Args:
            output_path: Path for output file (default: output/full_corpus.jsonl)

        Returns:
            True if successful
        """
        print("\n" + "=" * 70)
        print("EXPORTING ALL PASSAGES")
        print("=" * 70)

        if not self.all_passages:
            # Load data if not already loaded
            if not self.load_data():
                print("[ERROR] No passages found")
                return False

        output_file = Path(output_path) if output_path else self.output_dir / "full_corpus.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Exporting {len(self.all_passages)} passages to {output_file}")

        with open(output_file, "w", encoding="utf-8") as f:
            for passage in self.all_passages:
                # Create clean passage record for RAG indexing
                record = {
                    "passage_id": passage.get("passage_id"),
                    "text": passage.get("passage_text", ""),
                    "entity_qid": passage.get("entity_qid", ""),
                    "entity_label": passage.get("entity_label", ""),
                    "cluster_id": passage.get("cluster_id", ""),
                    "section": passage.get("section", "")
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"\nExported {len(self.all_passages)} passages to {output_file}")
        print("This file can be used for RAG indexing with any embedding model.")

        return True

    def run(self, test_mode: bool = False, test_queries: int = 6) -> bool:
        """Run the complete pipeline."""
        start_time = datetime.now()

        print("=" * 70)
        print("CROSS-ENTITY QA DATASET GENERATION")
        print("=" * 70)
        print(f"Timestamp: {start_time.isoformat()}")
        print(f"Provider: {self.provider}")
        print(f"Target queries: {self.target_queries}")
        print(f"Target sparsity: {self.target_sparsity:.1%}")
        print(f"Test mode: {test_mode}")
        print(f"Random seed: {self.seed}")

        # Step 1: Load data
        if not self.load_data():
            print("\n[ERROR] No valid clusters found")
            return False

        # Step 2: Generate queries with sub-sampling
        results = self.generate_queries(test_mode=test_mode, test_queries=test_queries)
        if not results:
            print("\n[ERROR] No queries generated")
            return False

        # Save intermediate results
        intermediate_path = self.output_dir / "generation_results.json"
        with open(intermediate_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "provider": self.provider,
                "test_mode": test_mode,
                "results": results
            }, f, indent=2, ensure_ascii=False)
        print(f"\nSaved intermediate results to {intermediate_path}")

        # Step 3: Process and filter
        metrics, filter_report = self.process_results(results)

        passed_queries = [r for r in results if r.get("is_valid", False)]
        if not passed_queries:
            print("\n[ERROR] No queries passed validation")
            return False

        # Step 4: Balance dataset (skip in test mode)
        if test_mode:
            selected_queries = passed_queries
        else:
            selected_queries = self.balance_dataset(passed_queries)

        if not selected_queries:
            print("\n[ERROR] No queries selected after balancing")
            return False

        # Step 5: Create passage mappings (1 passage per entity)
        mappings = self.create_passage_mappings(selected_queries)

        # Step 6: Assemble dataset
        dataset = self.assemble_dataset(selected_queries, mappings)

        # Step 7: Save dataset
        dataset_name = "CrossEntityQA_test" if test_mode else "CrossEntityQA"
        self.save_dataset(dataset, name=dataset_name)

        # Summary
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        print("\n" + "=" * 70)
        print("GENERATION COMPLETE")
        print("=" * 70)
        print(f"Total queries generated: {len(results)}")
        print(f"Valid queries: {metrics.valid_count}")
        print(f"Selected queries: {len(selected_queries)}")
        print(f"Corpus passages: {dataset.stats['total_passages']}")
        print(f"Elapsed time: {elapsed:.1f} seconds")
        print(f"Output directory: {self.output_dir / dataset_name}")

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate cross-entity QA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export all passages for RAG indexing (no LLM needed)
    python generate_dataset.py --export-corpus

    # Test with 6 queries
    python generate_dataset.py --test --queries 6

    # Full generation with OpenAI
    python generate_dataset.py --provider openai

    # Full generation with Anthropic
    python generate_dataset.py --provider anthropic

    # Custom target and output
    python generate_dataset.py --target 100 --output my_dataset
        """
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with small number of queries"
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=6,
        help="Number of queries to generate in test mode (default: 6)"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default=None,
        help="LLM provider (default: from .env or openai)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: provider default)"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=240,
        help="Target number of queries for full generation (default: 240)"
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.02,
        help="Target corpus sparsity (default: 0.02 = 2%%)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory with clusters and passages (default: data)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--export-corpus",
        action="store_true",
        help="Export all passages to a single corpus.jsonl for RAG indexing (no query generation)"
    )

    args = parser.parse_args()

    # Create generator
    generator = DatasetGenerator(
        provider=args.provider or os.getenv("LLM_PROVIDER", "openai"),
        model=args.model,
        data_dir=args.data_dir,
        output_dir=args.output,
        target_queries=args.target,
        target_sparsity=args.sparsity,
        seed=args.seed
    )

    # Export corpus mode (no LLM needed)
    if args.export_corpus:
        success = generator.export_all_passages()
        sys.exit(0 if success else 1)

    # Query generation mode - requires API key
    provider = args.provider or os.getenv("LLM_PROVIDER", "openai")
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("[ERROR] OPENAI_API_KEY not set")
            print("Configure it in .env file or set environment variable")
            sys.exit(1)
    else:
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("[ERROR] ANTHROPIC_API_KEY not set")
            print("Configure it in .env file or set environment variable")
            sys.exit(1)

    success = generator.run(
        test_mode=args.test,
        test_queries=args.queries
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
