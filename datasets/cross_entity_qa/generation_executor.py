"""
Generation Execution Module

Executes query generation with rate limiting, progress tracking,
and checkpointing.

Phase C.1.2: Generation Execution Protocol
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env.example in the same directory as this script
env_path = Path(__file__).parent / ".env.example"
load_dotenv(env_path)

from query_generator import QueryGenerator, GenerationConfig, QueryOutput, query_output_to_dict
from generation_planner import GenerationPlanner, ClusterAllocation


@dataclass
class GenerationResult:
    """Result of a single query generation."""
    query_id: str
    cluster_id: str
    cluster_type: str
    difficulty: str
    target_k: int
    query_text: str = ""
    expected_answer_type: str = ""
    entity_coverage: list = field(default_factory=list)
    difficulty_justification: dict = field(default_factory=dict)
    is_valid: bool = False
    validation_errors: list = field(default_factory=list)
    generation_timestamp: str = ""
    batch_id: str = ""
    status: str = "pending"  # pending, generated, failed, error


class GenerationExecutor:
    """
    Executes query generation according to plan.

    Features:
    - Rate limiting to avoid API throttling
    - Progress tracking and logging
    - Checkpointing for recovery
    - Quality monitoring
    """

    def __init__(
        self,
        config: GenerationConfig = None,
        output_dir: str = "generation_output",
        rate_limit_delay: float = 0.5,
        config_dir: str = "config"
    ):
        """
        Initialize executor.

        Args:
            config: LLM configuration
            output_dir: Directory for output files
            rate_limit_delay: Delay between API calls (seconds)
            config_dir: Directory containing config files
        """
        self.config = config or GenerationConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.config_dir = config_dir

        # Initialize generator
        self.generator = QueryGenerator(config=self.config, config_dir=config_dir)

        # Batch tracking
        self.batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []
        self.generated_count = 0
        self.failed_count = 0

        # Previous queries per cluster (for diversity)
        self.previous_queries = {}

    def execute(
        self,
        allocations: list,
        clusters: list,
        max_queries: int = None,
        checkpoint_interval: int = 25
    ) -> list:
        """
        Execute generation for all allocations.

        Args:
            allocations: List of ClusterAllocation objects
            clusters: List of cluster data dicts
            max_queries: Optional limit on total queries
            checkpoint_interval: Save checkpoint every N queries

        Returns:
            List of GenerationResult objects
        """
        print(f"\n{'='*60}")
        print(f"GENERATION EXECUTION")
        print(f"{'='*60}")
        print(f"Batch ID: {self.batch_id}")
        print(f"Allocations: {len(allocations)}")
        print(f"Rate limit: {self.rate_limit_delay}s between calls")

        # Build cluster lookup
        cluster_lookup = {c.get("cluster_id"): c for c in clusters}

        # Calculate total
        total_to_generate = sum(a.query_count for a in allocations)
        if max_queries:
            total_to_generate = min(total_to_generate, max_queries)
        print(f"Queries to generate: {total_to_generate}")
        print(f"{'='*60}\n")

        query_count = 0
        start_time = time.time()

        for alloc in allocations:
            cluster = cluster_lookup.get(alloc.cluster_id)
            if not cluster:
                print(f"  [WARN] Cluster not found: {alloc.cluster_id}")
                continue

            # Get previous queries for this cluster
            prev = self.previous_queries.get(alloc.cluster_id, [])

            for i in range(alloc.query_count):
                if max_queries and query_count >= max_queries:
                    break

                # Generate query
                result = self._generate_one(
                    cluster=cluster,
                    allocation=alloc,
                    index=i,
                    previous=prev
                )

                self.results.append(result)

                # Track for diversity
                if result.is_valid:
                    self.generated_count += 1
                    prev.append(result.query_text)
                    self.previous_queries[alloc.cluster_id] = prev
                else:
                    self.failed_count += 1

                query_count += 1

                # Progress
                if query_count % 5 == 0:
                    elapsed = time.time() - start_time
                    rate = query_count / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {query_count}/{total_to_generate} "
                          f"({self.generated_count} ok, {self.failed_count} failed) "
                          f"[{rate:.1f} q/s]")

                # Checkpoint
                if query_count % checkpoint_interval == 0:
                    self._save_checkpoint()

                # Rate limiting
                time.sleep(self.rate_limit_delay)

            if max_queries and query_count >= max_queries:
                print(f"\n  Reached max_queries limit: {max_queries}")
                break

        # Final save
        self._save_checkpoint()
        self._save_final()

        # Summary
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total generated: {self.generated_count}")
        print(f"Total failed: {self.failed_count}")
        print(f"Success rate: {self.generated_count/(query_count or 1):.1%}")
        print(f"Time elapsed: {elapsed:.1f}s")
        print(f"Output: {self.output_dir}")

        return self.results

    def _generate_one(
        self,
        cluster: dict,
        allocation: ClusterAllocation,
        index: int,
        previous: list
    ) -> GenerationResult:
        """Generate a single query."""
        query_id = f"{allocation.cluster_id}_{allocation.difficulty}_{index}"

        result = GenerationResult(
            query_id=query_id,
            cluster_id=allocation.cluster_id,
            cluster_type=allocation.cluster_type,
            difficulty=allocation.difficulty,
            target_k=allocation.target_k,
            generation_timestamp=datetime.now().isoformat(),
            batch_id=self.batch_id
        )

        try:
            output = self.generator.generate_query(
                cluster_data=cluster,
                difficulty=allocation.difficulty,
                target_k=allocation.target_k,
                previous_queries=previous
            )

            result.query_text = output.query
            result.expected_answer_type = output.expected_answer_type
            result.entity_coverage = output.entity_coverage
            result.difficulty_justification = output.difficulty_justification
            result.is_valid = output.is_valid
            result.validation_errors = output.validation_errors
            result.status = "generated" if output.is_valid else "failed"

        except Exception as e:
            result.status = "error"
            result.validation_errors = [str(e)]

        return result

    def _save_checkpoint(self):
        """Save checkpoint file."""
        path = self.output_dir / f"checkpoint_{self.batch_id}.json"
        self._save_to_file(path)

    def _save_final(self):
        """Save final output file."""
        path = self.output_dir / f"generated_{self.batch_id}.json"
        self._save_to_file(path)
        print(f"\nSaved to: {path}")

    def _save_to_file(self, path: Path):
        """Save results to JSON file."""
        data = {
            "batch_id": self.batch_id,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "provider": self.config.provider,
                "model": self.config.model,
                "temperature": self.config.temperature
            },
            "statistics": {
                "total": len(self.results),
                "generated": self.generated_count,
                "failed": self.failed_count,
                "success_rate": self.generated_count / len(self.results) if self.results else 0
            },
            "results": [self._result_to_dict(r) for r in self.results]
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _result_to_dict(self, result: GenerationResult) -> dict:
        """Convert result to dict."""
        return {
            "query_id": result.query_id,
            "cluster_id": result.cluster_id,
            "cluster_type": result.cluster_type,
            "difficulty": result.difficulty,
            "target_k": result.target_k,
            "query_text": result.query_text,
            "expected_answer_type": result.expected_answer_type,
            "entity_coverage": result.entity_coverage,
            "difficulty_justification": result.difficulty_justification,
            "is_valid": result.is_valid,
            "validation_errors": result.validation_errors,
            "status": result.status,
            "generation_timestamp": result.generation_timestamp,
            "batch_id": result.batch_id
        }

    def load_results(self, filepath: str) -> list:
        """Load results from a previous run."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.batch_id = data.get("batch_id", self.batch_id)
        self.results = []

        for r in data.get("results", []):
            self.results.append(GenerationResult(
                query_id=r.get("query_id", ""),
                cluster_id=r.get("cluster_id", ""),
                cluster_type=r.get("cluster_type", ""),
                difficulty=r.get("difficulty", ""),
                target_k=r.get("target_k", 0),
                query_text=r.get("query_text", ""),
                expected_answer_type=r.get("expected_answer_type", ""),
                entity_coverage=r.get("entity_coverage", []),
                difficulty_justification=r.get("difficulty_justification", {}),
                is_valid=r.get("is_valid", False),
                validation_errors=r.get("validation_errors", []),
                status=r.get("status", ""),
                generation_timestamp=r.get("generation_timestamp", ""),
                batch_id=r.get("batch_id", "")
            ))

        return self.results


if __name__ == "__main__":
    import os

    # Create sample clusters
    sample_clusters = [
        {
            "cluster_id": "creative_inception",
            "cluster_type": "creative_ensemble",
            "binding_description": "Main cast of Inception (2010)",
            "entities": [
                {"qid": "Q134773", "label": "Leonardo DiCaprio", "description": "American actor"},
                {"qid": "Q188018", "label": "Tom Hardy", "description": "English actor"},
                {"qid": "Q232163", "label": "Joseph Gordon-Levitt", "description": "American actor"},
            ]
        },
        {
            "cluster_id": "award_nobel_2020",
            "cluster_type": "award_recognition",
            "binding_description": "Nobel Prize in Physics 2020 laureates",
            "entities": [
                {"qid": "Q937", "label": "Roger Penrose", "description": "British physicist"},
                {"qid": "Q92600", "label": "Reinhard Genzel", "description": "German astrophysicist"},
            ]
        },
    ]

    # Create allocations
    planner = GenerationPlanner()
    allocations = planner.allocate_clusters(sample_clusters)

    print("Sample Allocations:")
    for alloc in allocations[:5]:
        print(f"  {alloc.cluster_id}: {alloc.difficulty} x{alloc.query_count}")

    # Check for API key
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"):
        print("\nRunning live test (3 queries)...")

        executor = GenerationExecutor(
            output_dir="generation_output",
            rate_limit_delay=0.5
        )

        results = executor.execute(
            allocations=allocations,
            clusters=sample_clusters,
            max_queries=3
        )

        print("\nGenerated Queries:")
        for r in results:
            if r.is_valid:
                print(f"\n  [{r.difficulty}] {r.query_text}")
    else:
        print("\n[SKIP] No API key - set ANTHROPIC_API_KEY or OPENAI_API_KEY")
