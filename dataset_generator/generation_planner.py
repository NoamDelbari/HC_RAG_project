"""
Generation Planning Module

Creates generation targets and cluster allocations for query generation.

Phase C.1.1: Generation Planning
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GenerationTarget:
    """Target for a specific K-range x difficulty cell."""
    k_range: tuple  # (min, max)
    difficulty: str
    target_final: int
    target_generate: int
    expected_yield: float


@dataclass
class ClusterAllocation:
    """Allocation of queries to a specific cluster."""
    cluster_id: str
    cluster_type: str
    difficulty: str
    query_count: int
    target_k: int


class GenerationPlanner:
    """
    Creates generation plans based on target distribution.

    The planner creates a matrix of targets:
    - K-ranges: 2-5, 6-10, 11-15, 16-20
    - Difficulties: easy, medium, hard
    - Cluster types: 7 types with percentage targets
    """

    # Target matrix from plan (Table in C.1.1)
    # Difficulty distribution: 90% easy, 7% medium, 3% hard
    TARGET_MATRIX = {
        (2, 5): {
            "easy":   {"final": 54, "generate": 90, "yield": 0.60},
            "medium": {"final": 4, "generate": 8, "yield": 0.50},
            "hard":   {"final": 2, "generate": 5, "yield": 0.40},
        },
        (6, 10): {
            "easy":   {"final": 68, "generate": 113, "yield": 0.60},
            "medium": {"final": 5, "generate": 10, "yield": 0.50},
            "hard":   {"final": 2, "generate": 5, "yield": 0.40},
        },
        (11, 15): {
            "easy":   {"final": 54, "generate": 90, "yield": 0.60},
            "medium": {"final": 4, "generate": 8, "yield": 0.50},
            "hard":   {"final": 2, "generate": 5, "yield": 0.40},
        },
        (16, 20): {
            "easy":   {"final": 40, "generate": 67, "yield": 0.60},
            "medium": {"final": 4, "generate": 8, "yield": 0.50},
            "hard":   {"final": 1, "generate": 3, "yield": 0.33},
        },
    }

    # Cluster type distribution (from B.1.3)
    TYPE_DISTRIBUTION = {
        "categorical": 0.20,
        "creative_ensemble": 0.15,
        "award_recognition": 0.15,
        "organizational": 0.15,
        "event_temporal": 0.15,
        "geographic_spatial": 0.10,
        "relational": 0.10,
    }

    def __init__(self):
        self.targets = []
        self._build_targets()

    def _build_targets(self):
        """Build the complete target list."""
        for k_range, difficulties in self.TARGET_MATRIX.items():
            for difficulty, values in difficulties.items():
                self.targets.append(GenerationTarget(
                    k_range=k_range,
                    difficulty=difficulty,
                    target_final=values["final"],
                    target_generate=values["generate"],
                    expected_yield=values["yield"]
                ))

    def get_matrix_display(self) -> str:
        """Get formatted matrix for display."""
        lines = [
            "Generation Target Matrix",
            "=" * 60,
            f"{'K Range':<10} {'Difficulty':<10} {'Generate':<10} {'Final':<10} {'Yield':<10}",
            "-" * 60
        ]

        for target in self.targets:
            k_label = f"{target.k_range[0]}-{target.k_range[1]}"
            lines.append(
                f"{k_label:<10} {target.difficulty:<10} "
                f"{target.target_generate:<10} {target.target_final:<10} "
                f"{target.expected_yield:.0%}"
            )

        total_generate = sum(t.target_generate for t in self.targets)
        total_final = sum(t.target_final for t in self.targets)
        lines.append("-" * 60)
        lines.append(f"{'TOTAL':<10} {'':<10} {total_generate:<10} {total_final:<10}")

        return "\n".join(lines)

    def get_type_distribution_display(self) -> str:
        """Get formatted type distribution for display."""
        lines = [
            "\nCluster Type Distribution Targets",
            "=" * 40,
            f"{'Type':<25} {'Percentage':<15}",
            "-" * 40
        ]

        for cluster_type, pct in self.TYPE_DISTRIBUTION.items():
            lines.append(f"{cluster_type:<25} {pct:.0%}")

        return "\n".join(lines)

    def allocate_clusters(self, clusters: list) -> list:
        """
        Allocate clusters to generation targets.

        Args:
            clusters: List of cluster dicts with cluster_id, cluster_type, entities

        Returns:
            List of ClusterAllocation objects
        """
        allocations = []

        # Group clusters by type and K-bucket
        by_type_k = {}
        for cluster in clusters:
            cluster_type = cluster.get("cluster_type", "categorical")
            entity_count = len(cluster.get("entities", []))
            k_bucket = self._get_k_bucket(entity_count)

            if k_bucket is None:
                continue

            key = (cluster_type, k_bucket)
            if key not in by_type_k:
                by_type_k[key] = []
            by_type_k[key].append(cluster)

        # For each target, allocate matching clusters
        for target in self.targets:
            # Get type share of this target
            for cluster_type, type_pct in self.TYPE_DISTRIBUTION.items():
                type_target = int(target.target_generate * type_pct)
                if type_target == 0:
                    continue

                key = (cluster_type, target.k_range)
                available = by_type_k.get(key, [])

                if not available:
                    continue

                # Distribute across clusters (max 4 per cluster)
                per_cluster = min(4, max(1, type_target // len(available)))

                for cluster in available:
                    allocations.append(ClusterAllocation(
                        cluster_id=cluster.get("cluster_id"),
                        cluster_type=cluster_type,
                        difficulty=target.difficulty,
                        query_count=per_cluster,
                        target_k=len(cluster.get("entities", []))
                    ))

        return allocations

    def _get_k_bucket(self, k: int) -> Optional[tuple]:
        """Get K-range bucket for a given K value."""
        for k_range in self.TARGET_MATRIX.keys():
            if k_range[0] <= k <= k_range[1]:
                return k_range
        if k > 20:
            return (16, 20)
        return None

    def get_allocation_summary(self, allocations: list) -> str:
        """Get summary of allocations."""
        lines = [
            "\nCluster Allocation Summary",
            "=" * 60
        ]

        by_cluster = {}
        for alloc in allocations:
            if alloc.cluster_id not in by_cluster:
                by_cluster[alloc.cluster_id] = []
            by_cluster[alloc.cluster_id].append(alloc)

        total_queries = 0
        for cluster_id, allocs in by_cluster.items():
            lines.append(f"\n{cluster_id}:")
            for alloc in allocs:
                lines.append(f"  - {alloc.difficulty}: {alloc.query_count} queries (K={alloc.target_k})")
                total_queries += alloc.query_count

        lines.append(f"\nTotal queries to generate: {total_queries}")

        return "\n".join(lines)

    def save_plan(self, allocations: list, output_path: str):
        """Save the generation plan to JSON."""
        plan = {
            "target_matrix": {
                f"{k[0]}-{k[1]}": {
                    diff: vals for diff, vals in diffs.items()
                }
                for k, diffs in self.TARGET_MATRIX.items()
            },
            "type_distribution": self.TYPE_DISTRIBUTION,
            "allocations": [
                {
                    "cluster_id": a.cluster_id,
                    "cluster_type": a.cluster_type,
                    "difficulty": a.difficulty,
                    "query_count": a.query_count,
                    "target_k": a.target_k
                }
                for a in allocations
            ],
            "totals": {
                "target_generate": sum(t.target_generate for t in self.targets),
                "target_final": sum(t.target_final for t in self.targets),
                "allocated_queries": sum(a.query_count for a in allocations)
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)

        print(f"Plan saved to {output_path}")


if __name__ == "__main__":
    # Demo the planner
    planner = GenerationPlanner()

    print(planner.get_matrix_display())
    print(planner.get_type_distribution_display())

    # Test with sample clusters
    sample_clusters = [
        {
            "cluster_id": "creative_inception",
            "cluster_type": "creative_ensemble",
            "entities": [{"qid": f"Q{i}"} for i in range(5)]
        },
        {
            "cluster_id": "award_nobel_2020",
            "cluster_type": "award_recognition",
            "entities": [{"qid": f"Q{i}"} for i in range(3)]
        },
        {
            "cluster_id": "geo_capitals",
            "cluster_type": "geographic_spatial",
            "entities": [{"qid": f"Q{i}"} for i in range(8)]
        },
    ]

    allocations = planner.allocate_clusters(sample_clusters)
    print(planner.get_allocation_summary(allocations))
