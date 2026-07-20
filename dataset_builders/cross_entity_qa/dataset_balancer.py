"""
Dataset Balancing and Final Selection Module

Selects final query set to match target distribution across
K-ranges, difficulty levels, and cluster types.

Phase C.4: Dataset Balancing
"""

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DistributionTarget:
    """Target distribution for a dimension."""
    dimension: str  # k_range, difficulty, cluster_type
    values: dict  # value -> target count


@dataclass
class SelectionResult:
    """Result of query selection."""
    selected_queries: list
    total_selected: int

    # Distribution alignment
    by_k_range: dict
    by_difficulty: dict
    by_cluster_type: dict

    # Shortfalls
    shortfalls: list

    # Statistics
    selection_stats: dict


class DatasetBalancer:
    """
    Balances the dataset to match target distributions.

    Selection priority:
    1. Fill under-represented cells (K-range × difficulty)
    2. Ensure cluster type diversity within each cell
    3. Maximize cluster instance diversity
    4. Prefer higher quality scores
    """

    # Target distribution from plan
    K_RANGE_TARGETS = {
        "2-5": 60,    # ~25%
        "6-10": 75,   # ~31%
        "11-15": 60,  # ~25%
        "16-20": 45,  # ~19%
    }

    # Difficulty distribution: 90% easy, 7% medium, 3% hard
    DIFFICULTY_TARGETS = {
        "easy": 216,   # 90%
        "medium": 17,  # 7%
        "hard": 7,     # 3%
    }

    CLUSTER_TYPE_TARGETS = {
        "categorical": 0.20,
        "creative_ensemble": 0.15,
        "award_recognition": 0.15,
        "organizational": 0.15,
        "event_temporal": 0.15,
        "geographic_spatial": 0.10,
        "relational": 0.10,
    }

    def __init__(self, total_target: int = 240):
        """
        Initialize balancer.

        Args:
            total_target: Target total number of queries
        """
        self.total_target = total_target

    def select_queries(
        self,
        queries: list,
        prioritize_quality: bool = True
    ) -> SelectionResult:
        """
        Select queries to match target distribution.

        Args:
            queries: List of query dicts (should be filtered first)
            prioritize_quality: If True, prefer higher-scoring queries

        Returns:
            SelectionResult with selected queries
        """
        # Sort by quality if requested
        if prioritize_quality:
            queries = sorted(queries, key=lambda q: q.get("total_score", 0), reverse=True)

        # Group queries by cell (K-range × difficulty)
        by_cell = defaultdict(list)
        for q in queries:
            k_range = self._get_k_range(q.get("actual_k", 0))
            difficulty = q.get("difficulty", "medium")
            cell = (k_range, difficulty)
            by_cell[cell].append(q)

        # Calculate targets per cell
        cell_targets = self._calculate_cell_targets()

        # Select queries
        selected = []
        shortfalls = []

        for cell, target in cell_targets.items():
            k_range, difficulty = cell
            available = by_cell.get(cell, [])

            # Select with cluster type diversity
            cell_selected = self._select_from_cell(available, target)
            selected.extend(cell_selected)

            # Track shortfalls
            if len(cell_selected) < target:
                shortfalls.append({
                    "cell": f"{k_range}/{difficulty}",
                    "target": target,
                    "selected": len(cell_selected),
                    "shortfall": target - len(cell_selected)
                })

        # Build result
        return self._build_result(selected, shortfalls)

    def _get_k_range(self, k: int) -> str:
        """Get K-range label for a K value."""
        if k <= 5:
            return "2-5"
        elif k <= 10:
            return "6-10"
        elif k <= 15:
            return "11-15"
        else:
            return "16-20"

    def _calculate_cell_targets(self) -> dict:
        """Calculate target count for each K-range × difficulty cell."""
        cell_targets = {}

        total_k = sum(self.K_RANGE_TARGETS.values())
        total_d = sum(self.DIFFICULTY_TARGETS.values())

        for k_range, k_target in self.K_RANGE_TARGETS.items():
            for difficulty, d_target in self.DIFFICULTY_TARGETS.items():
                # Proportional allocation
                k_pct = k_target / total_k
                d_pct = d_target / total_d
                cell_target = int(self.total_target * k_pct * d_pct)
                cell_targets[(k_range, difficulty)] = max(1, cell_target)

        return cell_targets

    def _select_from_cell(self, available: list, target: int) -> list:
        """Select queries from a cell with cluster type diversity."""
        if not available:
            return []

        if len(available) <= target:
            return available

        # Group by cluster type
        by_type = defaultdict(list)
        for q in available:
            ctype = q.get("cluster_type", "categorical")
            by_type[ctype].append(q)

        selected = []
        type_order = list(self.CLUSTER_TYPE_TARGETS.keys())
        random.shuffle(type_order)  # Randomize order for fairness

        # Round-robin selection from each type
        while len(selected) < target:
            made_progress = False
            for ctype in type_order:
                if len(selected) >= target:
                    break
                if by_type[ctype]:
                    # Take from this type (already sorted by quality)
                    selected.append(by_type[ctype].pop(0))
                    made_progress = True

            if not made_progress:
                break  # No more queries available

        return selected

    def _build_result(self, selected: list, shortfalls: list) -> SelectionResult:
        """Build selection result."""
        # Distribution counts
        by_k = Counter(self._get_k_range(q.get("actual_k", 0)) for q in selected)
        by_diff = Counter(q.get("difficulty", "") for q in selected)
        by_type = Counter(q.get("cluster_type", "") for q in selected)

        # Statistics
        stats = {
            "total_target": self.total_target,
            "total_selected": len(selected),
            "selection_rate": len(selected) / self.total_target if self.total_target > 0 else 0,
            "unique_clusters": len(set(q.get("cluster_id") for q in selected)),
            "avg_quality": sum(q.get("total_score", 0) for q in selected) / len(selected) if selected else 0
        }

        return SelectionResult(
            selected_queries=selected,
            total_selected=len(selected),
            by_k_range=dict(by_k),
            by_difficulty=dict(by_diff),
            by_cluster_type=dict(by_type),
            shortfalls=shortfalls,
            selection_stats=stats
        )

    def format_result(self, result: SelectionResult) -> str:
        """Format result as readable string."""
        lines = [
            "=" * 60,
            "DATASET BALANCING RESULT",
            "=" * 60,
            "",
            "SUMMARY",
            "-" * 40,
            f"Target:            {self.total_target}",
            f"Selected:          {result.total_selected}",
            f"Selection rate:    {result.selection_stats['selection_rate']:.1%}",
            f"Unique clusters:   {result.selection_stats['unique_clusters']}",
            f"Avg quality:       {result.selection_stats['avg_quality']:.2f}",
            "",
            "BY K-RANGE",
            "-" * 40,
        ]

        for k_range in ["2-5", "6-10", "11-15", "16-20"]:
            count = result.by_k_range.get(k_range, 0)
            target = self.K_RANGE_TARGETS.get(k_range, 0)
            pct = count / result.total_selected if result.total_selected > 0 else 0
            lines.append(f"  {k_range:<10} {count:>5} / {target:<5} ({pct:.1%})")

        lines.extend([
            "",
            "BY DIFFICULTY",
            "-" * 40,
        ])

        for diff in ["easy", "medium", "hard"]:
            count = result.by_difficulty.get(diff, 0)
            target = self.DIFFICULTY_TARGETS.get(diff, 0)
            pct = count / result.total_selected if result.total_selected > 0 else 0
            lines.append(f"  {diff:<10} {count:>5} / {target:<5} ({pct:.1%})")

        lines.extend([
            "",
            "BY CLUSTER TYPE",
            "-" * 40,
        ])

        for ctype in sorted(result.by_cluster_type.keys()):
            count = result.by_cluster_type[ctype]
            pct = count / result.total_selected if result.total_selected > 0 else 0
            lines.append(f"  {ctype:<25} {count:>5} ({pct:.1%})")

        if result.shortfalls:
            lines.extend([
                "",
                "SHORTFALLS",
                "-" * 40,
            ])
            for sf in result.shortfalls:
                lines.append(f"  {sf['cell']}: {sf['selected']}/{sf['target']} (missing {sf['shortfall']})")

        lines.append("=" * 60)
        return "\n".join(lines)


def save_selection_result(result: SelectionResult, output_path: str):
    """Save selection result to JSON."""
    data = {
        "total_selected": result.total_selected,
        "by_k_range": result.by_k_range,
        "by_difficulty": result.by_difficulty,
        "by_cluster_type": result.by_cluster_type,
        "shortfalls": result.shortfalls,
        "selection_stats": result.selection_stats,
        "selected_queries": result.selected_queries
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved selection result to {output_path}")


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("DATASET BALANCER DEMO")
    print("=" * 60)

    # Generate sample queries
    sample_queries = []
    k_values = [3, 5, 7, 10, 12, 15, 18]
    difficulties = ["easy", "medium", "hard"]
    cluster_types = ["categorical", "creative_ensemble", "award_recognition", "organizational"]

    for i in range(100):
        sample_queries.append({
            "query_id": f"q{i}",
            "query_text": f"Sample query {i}?",
            "cluster_id": f"c{i % 20}",
            "cluster_type": random.choice(cluster_types),
            "difficulty": random.choice(difficulties),
            "actual_k": random.choice(k_values),
            "total_score": random.random()
        })

    balancer = DatasetBalancer(total_target=50)
    result = balancer.select_queries(sample_queries)

    print(balancer.format_result(result))
