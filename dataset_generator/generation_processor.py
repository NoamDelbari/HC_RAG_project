"""
Generation Output Processor

Processes generated queries, computes statistics, and performs
initial quality checks.

Phase C.1.3: Initial Generation Output Processing
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class QualityMetrics:
    """Quality metrics for a batch of generated queries."""
    total_count: int
    valid_count: int
    invalid_count: int
    success_rate: float

    # Length statistics
    avg_length: float
    min_length: int
    max_length: int

    # Diversity metrics
    unique_count: int
    unique_rate: float

    # Distribution by difficulty
    by_difficulty: dict

    # Distribution by cluster type
    by_cluster_type: dict

    # Common patterns
    top_openings: list
    question_types: dict

    # Relevant passages per query statistics
    avg_passages_per_query: float
    min_passages_per_query: int
    max_passages_per_query: int
    median_passages_per_query: float
    passages_per_query_distribution: dict  # k_range -> count

    # Issues detected
    issues: list


class GenerationProcessor:
    """
    Processes generation output and computes quality metrics.

    Performs:
    - Basic quality flags (length, question mark, etc.)
    - Distribution analysis
    - Systematic issue detection
    """

    def __init__(self):
        self.results = []

    def load_results(self, filepath: str) -> list:
        """Load results from generation output file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.results = data.get("results", [])
        print(f"Loaded {len(self.results)} results from {filepath}")
        return self.results

    def compute_metrics(self, results: list = None) -> QualityMetrics:
        """Compute quality metrics for results."""
        results = results or self.results

        if not results:
            return QualityMetrics(
                total_count=0, valid_count=0, invalid_count=0,
                success_rate=0.0, avg_length=0.0, min_length=0, max_length=0,
                unique_count=0, unique_rate=0.0, by_difficulty={},
                by_cluster_type={}, top_openings=[], question_types={},
                avg_passages_per_query=0.0, min_passages_per_query=0,
                max_passages_per_query=0, median_passages_per_query=0.0,
                passages_per_query_distribution={},
                issues=[]
            )

        # Basic counts
        valid = [r for r in results if r.get("is_valid", False)]
        valid_queries = [r.get("query_text", "") for r in valid if r.get("query_text")]

        # Length statistics
        lengths = [len(q.split()) for q in valid_queries]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        min_len = min(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0

        # Uniqueness
        unique_queries = set(valid_queries)
        unique_rate = len(unique_queries) / len(valid_queries) if valid_queries else 0

        # By difficulty
        by_diff = Counter(r.get("difficulty") for r in valid)

        # By cluster type
        by_type = Counter(r.get("cluster_type") for r in valid)

        # Opening phrases (first 3 words)
        openings = []
        for q in valid_queries:
            words = q.split()[:3]
            if words:
                openings.append(" ".join(words))
        top_openings = Counter(openings).most_common(5)

        # Question types
        q_types = Counter()
        for q in valid_queries:
            q_lower = q.lower().strip()
            if q_lower.startswith("who"):
                q_types["who"] += 1
            elif q_lower.startswith("what"):
                q_types["what"] += 1
            elif q_lower.startswith("which"):
                q_types["which"] += 1
            elif q_lower.startswith("where"):
                q_types["where"] += 1
            elif q_lower.startswith("when"):
                q_types["when"] += 1
            elif q_lower.startswith("how"):
                q_types["how"] += 1
            elif q_lower.startswith("list"):
                q_types["list"] += 1
            elif q_lower.startswith("name"):
                q_types["name"] += 1
            else:
                q_types["other"] += 1

        # Detect issues
        issues = self._detect_issues(results, valid_queries, top_openings)

        # Passages per query statistics (using target_k or actual_k)
        k_values = []
        for r in valid:
            # Use actual_k if available, otherwise target_k
            k = r.get("actual_k") or r.get("target_k") or 0
            if k > 0:
                k_values.append(k)

        if k_values:
            avg_ppq = sum(k_values) / len(k_values)
            min_ppq = min(k_values)
            max_ppq = max(k_values)
            sorted_k = sorted(k_values)
            mid = len(sorted_k) // 2
            median_ppq = (sorted_k[mid] + sorted_k[~mid]) / 2
        else:
            avg_ppq = min_ppq = max_ppq = median_ppq = 0

        # Distribution by K-range
        ppq_dist = {"2-5": 0, "6-10": 0, "11-15": 0, "16-20": 0}
        for k in k_values:
            if k <= 5:
                ppq_dist["2-5"] += 1
            elif k <= 10:
                ppq_dist["6-10"] += 1
            elif k <= 15:
                ppq_dist["11-15"] += 1
            else:
                ppq_dist["16-20"] += 1

        return QualityMetrics(
            total_count=len(results),
            valid_count=len(valid),
            invalid_count=len(results) - len(valid),
            success_rate=len(valid) / len(results) if results else 0,
            avg_length=avg_len,
            min_length=min_len,
            max_length=max_len,
            unique_count=len(unique_queries),
            unique_rate=unique_rate,
            by_difficulty=dict(by_diff),
            by_cluster_type=dict(by_type),
            top_openings=top_openings,
            question_types=dict(q_types),
            avg_passages_per_query=avg_ppq,
            min_passages_per_query=min_ppq,
            max_passages_per_query=max_ppq,
            median_passages_per_query=median_ppq,
            passages_per_query_distribution=ppq_dist,
            issues=issues
        )

    def _detect_issues(
        self,
        results: list,
        valid_queries: list,
        top_openings: list
    ) -> list:
        """Detect systematic issues."""
        issues = []

        # Low success rate
        valid_count = len([r for r in results if r.get("is_valid")])
        if valid_count / len(results) < 0.8:
            issues.append({
                "type": "low_success_rate",
                "severity": "warning",
                "message": f"Success rate below 80%: {valid_count}/{len(results)}"
            })

        # Low uniqueness
        if valid_queries:
            unique_rate = len(set(valid_queries)) / len(valid_queries)
            if unique_rate < 0.9:
                issues.append({
                    "type": "low_uniqueness",
                    "severity": "warning",
                    "message": f"Unique rate below 90%: {unique_rate:.1%}"
                })

        # Repetitive openings
        if top_openings:
            top_phrase, top_count = top_openings[0]
            if top_count > len(valid_queries) * 0.2:
                issues.append({
                    "type": "repetitive_opening",
                    "severity": "info",
                    "message": f"Frequent opening phrase: '{top_phrase}' ({top_count} times)"
                })

        # Short queries
        short_count = sum(1 for q in valid_queries if len(q.split()) < 10)
        if short_count > len(valid_queries) * 0.1:
            issues.append({
                "type": "short_queries",
                "severity": "info",
                "message": f"Many short queries (<10 words): {short_count}"
            })

        # Check for common errors
        error_counter = Counter()
        for r in results:
            if not r.get("is_valid"):
                errors = r.get("validation_errors", [])
                for e in errors:
                    error_counter[e] += 1

        for error, count in error_counter.most_common(3):
            if count >= 3:
                issues.append({
                    "type": "common_error",
                    "severity": "warning",
                    "message": f"Common error ({count}x): {error[:50]}..."
                })

        return issues

    def format_report(self, metrics: QualityMetrics) -> str:
        """Format metrics as a readable report."""
        lines = [
            "=" * 60,
            "GENERATION QUALITY REPORT",
            "=" * 60,
            "",
            "SUMMARY",
            "-" * 40,
            f"Total queries:     {metrics.total_count}",
            f"Valid queries:     {metrics.valid_count}",
            f"Invalid queries:   {metrics.invalid_count}",
            f"Success rate:      {metrics.success_rate:.1%}",
            "",
            "QUERY LENGTH",
            "-" * 40,
            f"Average length:    {metrics.avg_length:.1f} words",
            f"Min length:        {metrics.min_length} words",
            f"Max length:        {metrics.max_length} words",
            "",
            "DIVERSITY",
            "-" * 40,
            f"Unique queries:    {metrics.unique_count}",
            f"Unique rate:       {metrics.unique_rate:.1%}",
            "",
            "BY DIFFICULTY",
            "-" * 40,
        ]

        for diff, count in sorted(metrics.by_difficulty.items()):
            lines.append(f"  {diff:<15} {count:>5}")

        lines.extend([
            "",
            "BY CLUSTER TYPE",
            "-" * 40,
        ])

        for ctype, count in sorted(metrics.by_cluster_type.items()):
            lines.append(f"  {ctype:<25} {count:>5}")

        lines.extend([
            "",
            "RELEVANT PASSAGES PER QUERY",
            "-" * 40,
            f"Min passages:      {metrics.min_passages_per_query}",
            f"Max passages:      {metrics.max_passages_per_query}",
            f"Avg passages:      {metrics.avg_passages_per_query:.1f}",
            f"Median passages:   {metrics.median_passages_per_query:.1f}",
            "",
            "Distribution by K-range:",
        ])

        total_with_k = sum(metrics.passages_per_query_distribution.values())
        for k_range in ["2-5", "6-10", "11-15", "16-20"]:
            count = metrics.passages_per_query_distribution.get(k_range, 0)
            pct = count / total_with_k if total_with_k > 0 else 0
            lines.append(f"  {k_range:<10} {count:>5} queries ({pct:.1%})")

        lines.extend([
            "",
            "QUESTION TYPES",
            "-" * 40,
        ])

        for qtype, count in sorted(metrics.question_types.items(), key=lambda x: -x[1]):
            lines.append(f"  {qtype:<15} {count:>5}")

        lines.extend([
            "",
            "TOP OPENING PHRASES",
            "-" * 40,
        ])

        for phrase, count in metrics.top_openings:
            lines.append(f"  \"{phrase}...\" ({count})")

        if metrics.issues:
            lines.extend([
                "",
                "ISSUES DETECTED",
                "-" * 40,
            ])
            for issue in metrics.issues:
                lines.append(f"  [{issue['severity'].upper()}] {issue['message']}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def filter_valid(self, results: list = None) -> list:
        """Return only valid results."""
        results = results or self.results
        return [r for r in results if r.get("is_valid", False)]

    def filter_by_difficulty(self, difficulty: str, results: list = None) -> list:
        """Filter results by difficulty level."""
        results = results or self.results
        return [r for r in results if r.get("difficulty") == difficulty]

    def filter_by_cluster_type(self, cluster_type: str, results: list = None) -> list:
        """Filter results by cluster type."""
        results = results or self.results
        return [r for r in results if r.get("cluster_type") == cluster_type]

    def save_valid_queries(self, output_path: str, results: list = None):
        """Save only valid queries to a file."""
        valid = self.filter_valid(results)

        output = {
            "count": len(valid),
            "queries": [
                {
                    "query_id": r.get("query_id"),
                    "query_text": r.get("query_text"),
                    "difficulty": r.get("difficulty"),
                    "cluster_id": r.get("cluster_id"),
                    "cluster_type": r.get("cluster_type"),
                    "target_k": r.get("target_k"),
                    "expected_answer_type": r.get("expected_answer_type"),
                    "entity_coverage": r.get("entity_coverage", [])
                }
                for r in valid
            ]
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(valid)} valid queries to {output_path}")


if __name__ == "__main__":
    # Demo with sample data
    sample_results = [
        {
            "query_id": "test_1",
            "query_text": "Who are the main actors in the film Inception?",
            "difficulty": "easy",
            "cluster_type": "creative_ensemble",
            "cluster_id": "inception_cast",
            "target_k": 5,
            "is_valid": True,
            "expected_answer_type": "list",
            "entity_coverage": ["Q134773", "Q188018"],
            "validation_errors": []
        },
        {
            "query_id": "test_2",
            "query_text": "What nationalities do the cast of Nolan's 2010 thriller have?",
            "difficulty": "medium",
            "cluster_type": "creative_ensemble",
            "cluster_id": "inception_cast",
            "target_k": 5,
            "is_valid": True,
            "expected_answer_type": "list",
            "entity_coverage": ["Q134773"],
            "validation_errors": []
        },
        {
            "query_id": "test_3",
            "query_text": "List the Nobel Prize winners",
            "difficulty": "easy",
            "cluster_type": "award_recognition",
            "cluster_id": "nobel_2020",
            "target_k": 3,
            "is_valid": False,
            "expected_answer_type": "list",
            "entity_coverage": [],
            "validation_errors": ["Query does not end with question mark"]
        },
        {
            "query_id": "test_4",
            "query_text": "Who received the Nobel Prize in Physics in 2020?",
            "difficulty": "easy",
            "cluster_type": "award_recognition",
            "cluster_id": "nobel_2020",
            "target_k": 3,
            "is_valid": True,
            "expected_answer_type": "list",
            "entity_coverage": ["Q937", "Q92600"],
            "validation_errors": []
        },
    ]

    processor = GenerationProcessor()
    processor.results = sample_results

    metrics = processor.compute_metrics()
    report = processor.format_report(metrics)
    print(report)
