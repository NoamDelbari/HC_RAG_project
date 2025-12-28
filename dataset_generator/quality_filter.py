"""
Multi-Stage Quality Filtering Module

Applies quality filters to generated queries:
- K-alignment filter
- Diversity filter (semantic similarity)
- Answerability verification

Phase C.3: Multi-Stage Quality Filtering
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Callable
import re


@dataclass
class FilterResult:
    """Result of applying a filter to a query."""
    query_id: str
    filter_name: str
    passed: bool
    score: float  # 0-1, higher = better
    reason: str = ""


@dataclass
class FilteredQuery:
    """A query with all filter results."""
    query_id: str
    query_text: str
    cluster_id: str
    cluster_type: str
    difficulty: str
    target_k: int
    actual_k: int

    # Filter results
    filter_results: list = field(default_factory=list)

    # Aggregate
    passed_all: bool = True
    total_score: float = 0.0
    failed_filters: list = field(default_factory=list)


@dataclass
class FilteringReport:
    """Report from multi-stage filtering."""
    total_input: int
    total_passed: int
    total_failed: int
    pass_rate: float

    # By filter
    by_filter: dict

    # By stage
    cumulative_counts: list

    # Final queries
    passed_queries: list


class KAlignmentFilter:
    """
    Filter based on K-value alignment.

    Accepts queries where actual K is within tolerance of target K.
    """

    def __init__(self, tolerance: int = 2, min_k: int = 2, max_k: int = 25):
        self.tolerance = tolerance
        self.min_k = min_k
        self.max_k = max_k
        self.name = "k_alignment"

    def apply(self, query: dict) -> FilterResult:
        """Apply K-alignment filter."""
        query_id = query.get("query_id", "")
        target_k = query.get("target_k", 0)
        actual_k = query.get("actual_k", 0)

        # Check bounds
        if actual_k < self.min_k:
            return FilterResult(
                query_id=query_id,
                filter_name=self.name,
                passed=False,
                score=0.0,
                reason=f"Actual K too low: {actual_k} < {self.min_k}"
            )

        if actual_k > self.max_k:
            return FilterResult(
                query_id=query_id,
                filter_name=self.name,
                passed=False,
                score=0.0,
                reason=f"Actual K too high: {actual_k} > {self.max_k}"
            )

        # Check alignment
        discrepancy = abs(actual_k - target_k)
        if discrepancy > self.tolerance:
            return FilterResult(
                query_id=query_id,
                filter_name=self.name,
                passed=False,
                score=1.0 - (discrepancy / (target_k or 1)),
                reason=f"K discrepancy too large: {discrepancy} > {self.tolerance}"
            )

        # Calculate score (1.0 for exact match, decreasing with discrepancy)
        score = 1.0 - (discrepancy / (self.tolerance + 1))

        return FilterResult(
            query_id=query_id,
            filter_name=self.name,
            passed=True,
            score=score
        )


class DiversityFilter:
    """
    Filter for query diversity.

    Removes near-duplicate queries based on text similarity.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold
        self.name = "diversity"
        self.seen_queries = []

    def reset(self):
        """Reset seen queries."""
        self.seen_queries = []

    def apply(self, query: dict) -> FilterResult:
        """Apply diversity filter."""
        query_id = query.get("query_id", "")
        query_text = query.get("query_text", "")

        if not query_text:
            return FilterResult(
                query_id=query_id,
                filter_name=self.name,
                passed=False,
                score=0.0,
                reason="Empty query text"
            )

        # Check similarity with seen queries
        max_sim = 0.0
        most_similar = ""

        for seen in self.seen_queries:
            sim = self._compute_similarity(query_text, seen)
            if sim > max_sim:
                max_sim = sim
                most_similar = seen

        if max_sim > self.threshold:
            return FilterResult(
                query_id=query_id,
                filter_name=self.name,
                passed=False,
                score=1.0 - max_sim,
                reason=f"Too similar to existing query (sim={max_sim:.2f})"
            )

        # Add to seen
        self.seen_queries.append(query_text)

        return FilterResult(
            query_id=query_id,
            filter_name=self.name,
            passed=True,
            score=1.0 - max_sim
        )

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts."""
        # Tokenize
        tokens1 = set(self._tokenize(text1))
        tokens2 = set(self._tokenize(text2))

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _tokenize(self, text: str) -> list:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
                     'to', 'for', 'of', 'and', 'or', 'that', 'this', 'with', 'by'}
        return [t for t in tokens if t not in stopwords]


class AnswerabilityFilter:
    """
    Filter for query answerability.

    Checks if query appears answerable from the ground truth passages.
    """

    def __init__(self, min_coverage: float = 0.5, min_passages: int = 1):
        self.min_coverage = min_coverage
        self.min_passages = min_passages
        self.name = "answerability"

    def apply(self, query: dict) -> FilterResult:
        """Apply answerability filter."""
        query_id = query.get("query_id", "")
        entity_coverage = query.get("entity_coverage", [])
        passage_count = query.get("passage_count", 0)
        query_text = query.get("query_text", "")

        # Check passage availability
        if passage_count < self.min_passages:
            return FilterResult(
                query_id=query_id,
                filter_name=self.name,
                passed=False,
                score=0.0,
                reason=f"Too few passages: {passage_count} < {self.min_passages}"
            )

        # Check entity coverage - handle both list and rate formats
        if isinstance(entity_coverage, list):
            # Calculate coverage rate from list and target_k
            target_k = query.get("target_k", 1) or 1  # Avoid division by zero
            coverage_rate = len(entity_coverage) / target_k
        else:
            # Use entity_coverage_rate if available, otherwise use entity_coverage as-is
            coverage_rate = query.get("entity_coverage_rate", entity_coverage)
        
        if coverage_rate < self.min_coverage:
            return FilterResult(
                query_id=query_id,
                filter_name=self.name,
                passed=False,
                score=coverage_rate if isinstance(coverage_rate, (int, float)) else 0.0,
                reason=f"Low entity coverage: {coverage_rate:.1%} < {self.min_coverage:.1%}"
            )

        # Basic query structure check
        if not query_text.strip().endswith("?"):
            return FilterResult(
                query_id=query_id,
                filter_name=self.name,
                passed=False,
                score=0.5,
                reason="Query does not end with question mark"
            )

        # Passed
        score = (coverage_rate + (passage_count / 10)) / 2
        score = min(1.0, score)

        return FilterResult(
            query_id=query_id,
            filter_name=self.name,
            passed=True,
            score=score
        )


class DifficultyVerificationFilter:
    """
    Filter to verify difficulty level.

    Checks if query achieves target difficulty based on lexical characteristics.
    """

    def __init__(self, strict: bool = False):
        self.strict = strict
        self.name = "difficulty_verification"

        # Indicators for each difficulty
        self.easy_indicators = ['who', 'what', 'list', 'name', 'which']
        self.hard_indicators = ['how', 'compare', 'analyze', 'pattern', 'relationship']

    def apply(self, query: dict) -> FilterResult:
        """Apply difficulty verification filter."""
        query_id = query.get("query_id", "")
        query_text = query.get("query_text", "").lower()
        difficulty = query.get("difficulty", "")

        # Count indicators
        easy_count = sum(1 for ind in self.easy_indicators if ind in query_text)
        hard_count = sum(1 for ind in self.hard_indicators if ind in query_text)

        # Word count as proxy for complexity
        word_count = len(query_text.split())

        # Simple heuristics
        is_short = word_count < 12
        is_long = word_count > 25

        # Difficulty alignment
        alignment_score = 0.5  # Neutral

        if difficulty == "easy":
            if is_short and easy_count > 0:
                alignment_score = 1.0
            elif is_long or hard_count > 0:
                alignment_score = 0.3
            else:
                alignment_score = 0.7

        elif difficulty == "hard":
            if is_long or hard_count > 0:
                alignment_score = 1.0
            elif is_short and easy_count > 0:
                alignment_score = 0.3
            else:
                alignment_score = 0.6

        else:  # medium
            if not is_short and not is_long:
                alignment_score = 0.8
            else:
                alignment_score = 0.6

        # In strict mode, reject low alignment
        if self.strict and alignment_score < 0.4:
            return FilterResult(
                query_id=query_id,
                filter_name=self.name,
                passed=False,
                score=alignment_score,
                reason=f"Difficulty mismatch for {difficulty}"
            )

        return FilterResult(
            query_id=query_id,
            filter_name=self.name,
            passed=True,
            score=alignment_score
        )


class QualityFilterPipeline:
    """
    Pipeline of quality filters applied in sequence.
    """

    def __init__(self, filters: list = None):
        """
        Initialize pipeline.

        Args:
            filters: List of filter objects. If None, uses default filters.
        """
        if filters is None:
            self.filters = [
                KAlignmentFilter(),
                DiversityFilter(),
                AnswerabilityFilter(),
                DifficultyVerificationFilter()
            ]
        else:
            self.filters = filters

    def filter_queries(self, queries: list) -> FilteringReport:
        """
        Apply all filters to a list of queries.

        Args:
            queries: List of query dicts

        Returns:
            FilteringReport with results
        """
        # Reset stateful filters
        for f in self.filters:
            if hasattr(f, 'reset'):
                f.reset()

        filtered_queries = []
        cumulative = [(0, len(queries))]  # (passed, total)

        for query in queries:
            fq = FilteredQuery(
                query_id=query.get("query_id", ""),
                query_text=query.get("query_text", ""),
                cluster_id=query.get("cluster_id", ""),
                cluster_type=query.get("cluster_type", ""),
                difficulty=query.get("difficulty", ""),
                target_k=query.get("target_k", 0),
                actual_k=query.get("actual_k", 0)
            )

            total_score = 0.0

            for filter_obj in self.filters:
                result = filter_obj.apply(query)
                fq.filter_results.append(result)
                total_score += result.score

                if not result.passed:
                    fq.passed_all = False
                    fq.failed_filters.append(filter_obj.name)

            fq.total_score = total_score / len(self.filters) if self.filters else 0
            filtered_queries.append(fq)

        # Build report
        return self._build_report(filtered_queries)

    def _build_report(self, filtered_queries: list) -> FilteringReport:
        """Build filtering report."""
        passed = [fq for fq in filtered_queries if fq.passed_all]
        failed = [fq for fq in filtered_queries if not fq.passed_all]

        # By filter
        by_filter = {}
        for f in self.filters:
            filter_results = [
                fr for fq in filtered_queries
                for fr in fq.filter_results
                if fr.filter_name == f.name
            ]
            passed_count = sum(1 for fr in filter_results if fr.passed)
            by_filter[f.name] = {
                "total": len(filter_results),
                "passed": passed_count,
                "pass_rate": passed_count / len(filter_results) if filter_results else 0
            }

        # Cumulative
        cumulative = []
        remaining = len(filtered_queries)
        for f in self.filters:
            failed_by_filter = sum(
                1 for fq in filtered_queries
                if f.name in fq.failed_filters
            )
            remaining -= failed_by_filter
            cumulative.append({
                "filter": f.name,
                "remaining": remaining,
                "pass_rate": remaining / len(filtered_queries) if filtered_queries else 0
            })

        return FilteringReport(
            total_input=len(filtered_queries),
            total_passed=len(passed),
            total_failed=len(failed),
            pass_rate=len(passed) / len(filtered_queries) if filtered_queries else 0,
            by_filter=by_filter,
            cumulative_counts=cumulative,
            passed_queries=[self._fq_to_dict(fq) for fq in passed]
        )

    def _fq_to_dict(self, fq: FilteredQuery) -> dict:
        """Convert FilteredQuery to dict."""
        return {
            "query_id": fq.query_id,
            "query_text": fq.query_text,
            "cluster_id": fq.cluster_id,
            "cluster_type": fq.cluster_type,
            "difficulty": fq.difficulty,
            "target_k": fq.target_k,
            "actual_k": fq.actual_k,
            "total_score": fq.total_score,
            "filter_scores": {
                fr.filter_name: fr.score
                for fr in fq.filter_results
            }
        }

    def format_report(self, report: FilteringReport) -> str:
        """Format report as readable string."""
        lines = [
            "=" * 60,
            "QUALITY FILTERING REPORT",
            "=" * 60,
            "",
            "SUMMARY",
            "-" * 40,
            f"Input queries:      {report.total_input}",
            f"Passed:             {report.total_passed}",
            f"Failed:             {report.total_failed}",
            f"Pass rate:          {report.pass_rate:.1%}",
            "",
            "BY FILTER",
            "-" * 40,
        ]

        for name, stats in report.by_filter.items():
            lines.append(f"  {name:<25} {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1%})")

        lines.extend([
            "",
            "CUMULATIVE PASS-THROUGH",
            "-" * 40,
        ])

        for stage in report.cumulative_counts:
            lines.append(f"  After {stage['filter']:<20} {stage['remaining']} ({stage['pass_rate']:.1%})")

        lines.append("=" * 60)
        return "\n".join(lines)


def save_filtering_results(report: FilteringReport, output_path: str):
    """Save filtering results to JSON."""
    data = {
        "total_input": report.total_input,
        "total_passed": report.total_passed,
        "total_failed": report.total_failed,
        "pass_rate": report.pass_rate,
        "by_filter": report.by_filter,
        "cumulative_counts": report.cumulative_counts,
        "passed_queries": report.passed_queries
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved filtering results to {output_path}")


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("QUALITY FILTER PIPELINE DEMO")
    print("=" * 60)

    # Sample queries
    sample_queries = [
        {
            "query_id": "q1",
            "query_text": "Who are the main actors in Inception?",
            "cluster_id": "c1",
            "cluster_type": "creative_ensemble",
            "difficulty": "easy",
            "target_k": 5,
            "actual_k": 5,
            "entity_coverage": 1.0,
            "passage_count": 5
        },
        {
            "query_id": "q2",
            "query_text": "Who are the main actors in Inception?",  # Duplicate
            "cluster_id": "c1",
            "cluster_type": "creative_ensemble",
            "difficulty": "easy",
            "target_k": 5,
            "actual_k": 5,
            "entity_coverage": 1.0,
            "passage_count": 5
        },
        {
            "query_id": "q3",
            "query_text": "What nationalities do the cast of Nolan's film have?",
            "cluster_id": "c1",
            "cluster_type": "creative_ensemble",
            "difficulty": "medium",
            "target_k": 5,
            "actual_k": 4,
            "entity_coverage": 0.8,
            "passage_count": 4
        },
        {
            "query_id": "q4",
            "query_text": "Complex analysis of patterns",  # No question mark
            "cluster_id": "c2",
            "cluster_type": "award_recognition",
            "difficulty": "hard",
            "target_k": 3,
            "actual_k": 1,  # Too low
            "entity_coverage": 0.3,
            "passage_count": 1
        },
    ]

    pipeline = QualityFilterPipeline()
    report = pipeline.filter_queries(sample_queries)

    print(pipeline.format_report(report))

    print("\nPassed Queries:")
    for pq in report.passed_queries:
        print(f"  {pq['query_id']}: {pq['query_text'][:50]}...")
