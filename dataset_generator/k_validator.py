"""
K Validation and Discrepancy Analysis Module

Validates actual K values against targets and analyzes discrepancies.

Phase C.2.3: K Validation and Discrepancy Analysis
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KValidationResult:
    """K validation result for a single query."""
    query_id: str
    cluster_id: str
    difficulty: str

    # K values
    target_k: int
    entity_k: int  # Number of entities
    passage_k: int  # Number of passages
    actual_k: int  # Final K (typically passage_k)

    # Discrepancy
    k_discrepancy: int
    discrepancy_category: str  # exact, close, moderate, large

    # Coverage
    entity_coverage: float  # % of entities with passages
    passage_coverage: float  # passages vs target

    # Decision
    accepted: bool
    rejection_reason: str = ""


@dataclass
class KValidationReport:
    """Validation report for a batch of queries."""
    total_queries: int
    accepted_queries: int
    rejected_queries: int
    acceptance_rate: float

    # By discrepancy category
    by_category: dict

    # By difficulty
    by_difficulty: dict

    # Discrepancy statistics
    avg_discrepancy: float
    max_discrepancy: int

    # Detailed results
    results: list


class KValidator:
    """
    Validates K values and makes acceptance decisions.

    Acceptance criteria:
    - Actual K within ±2 of target (close match)
    - Entity coverage >= 80%
    - Actual K >= 2 (at least 2 passages needed)
    """

    # Discrepancy thresholds
    CLOSE_THRESHOLD = 2  # Accept if |discrepancy| <= 2
    MODERATE_THRESHOLD = 5  # Flag if 3-5

    # Minimum requirements
    MIN_K = 2
    MIN_ENTITY_COVERAGE = 0.6

    def __init__(
        self,
        close_threshold: int = 2,
        min_k: int = 2,
        min_entity_coverage: float = 0.6
    ):
        self.close_threshold = close_threshold
        self.min_k = min_k
        self.min_entity_coverage = min_entity_coverage

    def validate_query(
        self,
        query_id: str,
        cluster_id: str,
        difficulty: str,
        target_k: int,
        entity_qids: list,
        passage_count: int,
        entity_coverage: float
    ) -> KValidationResult:
        """
        Validate K for a single query.

        Args:
            query_id: Query identifier
            cluster_id: Cluster identifier
            difficulty: Difficulty level
            target_k: Target K value
            entity_qids: List of entity QIDs
            passage_count: Number of matched passages
            entity_coverage: % of entities with passages

        Returns:
            KValidationResult
        """
        entity_k = len(entity_qids)
        passage_k = passage_count
        actual_k = passage_k  # Use passage count as actual K

        # Calculate discrepancy
        k_discrepancy = actual_k - target_k

        # Categorize discrepancy
        abs_disc = abs(k_discrepancy)
        if abs_disc == 0:
            category = "exact"
        elif abs_disc <= self.close_threshold:
            category = "close"
        elif abs_disc <= self.MODERATE_THRESHOLD:
            category = "moderate"
        else:
            category = "large"

        # Coverage calculations
        passage_coverage = actual_k / target_k if target_k > 0 else 0

        # Acceptance decision
        accepted = True
        rejection_reason = ""

        if actual_k < self.min_k:
            accepted = False
            rejection_reason = f"Actual K too low: {actual_k} < {self.min_k}"
        elif entity_coverage < self.min_entity_coverage:
            accepted = False
            rejection_reason = f"Entity coverage too low: {entity_coverage:.1%} < {self.min_entity_coverage:.1%}"
        elif category == "large":
            accepted = False
            rejection_reason = f"K discrepancy too large: {k_discrepancy}"
        elif actual_k > 25:
            accepted = False
            rejection_reason = f"Actual K too high: {actual_k} > 25"

        return KValidationResult(
            query_id=query_id,
            cluster_id=cluster_id,
            difficulty=difficulty,
            target_k=target_k,
            entity_k=entity_k,
            passage_k=passage_k,
            actual_k=actual_k,
            k_discrepancy=k_discrepancy,
            discrepancy_category=category,
            entity_coverage=entity_coverage,
            passage_coverage=passage_coverage,
            accepted=accepted,
            rejection_reason=rejection_reason
        )

    def validate_batch(
        self,
        queries: list,
        ground_truths: dict,
        passage_mappings: dict
    ) -> KValidationReport:
        """
        Validate a batch of queries.

        Args:
            queries: List of query dicts
            ground_truths: Dict mapping query_id to ground truth
            passage_mappings: Dict mapping query_id to passage mapping

        Returns:
            KValidationReport with all results
        """
        results = []

        for query in queries:
            query_id = query.get("query_id")

            gt = ground_truths.get(query_id, {})
            pm = passage_mappings.get(query_id, {})

            result = self.validate_query(
                query_id=query_id,
                cluster_id=query.get("cluster_id", ""),
                difficulty=query.get("difficulty", ""),
                target_k=query.get("target_k", 0),
                entity_qids=gt.get("verified_entities", []),
                passage_count=pm.get("passage_count", 0),
                entity_coverage=pm.get("coverage_rate", 0)
            )
            results.append(result)

        return self._build_report(results)

    def _build_report(self, results: list) -> KValidationReport:
        """Build validation report from results."""
        accepted = [r for r in results if r.accepted]
        rejected = [r for r in results if not r.accepted]

        # By category
        by_category = Counter(r.discrepancy_category for r in results)

        # By difficulty
        by_difficulty = {}
        for diff in ["easy", "medium", "hard"]:
            diff_results = [r for r in results if r.difficulty == diff]
            if diff_results:
                by_difficulty[diff] = {
                    "total": len(diff_results),
                    "accepted": len([r for r in diff_results if r.accepted]),
                    "acceptance_rate": len([r for r in diff_results if r.accepted]) / len(diff_results)
                }

        # Discrepancy stats
        discrepancies = [abs(r.k_discrepancy) for r in results]
        avg_disc = sum(discrepancies) / len(discrepancies) if discrepancies else 0
        max_disc = max(discrepancies) if discrepancies else 0

        return KValidationReport(
            total_queries=len(results),
            accepted_queries=len(accepted),
            rejected_queries=len(rejected),
            acceptance_rate=len(accepted) / len(results) if results else 0,
            by_category=dict(by_category),
            by_difficulty=by_difficulty,
            avg_discrepancy=avg_disc,
            max_discrepancy=max_disc,
            results=results
        )

    def format_report(self, report: KValidationReport) -> str:
        """Format report as readable string."""
        lines = [
            "=" * 60,
            "K VALIDATION REPORT",
            "=" * 60,
            "",
            "SUMMARY",
            "-" * 40,
            f"Total queries:      {report.total_queries}",
            f"Accepted:           {report.accepted_queries}",
            f"Rejected:           {report.rejected_queries}",
            f"Acceptance rate:    {report.acceptance_rate:.1%}",
            "",
            "K DISCREPANCY",
            "-" * 40,
            f"Average discrepancy: {report.avg_discrepancy:.1f}",
            f"Max discrepancy:     {report.max_discrepancy}",
            "",
            "BY DISCREPANCY CATEGORY",
            "-" * 40,
        ]

        for cat in ["exact", "close", "moderate", "large"]:
            count = report.by_category.get(cat, 0)
            lines.append(f"  {cat:<15} {count:>5}")

        lines.extend([
            "",
            "BY DIFFICULTY",
            "-" * 40,
        ])

        for diff, stats in report.by_difficulty.items():
            lines.append(f"  {diff:<10} {stats['accepted']}/{stats['total']} "
                         f"({stats['acceptance_rate']:.1%})")

        # Rejection reasons
        rejection_reasons = Counter()
        for r in report.results:
            if not r.accepted:
                rejection_reasons[r.rejection_reason] += 1

        if rejection_reasons:
            lines.extend([
                "",
                "REJECTION REASONS",
                "-" * 40,
            ])
            for reason, count in rejection_reasons.most_common():
                lines.append(f"  {reason}: {count}")

        lines.append("=" * 60)
        return "\n".join(lines)


def k_validation_result_to_dict(result: KValidationResult) -> dict:
    """Convert KValidationResult to dict."""
    return {
        "query_id": result.query_id,
        "cluster_id": result.cluster_id,
        "difficulty": result.difficulty,
        "target_k": result.target_k,
        "entity_k": result.entity_k,
        "passage_k": result.passage_k,
        "actual_k": result.actual_k,
        "k_discrepancy": result.k_discrepancy,
        "discrepancy_category": result.discrepancy_category,
        "entity_coverage": result.entity_coverage,
        "passage_coverage": result.passage_coverage,
        "accepted": result.accepted,
        "rejection_reason": result.rejection_reason
    }


def save_validation_report(report: KValidationReport, output_path: str):
    """Save validation report to JSON."""
    data = {
        "total_queries": report.total_queries,
        "accepted_queries": report.accepted_queries,
        "rejected_queries": report.rejected_queries,
        "acceptance_rate": report.acceptance_rate,
        "by_category": report.by_category,
        "by_difficulty": report.by_difficulty,
        "avg_discrepancy": report.avg_discrepancy,
        "max_discrepancy": report.max_discrepancy,
        "results": [k_validation_result_to_dict(r) for r in report.results]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved K validation report to {output_path}")


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("K VALIDATOR DEMO")
    print("=" * 60)

    validator = KValidator()

    # Sample validations
    samples = [
        ("q1", "c1", "easy", 5, ["Q1", "Q2", "Q3", "Q4", "Q5"], 5, 1.0),  # Exact
        ("q2", "c1", "medium", 5, ["Q1", "Q2", "Q3"], 4, 0.8),  # Close
        ("q3", "c2", "hard", 10, ["Q1", "Q2"], 3, 0.5),  # Low coverage - reject
        ("q4", "c2", "easy", 5, ["Q1"], 1, 0.2),  # Too few - reject
        ("q5", "c3", "medium", 8, ["Q1"]*8, 10, 1.0),  # Close (+2)
    ]

    results = []
    for query_id, cluster_id, diff, target, entities, passages, cov in samples:
        result = validator.validate_query(
            query_id=query_id,
            cluster_id=cluster_id,
            difficulty=diff,
            target_k=target,
            entity_qids=entities,
            passage_count=passages,
            entity_coverage=cov
        )
        results.append(result)
        status = "ACCEPT" if result.accepted else "REJECT"
        print(f"\n  {query_id}: {status}")
        print(f"    Target K: {target}, Actual K: {result.actual_k}, Discrepancy: {result.k_discrepancy}")
        print(f"    Category: {result.discrepancy_category}")
        if result.rejection_reason:
            print(f"    Reason: {result.rejection_reason}")

    # Build report
    report = validator._build_report(results)
    print("\n" + validator.format_report(report))
