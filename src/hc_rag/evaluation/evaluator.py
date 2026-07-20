"""
Retrieval Evaluator - IR Metrics Only

Implements standard Information Retrieval metrics for evaluating retrieval quality
WITHOUT LLM generation. This is Phase 1 evaluation focused purely on retrieval.

Metrics:
- Recall@k: % of relevant items retrieved
- Precision@k: % of retrieved items that are relevant
- MRR (Mean Reciprocal Rank): Inverse rank of first relevant item
- NDCG@k: Normalized Discounted Cumulative Gain (binary relevance)
- Hit Rate: Whether at least one relevant item was retrieved
- MAP@k: Mean Average Precision (truncated at k)
- Coverage: Query coverage and catalog coverage

Note: Accuracy and hallucination metrics are in Phase 4 (end-to-end RAG evaluation)
"""

import numpy as np
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field
import logging

# Only get logger, don't configure (let entrypoint configure)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Stores retrieval results for a single query.
    """
    query_id: str
    retrieved_ids: List[str]  # Retrieved document/chunk IDs in rank order (deduplicated)
    retrieved_scores: List[float]  # Similarity scores (after deduplication)
    relevant_ids: Set[str]  # Ground truth relevant IDs
    k: int  # Number retrieved (can be variable for HC)

    # Flags
    has_relevant_labels: bool = True  # Whether this query has any labeled relevant items
    had_duplicates: bool = False  # Whether duplicates were removed

    # Computed metrics
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    reciprocal_rank: float = 0.0
    ndcg_at_k: float = 0.0
    hit_rate: float = 0.0
    average_precision: float = 0.0

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregateMetrics:
    """
    Aggregated metrics across multiple queries.
    """
    n_queries: int
    n_labeled_queries: int  # Queries with relevant_ids > 0
    n_unlabeled_queries: int  # Queries with relevant_ids == 0

    # Mean metrics (across all queries)
    mean_recall_at_k: float
    mean_precision_at_k: float
    mean_reciprocal_rank: float
    mean_ndcg_at_k: float
    mean_hit_rate: float
    mean_average_precision: float

    # Mean metrics (across labeled queries only)
    mean_recall_at_k_labeled: float
    mean_precision_at_k_labeled: float
    mean_reciprocal_rank_labeled: float
    mean_ndcg_at_k_labeled: float
    mean_hit_rate_labeled: float
    mean_average_precision_labeled: float

    # Coverage metrics
    query_coverage: float  # Fraction of queries that retrieved at least one document
    catalog_coverage: float  # Fraction of all relevant items retrieved at least once

    # Retrieval stats
    mean_k: float  # Average number of items retrieved (useful for HC)
    min_k: int
    max_k: int

    # Quality flags
    n_queries_with_duplicates: int  # Number of queries that had duplicates removed

    def __repr__(self):
        return (f"AggregateMetrics(\n"
                f"  n_queries={self.n_queries} ({self.n_labeled_queries} labeled, {self.n_unlabeled_queries} unlabeled)\n"
                f"  \n"
                f"  All queries:\n"
                f"    Recall@k={self.mean_recall_at_k:.3f}\n"
                f"    Precision@k={self.mean_precision_at_k:.3f}\n"
                f"    MRR={self.mean_reciprocal_rank:.3f}\n"
                f"    NDCG@k={self.mean_ndcg_at_k:.3f}\n"
                f"    Hit Rate={self.mean_hit_rate:.3f}\n"
                f"    MAP@k={self.mean_average_precision:.3f}\n"
                f"  \n"
                f"  Labeled queries only:\n"
                f"    Recall@k={self.mean_recall_at_k_labeled:.3f}\n"
                f"    Precision@k={self.mean_precision_at_k_labeled:.3f}\n"
                f"    MRR={self.mean_reciprocal_rank_labeled:.3f}\n"
                f"    NDCG@k={self.mean_ndcg_at_k_labeled:.3f}\n"
                f"    Hit Rate={self.mean_hit_rate_labeled:.3f}\n"
                f"    MAP@k={self.mean_average_precision_labeled:.3f}\n"
                f"  \n"
                f"  Coverage:\n"
                f"    Query coverage={self.query_coverage:.3f}\n"
                f"    Catalog coverage={self.catalog_coverage:.3f}\n"
                f"  \n"
                f"  Retrieval stats:\n"
                f"    Avg k={self.mean_k:.1f} (range: {self.min_k}-{self.max_k})\n"
                f"    Queries with duplicates={self.n_queries_with_duplicates}\n"
                f")")


class RetrievalEvaluator:
    """
    Evaluates retrieval quality using standard IR metrics.

    NO LLM generation, NO accuracy metrics, NO hallucination detection.
    Pure retrieval evaluation for Phase 1.
    """

    def __init__(self):
        """Initialize retrieval evaluator."""
        logger.info("RetrievalEvaluator initialized (IR metrics only)")

    # =========================================================================
    # Utilities
    # =========================================================================

    def _deduplicate_preserving_order(
        self,
        retrieved_ids: List[str],
        retrieved_scores: List[float]
    ) -> tuple[List[str], List[float], bool]:
        """
        Remove duplicates from retrieved_ids while preserving order.

        Keep first occurrence of each ID (highest rank).

        Args:
            retrieved_ids: Retrieved document IDs
            retrieved_scores: Corresponding similarity scores

        Returns:
            Tuple of (deduplicated_ids, deduplicated_scores, had_duplicates)
        """
        seen = set()
        dedup_ids = []
        dedup_scores = []

        for doc_id, score in zip(retrieved_ids, retrieved_scores):
            if doc_id not in seen:
                seen.add(doc_id)
                dedup_ids.append(doc_id)
                dedup_scores.append(score)

        had_duplicates = len(dedup_ids) < len(retrieved_ids)

        return dedup_ids, dedup_scores, had_duplicates

    # =========================================================================
    # Core IR Metrics
    # =========================================================================

    def compute_recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str]
    ) -> float:
        """
        Compute Recall@k: fraction of relevant items retrieved.

        Recall@k = |Retrieved ∩ Relevant| / |Relevant|

        Args:
            retrieved_ids: Retrieved document IDs (top-k)
            relevant_ids: Ground truth relevant document IDs

        Returns:
            Recall@k score (0-1)
        """
        if not relevant_ids:
            return 0.0

        retrieved_set = set(retrieved_ids)
        num_relevant_retrieved = len(retrieved_set & relevant_ids)

        return num_relevant_retrieved / len(relevant_ids)

    def compute_precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str]
    ) -> float:
        """
        Compute Precision@k: fraction of retrieved items that are relevant.

        Precision@k = |Retrieved ∩ Relevant| / k

        Args:
            retrieved_ids: Retrieved document IDs (top-k)
            relevant_ids: Ground truth relevant document IDs

        Returns:
            Precision@k score (0-1)
        """
        if not retrieved_ids:
            return 0.0

        retrieved_set = set(retrieved_ids)
        num_relevant_retrieved = len(retrieved_set & relevant_ids)

        return num_relevant_retrieved / len(retrieved_ids)

    def compute_reciprocal_rank(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str]
    ) -> float:
        """
        Compute Reciprocal Rank: inverse of rank of first relevant item.

        RR = 1 / rank_of_first_relevant_item

        Args:
            retrieved_ids: Retrieved document IDs in rank order
            relevant_ids: Ground truth relevant document IDs

        Returns:
            Reciprocal rank (0-1)
        """
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank

        return 0.0  # No relevant items found

    def compute_ndcg_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str]
    ) -> float:
        """
        Compute NDCG@k: Normalized Discounted Cumulative Gain.

        DCG@k = Σ (rel_i / log2(i+1)) for i=1 to k
        NDCG@k = DCG@k / IDCG@k

        Assumes binary relevance: relevant=1, not relevant=0.
        For graded relevance, replace the binary check with a relevance score lookup.

        Args:
            retrieved_ids: Retrieved document IDs in rank order
            relevant_ids: Ground truth relevant document IDs

        Returns:
            NDCG@k score (0-1)
        """
        if not retrieved_ids or not relevant_ids:
            return 0.0

        # Compute DCG (binary relevance: 1 if relevant, 0 otherwise)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                dcg += 1.0 / np.log2(i + 1)

        # Compute IDCG (ideal DCG - all relevant items ranked first)
        idcg = 0.0
        for i in range(1, min(len(relevant_ids), len(retrieved_ids)) + 1):
            idcg += 1.0 / np.log2(i + 1)

        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    def compute_hit_rate(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str]
    ) -> float:
        """
        Compute Hit Rate: whether at least one relevant item was retrieved.

        Hit Rate = 1 if |Retrieved ∩ Relevant| > 0, else 0

        Args:
            retrieved_ids: Retrieved document IDs
            relevant_ids: Ground truth relevant document IDs

        Returns:
            1.0 if hit, 0.0 otherwise
        """
        retrieved_set = set(retrieved_ids)
        return 1.0 if len(retrieved_set & relevant_ids) > 0 else 0.0

    def compute_average_precision(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str]
    ) -> float:
        """
        Compute Average Precision: mean precision at each relevant item position.

        AP@k = (Σ P(i) * rel(i)) / |Relevant|
        where P(i) = precision at rank i, rel(i) = 1 if item at i is relevant

        Note: This is AP@k (truncated), since we only see the top-k retrieved list.
        The denominator is the total number of relevant items in the corpus,
        not just those retrieved.

        Args:
            retrieved_ids: Retrieved document IDs in rank order
            relevant_ids: Ground truth relevant document IDs

        Returns:
            Average Precision (0-1)
        """
        if not relevant_ids:
            return 0.0

        num_relevant_retrieved = 0
        sum_precisions = 0.0

        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                num_relevant_retrieved += 1
                precision_at_rank = num_relevant_retrieved / rank
                sum_precisions += precision_at_rank

        if num_relevant_retrieved == 0:
            return 0.0

        return sum_precisions / len(relevant_ids)

    # =========================================================================
    # Evaluation
    # =========================================================================

    def evaluate_single(
        self,
        query_id: str,
        retrieved_ids: List[str],
        retrieved_scores: List[float],
        relevant_ids: Set[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Evaluate retrieval for a single query.

        Args:
            query_id: Unique query identifier
            retrieved_ids: Retrieved document IDs in rank order
            retrieved_scores: Similarity scores for retrieved documents
            relevant_ids: Ground truth relevant document IDs
            metadata: Optional additional metadata

        Returns:
            RetrievalResult with all metrics
        """
        # Validate input lengths
        if len(retrieved_ids) != len(retrieved_scores):
            raise ValueError(
                f"Length mismatch: retrieved_ids ({len(retrieved_ids)}) "
                f"vs retrieved_scores ({len(retrieved_scores)}) for query {query_id}"
            )

        # Deduplicate while preserving order
        dedup_ids, dedup_scores, had_duplicates = self._deduplicate_preserving_order(
            retrieved_ids, retrieved_scores
        )

        if had_duplicates:
            logger.warning(
                f"Query {query_id}: Removed {len(retrieved_ids) - len(dedup_ids)} "
                f"duplicate(s) from retrieved list"
            )

        # Check if query has labels
        has_relevant_labels = len(relevant_ids) > 0

        # Compute all metrics (will return 0 if no labels, which is expected)
        recall = self.compute_recall_at_k(dedup_ids, relevant_ids)
        precision = self.compute_precision_at_k(dedup_ids, relevant_ids)
        rr = self.compute_reciprocal_rank(dedup_ids, relevant_ids)
        ndcg = self.compute_ndcg_at_k(dedup_ids, relevant_ids)
        hit = self.compute_hit_rate(dedup_ids, relevant_ids)
        ap = self.compute_average_precision(dedup_ids, relevant_ids)

        return RetrievalResult(
            query_id=query_id,
            retrieved_ids=dedup_ids,
            retrieved_scores=dedup_scores,
            relevant_ids=relevant_ids,
            k=len(dedup_ids),
            has_relevant_labels=has_relevant_labels,
            had_duplicates=had_duplicates,
            recall_at_k=recall,
            precision_at_k=precision,
            reciprocal_rank=rr,
            ndcg_at_k=ndcg,
            hit_rate=hit,
            average_precision=ap,
            metadata=metadata or {}
        )

    def evaluate_batch(
        self,
        results: List[RetrievalResult]
    ) -> AggregateMetrics:
        """
        Compute aggregate metrics from individual results.

        Handles unlabeled queries (relevant_ids empty) by computing metrics
        both across all queries and across labeled queries only.

        Args:
            results: List of RetrievalResult objects

        Returns:
            AggregateMetrics with aggregated scores
        """
        if not results:
            raise ValueError("Cannot compute metrics from empty results")

        n_queries = len(results)

        # Separate labeled and unlabeled queries
        labeled_results = [r for r in results if r.has_relevant_labels]
        unlabeled_results = [r for r in results if not r.has_relevant_labels]

        n_labeled = len(labeled_results)
        n_unlabeled = len(unlabeled_results)

        logger.info(f"Evaluating {n_queries} queries: {n_labeled} labeled, {n_unlabeled} unlabeled")

        if n_unlabeled > 0:
            logger.warning(
                f"{n_unlabeled} queries have no labeled relevant items. "
                f"Metrics will be computed separately for labeled queries."
            )

        # Aggregate metrics across ALL queries
        mean_recall = np.mean([r.recall_at_k for r in results])
        mean_precision = np.mean([r.precision_at_k for r in results])
        mean_mrr = np.mean([r.reciprocal_rank for r in results])
        mean_ndcg = np.mean([r.ndcg_at_k for r in results])
        mean_hit = np.mean([r.hit_rate for r in results])
        mean_ap = np.mean([r.average_precision for r in results])

        # Aggregate metrics across LABELED queries only
        if n_labeled > 0:
            mean_recall_labeled = np.mean([r.recall_at_k for r in labeled_results])
            mean_precision_labeled = np.mean([r.precision_at_k for r in labeled_results])
            mean_mrr_labeled = np.mean([r.reciprocal_rank for r in labeled_results])
            mean_ndcg_labeled = np.mean([r.ndcg_at_k for r in labeled_results])
            mean_hit_labeled = np.mean([r.hit_rate for r in labeled_results])
            mean_ap_labeled = np.mean([r.average_precision for r in labeled_results])
        else:
            mean_recall_labeled = 0.0
            mean_precision_labeled = 0.0
            mean_mrr_labeled = 0.0
            mean_ndcg_labeled = 0.0
            mean_hit_labeled = 0.0
            mean_ap_labeled = 0.0

        # Coverage metrics
        # Query coverage: fraction of queries that retrieved at least one document
        query_coverage = np.mean([1.0 if r.k > 0 else 0.0 for r in results])

        # Catalog coverage: fraction of all relevant items retrieved at least once
        all_relevant_ids = set()
        all_retrieved_ids = set()
        for r in results:
            all_relevant_ids.update(r.relevant_ids)
            all_retrieved_ids.update(r.retrieved_ids)

        if len(all_relevant_ids) > 0:
            catalog_coverage = len(all_retrieved_ids & all_relevant_ids) / len(all_relevant_ids)
        else:
            catalog_coverage = 0.0

        # k statistics
        ks = [r.k for r in results]
        mean_k = np.mean(ks)
        min_k = min(ks) if ks else 0
        max_k = max(ks) if ks else 0

        # Duplicate count
        n_queries_with_duplicates = sum(1 for r in results if r.had_duplicates)

        return AggregateMetrics(
            n_queries=n_queries,
            n_labeled_queries=n_labeled,
            n_unlabeled_queries=n_unlabeled,
            mean_recall_at_k=mean_recall,
            mean_precision_at_k=mean_precision,
            mean_reciprocal_rank=mean_mrr,
            mean_ndcg_at_k=mean_ndcg,
            mean_hit_rate=mean_hit,
            mean_average_precision=mean_ap,
            mean_recall_at_k_labeled=mean_recall_labeled,
            mean_precision_at_k_labeled=mean_precision_labeled,
            mean_reciprocal_rank_labeled=mean_mrr_labeled,
            mean_ndcg_at_k_labeled=mean_ndcg_labeled,
            mean_hit_rate_labeled=mean_hit_labeled,
            mean_average_precision_labeled=mean_ap_labeled,
            query_coverage=query_coverage,
            catalog_coverage=catalog_coverage,
            mean_k=mean_k,
            min_k=min_k,
            max_k=max_k,
            n_queries_with_duplicates=n_queries_with_duplicates
        )


def main():
    """Example usage of RetrievalEvaluator."""
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("Retrieval Evaluator Demo (IR Metrics Only)")
    print("="*70 + "\n")

    evaluator = RetrievalEvaluator()

    # Example 1: Perfect retrieval
    print("Example 1: Perfect retrieval (all 3 relevant items in top-5)")
    print("-" * 70)

    result1 = evaluator.evaluate_single(
        query_id="q1",
        retrieved_ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
        retrieved_scores=[0.95, 0.90, 0.85, 0.70, 0.65],
        relevant_ids={"doc1", "doc2", "doc3"}
    )

    print(f"Retrieved: {result1.retrieved_ids[:5]}")
    print(f"Relevant: {result1.relevant_ids}")
    print(f"\nMetrics:")
    print(f"  Recall@5: {result1.recall_at_k:.3f}")
    print(f"  Precision@5: {result1.precision_at_k:.3f}")
    print(f"  MRR: {result1.reciprocal_rank:.3f}")
    print(f"  NDCG@5: {result1.ndcg_at_k:.3f}")
    print(f"  Hit Rate: {result1.hit_rate:.0f}")
    print(f"  AP@5: {result1.average_precision:.3f}")

    # Example 2: Partial retrieval
    print("\n\nExample 2: Partial retrieval (1 of 3 relevant items)")
    print("-" * 70)

    result2 = evaluator.evaluate_single(
        query_id="q2",
        retrieved_ids=["doc10", "doc11", "doc1", "doc12", "doc13"],
        retrieved_scores=[0.80, 0.75, 0.70, 0.65, 0.60],
        relevant_ids={"doc1", "doc2", "doc3"}
    )

    print(f"Retrieved: {result2.retrieved_ids[:5]}")
    print(f"Relevant: {result2.relevant_ids}")
    print(f"\nMetrics:")
    print(f"  Recall@5: {result2.recall_at_k:.3f}")
    print(f"  Precision@5: {result2.precision_at_k:.3f}")
    print(f"  MRR: {result2.reciprocal_rank:.3f} (first relevant at rank 3)")
    print(f"  NDCG@5: {result2.ndcg_at_k:.3f}")
    print(f"  Hit Rate: {result2.hit_rate:.0f}")
    print(f"  AP@5: {result2.average_precision:.3f}")

    # Example 3: Retrieval with duplicates
    print("\n\nExample 3: Retrieval with duplicates (will be deduplicated)")
    print("-" * 70)

    result3 = evaluator.evaluate_single(
        query_id="q3",
        retrieved_ids=["doc1", "doc2", "doc1", "doc3", "doc2"],  # Duplicates!
        retrieved_scores=[0.95, 0.90, 0.85, 0.80, 0.75],
        relevant_ids={"doc1", "doc2", "doc3"}
    )

    print(f"Original: ['doc1', 'doc2', 'doc1', 'doc3', 'doc2']")
    print(f"Deduplicated: {result3.retrieved_ids}")
    print(f"Had duplicates: {result3.had_duplicates}")
    print(f"\nMetrics (after deduplication):")
    print(f"  Recall@3: {result3.recall_at_k:.3f}")
    print(f"  Precision@3: {result3.precision_at_k:.3f}")

    # Example 4: Unlabeled query (no relevant items)
    print("\n\nExample 4: Unlabeled query (no relevant items)")
    print("-" * 70)

    result4 = evaluator.evaluate_single(
        query_id="q4",
        retrieved_ids=["doc10", "doc11", "doc12"],
        retrieved_scores=[0.80, 0.75, 0.70],
        relevant_ids=set()  # No labels
    )

    print(f"Retrieved: {result4.retrieved_ids}")
    print(f"Relevant: {result4.relevant_ids} (unlabeled)")
    print(f"Has labels: {result4.has_relevant_labels}")
    print(f"\nMetrics (will be 0 for unlabeled):")
    print(f"  Recall@3: {result4.recall_at_k:.3f}")

    # Aggregate metrics
    print("\n\nAggregate Metrics")
    print("-" * 70)

    aggregate = evaluator.evaluate_batch([result1, result2, result3, result4])
    print(aggregate)

    print("\n" + "="*70)
    print("Note: This evaluator uses IR metrics only (no LLM, no accuracy)")
    print("Accuracy and hallucination metrics are in Phase 4")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
