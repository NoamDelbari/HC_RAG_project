"""
End-to-End RAG Evaluator

Combines retrieval IR metrics with LLM judge scores for full RAG pipeline evaluation.

Title matching uses a multi-tier approach inspired by SQuAD, RAGAS, and CRAG benchmarks:
  - SQuAD-style Token F1: bag-of-words token overlap on the full answer string
  - Fuzzy title matching: token-set overlap ratio for matching individual titles
  - Exact title matching: normalized exact string comparison (original strict metric)
"""

import re
import string
import numpy as np
from collections import Counter
from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass, field

from .evaluator import RetrievalResult, AggregateMetrics, RetrievalEvaluator

# Default threshold for fuzzy title matching (token-set overlap ratio)
FUZZY_MATCH_THRESHOLD = 0.8


@dataclass
class E2EResult:
    """Stores end-to-end RAG evaluation results for a single query."""
    query_id: str
    retrieval_result: RetrievalResult
    generated_answer: str
    gold_answer: str
    judge_scores: Dict[str, dict]  # 4 metrics, each with 'rating' and 'reasoning'
    exact_match: float = 0.0  # 1.0 if full string matches, else 0.0

    # SQuAD-style token-level metrics (on full answer string)
    token_precision: float = 0.0
    token_recall: float = 0.0
    token_f1: float = 0.0

    # Title-level set metrics (exact matching)
    title_precision: float = 0.0
    title_recall: float = 0.0
    title_f1: float = 0.0

    # Title-level set metrics (fuzzy matching)
    fuzzy_title_precision: float = 0.0
    fuzzy_title_recall: float = 0.0
    fuzzy_title_f1: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class E2EAggregateMetrics:
    """Aggregated E2E metrics across multiple queries."""
    n_queries: int
    retrieval_metrics: AggregateMetrics

    # Exact Match
    mean_exact_match: float

    # SQuAD-style token-level metrics
    mean_token_precision: float
    mean_token_recall: float
    mean_token_f1: float

    # Title-level set metrics (exact)
    mean_title_precision: float
    mean_title_recall: float
    mean_title_f1: float

    # Title-level set metrics (fuzzy)
    mean_fuzzy_title_precision: float
    mean_fuzzy_title_recall: float
    mean_fuzzy_title_f1: float

    # Mean judge scores (1-5 scale)
    mean_context_precision: float
    mean_context_recall: float
    mean_answer_faithfulness: float
    mean_answer_correctness: float

    def __repr__(self):
        return (
            f"E2EAggregateMetrics(\n"
            f"  n_queries={self.n_queries}\n"
            f"  \n"
            f"  Retrieval (labeled queries):\n"
            f"    Recall@k={self.retrieval_metrics.mean_recall_at_k_labeled:.3f}\n"
            f"    Precision@k={self.retrieval_metrics.mean_precision_at_k_labeled:.3f}\n"
            f"    MRR={self.retrieval_metrics.mean_reciprocal_rank_labeled:.3f}\n"
            f"    NDCG@k={self.retrieval_metrics.mean_ndcg_at_k_labeled:.3f}\n"
            f"    MAP@k={self.retrieval_metrics.mean_average_precision_labeled:.3f}\n"
            f"    Mean k={self.retrieval_metrics.mean_k:.1f}\n"
            f"  \n"
            f"  QA Metrics:\n"
            f"    Exact Match={self.mean_exact_match:.3f}\n"
            f"    Token P={self.mean_token_precision:.3f}  R={self.mean_token_recall:.3f}  F1={self.mean_token_f1:.3f}\n"
            f"    Title P={self.mean_title_precision:.3f}  R={self.mean_title_recall:.3f}  F1={self.mean_title_f1:.3f}  (exact)\n"
            f"    Title P={self.mean_fuzzy_title_precision:.3f}  R={self.mean_fuzzy_title_recall:.3f}  F1={self.mean_fuzzy_title_f1:.3f}  (fuzzy)\n"
            f"  \n"
            f"  Judge Scores (1-5):\n"
            f"    Context Precision={self.mean_context_precision:.2f}\n"
            f"    Context Recall={self.mean_context_recall:.2f}\n"
            f"    Answer Faithfulness={self.mean_answer_faithfulness:.2f}\n"
            f"    Answer Correctness={self.mean_answer_correctness:.2f}\n"
            f")"
        )


class E2EEvaluator:
    """Evaluates full RAG pipeline: retrieval + generation + judging.

    Provides three tiers of answer comparison metrics:
      1. SQuAD-style Token F1 — bag-of-words overlap on full answer string
      2. Title-level Fuzzy F1 — per-title token-set overlap with greedy matching
      3. Title-level Exact F1 — strict normalized string equality per title
    """

    def __init__(self):
        self.retrieval_evaluator = RetrievalEvaluator()

    # ── Normalization ────────────────────────────────────────────────

    @staticmethod
    def _normalize_title(s: str) -> str:
        """Normalize a title for comparison: lowercase, strip, collapse whitespace."""
        s = s.lower().strip()
        s = re.sub(r"\s+", " ", s)
        return s

    @staticmethod
    def _normalize_answer_squad(s: str) -> str:
        """SQuAD-style normalization: lowercase, remove punctuation & articles, collapse whitespace."""
        s = s.lower()
        # Remove punctuation
        s = "".join(ch for ch in s if ch not in string.punctuation)
        # Remove articles
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        # Collapse whitespace
        s = " ".join(s.split())
        return s

    # ── Title splitting ──────────────────────────────────────────────

    @staticmethod
    def _split_titles(answer: str) -> List[str]:
        """Split a comma-separated answer into a list of normalized titles."""
        titles = []
        for t in answer.split(","):
            norm = E2EEvaluator._normalize_title(t)
            if norm:
                titles.append(norm)
        return titles

    # ── Exact Match ──────────────────────────────────────────────────

    @staticmethod
    def compute_exact_match(generated: str, gold: str) -> float:
        """Compute Exact Match (1.0 if normalized strings match, else 0.0)."""
        return 1.0 if E2EEvaluator._normalize_title(generated) == E2EEvaluator._normalize_title(gold) else 0.0

    # ── SQuAD-style Token F1 ─────────────────────────────────────────

    @staticmethod
    def compute_token_f1(generated: str, gold: str) -> Tuple[float, float, float]:
        """
        SQuAD-style token-level Precision, Recall, and F1 on the full answer.

        Normalizes both strings (lowercase, remove punctuation & articles),
        tokenizes by whitespace, then computes bag-of-words overlap using
        multiset intersection (Counter &).

        Returns:
            (precision, recall, f1)
        """
        gen_toks = E2EEvaluator._normalize_answer_squad(generated).split()
        gold_toks = E2EEvaluator._normalize_answer_squad(gold).split()

        if not gen_toks and not gold_toks:
            return 1.0, 1.0, 1.0
        if not gen_toks or not gold_toks:
            return 0.0, 0.0, 0.0

        common = Counter(gen_toks) & Counter(gold_toks)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0, 0.0, 0.0

        precision = num_same / len(gen_toks)
        recall = num_same / len(gold_toks)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    # ── Title-level Exact F1 ─────────────────────────────────────────

    @staticmethod
    def compute_title_f1(generated: str, gold: str) -> Tuple[float, float, float]:
        """
        Compute set-based title-level Precision, Recall, and F1 (exact matching).

        Splits both answers by comma into title sets, normalizes each title,
        then computes set overlap metrics.

        Returns:
            (precision, recall, f1)
        """
        gen_titles = set(E2EEvaluator._split_titles(generated))
        gold_titles = set(E2EEvaluator._split_titles(gold))

        if not gen_titles and not gold_titles:
            return 1.0, 1.0, 1.0
        if not gen_titles or not gold_titles:
            return 0.0, 0.0, 0.0

        overlap = gen_titles & gold_titles
        precision = len(overlap) / len(gen_titles)
        recall = len(overlap) / len(gold_titles)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    # ── Fuzzy title matching ─────────────────────────────────────────

    @staticmethod
    def _token_set_ratio(s1: str, s2: str) -> float:
        """
        Compute token-set overlap ratio between two strings (similar to
        fuzzywuzzy's token_set_ratio but without Levenshtein dependency).

        Tokenizes both strings after SQuAD normalization, computes:
          intersection = tokens in common
          remainder1   = tokens only in s1
          remainder2   = tokens only in s2

        Returns the best Jaccard-like ratio among:
          - sorted(intersection) vs sorted(intersection + remainder1)
          - sorted(intersection) vs sorted(intersection + remainder2)
          - sorted(intersection + remainder1) vs sorted(intersection + remainder2)

        Each comparison uses: 2 * |common_tokens| / (|a_tokens| + |b_tokens|)
        (Dice coefficient on token bags).

        Returns:
            float in [0, 1], where 1.0 = perfect match.
        """
        toks1 = set(E2EEvaluator._normalize_answer_squad(s1).split())
        toks2 = set(E2EEvaluator._normalize_answer_squad(s2).split())

        if not toks1 and not toks2:
            return 1.0
        if not toks1 or not toks2:
            return 0.0

        intersection = toks1 & toks2
        remainder1 = toks1 - toks2
        remainder2 = toks2 - toks1

        # Compare sorted token strings (as fuzzywuzzy does)
        t_int = sorted(intersection)
        t_int_r1 = sorted(intersection | remainder1)
        t_int_r2 = sorted(intersection | remainder2)

        def _dice(a_toks, b_toks):
            """Dice coefficient on two token lists."""
            a_set, b_set = set(a_toks), set(b_toks)
            if not a_set and not b_set:
                return 1.0
            if not a_set or not b_set:
                return 0.0
            return 2 * len(a_set & b_set) / (len(a_set) + len(b_set))

        return max(
            _dice(t_int, t_int_r1),
            _dice(t_int, t_int_r2),
            _dice(t_int_r1, t_int_r2),
        )

    @staticmethod
    def compute_fuzzy_title_f1(
        generated: str,
        gold: str,
        threshold: float = FUZZY_MATCH_THRESHOLD,
    ) -> Tuple[float, float, float]:
        """
        Compute title-level Precision, Recall, and F1 with fuzzy matching.

        For each generated title, finds the best-matching gold title using
        token-set overlap ratio. If the best score >= threshold, it counts
        as a match (greedy bipartite matching — each gold title matched at most once).

        Returns:
            (precision, recall, f1)
        """
        gen_titles = E2EEvaluator._split_titles(generated)
        gold_titles = E2EEvaluator._split_titles(gold)

        if not gen_titles and not gold_titles:
            return 1.0, 1.0, 1.0
        if not gen_titles or not gold_titles:
            return 0.0, 0.0, 0.0

        # Build similarity matrix and greedy match
        matched_gold = set()  # indices of matched gold titles
        matches = 0

        # Sort gen titles by best available match (descending) for better greedy results
        gen_scores = []
        for gi, gt in enumerate(gen_titles):
            best_score = 0.0
            best_gidx = -1
            for goi, go in enumerate(gold_titles):
                score = E2EEvaluator._token_set_ratio(gt, go)
                if score > best_score:
                    best_score = score
                    best_gidx = goi
            gen_scores.append((best_score, gi, best_gidx))

        # Greedy match: highest scores first
        gen_scores.sort(reverse=True)
        for best_score, gi, best_gidx in gen_scores:
            if best_score < threshold:
                continue
            # Re-check: find best unmatched gold title for this gen title
            best_unmatched_score = 0.0
            best_unmatched_idx = -1
            for goi, go in enumerate(gold_titles):
                if goi in matched_gold:
                    continue
                score = E2EEvaluator._token_set_ratio(gen_titles[gi], go)
                if score > best_unmatched_score:
                    best_unmatched_score = score
                    best_unmatched_idx = goi
            if best_unmatched_score >= threshold and best_unmatched_idx >= 0:
                matched_gold.add(best_unmatched_idx)
                matches += 1

        precision = matches / len(gen_titles)
        recall = matches / len(gold_titles)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    # ── Single-query evaluation ──────────────────────────────────────

    def evaluate_single(
        self,
        query_id: str,
        retrieved_ids: List[str],
        retrieved_scores: List[float],
        relevant_ids: set,
        generated_answer: str,
        gold_answer: str,
        judge_scores: Dict[str, dict],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> E2EResult:
        """Evaluate a single query end-to-end."""
        retrieval_result = self.retrieval_evaluator.evaluate_single(
            query_id=query_id,
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            relevant_ids=relevant_ids,
            metadata=metadata,
        )

        em = self.compute_exact_match(generated_answer, gold_answer)
        tok_p, tok_r, tok_f1 = self.compute_token_f1(generated_answer, gold_answer)
        t_prec, t_rec, t_f1 = self.compute_title_f1(generated_answer, gold_answer)
        ft_prec, ft_rec, ft_f1 = self.compute_fuzzy_title_f1(generated_answer, gold_answer)

        return E2EResult(
            query_id=query_id,
            retrieval_result=retrieval_result,
            generated_answer=generated_answer,
            gold_answer=gold_answer,
            judge_scores=judge_scores,
            exact_match=em,
            token_precision=tok_p,
            token_recall=tok_r,
            token_f1=tok_f1,
            title_precision=t_prec,
            title_recall=t_rec,
            title_f1=t_f1,
            fuzzy_title_precision=ft_prec,
            fuzzy_title_recall=ft_rec,
            fuzzy_title_f1=ft_f1,
            metadata=metadata or {},
        )

    # ── Batch aggregation ────────────────────────────────────────────

    def evaluate_batch(self, results: List[E2EResult]) -> E2EAggregateMetrics:
        """Compute aggregate E2E metrics from individual results."""
        if not results:
            raise ValueError("Cannot compute metrics from empty results")

        # Aggregate retrieval metrics
        retrieval_results = [r.retrieval_result for r in results]
        retrieval_metrics = self.retrieval_evaluator.evaluate_batch(retrieval_results)

        # Aggregate judge scores
        metric_keys = ["context_precision", "context_recall", "answer_faithfulness", "answer_correctness"]
        mean_scores = {}
        for key in metric_keys:
            ratings = [
                r.judge_scores[key]["rating"]
                for r in results
                if key in r.judge_scores and "rating" in r.judge_scores[key]
            ]
            mean_scores[key] = float(np.mean(ratings)) if ratings else 0.0

        # Aggregate QA metrics
        mean_em = float(np.mean([r.exact_match for r in results]))

        mean_tok_p = float(np.mean([r.token_precision for r in results]))
        mean_tok_r = float(np.mean([r.token_recall for r in results]))
        mean_tok_f1 = float(np.mean([r.token_f1 for r in results]))

        mean_t_prec = float(np.mean([r.title_precision for r in results]))
        mean_t_rec = float(np.mean([r.title_recall for r in results]))
        mean_t_f1 = float(np.mean([r.title_f1 for r in results]))

        mean_ft_prec = float(np.mean([r.fuzzy_title_precision for r in results]))
        mean_ft_rec = float(np.mean([r.fuzzy_title_recall for r in results]))
        mean_ft_f1 = float(np.mean([r.fuzzy_title_f1 for r in results]))

        return E2EAggregateMetrics(
            n_queries=len(results),
            retrieval_metrics=retrieval_metrics,
            mean_exact_match=mean_em,
            mean_token_precision=mean_tok_p,
            mean_token_recall=mean_tok_r,
            mean_token_f1=mean_tok_f1,
            mean_title_precision=mean_t_prec,
            mean_title_recall=mean_t_rec,
            mean_title_f1=mean_t_f1,
            mean_fuzzy_title_precision=mean_ft_prec,
            mean_fuzzy_title_recall=mean_ft_rec,
            mean_fuzzy_title_f1=mean_ft_f1,
            mean_context_precision=mean_scores["context_precision"],
            mean_context_recall=mean_scores["context_recall"],
            mean_answer_faithfulness=mean_scores["answer_faithfulness"],
            mean_answer_correctness=mean_scores["answer_correctness"],
        )
