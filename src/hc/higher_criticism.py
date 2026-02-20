"""
Higher Criticism (HC) Module

Implements Higher Criticism statistics for adaptive document thresholding in RAG.
Based on the HC-RAG paper's approach to statistical significance testing.
"""

import numpy as np
from typing import Optional, Tuple, NamedTuple
import logging
import sys
from pathlib import Path

# Handle both direct execution and module import
try:
    from .null_distribution import NullDistribution
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from null_distribution import NullDistribution

logger = logging.getLogger(__name__)


class HCThresholdResult(NamedTuple):
    """
    Result from HC threshold computation.

    Attributes:
        threshold: Similarity threshold for document selection
        hc_statistic: Maximum HC statistic value
        k: Number of documents to retrieve (0 for empty set)
    """
    threshold: float
    hc_statistic: float
    k: int


class HigherCriticism:
    """
    Higher Criticism statistical test for adaptive document retrieval.

    Uses HC statistics to identify which retrieved documents are
    statistically significant (likely relevant) vs random noise.
    
    Supports both global and per-query null distributions. Per-query nulls
    are critical for correctness when queries have different similarity baselines.
    """

    def __init__(self, null_distribution: Optional[NullDistribution] = None):
        """
        Initialize Higher Criticism module.

        Args:
            null_distribution: Pre-computed null distribution (optional).
                              Can be overridden per-query using set_null_distribution().
        """
        self.null_distribution = null_distribution
        logger.info("HigherCriticism module initialized")

        if null_distribution is not None:
            logger.info(f"  Loaded null distribution: {null_distribution}")
    
    def set_null_distribution(self, null_distribution: NullDistribution):
        """
        Set null distribution (allows per-query null distributions).
        
        Args:
            null_distribution: The null distribution to use for p-value computation.
        """
        self.null_distribution = null_distribution

    # =========================================================================
    # Higher Criticism Statistic Calculation
    # =========================================================================

    def compute_p_values(self, similarities: np.ndarray) -> np.ndarray:
        """
        Compute p-values for similarity scores using the null distribution.

        P-value = P(null similarity >= observed similarity)

        Uses vectorized searchsorted for O(n log N) performance instead of O(n*N).
        Applies proper Monte Carlo calibration: smallest p-value is ~1/(N+1).

        Args:
            similarities: Array of observed similarity scores

        Returns:
            Array of p-values (same shape as similarities)
        """
        if self.null_distribution is None:
            raise ValueError("No null distribution available. Call fit_null_distribution() first.")

        null_sims = self.null_distribution.similarities
        N_null = len(null_sims)

        # Sort null distribution once for binary search
        sorted_null = np.sort(null_sims)

        # Vectorized p-value computation using searchsorted
        # For each observed sim, find how many null samples are >= sim
        # searchsorted with 'left' gives first index where null >= observed
        # All values at that index and beyond satisfy null >= observed
        indices = np.searchsorted(sorted_null, similarities, side='left')
        n_greater_equal = N_null - indices

        # Monte Carlo calibration: p = (n_greater_equal + 1) / (N_null + 1)
        # +1 smoothing ensures p-values are in (0, 1] and respects finite sample resolution
        p_values = (n_greater_equal + 1.0) / (N_null + 1.0)

        # Clip to principled bounds to prevent 0/1 in HC denominator
        # Smallest meaningful p-value is ~1/(N_null+1)
        eps = 1.0 / (N_null + 1.0)
        p_values = np.clip(p_values, eps, 1.0 - eps)

        return p_values

    def compute_hc_from_pvalues(
        self,
        p_values: np.ndarray,
        gamma: float = 0.1
    ) -> Tuple[float, int]:
        """
        Compute Higher Criticism statistic from pre-computed p-values.
        
        Use this when p-values are computed externally (e.g., position-aware nulls).
        
        Args:
            p_values: Pre-computed p-values (must be sorted in ascending order)
            gamma: Fraction of top scores to search for HC maximum (0 < gamma <= 1)
            
        Returns:
            Tuple of (hc_statistic, best_index)
        """
        # Validate gamma
        if not (0 < gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        
        if len(p_values) == 0:
            return 0.0, 0
        
        # Sort p-values ascending (smallest first)
        sorted_pvals = np.sort(p_values)
        n = len(sorted_pvals)
        
        # Clip p-values to avoid edge cases
        sorted_pvals = np.clip(sorted_pvals, 1e-10, 1 - 1e-10)
        
        # Only consider top gamma fraction
        n_gamma = min(n, max(1, int(np.floor(gamma * n))))
        
        # Compute HC statistic
        hc_values = []
        for i in range(1, n_gamma + 1):
            expected_quantile = i / n
            p_i = sorted_pvals[i - 1]
            
            numerator = expected_quantile - p_i
            denominator = np.sqrt(p_i * (1 - p_i) / n)
            
            if denominator > 0:
                hc_i = numerator / denominator
                hc_values.append(hc_i)
            else:
                hc_values.append(0.0)
        
        if hc_values:
            max_hc = float(np.max(hc_values))
            best_idx = int(np.argmax(hc_values))
        else:
            max_hc = 0.0
            best_idx = 0
        
        return max_hc, best_idx

    def compute_hc_threshold_from_pvalues(
        self,
        p_values: np.ndarray,
        similarities: np.ndarray,
        gamma: float = 0.1,
        min_hc: float = 0.0,
        allow_empty: bool = True
    ) -> 'HCThresholdResult':
        """
        Compute HC-based threshold from pre-computed p-values.
        
        IMPORTANT: With position-aware nulls, p-values are not monotonic with similarity.
        HC determines how many of the SMALLEST p-values are significant, then we
        return the k documents with highest similarity (standard top-k retrieval).
        
        This preserves compatibility with downstream systems that expect similarity ranking.
        
        Args:
            p_values: Pre-computed p-values (same order as similarities)
            similarities: Original similarity scores (for threshold computation)
            gamma: HC gamma parameter
            min_hc: Minimum HC statistic threshold
            allow_empty: Allow empty result set
            
        Returns:
            HCThresholdResult with threshold, hc_statistic, and k
        """
        if len(p_values) == 0:
            return HCThresholdResult(threshold=np.inf, hc_statistic=0.0, k=0)
        
        n = len(p_values)
        
        # Compute HC from p-values (sorts internally)
        hc_stat, best_idx = self.compute_hc_from_pvalues(p_values, gamma=gamma)
        
        # Check HC gate
        if allow_empty and hc_stat < min_hc:
            return HCThresholdResult(threshold=np.inf, hc_statistic=float(hc_stat), k=0)
        
        # k is number of "significant" p-values (smallest p-values)
        k = max(0, min(n, best_idx + 1))
        
        if not allow_empty and k == 0:
            k = 1
        
        # Return top-k by SIMILARITY (not by p-value ranking)
        # This maintains compatibility with standard retrieval
        sorted_sims = np.sort(similarities)[::-1]  # Sort descending
        
        if k > 0:
            threshold = float(sorted_sims[k - 1])
        else:
            threshold = np.inf
        
        return HCThresholdResult(threshold=threshold, hc_statistic=float(hc_stat), k=int(k))

    def compute_hc_statistic(
        self,
        similarities: np.ndarray,
        gamma: float = 0.1
    ) -> Tuple[float, int]:
        """
        Compute Higher Criticism statistic.

        HC statistic measures the maximum deviation between observed p-values
        and uniform distribution, focusing on small p-values (likely relevant docs).

        Maximizes HC over the first floor(gamma*n) order statistics and returns
        the argmax index (not a quantile threshold).

        Formula: HC_i = sqrt(n) * (i/n - p_(i)) / sqrt(p_(i) * (1 - p_(i)))

        Args:
            similarities: Array of similarity scores (will be sorted descending)
            gamma: Fraction of top scores to search for HC maximum (0 < gamma <= 1)

        Returns:
            Tuple of (hc_statistic, best_index)
            - hc_statistic: Maximum HC value across the search window
            - best_index: Index (0-based) where maximum occurs

        Raises:
            ValueError: If gamma is not in (0, 1]
        """
        # Validate gamma
        if not (0 < gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")

        # Handle empty input
        if len(similarities) == 0:
            logger.debug("compute_hc_statistic: empty input, returning (0.0, 0)")
            return 0.0, 0

        # Sort similarities in descending order (highest first)
        sorted_sims = np.sort(similarities)[::-1]

        # Compute p-values for sorted similarities
        p_values = self.compute_p_values(sorted_sims)

        # Number of documents
        n = len(p_values)

        # Only consider top gamma fraction of smallest p-values
        # Ensure window is in valid range [1, n]
        n_gamma = min(n, max(1, int(np.floor(gamma * n))))

        if n_gamma == 1 and gamma * n < 1:
            logger.debug(f"gamma={gamma} with n={n} yields search window < 1, using n_gamma=1")

        # Compute HC statistic for each position i (1-indexed in formula, 0-indexed in code)
        hc_values = []
        for i in range(1, n_gamma + 1):
            # Expected quantile under uniform distribution
            expected_quantile = i / n

            # Observed p-value (sorted, so this is the i-th smallest p-value)
            p_i = p_values[i - 1]

            # HC statistic at position i
            # Note: p_i is already clipped in compute_p_values to avoid division by zero
            numerator = expected_quantile - p_i
            denominator = np.sqrt(p_i * (1 - p_i) / n)

            # Denominator should never be zero due to p-value clipping, but check anyway
            if denominator > 0:
                hc_i = numerator / denominator
                hc_values.append(hc_i)
            else:
                hc_values.append(0.0)

        # Find maximum HC value
        if hc_values:
            max_hc = float(np.max(hc_values))
            best_idx = int(np.argmax(hc_values))
        else:
            max_hc = 0.0
            best_idx = 0

        return max_hc, best_idx

    def compute_hc_threshold(
        self,
        similarities: np.ndarray,
        gamma: float = 0.1,
        min_hc: float = 0.0,
        allow_empty: bool = True
    ) -> HCThresholdResult:
        """
        Compute HC-based threshold for adaptive retrieval.

        The threshold is determined by the position where HC is maximized.
        Selection rule: retrieve the top-k documents where k is determined by HC.
        The threshold is the score of the k-th document (informational).

        Can return empty set (k=0, threshold=+inf) if HC statistic is below min_hc.

        Args:
            similarities: Array of similarity scores from retrieval
            gamma: Fraction of top scores to search for HC maximum (0 < gamma <= 1)
            min_hc: Minimum HC statistic to accept any documents. If HC < min_hc,
                    returns empty set (threshold=+inf, k=0). Set to 0.0 to disable.
            allow_empty: If True, can return empty set when HC < min_hc.
                        If False, always returns at least 1 document.

        Returns:
            HCThresholdResult with:
                - threshold: Score of k-th document (or +inf for empty set)
                - hc_statistic: Maximum HC value
                - k: Number of documents to retrieve (0 for empty set)

        Examples:
            >>> # Standard usage
            >>> result = hc.compute_hc_threshold(sims, gamma=0.1)
            >>> print(f"Retrieve top {result.k} docs with threshold {result.threshold:.3f}")

            >>> # With empty result gate (for pure noise queries)
            >>> result = hc.compute_hc_threshold(
            ...     sims, gamma=0.1, min_hc=2.0, allow_empty=True
            ... )
            >>> if result.k == 0:
            ...     print("No signal detected, returning empty set")
        """
        # Handle empty input
        if len(similarities) == 0:
            logger.debug("compute_hc_threshold: empty input")
            return HCThresholdResult(threshold=np.inf, hc_statistic=0.0, k=0)

        # Sort similarities descending
        sorted_sims = np.sort(similarities)[::-1]
        n = len(sorted_sims)

        # Compute HC statistic and find best cutoff
        hc_stat, best_idx = self.compute_hc_statistic(sorted_sims, gamma=gamma)

        # Check HC gate: if below minimum, return empty set
        if allow_empty and hc_stat < min_hc:
            logger.debug(
                f"HC statistic {hc_stat:.3f} below min_hc={min_hc:.3f}, "
                f"returning empty set"
            )
            return HCThresholdResult(threshold=np.inf, hc_statistic=float(hc_stat), k=0)

        # Guard boundary conditions for index → k conversion
        # best_idx is 0-based, k = best_idx + 1 is the number of docs to retrieve
        k = max(0, min(n, best_idx + 1))

        # Ensure at least 1 document if allow_empty=False
        if not allow_empty and k == 0:
            k = 1
            logger.debug("allow_empty=False, forcing k=1")

        # Determine threshold based on k
        # Selection rule: top-k documents
        # Threshold is the score of the k-th document (informational)
        if k > 0:
            # Threshold is the similarity at position k-1 (0-indexed)
            threshold = float(sorted_sims[k - 1])
        else:
            # Empty set: threshold = +inf (no documents pass)
            threshold = np.inf

        return HCThresholdResult(threshold=threshold, hc_statistic=float(hc_stat), k=int(k))


def main():
    """Example usage of Higher Criticism for adaptive retrieval."""
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(name)s - %(message)s'
    )

    # Set random seed for reproducibility
    np.random.seed(42)

    print("\n" + "="*70)
    print("Higher Criticism - Adaptive Retrieval Demo")
    print("="*70 + "\n")

    # =========================================================================
    # Step 1: Generate Null Distribution
    # =========================================================================
    print("Step 1: Generating null distribution...")
    print("-" * 70)

    # Simulate null distribution (random pairs of normalized vectors)
    n_samples = 10000
    null_similarities = np.random.normal(0, 0.15, n_samples)  # Mean ~0, std ~0.15
    null_similarities = np.clip(null_similarities, -1, 1)  # Clip to valid range

    print(f"Generated {n_samples} random similarity scores")

    # Create null distribution
    null_dist = NullDistribution(
        similarities=null_similarities,
        mean=float(np.mean(null_similarities)),
        std=float(np.std(null_similarities)),
        min_val=float(np.min(null_similarities)),
        max_val=float(np.max(null_similarities)),
        n_samples=len(null_similarities)
    )

    hc = HigherCriticism(null_distribution=null_dist)
    print()

    # =========================================================================
    # Step 2: Simulate Query Retrieval
    # =========================================================================
    print("Step 2: Simulating query retrieval...")
    print("-" * 70)

    # Simulate retrieved documents with mixed relevant/irrelevant
    # Relevant docs: high similarity (signal)
    n_relevant = 5
    relevant_sims = np.random.uniform(0.6, 0.9, n_relevant)

    # Irrelevant docs: random similarity (noise from null distribution)
    n_irrelevant = 15
    irrelevant_sims = np.random.normal(0.0, 0.15, n_irrelevant)
    irrelevant_sims = np.clip(irrelevant_sims, -1, 1)

    # Combine and shuffle
    query_similarities = np.concatenate([relevant_sims, irrelevant_sims])
    np.random.shuffle(query_similarities)

    print(f"Retrieved {len(query_similarities)} documents:")
    print(f"  {n_relevant} relevant (high similarity)")
    print(f"  {n_irrelevant} irrelevant (noise)")
    print(f"  Similarity range: [{query_similarities.min():.3f}, {query_similarities.max():.3f}]")
    print()

    # =========================================================================
    # Step 3: Compute HC Threshold
    # =========================================================================
    print("Step 3: Computing HC threshold (gamma=0.1)...")
    print("-" * 70)

    result = hc.compute_hc_threshold(
        query_similarities,
        gamma=0.1
    )

    print(f"HC Statistic: {result.hc_statistic:.3f}")
    print(f"Threshold: {result.threshold:.3f}")
    print(f"Documents selected: {result.k} / {len(query_similarities)}")

    # Show which docs would be retrieved
    sorted_sims = np.sort(query_similarities)[::-1]
    selected_sims = sorted_sims[:result.k]

    print(f"\nSelected documents (top {result.k}):")
    for i, sim in enumerate(selected_sims, 1):
        is_relevant = sim >= 0.6  # Assume > 0.6 = truly relevant
        marker = "✓" if is_relevant else "✗"
        print(f"  {marker} Doc {i}: {sim:.3f}")

    print()

    # =========================================================================
    # Step 4: Compare Different Gamma Values
    # =========================================================================
    print("Step 4: Comparing different gamma values...")
    print("-" * 70)

    gamma_values = [0.07, 0.1, 0.4]

    print(f"{'Gamma':<10} {'HC Stat':<12} {'Threshold':<12} {'Docs Retrieved':<15}")
    print("-" * 70)

    for gamma in gamma_values:
        result = hc.compute_hc_threshold(
            query_similarities,
            gamma=gamma
        )
        print(f"{gamma:<10.2f} {result.hc_statistic:<12.3f} {result.threshold:<12.3f} {result.k:<15d}")

    print()

    # =========================================================================
    # Step 5: Save Null Distribution
    # =========================================================================
    print("Step 5: Saving null distribution...")
    print("-" * 70)

    save_path = "test_null_distribution"
    null_dist.save(save_path)

    # Test loading
    loaded_null_dist = NullDistribution.load(save_path)
    hc_loaded = HigherCriticism(null_distribution=loaded_null_dist)
    print(f"✓ Successfully saved and loaded null distribution")

    print()
    print("="*70)
    print("✓ Higher Criticism demo completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
