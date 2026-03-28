"""
Higher Criticism (HC) Module

Implements Higher Criticism statistics for adaptive document thresholding in RAG.
Based on the HC-RAG paper's approach to statistical significance testing.
"""

import numpy as np
from scipy.stats import norm
from typing import Optional, Tuple, NamedTuple
import logging

from hc_rag.hc.null_distribution import NullDistribution

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
    """

    def __init__(self, null_distribution: Optional[NullDistribution] = None):
        """
        Initialize Higher Criticism module.

        Args:
            null_distribution: Pre-computed null distribution (optional)
        """
        self.null_distribution = null_distribution
        logger.info("HigherCriticism module initialized")

        if null_distribution is not None:
            logger.info(f"  Loaded null distribution: {null_distribution}")

    # =========================================================================
    # Higher Criticism Statistic Calculation
    # =========================================================================

    def _compute_p_values_empirical(self, values: np.ndarray) -> np.ndarray:
        """
        Compute p-values for values against the null distribution using empirical CDF.

        P-value = P(null value >= observed value)

        Uses vectorized searchsorted for O(n log N) performance instead of O(n*N).
        Applies proper Monte Carlo calibration: smallest p-value is ~1/(N+1).

        Args:
            values: Array of observed values (raw similarities or z-scores)

        Returns:
            Array of p-values (same shape as values)
        """
        if self.null_distribution is None:
            raise ValueError("No null distribution available. Call fit_null_distribution() first.")

        null_sims = self.null_distribution.similarities
        N_null = len(null_sims)

        # Sort null distribution once for binary search
        sorted_null = np.sort(null_sims)

        # Vectorized p-value computation using searchsorted
        # For each observed value, find how many null samples are >= value
        indices = np.searchsorted(sorted_null, values, side='left')
        n_greater_equal = N_null - indices

        # Monte Carlo calibration: p = (n_greater_equal + 1) / (N_null + 1)
        # +1 smoothing ensures p-values are in (0, 1] and respects finite sample resolution
        p_values = (n_greater_equal + 1.0) / (N_null + 1.0)

        # Clip to principled bounds to prevent 0/1 in HC denominator
        # Smallest meaningful p-value is ~1/(N_null+1)
        eps = 1.0 / (N_null + 1.0)
        p_values = np.clip(p_values, eps, 1.0 - eps)

        return p_values

    def _compute_p_values_parametric(self, z_scores: np.ndarray) -> np.ndarray:
        """
        Compute p-values from z-scores using the Gaussian CDF.

        p_i = 1 - Phi(z_i) where Phi is the standard normal CDF.
        Eliminates the empirical CDF resolution floor, allowing much smaller
        p-values for strong signals.

        Args:
            z_scores: Array of z-score standardized values

        Returns:
            Array of p-values (same shape as z_scores)
        """
        p_values = norm.sf(z_scores)  # sf = 1 - cdf, more numerically stable
        # Clip to prevent exact 0 or 1
        eps = 1e-15  # much smaller floor than empirical method
        p_values = np.clip(p_values, eps, 1.0 - eps)
        return p_values

    def _compute_p_values_gpd_composite(self, values: np.ndarray) -> np.ndarray:
        """
        Compute p-values using composite empirical + GPD tail extrapolation.

        For z-scores within the null's observed range: use empirical CDF.
        For z-scores beyond the GPD threshold: use GPD survival function.

        This eliminates the empirical CDF floor problem while preserving
        the empirical distribution's accuracy in the body of the null.

        Requires null_distribution to have GPD parameters fitted
        (gpd_shape, gpd_scale, gpd_threshold, gpd_rate_above).
        """
        if self.null_distribution is None:
            raise ValueError("No null distribution available.")

        nd = self.null_distribution
        if nd.gpd_shape is None or nd.gpd_scale is None:
            raise ValueError(
                "NullDistribution does not have GPD parameters. "
                "Rebuild the null with GPD fitting enabled."
            )

        # Empirical CDF for all values first
        null_sims = nd.similarities
        N_null = len(null_sims)
        sorted_null = np.sort(null_sims)

        indices = np.searchsorted(sorted_null, values, side='left')
        n_greater_equal = N_null - indices
        p_empirical = (n_greater_equal + 1.0) / (N_null + 1.0)

        # For values above GPD threshold, use GPD survival function
        from scipy.stats import genpareto
        above_mask = values > nd.gpd_threshold
        if np.any(above_mask):
            exceedances = values[above_mask] - nd.gpd_threshold
            # GPD survival: P(X > x) = (1 + shape * x / scale) ^ (-1/shape)
            gpd_survival = genpareto.sf(exceedances, nd.gpd_shape, loc=0, scale=nd.gpd_scale)
            # Scale by the rate of exceeding the threshold in the null
            p_gpd = nd.gpd_rate_above * gpd_survival
            p_empirical[above_mask] = p_gpd

        # Clip to prevent 0 or 1
        # Use 1e-15 floor (not 1/(N+1)) since GPD can legitimately produce very small p-values
        eps = 1e-15
        p_values = np.clip(p_empirical, eps, 1.0 - eps)

        return p_values

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
        return self._compute_p_values_empirical(similarities)

    def compute_p_values_zscore(
        self, similarities: np.ndarray, fraction_for_null: float = 0.8,
        parametric: bool = False, z_cap: Optional[float] = None,
        gpd_composite: bool = False
    ) -> np.ndarray:
        """
        Compute p-values using z-score standardization against a z-score null.

        Flow:
        1. Estimate per-query null mean/std from the bottom portion of candidates
        2. Z-score standardize all candidate similarities
        3. Optionally clip z-scores at z_cap (controlled floor effect)
        4. Compute p-values of z-scores against the z-score global null (empirical or parametric)

        Args:
            similarities: Array of observed similarity scores (raw cosine)
            fraction_for_null: Fraction of bottom scores used to estimate null params
            parametric: If True, use Gaussian CDF instead of empirical CDF for p-values.
                       Eliminates the p-value floor saturation problem.
            z_cap: If not None, clip z-scores at this upper bound before computing p-values.
                   Creates a controlled floor effect that prevents extreme z-scores from
                   dominating the HC statistic.

        Returns:
            Array of p-values (same shape as similarities)
        """
        # Sort descending to identify bottom portion
        sorted_desc = np.sort(similarities)[::-1]
        n = len(sorted_desc)
        n_null_est = max(int(n * fraction_for_null), 10)

        # Use bottom fraction to estimate null mean/std
        bottom = sorted_desc[-n_null_est:]  # lowest similarities
        mu_est = np.mean(bottom)
        sigma_est = np.std(bottom)
        if sigma_est < 1e-10:
            sigma_est = 1e-10  # prevent division by zero

        # Z-score standardize ALL similarities
        z_scores = (similarities - mu_est) / sigma_est

        # Apply z-score capping if specified
        if z_cap is not None:
            z_scores = np.clip(z_scores, None, z_cap)

        # Compute p-values against z-null
        if gpd_composite:
            return self._compute_p_values_gpd_composite(z_scores)
        elif parametric:
            return self._compute_p_values_parametric(z_scores)
        else:
            return self._compute_p_values_empirical(z_scores)

    def _compute_hc_values(self, p_values: np.ndarray, n_gamma: int, n: int) -> np.ndarray:
        """
        Compute Higher Criticism statistic values for each position.

        Formula: HC_i = (i/n - p_i) / sqrt(p_i * (1 - p_i) / n)

        Args:
            p_values: Sorted p-values (smallest first, corresponding to descending similarities)
            n_gamma: Number of positions to evaluate
            n: Total number of candidates

        Returns:
            Array of HC values for positions 1..n_gamma
        """
        hc_values = np.zeros(n_gamma)
        for i in range(1, n_gamma + 1):
            expected_quantile = i / n
            p_i = p_values[i - 1]
            denominator = np.sqrt(p_i * (1 - p_i) / n)
            if denominator > 0:
                hc_values[i - 1] = (expected_quantile - p_i) / denominator
        return hc_values

    def _compute_berk_jones_values(self, p_values: np.ndarray, n_gamma: int, n: int) -> np.ndarray:
        """
        Compute Berk-Jones statistic values for each position.

        Formula: BJ_i = n * KL(i/n || p_i)
        where KL(a||b) = a*log(a/b) + (1-a)*log((1-a)/(1-b))

        BJ avoids the p-value floor pathology of HC because it grows
        logarithmically at the floor instead of linearly, allowing it to
        detect signal beyond the null max boundary.

        Args:
            p_values: Sorted p-values (smallest first, corresponding to descending similarities)
            n_gamma: Number of positions to evaluate
            n: Total number of candidates

        Returns:
            Array of BJ values for positions 1..n_gamma
        """
        bj_values = np.zeros(n_gamma)
        for i in range(1, n_gamma + 1):
            a = np.clip(i / n, 1e-15, 1.0 - 1e-15)
            b = np.clip(p_values[i - 1], 1e-15, 1.0 - 1e-15)
            kl = a * np.log(a / b) + (1.0 - a) * np.log((1.0 - a) / (1.0 - b))
            bj_values[i - 1] = n * kl
        return bj_values

    def compute_hc_statistic(
        self,
        similarities: np.ndarray,
        gamma: float = 0.1,
        use_zscore: bool = False,
        parametric: bool = False,
        z_cap: Optional[float] = None,
        method: str = "hc",
        gpd_composite: bool = False
    ) -> Tuple[float, int]:
        """
        Compute Higher Criticism or Berk-Jones statistic.

        HC statistic measures the maximum deviation between observed p-values
        and uniform distribution, focusing on small p-values (likely relevant docs).

        Maximizes the chosen statistic over the first floor(gamma*n) order statistics
        and returns the argmax index (not a quantile threshold).

        Formulas:
            HC:  HC_i = sqrt(n) * (i/n - p_(i)) / sqrt(p_(i) * (1 - p_(i)))
            BJ:  BJ_i = n * KL(i/n || p_i)  where KL is the KL divergence

        Args:
            similarities: Array of similarity scores (will be sorted descending)
            gamma: Fraction of top scores to search for maximum (0 < gamma <= 1)
            use_zscore: If True, use z-score standardized p-values
            parametric: If True and use_zscore=True, use Gaussian CDF instead of
                       empirical CDF for p-value computation. Ignored when use_zscore=False.
            z_cap: If not None and use_zscore=True, clip z-scores at this upper bound.
            method: Statistical method to use. "hc" for Higher Criticism (default),
                    "berk_jones" for Berk-Jones statistic.

        Returns:
            Tuple of (statistic_value, best_index)
            - statistic_value: Maximum statistic value across the search window
            - best_index: Index (0-based) where maximum occurs

        Raises:
            ValueError: If gamma is not in (0, 1] or method is invalid
        """
        # Validate inputs
        if not (0 < gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if method not in ("hc", "berk_jones"):
            raise ValueError(f"method must be 'hc' or 'berk_jones', got '{method}'")

        # Handle empty input
        if len(similarities) == 0:
            logger.debug("compute_hc_statistic: empty input, returning (0.0, 0)")
            return 0.0, 0

        # Sort similarities in descending order (highest first)
        sorted_sims = np.sort(similarities)[::-1]

        # Compute p-values for sorted similarities
        if use_zscore:
            p_values = self.compute_p_values_zscore(
                sorted_sims, parametric=parametric, z_cap=z_cap, gpd_composite=gpd_composite
            )
        else:
            p_values = self.compute_p_values(sorted_sims)

        # Number of documents
        n = len(p_values)

        # Only consider top gamma fraction of smallest p-values
        # Ensure window is in valid range [1, n]
        n_gamma = min(n, max(1, int(np.floor(gamma * n))))

        if n_gamma == 1 and gamma * n < 1:
            logger.debug(f"gamma={gamma} with n={n} yields search window < 1, using n_gamma=1")

        # Compute statistic values using the selected method
        if method == "berk_jones":
            stat_values = self._compute_berk_jones_values(p_values, n_gamma, n)
        else:
            stat_values = self._compute_hc_values(p_values, n_gamma, n)

        # Find maximum statistic value
        if len(stat_values) > 0:
            max_stat = float(np.max(stat_values))
            best_idx = int(np.argmax(stat_values))
        else:
            max_stat = 0.0
            best_idx = 0

        return max_stat, best_idx

    def compute_hc_threshold(
        self,
        similarities: np.ndarray,
        gamma: float = 0.1,
        min_hc: float = 0.0,
        allow_empty: bool = True,
        use_zscore: bool = False,
        parametric: bool = False,
        z_cap: Optional[float] = None,
        method: str = "hc",
        gpd_composite: bool = False
    ) -> HCThresholdResult:
        """
        Compute statistic-based threshold for adaptive retrieval.

        The threshold is determined by the position where the chosen statistic
        (HC or Berk-Jones) is maximized. Selection rule: retrieve the top-k
        documents where k is determined by the statistic.
        The threshold is the score of the k-th document (informational).

        Can return empty set (k=0, threshold=+inf) if the statistic is below min_hc.

        Args:
            similarities: Array of similarity scores from retrieval
            gamma: Fraction of top scores to search for maximum (0 < gamma <= 1)
            min_hc: Minimum statistic value to accept any documents. If stat < min_hc,
                    returns empty set (threshold=+inf, k=0). Set to 0.0 to disable.
            allow_empty: If True, can return empty set when stat < min_hc.
                        If False, always returns at least 1 document.
            use_zscore: If True, use z-score standardized p-values for computation.
            parametric: If True and use_zscore=True, use Gaussian CDF instead of
                       empirical CDF for p-value computation. Ignored when use_zscore=False.
            z_cap: If not None and use_zscore=True, clip z-scores at this upper bound.
            method: Statistical method to use. "hc" for Higher Criticism (default),
                    "berk_jones" for Berk-Jones statistic.

        Returns:
            HCThresholdResult with:
                - threshold: Score of k-th document (or +inf for empty set)
                - hc_statistic: Maximum statistic value (HC or BJ)
                - k: Number of documents to retrieve (0 for empty set)

        Examples:
            >>> # Standard usage with HC
            >>> result = hc.compute_hc_threshold(sims, gamma=0.1)
            >>> print(f"Retrieve top {result.k} docs with threshold {result.threshold:.3f}")

            >>> # Using Berk-Jones statistic
            >>> result = hc.compute_hc_threshold(sims, gamma=0.1, method="berk_jones")
            >>> print(f"BJ selected {result.k} docs")

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

        # Compute statistic and find best cutoff
        hc_stat, best_idx = self.compute_hc_statistic(
            sorted_sims, gamma=gamma, use_zscore=use_zscore,
            parametric=parametric, z_cap=z_cap, method=method,
            gpd_composite=gpd_composite
        )

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
