"""
Validate Null Distributions (HARD GATE)

Runs KS and Anderson-Darling tests to verify that p-values from null
distributions are Uniform(0,1). Produces diagnostic plots.

Tests two p-value methods:
  1. Empirical split-half (reference)
  2. Gaussian fit (fallback for small corpus)

PASS CRITERIA:
  - KS p-value > 0.05 for >= 90% of queries (either method)
  - Mean KS statistic < 0.1
"""

import sys
import numpy as np
from pathlib import Path
from scipy import stats
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hc.null_distribution import QueryNullDistributions


def multi_seed_ks_test(null_dist, method: str = "empirical", n_seeds: int = 10) -> tuple:
    """
    Run KS uniformity test across multiple random seeds and return
    the median KS statistic and geometric mean p-value.

    This averages out the randomness of the split-half procedure,
    giving a more robust assessment of null quality.
    """
    ks_stats = []
    ks_pvalues = []

    for seed in range(42, 42 + n_seeds):
        rng = np.random.RandomState(seed)
        sims = null_dist.similarities.copy()
        rng.shuffle(sims)

        n = len(sims)
        half = n // 2
        ref_half = sims[:half]
        test_half = sims[half:2 * half]

        if method == "gaussian":
            mu, sigma = np.mean(ref_half), np.std(ref_half)
            p_values = stats.norm.sf(test_half, loc=mu, scale=sigma)
            eps = 1e-10
            p_values = np.clip(p_values, eps, 1.0 - eps)
        else:
            sorted_ref = np.sort(ref_half)
            N = len(sorted_ref)
            indices = np.searchsorted(sorted_ref, test_half, side='left')
            n_geq = N - indices
            p_values = (n_geq + 1.0) / (N + 1.0)

        ks_stat, ks_pvalue = stats.kstest(p_values, 'uniform')
        ks_stats.append(ks_stat)
        ks_pvalues.append(ks_pvalue)

    median_stat = float(np.median(ks_stats))
    # Use median p-value (more robust than geometric mean)
    median_pvalue = float(np.median(ks_pvalues))

    return median_stat, median_pvalue


def validate_query(query_id: str, null_dist, verbose: bool = False) -> dict:
    """Validate a single query's null distribution using multi-seed tests."""
    n_seeds = 20

    # Method 1: Empirical split-half (multi-seed)
    emp_ks_stat, emp_ks_pvalue = multi_seed_ks_test(null_dist, "empirical", n_seeds)

    # Method 2: Gaussian fit (multi-seed)
    gauss_ks_stat, gauss_ks_pvalue = multi_seed_ks_test(null_dist, "gaussian", n_seeds)

    # Pass if either method passes
    emp_pass = emp_ks_pvalue > 0.05
    gauss_pass = gauss_ks_pvalue > 0.05
    best_method = "gaussian" if gauss_ks_pvalue > emp_ks_pvalue else "empirical"

    result = {
        "query_id": query_id,
        "n_samples": len(null_dist.similarities),
        "null_mean": float(null_dist.mean),
        "null_std": float(null_dist.std),
        "empirical_ks_stat": float(emp_ks_stat),
        "empirical_ks_pvalue": float(emp_ks_pvalue),
        "empirical_pass": bool(emp_pass),
        "gaussian_ks_stat": float(gauss_ks_stat),
        "gaussian_ks_pvalue": float(gauss_ks_pvalue),
        "gaussian_pass": bool(gauss_pass),
        "best_method": best_method,
        "pass": bool(emp_pass or gauss_pass),
        "best_ks_stat": float(min(emp_ks_stat, gauss_ks_stat)),
    }

    if verbose:
        status = "PASS" if result["pass"] else "FAIL"
        print(f"  {query_id}: emp_KS={emp_ks_stat:.4f}(p={emp_ks_pvalue:.3f}) "
              f"gauss_KS={gauss_ks_stat:.4f}(p={gauss_ks_pvalue:.3f}) "
              f"[{status}] best={best_method}")

    return result


def generate_plots(results: list, null_dists: QueryNullDistributions, output_dir: Path):
    """Generate diagnostic plots showing actual null distribution quality."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    plot_dir = output_dir / "validation"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: Raw null distribution diagnostics (3 sample queries) ----
    sorted_queries = sorted(null_dists.distributions.keys())
    # Pick one from each bucket size by choosing spread-out indices
    sample_indices = [0, len(sorted_queries) // 2, len(sorted_queries) - 1]
    sample_qids = [sorted_queries[i] for i in sample_indices]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for col, qid in enumerate(sample_qids):
        nd = null_dists.distributions[qid]
        sims = nd.similarities

        # Top row: Null similarity histogram + Gaussian overlay
        ax = axes[0, col]
        ax.hist(sims, bins=40, density=True, alpha=0.7, edgecolor='black',
                label=f'Empirical (n={len(sims)})')
        x_range = np.linspace(sims.min(), sims.max(), 200)
        gaussian_pdf = stats.norm.pdf(x_range, loc=nd.mean, scale=nd.std)
        ax.plot(x_range, gaussian_pdf, 'r-', linewidth=2,
                label=f'Gaussian fit\n(mu={nd.mean:.3f}, sigma={nd.std:.3f})')
        ax.set_xlabel('Cosine similarity')
        ax.set_ylabel('Density')
        ax.set_title(f'Null Distribution: {qid}')
        ax.legend(fontsize=7)

        # Bottom row: Split-half p-value histogram (should be uniform)
        rng = np.random.RandomState(42)
        sims_shuffled = sims.copy()
        rng.shuffle(sims_shuffled)
        half = len(sims_shuffled) // 2
        ref = sims_shuffled[:half]
        test = sims_shuffled[half:2 * half]
        sorted_ref = np.sort(ref)
        N_ref = len(sorted_ref)
        indices = np.searchsorted(sorted_ref, test, side='left')
        n_geq = N_ref - indices
        p_vals = (n_geq + 1.0) / (N_ref + 1.0)

        ax2 = axes[1, col]
        ax2.hist(p_vals, bins=10, range=(0, 1), density=True, alpha=0.7,
                 edgecolor='black', color='orange')
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5,
                    label='Uniform(0,1)')
        ax2.set_xlabel('p-value')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Split-half p-values: {qid}')
        ax2.set_ylim(0, 2.0)
        ax2.legend(fontsize=8)

    plt.suptitle('Null Distribution Diagnostics (3 sample queries)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(plot_dir / "null_distribution_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ---- Figure 2: Validation summary across all queries ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: All null means and stds
    means = [null_dists.distributions[qid].mean for qid in sorted_queries]
    stds = [null_dists.distributions[qid].std for qid in sorted_queries]
    x = np.arange(len(sorted_queries))
    axes[0].errorbar(x, means, yerr=stds, fmt='o', markersize=4, capsize=2, alpha=0.7)
    axes[0].set_xlabel('Query index')
    axes[0].set_ylabel('Similarity')
    axes[0].set_title('Per-query null: mean +/- std')
    axes[0].axhline(y=np.mean(means), color='red', linestyle='--', alpha=0.5,
                    label=f'Grand mean={np.mean(means):.3f}')
    axes[0].legend()

    # Plot 2: KS statistics (both methods)
    emp_stats = [r["empirical_ks_stat"] for r in results]
    gauss_stats = [r["gaussian_ks_stat"] for r in results]
    axes[1].scatter(emp_stats, gauss_stats, s=30, alpha=0.7)
    max_val = max(max(emp_stats), max(gauss_stats)) + 0.01
    axes[1].plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    axes[1].axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='threshold')
    axes[1].axvline(x=0.1, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Empirical KS statistic')
    axes[1].set_ylabel('Gaussian KS statistic')
    axes[1].set_title('KS Statistics: Empirical vs Gaussian')
    axes[1].legend()

    # Plot 3: Pass/fail summary bar chart
    n_total = len(results)
    n_emp = sum(1 for r in results if r["empirical_pass"])
    n_gauss = sum(1 for r in results if r["gaussian_pass"])
    n_either = sum(1 for r in results if r["pass"])
    categories = ['Empirical', 'Gaussian', 'Either']
    pass_counts = [n_emp, n_gauss, n_either]
    fail_counts = [n_total - n_emp, n_total - n_gauss, n_total - n_either]
    bar_x = np.arange(len(categories))
    axes[2].bar(bar_x, pass_counts, 0.6, label='Pass', color='green', alpha=0.7)
    axes[2].bar(bar_x, fail_counts, 0.6, bottom=pass_counts, label='Fail', color='red', alpha=0.7)
    axes[2].axhline(y=0.9 * n_total, color='orange', linestyle='--',
                    label=f'90% threshold ({int(0.9*n_total)})')
    axes[2].set_xticks(bar_x)
    axes[2].set_xticklabels(categories)
    axes[2].set_ylabel('Number of queries')
    axes[2].set_title('Validation Pass/Fail (alpha=0.05)')
    axes[2].legend(fontsize=8)
    for i, (p, f) in enumerate(zip(pass_counts, fail_counts)):
        axes[2].text(i, p + f + 0.3, f'{p}/{n_total}', ha='center', fontsize=10)

    plt.suptitle('Null Validation Summary', fontsize=14)
    plt.tight_layout()
    plt.savefig(plot_dir / "null_validation.png", dpi=150)
    plt.close()

    print(f"  Diagnostics saved to {plot_dir / 'null_distribution_diagnostics.png'}")
    print(f"  Summary saved to {plot_dir / 'null_validation.png'}")


def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    null_path = str(output_dir / "null_distributions" / "category_key_per_query_null")

    print("=" * 70)
    print("Null Distribution Validation (HARD GATE)")
    print("=" * 70)

    # Load null distributions
    print("\nLoading null distributions...")
    null_dists = QueryNullDistributions.load(null_path)
    print(f"  Loaded {len(null_dists.distributions)} distributions")

    # Validate each query
    print("\nValidating per-query null distributions...")
    results = []
    for query_id, null_dist in sorted(null_dists.distributions.items()):
        result = validate_query(query_id, null_dist, verbose=True)
        results.append(result)

    # Aggregate results
    n_total = len(results)
    n_emp_pass = sum(1 for r in results if r["empirical_pass"])
    n_gauss_pass = sum(1 for r in results if r["gaussian_pass"])
    n_either_pass = sum(1 for r in results if r["pass"])
    best_ks_stats = [r["best_ks_stat"] for r in results]
    mean_best_ks = np.mean(best_ks_stats)

    emp_rate = n_emp_pass / n_total
    gauss_rate = n_gauss_pass / n_total
    either_rate = n_either_pass / n_total

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"  Total queries: {n_total}")
    print(f"  Empirical split-half pass rate: {n_emp_pass}/{n_total} ({emp_rate:.1%})")
    print(f"  Gaussian fit pass rate: {n_gauss_pass}/{n_total} ({gauss_rate:.1%})")
    print(f"  Either method pass rate: {n_either_pass}/{n_total} ({either_rate:.1%})")
    print(f"  Mean best KS statistic: {mean_best_ks:.4f}")

    # Determine which method to use for HC
    if gauss_rate >= emp_rate:
        recommended_method = "gaussian"
        primary_rate = gauss_rate
    else:
        recommended_method = "empirical"
        primary_rate = emp_rate
    print(f"\n  Recommended p-value method: {recommended_method}")

    # GATE CHECK: pass if either method achieves >= 90% AND mean KS < 0.1
    print("\n" + "-" * 70)
    gate_pass = either_rate >= 0.90 and mean_best_ks < 0.1

    if gate_pass:
        print("  >>> GATE: PASS <<<")
        print(f"  Either-method pass rate {either_rate:.1%} >= 90%: OK")
        print(f"  Mean best KS stat {mean_best_ks:.4f} < 0.1: OK")
    else:
        print("  >>> GATE: FAIL <<<")
        if either_rate < 0.90:
            print(f"  Either-method pass rate {either_rate:.1%} < 90%: FAIL")
        if mean_best_ks >= 0.1:
            print(f"  Mean best KS stat {mean_best_ks:.4f} >= 0.1: FAIL")
        print("\n  Suggested fallbacks:")
        print("  1. Increase negatives_per_query further")
        print("  2. Try KDE-based null")
        print("  3. Pool all negatives across queries into single null")

    print("-" * 70)

    # Generate plots
    print("\nGenerating diagnostic plots...")
    generate_plots(results, null_dists, output_dir)

    # Save validation report
    report = {
        "gate_pass": bool(gate_pass),
        "recommended_method": recommended_method,
        "n_queries": n_total,
        "empirical_pass_rate": float(emp_rate),
        "gaussian_pass_rate": float(gauss_rate),
        "either_pass_rate": float(either_rate),
        "mean_best_ks_statistic": float(mean_best_ks),
        "per_query_results": results
    }
    report_path = output_dir / "validation" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to {report_path}")

    # Return exit code based on gate
    if not gate_pass:
        print("\nWARNING: Null validation FAILED. Do not proceed with experiments.")
        sys.exit(1)

    print("\nNull validation PASSED. Safe to proceed with experiments.")


if __name__ == "__main__":
    main()
