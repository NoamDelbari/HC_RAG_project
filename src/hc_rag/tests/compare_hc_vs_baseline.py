"""
Compare HC-based Adaptive Retrieval vs Baseline Top-k

Loads results from both baseline and HC experiments and creates detailed comparisons.

Usage:
    python src/tests/compare_hc_vs_baseline.py
"""

import json
from pathlib import Path
import numpy as np
from typing import Dict, List


def load_baseline_results(baseline_dir: Path) -> Dict:
    """Load baseline experiment results."""
    comparison_file = baseline_dir / "baseline_comparison.json"

    if not comparison_file.exists():
        raise FileNotFoundError(f"Baseline comparison file not found: {comparison_file}")

    with open(comparison_file, "r") as f:
        return json.load(f)


def load_hc_results(hc_dir: Path) -> Dict:
    """Load HC experiment results."""
    summary_file = hc_dir / "hc_comparison_summary.json"

    if not summary_file.exists():
        raise FileNotFoundError(f"HC summary file not found: {summary_file}")

    with open(summary_file, "r") as f:
        return json.load(f)


def print_comparison_table(baseline_data: Dict, hc_data: Dict):
    """Print detailed comparison table."""

    print("\n" + "="*100)
    print("RETRIEVAL METHOD COMPARISON (Labeled Queries Only)")
    print("="*100)

    print(f"\n{'Method':<20} {'Config':<20} {'Avg k':<10} {'Recall@k':<12} {'Precision@k':<15} {'MRR':<10} {'NDCG@k':<10}")
    print("-" * 100)

    # Baseline results
    baseline_metrics = baseline_data["metrics"]
    for k_str, metrics in baseline_metrics.items():
        k = int(k_str)
        print(
            f"{'Baseline Top-k':<20} "
            f"{'k=' + str(k):<20} "
            f"{k:<10} "
            f"{metrics['recall_at_k']:<12.3f} "
            f"{metrics['precision_at_k']:<15.3f} "
            f"{metrics['mrr']:<10.3f} "
            f"{metrics['ndcg_at_k']:<10.3f}"
        )

    print("-" * 100)

    # HC results
    for config in hc_data["configurations"]:
        gamma = config["gamma"]
        min_hc = config["min_hc"]
        avg_k = config["retrieval_stats"]["mean_k"]

        config_str = f"γ={gamma}"
        if min_hc > 0:
            config_str += f", HC≥{min_hc}"

        print(
            f"{'HC Adaptive':<20} "
            f"{config_str:<20} "
            f"{avg_k:<10.2f} "
            f"{config['metrics']['recall_at_k']:<12.3f} "
            f"{config['metrics']['precision_at_k']:<15.3f} "
            f"{config['metrics']['mrr']:<10.3f} "
            f"{config['metrics']['ndcg_at_k']:<10.3f}"
        )

    print("="*100)


def find_best_configurations(baseline_data: Dict, hc_data: Dict):
    """Find and print best configurations for each metric."""

    print("\n" + "="*100)
    print("BEST CONFIGURATIONS BY METRIC")
    print("="*100)

    metrics_to_compare = ["recall_at_k", "precision_at_k", "mrr", "ndcg_at_k"]

    for metric_name in metrics_to_compare:
        print(f"\n{metric_name.upper().replace('_', ' ')}:")
        print("-" * 100)

        # Find best baseline
        best_baseline_k = None
        best_baseline_value = -1
        for k_str, metrics in baseline_data["metrics"].items():
            if metrics[metric_name] > best_baseline_value:
                best_baseline_value = metrics[metric_name]
                best_baseline_k = int(k_str)

        print(f"  Best Baseline: k={best_baseline_k:2d}, {metric_name}={best_baseline_value:.4f}")

        # Find best HC
        best_hc_config = None
        best_hc_value = -1
        for config in hc_data["configurations"]:
            if config["metrics"][metric_name] > best_hc_value:
                best_hc_value = config["metrics"][metric_name]
                best_hc_config = config

        gamma = best_hc_config["gamma"]
        min_hc = best_hc_config["min_hc"]
        avg_k = best_hc_config["retrieval_stats"]["mean_k"]

        print(f"  Best HC:       γ={gamma:.2f}, min_hc={min_hc:.1f}, avg_k={avg_k:.2f}, {metric_name}={best_hc_value:.4f}")

        # Compare
        improvement = ((best_hc_value - best_baseline_value) / best_baseline_value) * 100
        if improvement > 0:
            print(f"  → HC improves by {improvement:+.2f}%")
        else:
            print(f"  → Baseline better by {-improvement:.2f}%")

    print("="*100)


def analyze_hc_adaptivity(hc_data: Dict):
    """Analyze HC adaptivity characteristics."""

    print("\n" + "="*100)
    print("HC ADAPTIVITY ANALYSIS")
    print("="*100)

    for config in hc_data["configurations"]:
        gamma = config["gamma"]
        min_hc = config["min_hc"]
        stats = config["retrieval_stats"]

        print(f"\nγ={gamma:.2f}, min_hc={min_hc:.1f}:")
        print(f"  Mean k:   {stats['mean_k']:.2f}")

        if 'median_k' in stats:
            print(f"  Median k: {stats['median_k']:.0f}")

        if 'min_k' in stats:
            print(f"  Min k:    {stats['min_k']}")

        if 'max_k' in stats:
            print(f"  Max k:    {stats['max_k']}")

        if 'std_k' in stats:
            print(f"  Std k:    {stats['std_k']:.2f}")

        if 'n_empty' in stats:
            n_empty = stats['n_empty']
            n_queries = hc_data['n_queries']
            print(f"  Empty results: {n_empty} / {n_queries} ({100*n_empty/n_queries:.1f}%)")

        if 'empty_rate' in stats:
            print(f"  Empty rate: {stats['empty_rate']*100:.1f}%")

    print("="*100)


def create_recommendations(baseline_data: Dict, hc_data: Dict):
    """Create recommendations based on analysis."""

    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100)

    # Find best overall configurations
    best_baseline_ndcg = max(
        [(int(k), m["ndcg_at_k"]) for k, m in baseline_data["metrics"].items()],
        key=lambda x: x[1]
    )

    best_hc_ndcg = max(
        [
            (c, c["metrics"]["ndcg_at_k"])
            for c in hc_data["configurations"]
        ],
        key=lambda x: x[1]
    )

    print(f"\n1. For Maximum Retrieval Quality (NDCG@k):")
    print(f"   Baseline: k={best_baseline_ndcg[0]}, NDCG={best_baseline_ndcg[1]:.4f}")
    print(f"   HC: γ={best_hc_ndcg[0]['gamma']:.2f}, "
          f"avg_k={best_hc_ndcg[0]['retrieval_stats']['mean_k']:.2f}, "
          f"NDCG={best_hc_ndcg[1]:.4f}")

    # Precision-focused recommendation
    best_baseline_precision = max(
        [(int(k), m["precision_at_k"]) for k, m in baseline_data["metrics"].items()],
        key=lambda x: x[1]
    )

    best_hc_precision = max(
        [
            (c, c["metrics"]["precision_at_k"])
            for c in hc_data["configurations"]
        ],
        key=lambda x: x[1]
    )

    print(f"\n2. For Maximum Precision:")
    print(f"   Baseline: k={best_baseline_precision[0]}, Precision={best_baseline_precision[1]:.4f}")
    print(f"   HC: γ={best_hc_precision[0]['gamma']:.2f}, "
          f"avg_k={best_hc_precision[0]['retrieval_stats']['mean_k']:.2f}, "
          f"Precision={best_hc_precision[1]:.4f}")

    # Recall-focused recommendation
    best_baseline_recall = max(
        [(int(k), m["recall_at_k"]) for k, m in baseline_data["metrics"].items()],
        key=lambda x: x[1]
    )

    best_hc_recall = max(
        [
            (c, c["metrics"]["recall_at_k"])
            for c in hc_data["configurations"]
        ],
        key=lambda x: x[1]
    )

    print(f"\n3. For Maximum Recall:")
    print(f"   Baseline: k={best_baseline_recall[0]}, Recall={best_baseline_recall[1]:.4f}")
    print(f"   HC: γ={best_hc_recall[0]['gamma']:.2f}, "
          f"avg_k={best_hc_recall[0]['retrieval_stats']['mean_k']:.2f}, "
          f"Recall={best_hc_recall[1]:.4f}")

    print("\n4. General Recommendation:")
    print("   • Use HC with γ=0.1 for balanced precision/recall")
    print("   • Use smaller γ (0.07) for higher precision, lower avg k")
    print("   • Use larger γ (0.4) for higher recall, higher avg k")
    print("   • Set min_hc > 0 to filter pure noise queries (returns empty set)")

    print("="*100)


def main():
    print("\n" + "="*100)
    print("HC-RAG vs BASELINE COMPARISON ANALYSIS")
    print("="*100)

    # Paths
    baseline_dir = Path("results/baseline_experiments")
    hc_dir = Path("results/hc_experiments")

    # Load results
    print("\nLoading results...")
    try:
        baseline_data = load_baseline_results(baseline_dir)
        print(f"✓ Loaded baseline results: {baseline_data['n_labeled_queries']} queries")
    except FileNotFoundError as e:
        print(f"✗ Error loading baseline results: {e}")
        print("\n  Run baseline experiments first:")
        print("    python src/tests/run_baseline_experiments.py")
        return

    try:
        hc_data = load_hc_results(hc_dir)
        print(f"✓ Loaded HC results: {hc_data['n_labeled_queries']} queries, "
              f"{len(hc_data['configurations'])} configurations")
    except FileNotFoundError as e:
        print(f"✗ Error loading HC results: {e}")
        print("\n  Run HC experiments first:")
        print("    python src/tests/run_hc_experiments.py")
        return

    # Verify same dataset
    if baseline_data["n_labeled_queries"] != hc_data["n_labeled_queries"]:
        print(f"\n⚠ Warning: Different number of queries!")
        print(f"  Baseline: {baseline_data['n_labeled_queries']}")
        print(f"  HC: {hc_data['n_labeled_queries']}")

    # Comparison table
    print_comparison_table(baseline_data, hc_data)

    # Best configurations
    find_best_configurations(baseline_data, hc_data)

    # HC adaptivity analysis
    analyze_hc_adaptivity(hc_data)

    # Recommendations
    create_recommendations(baseline_data, hc_data)

    # Save detailed comparison
    output_file = Path("results/hc_vs_baseline_comparison.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    comparison_data = {
        "baseline": baseline_data,
        "hc": hc_data,
        "summary": {
            "dataset": {
                "n_queries": baseline_data["n_queries"],
                "n_labeled_queries": baseline_data["n_labeled_queries"]
            },
            "best_configurations": {
                "baseline_best_ndcg": {
                    "k": max(
                        [(int(k), m["ndcg_at_k"]) for k, m in baseline_data["metrics"].items()],
                        key=lambda x: x[1]
                    )[0],
                    "ndcg": max(
                        [(int(k), m["ndcg_at_k"]) for k, m in baseline_data["metrics"].items()],
                        key=lambda x: x[1]
                    )[1]
                },
                "hc_best_ndcg": {
                    "gamma": max(
                        hc_data["configurations"],
                        key=lambda c: c["metrics"]["ndcg_at_k"]
                    )["gamma"],
                    "ndcg": max(
                        hc_data["configurations"],
                        key=lambda c: c["metrics"]["ndcg_at_k"]
                    )["metrics"]["ndcg_at_k"]
                }
            }
        }
    }

    with open(output_file, "w") as f:
        json.dump(comparison_data, f, indent=2)

    print(f"\n✓ Saved detailed comparison to: {output_file}")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
