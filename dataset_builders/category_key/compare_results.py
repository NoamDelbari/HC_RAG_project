"""
Compare HC vs Baseline Results on Category-Key Dataset

The "money analysis" for the advisor:
- Per-query K analysis: true_K vs baseline_K vs HC_K
- Recall grouped by K bucket (small/medium/large)
- Scatter: true_K vs recall for baseline vs HC
- False negative analysis
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_results(results_dir: Path) -> dict:
    """Load baseline and HC results."""
    baseline_path = results_dir / "baseline" / "baseline_results.json"
    hc_path = results_dir / "hc" / "hc_results.json"

    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(hc_path) as f:
        hc = json.load(f)

    return baseline, hc


def bucket_analysis(per_query: list) -> dict:
    """Group results by size bucket and compute per-bucket metrics."""
    buckets = defaultdict(list)
    for r in per_query:
        buckets[r["size_bucket"]].append(r)

    bucket_stats = {}
    for bucket, results in sorted(buckets.items()):
        recalls = [r["recall"] for r in results]
        precisions = [r["precision"] for r in results]
        true_ks = [r["true_k"] for r in results]
        bucket_stats[bucket] = {
            "n_queries": len(results),
            "mean_recall": np.mean(recalls),
            "mean_precision": np.mean(precisions),
            "mean_true_k": np.mean(true_ks),
            "k_range": f"{min(true_ks)}-{max(true_ks)}"
        }

    return bucket_stats


def false_negative_analysis(baseline_pq: list, hc_pq: list, qrels_from_baseline: dict) -> dict:
    """Analyze docs missed by baseline but found by HC."""
    # Build lookup by query_id
    baseline_by_q = {r["query_id"]: set(r["retrieved_ids"]) for r in baseline_pq}
    hc_by_q = {r["query_id"]: set(r["retrieved_ids"]) for r in hc_pq}

    total_fn_baseline = 0
    total_fn_hc = 0
    hc_recovered = 0

    for r in baseline_pq:
        qid = r["query_id"]
        true_k = r["true_k"]
        # We don't have the actual relevant IDs in per_query export,
        # but we can compare recall: if recall < 1, there are false negatives
        baseline_retrieved = baseline_by_q.get(qid, set())
        hc_retrieved = hc_by_q.get(qid, set())

        # Count unique docs HC found that baseline missed
        hc_extra = hc_retrieved - baseline_retrieved
        baseline_extra = baseline_retrieved - hc_retrieved

        # We use recall to approximate
        baseline_recall = r["recall"]
        baseline_fn = int(round(true_k * (1 - baseline_recall)))
        total_fn_baseline += baseline_fn

    for r in hc_pq:
        hc_recall = r["recall"]
        true_k = r["true_k"]
        hc_fn = int(round(true_k * (1 - hc_recall)))
        total_fn_hc += hc_fn

    return {
        "total_false_negatives_baseline": total_fn_baseline,
        "total_false_negatives_hc": total_fn_hc,
        "fn_reduction": total_fn_baseline - total_fn_hc,
        "fn_reduction_pct": ((total_fn_baseline - total_fn_hc) / max(total_fn_baseline, 1)) * 100
    }


def generate_plots(baseline: dict, hc: dict, output_dir: Path):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    plot_dir = output_dir / "comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Find best baseline k and best HC gamma
    best_baseline_k = max(baseline.keys(), key=lambda k: baseline[k]["aggregate"]["recall"])
    best_hc_gamma = max(hc.keys(), key=lambda g: hc[g]["aggregate"]["recall"])

    bl_pq = baseline[best_baseline_k]["per_query"]
    hc_pq = hc[best_hc_gamma]["per_query"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Recall by size bucket
    bl_buckets = bucket_analysis(bl_pq)
    hc_buckets = bucket_analysis(hc_pq)

    bucket_names = sorted(set(list(bl_buckets.keys()) + list(hc_buckets.keys())),
                         key=lambda x: {"small": 0, "medium": 1, "large": 2}.get(x, 3))
    x = np.arange(len(bucket_names))
    width = 0.35

    bl_recalls = [bl_buckets.get(b, {}).get("mean_recall", 0) for b in bucket_names]
    hc_recalls = [hc_buckets.get(b, {}).get("mean_recall", 0) for b in bucket_names]

    axes[0, 0].bar(x - width/2, bl_recalls, width, label=f'Baseline k={best_baseline_k}', color='steelblue')
    axes[0, 0].bar(x + width/2, hc_recalls, width, label=f'HC gamma={best_hc_gamma}', color='coral')
    axes[0, 0].set_xlabel('Size Bucket')
    axes[0, 0].set_ylabel('Mean Recall')
    axes[0, 0].set_title('Recall by Country Size Bucket')
    axes[0, 0].set_xticks(x)
    xticklabels = []
    for b in bucket_names:
        k_range = bl_buckets.get(b, {}).get("k_range", "?")
        xticklabels.append(f"{b}\n(K={k_range})")
    axes[0, 0].set_xticklabels(xticklabels)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1.1)

    # Plot 2: Scatter - true_K vs recall
    bl_true_k = [r["true_k"] for r in bl_pq]
    bl_recall = [r["recall"] for r in bl_pq]
    hc_true_k = [r["true_k"] for r in hc_pq]
    hc_recall = [r["recall"] for r in hc_pq]

    axes[0, 1].scatter(bl_true_k, bl_recall, alpha=0.6, label=f'Baseline k={best_baseline_k}',
                       color='steelblue', s=40)
    axes[0, 1].scatter(hc_true_k, hc_recall, alpha=0.6, label=f'HC gamma={best_hc_gamma}',
                       color='coral', s=40, marker='^')
    axes[0, 1].set_xlabel('True K (# relevant docs)')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_title('True K vs Recall')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(-0.05, 1.1)

    # Plot 3: true_K vs selected_K (HC)
    hc_selected_k = [r["hc_k"] for r in hc_pq]
    axes[1, 0].scatter(hc_true_k, hc_selected_k, alpha=0.6, color='coral', s=40)
    max_k = max(max(hc_true_k), max(hc_selected_k)) + 2
    axes[1, 0].plot([0, max_k], [0, max_k], 'k--', alpha=0.5, label='y=x (ideal)')
    axes[1, 0].set_xlabel('True K')
    axes[1, 0].set_ylabel('HC Selected K')
    axes[1, 0].set_title(f'HC K Selection (gamma={best_hc_gamma})')
    axes[1, 0].legend()

    # Plot 4: Baseline recall across different k values
    k_values = sorted(baseline.keys(), key=lambda x: int(x))
    overall_recalls = [baseline[k]["aggregate"]["recall"] for k in k_values]

    axes[1, 1].plot([int(k) for k in k_values], overall_recalls, 'o-',
                    color='steelblue', label='Baseline (varying k)')

    # Add HC as horizontal lines
    for gamma_key in sorted(hc.keys(), key=lambda x: float(x)):
        hc_recall_val = hc[gamma_key]["aggregate"]["recall"]
        axes[1, 1].axhline(y=hc_recall_val, linestyle='--', alpha=0.7,
                          label=f'HC gamma={gamma_key} (recall={hc_recall_val:.3f})')

    axes[1, 1].set_xlabel('Baseline k')
    axes[1, 1].set_ylabel('Mean Recall')
    axes[1, 1].set_title('Overall Recall: Baseline vs HC')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(plot_dir / "hc_vs_baseline.png", dpi=150)
    plt.close()
    print(f"  Plots saved to {plot_dir / 'hc_vs_baseline.png'}")


def main():
    results_dir = Path(PROJECT_ROOT) / "results" / "category_key"

    print("=" * 70)
    print("HC vs Baseline Comparison (Category-Key Dataset)")
    print("=" * 70)

    # Load results
    print("\nLoading results...")
    baseline, hc = load_results(results_dir)

    # Overall comparison table
    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)
    print(f"{'k':<8} {'Recall':<10} {'Precision':<10} {'MRR':<10} {'NDCG':<10} {'MAP':<10}")
    print("-" * 58)
    for k in sorted(baseline.keys(), key=lambda x: int(x)):
        r = baseline[k]["aggregate"]
        print(f"{k:<8} {r['recall']:<10.4f} {r['precision']:<10.4f} "
              f"{r['mrr']:<10.4f} {r['ndcg']:<10.4f} {r['map']:<10.4f}")

    print("\n" + "=" * 70)
    print("HC RESULTS")
    print("=" * 70)
    print(f"{'gamma':<8} {'Recall':<10} {'Precision':<10} {'MRR':<10} {'NDCG':<10} {'MAP':<10} {'Mean K':<10}")
    print("-" * 68)
    for gamma in sorted(hc.keys(), key=lambda x: float(x)):
        r = hc[gamma]["aggregate"]
        print(f"{gamma:<8} {r['recall']:<10.4f} {r['precision']:<10.4f} "
              f"{r['mrr']:<10.4f} {r['ndcg']:<10.4f} {r['map']:<10.4f} {r['mean_k']:<10.1f}")

    # Per-bucket analysis
    print("\n" + "=" * 70)
    print("PER-BUCKET ANALYSIS")
    print("=" * 70)

    # Use best baseline (by recall) and best HC (by recall)
    best_bl_k = max(baseline.keys(), key=lambda k: baseline[k]["aggregate"]["recall"])
    best_hc_gamma = max(hc.keys(), key=lambda g: hc[g]["aggregate"]["recall"])

    print(f"\nBest baseline: k={best_bl_k} (recall={baseline[best_bl_k]['aggregate']['recall']:.4f})")
    print(f"Best HC: gamma={best_hc_gamma} (recall={hc[best_hc_gamma]['aggregate']['recall']:.4f})")

    bl_pq = baseline[best_bl_k]["per_query"]
    hc_pq = hc[best_hc_gamma]["per_query"]

    bl_buckets = bucket_analysis(bl_pq)
    hc_buckets = bucket_analysis(hc_pq)

    print(f"\n{'Bucket':<10} {'K range':<10} {'BL Recall':<12} {'HC Recall':<12} {'Delta':<10}")
    print("-" * 54)
    for bucket in ["small", "medium", "large"]:
        if bucket in bl_buckets:
            bl_r = bl_buckets[bucket]["mean_recall"]
            hc_r = hc_buckets.get(bucket, {}).get("mean_recall", 0)
            k_range = bl_buckets[bucket]["k_range"]
            delta = hc_r - bl_r
            sign = "+" if delta > 0 else ""
            print(f"{bucket:<10} {k_range:<10} {bl_r:<12.4f} {hc_r:<12.4f} {sign}{delta:<10.4f}")

    # False negative analysis
    print("\n" + "=" * 70)
    print("FALSE NEGATIVE ANALYSIS")
    print("=" * 70)

    fn_analysis = false_negative_analysis(bl_pq, hc_pq, {})
    print(f"  Baseline false negatives: {fn_analysis['total_false_negatives_baseline']}")
    print(f"  HC false negatives: {fn_analysis['total_false_negatives_hc']}")
    print(f"  Reduction: {fn_analysis['fn_reduction']} ({fn_analysis['fn_reduction_pct']:.1f}%)")

    # Per-query K comparison (sample)
    print("\n" + "=" * 70)
    print("PER-QUERY K COMPARISON (sample)")
    print("=" * 70)

    # Build HC lookup
    hc_by_q = {r["query_id"]: r for r in hc_pq}

    print(f"{'Query':<30} {'Bucket':<8} {'True K':<8} {'BL K':<8} {'HC K':<8} {'BL Rec':<8} {'HC Rec':<8}")
    print("-" * 78)
    for r in sorted(bl_pq, key=lambda x: x["true_k"], reverse=True)[:20]:
        qid = r["query_id"]
        hc_r = hc_by_q.get(qid, {})
        print(f"{qid:<30} {r['size_bucket']:<8} {r['true_k']:<8} "
              f"{r['k']:<8} {hc_r.get('hc_k', '?'):<8} "
              f"{r['recall']:<8.3f} {hc_r.get('recall', 0):<8.3f}")

    # Generate plots
    print("\nGenerating comparison plots...")
    generate_plots(baseline, hc, results_dir)

    # Save comparison report
    report = {
        "best_baseline_k": best_bl_k,
        "best_hc_gamma": best_hc_gamma,
        "best_baseline_recall": baseline[best_bl_k]["aggregate"]["recall"],
        "best_hc_recall": hc[best_hc_gamma]["aggregate"]["recall"],
        "recall_improvement": hc[best_hc_gamma]["aggregate"]["recall"] - baseline[best_bl_k]["aggregate"]["recall"],
        "bucket_analysis": {
            "baseline": {k: {kk: vv for kk, vv in v.items()} for k, v in bl_buckets.items()},
            "hc": {k: {kk: vv for kk, vv in v.items()} for k, v in hc_buckets.items()}
        },
        "false_negative_analysis": fn_analysis
    }

    report_path = results_dir / "comparison" / "comparison_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {report_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
