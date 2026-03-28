"""
Compare Baseline vs HC Results on Amazon Categories Dataset

Loads results from both experiments, produces comparison tables
and analysis by K bucket.
"""

import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_results():
    """Load baseline and HC results."""
    baseline_path = PROJECT_ROOT / "results" / "amazon_categories" / "baseline" / "baseline_results.json"
    hc_path = PROJECT_ROOT / "results" / "amazon_categories" / "hc" / "hc_results.json"

    baseline = None
    hc = None

    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"Loaded baseline results: {len(baseline)} configs")
    else:
        print(f"WARNING: No baseline results at {baseline_path}")

    if hc_path.exists():
        with open(hc_path) as f:
            hc = json.load(f)
        print(f"Loaded HC results: {len(hc)} configs")
    else:
        print(f"WARNING: No HC results at {hc_path}")

    return baseline, hc


def print_baseline_table(baseline):
    """Print baseline results table."""
    print("\n" + "=" * 80)
    print("BASELINE TOP-K RESULTS")
    print("=" * 80)

    header = f"{'k':<6} {'Recall':<10} {'Precision':<10} {'F1':<10} {'MRR':<10} {'NDCG':<10} {'MAP':<10}"
    print(header)
    print("-" * len(header))

    for k_str in sorted(baseline.keys(), key=lambda x: int(x)):
        r = baseline[k_str]["aggregate"]
        rec, prec = r["recall"], r["precision"]
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
        print(
            f"{k_str:<6} {rec:<10.4f} {prec:<10.4f} {f1:<10.4f} "
            f"{r['mrr']:<10.4f} {r['ndcg']:<10.4f} {r['map']:<10.4f}"
        )


def print_hc_table(hc):
    """Print HC results table sorted by F1."""
    print("\n" + "=" * 80)
    print("HC RESULTS (sorted by F1)")
    print("=" * 80)

    header = f"{'Config':<25} {'Recall':<8} {'Prec':<8} {'F1':<8} {'MRR':<8} {'NDCG':<8} {'MAP':<8} {'Mean K':<8}"
    print(header)
    print("-" * len(header))

    sorted_configs = sorted(
        hc.items(), key=lambda x: x[1]["aggregate"].get("f1", 0), reverse=True
    )
    for config_key, data in sorted_configs:
        r = data["aggregate"]
        f1 = r.get("f1", 0)
        if f1 == 0:
            rec, prec = r["recall"], r["precision"]
            f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
        print(
            f"{config_key:<25} {r['recall']:<8.4f} {r['precision']:<8.4f} "
            f"{f1:<8.4f} {r['mrr']:<8.4f} {r['ndcg']:<8.4f} {r['map']:<8.4f} "
            f"{r['mean_k']:<8.1f}"
        )


def print_bucket_comparison(baseline, hc):
    """Compare best HC config vs best baseline by K bucket."""
    print("\n" + "=" * 80)
    print("COMPARISON BY K BUCKET")
    print("=" * 80)

    # Find best baseline by F1
    best_baseline_k = None
    best_baseline_f1 = -1
    for k_str, data in baseline.items():
        r = data["aggregate"]
        rec, prec = r["recall"], r["precision"]
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
        if f1 > best_baseline_f1:
            best_baseline_f1 = f1
            best_baseline_k = k_str

    # Find best HC by F1
    best_hc_key = None
    best_hc_f1 = -1
    for config_key, data in hc.items():
        r = data["aggregate"]
        f1 = r.get("f1", 0)
        if f1 == 0:
            rec, prec = r["recall"], r["precision"]
            f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
        if f1 > best_hc_f1:
            best_hc_f1 = f1
            best_hc_key = config_key

    print(f"\nBest baseline: k={best_baseline_k} (F1={best_baseline_f1:.4f})")
    print(f"Best HC: {best_hc_key} (F1={best_hc_f1:.4f})")

    baseline_buckets = baseline[best_baseline_k].get("by_bucket", {})
    hc_buckets = hc[best_hc_key].get("by_bucket", {})

    if baseline_buckets and hc_buckets:
        print(f"\n{'Bucket':<10} {'Method':<12} {'n':<5} {'Recall':<8} {'Prec':<8} {'F1':<8} {'Mean K':<8}")
        print("-" * 59)

        for bucket in ["small", "medium", "large"]:
            b_base = baseline_buckets.get(bucket)
            b_hc = hc_buckets.get(bucket)

            if b_base:
                rec, prec = b_base["recall"], b_base["precision"]
                f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
                print(
                    f"{bucket:<10} {'Baseline':<12} {b_base['n_queries']:<5} "
                    f"{rec:<8.4f} {prec:<8.4f} {f1:<8.4f} {int(best_baseline_k):<8}"
                )
            if b_hc:
                rec, prec = b_hc["recall"], b_hc["precision"]
                f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
                mean_k = b_hc.get("mean_hc_k", b_hc.get("mean_k", "?"))
                print(
                    f"{'':10} {'HC':<12} {b_hc['n_queries']:<5} "
                    f"{rec:<8.4f} {prec:<8.4f} {f1:<8.4f} {mean_k:<8.1f}"
                )
            print()


def print_head_to_head(baseline, hc):
    """Compare each HC config against the oracle baseline (best fixed k per query)."""
    print("\n" + "=" * 80)
    print("HC vs ORACLE BASELINE (best fixed k)")
    print("=" * 80)

    # For each HC config, find which fixed k would be closest to the true K
    # and compare HC's adaptive K selection
    if not hc:
        return

    best_hc_key = max(
        hc.keys(),
        key=lambda k: hc[k]["aggregate"].get("f1", 0),
    )
    best_hc = hc[best_hc_key]

    print(f"\nBest HC config: {best_hc_key}")
    print(f"\nPer-query K selection analysis:")

    hc_ks = [q["hc_k"] for q in best_hc["per_query"]]
    true_ks = [q["true_k"] for q in best_hc["per_query"]]

    if len(set(hc_ks)) > 1 and len(set(true_ks)) > 1:
        corr = np.corrcoef(true_ks, hc_ks)[0, 1]
        print(f"  Correlation(true_K, HC_K): {corr:.4f}")

    print(f"  Mean true K: {np.mean(true_ks):.1f} (range: {min(true_ks)}-{max(true_ks)})")
    print(f"  Mean HC K: {np.mean(hc_ks):.1f} (range: {min(hc_ks)}-{max(hc_ks)})")

    # Ratio analysis
    ratios = [h / t if t > 0 else 0 for h, t in zip(hc_ks, true_ks)]
    print(f"  Mean HC_K/true_K ratio: {np.mean(ratios):.2f}")
    print(f"  Median HC_K/true_K ratio: {np.median(ratios):.2f}")

    over = sum(1 for r in ratios if r > 1.0)
    under = sum(1 for r in ratios if r < 1.0)
    exact = sum(1 for r in ratios if r == 1.0)
    print(f"  Over-retrieval: {over}/{len(ratios)} queries")
    print(f"  Under-retrieval: {under}/{len(ratios)} queries")
    print(f"  Exact: {exact}/{len(ratios)} queries")


def save_comparison(baseline, hc):
    """Save combined comparison to JSON."""
    output_dir = PROJECT_ROOT / "results" / "amazon_categories"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = {
        "baseline_configs": {},
        "hc_configs": {},
    }

    if baseline:
        for k_str, data in baseline.items():
            r = data["aggregate"]
            rec, prec = r["recall"], r["precision"]
            f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
            comparison["baseline_configs"][f"top_k={k_str}"] = {
                "recall": rec,
                "precision": prec,
                "f1": f1,
                "mrr": r["mrr"],
                "ndcg": r["ndcg"],
                "map": r["map"],
                "mean_k": float(k_str),
                "by_bucket": data.get("by_bucket", {}),
            }

    if hc:
        for config_key, data in hc.items():
            r = data["aggregate"]
            f1 = r.get("f1", 0)
            if f1 == 0:
                rec, prec = r["recall"], r["precision"]
                f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
            comparison["hc_configs"][config_key] = {
                "recall": r["recall"],
                "precision": r["precision"],
                "f1": f1,
                "mrr": r["mrr"],
                "ndcg": r["ndcg"],
                "map": r["map"],
                "mean_k": r["mean_k"],
                "by_bucket": data.get("by_bucket", {}),
            }

    path = output_dir / "comparison.json"
    with open(path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to {path}")


def main():
    print("=" * 80)
    print("Amazon Categories: Baseline vs HC Comparison")
    print("=" * 80)

    baseline, hc = load_results()

    if baseline:
        print_baseline_table(baseline)
    if hc:
        print_hc_table(hc)
    if baseline and hc:
        print_bucket_comparison(baseline, hc)
        print_head_to_head(baseline, hc)
        save_comparison(baseline, hc)

    print("\nDone!")


if __name__ == "__main__":
    main()
