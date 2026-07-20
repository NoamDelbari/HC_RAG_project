"""Generate HC vs Baseline comparison plots."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "category_key"
OUT_DIR = RESULTS_DIR / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load results
with open(RESULTS_DIR / "baseline" / "baseline_results.json") as f:
    baseline = json.load(f)
with open(RESULTS_DIR / "hc" / "hc_results.json") as f:
    hc = json.load(f)

bucket_colors = {"small": "royalblue", "medium": "darkorange", "large": "forestgreen"}
baseline_ks = ["5", "8", "10", "12", "15", "20"]
hc_gammas = ["0.07", "0.1", "0.2", "0.4"]


# =========================================================================
# PLOT 1: Recall-Precision tradeoff curve
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
b_recalls = [baseline[k]["aggregate"]["recall"] for k in baseline_ks]
b_precisions = [baseline[k]["aggregate"]["precision"] for k in baseline_ks]
b_mean_ks = [baseline[k]["aggregate"]["mean_k"] for k in baseline_ks]
ax.plot(b_recalls, b_precisions, "o-", color="gray", linewidth=2, markersize=8,
        label="Baseline (fixed K)", zorder=2)
for i, k in enumerate(baseline_ks):
    ax.annotate(f"k={k}", (b_recalls[i], b_precisions[i]),
               textcoords="offset points", xytext=(8, 5), fontsize=8, color="gray")

h_recalls = [hc[g]["aggregate"]["recall"] for g in hc_gammas]
h_precisions = [hc[g]["aggregate"]["precision"] for g in hc_gammas]
h_mean_ks = [hc[g]["aggregate"]["mean_k"] for g in hc_gammas]
ax.plot(h_recalls, h_precisions, "s-", color="crimson", linewidth=2, markersize=8,
        label="HC (adaptive K)", zorder=3)
for i, g in enumerate(hc_gammas):
    ax.annotate(f"\u03b3={g}\n(K\u2248{h_mean_ks[i]:.0f})", (h_recalls[i], h_precisions[i]),
               textcoords="offset points", xytext=(8, -12), fontsize=8, color="crimson")

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Recall-Precision Tradeoff", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim(0.5, 1.0)
ax.set_ylim(0.1, 1.05)
ax.grid(True, alpha=0.3)

# Right: Recall vs Mean K
ax = axes[1]
ax.plot(b_mean_ks, b_recalls, "o-", color="gray", linewidth=2, markersize=8,
        label="Baseline (fixed K)", zorder=2)
for i, k in enumerate(baseline_ks):
    ax.annotate(f"k={k}", (b_mean_ks[i], b_recalls[i]),
               textcoords="offset points", xytext=(5, 5), fontsize=8, color="gray")

ax.plot(h_mean_ks, h_recalls, "s-", color="crimson", linewidth=2, markersize=8,
        label="HC (adaptive K)", zorder=3)
for i, g in enumerate(hc_gammas):
    ax.annotate(f"\u03b3={g}", (h_mean_ks[i], h_recalls[i]),
               textcoords="offset points", xytext=(5, -10), fontsize=8, color="crimson")

ax.set_xlabel("Mean K (documents retrieved)", fontsize=12)
ax.set_ylabel("Recall", fontsize=12)
ax.set_title("Recall vs Mean K", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle("HC vs Baseline: Overall Performance", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "1_recall_precision_tradeoff.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 1_recall_precision_tradeoff.png")


# =========================================================================
# PLOT 2: Per-bucket recall comparison at matched mean K
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

matched_pairs = [
    ("5", "0.07", "k=5 vs \u03b3=0.07\n(mean K\u22485-6)"),
    ("8", "0.1",  "k=8 vs \u03b3=0.1\n(mean K\u22488)"),
    ("15", "0.2", "k=15 vs \u03b3=0.2\n(mean K\u224812-15)"),
]

for idx, (bk, hg, title) in enumerate(matched_pairs):
    ax = axes[idx]
    bucket_order = ["small", "medium", "large"]
    x = np.arange(len(bucket_order))
    width = 0.35

    b_bucket_recalls = {}
    for q in baseline[bk]["per_query"]:
        b_bucket_recalls.setdefault(q["size_bucket"], []).append(q["recall"])

    h_bucket_recalls = {}
    for q in hc[hg]["per_query"]:
        h_bucket_recalls.setdefault(q["size_bucket"], []).append(q["recall"])

    b_means = [np.mean(b_bucket_recalls.get(b, [0])) for b in bucket_order]
    h_means = [np.mean(h_bucket_recalls.get(b, [0])) for b in bucket_order]
    b_stds = [np.std(b_bucket_recalls.get(b, [0])) for b in bucket_order]
    h_stds = [np.std(h_bucket_recalls.get(b, [0])) for b in bucket_order]

    ax.bar(x - width/2, b_means, width, yerr=b_stds, capsize=4,
           color="lightgray", edgecolor="black", linewidth=0.5,
           label=f"Baseline k={bk}")
    ax.bar(x + width/2, h_means, width, yerr=h_stds, capsize=4,
           color="crimson", edgecolor="black", linewidth=0.5, alpha=0.7,
           label=f"HC \u03b3={hg}")

    for i in range(len(bucket_order)):
        ax.text(x[i] - width/2, b_means[i] + b_stds[i] + 0.02,
               f"{b_means[i]:.2f}", ha="center", fontsize=8, color="gray")
        ax.text(x[i] + width/2, h_means[i] + h_stds[i] + 0.02,
               f"{h_means[i]:.2f}", ha="center", fontsize=8, color="crimson")

    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in bucket_order], fontsize=11)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)

plt.suptitle("HC vs Baseline: Recall by Country Size (at Matched Mean K)",
            fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "2_per_bucket_recall.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 2_per_bucket_recall.png")


# =========================================================================
# PLOT 3: Per-query scatter — true K vs recall
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

configs = [
    (baseline, "10", "Baseline k=10"),
    (hc, "0.1", "HC \u03b3=0.1"),
]

for idx, (source, key, label) in enumerate(configs):
    ax = axes[idx]
    per_query = source[key]["per_query"]

    for q in per_query:
        c = bucket_colors[q["size_bucket"]]
        ax.scatter(q["true_k"], q["recall"], s=50, color=c,
                  edgecolors="black", linewidth=0.5, alpha=0.7, zorder=3)

    ax.set_xlabel("True K (ground truth)", fontsize=12)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
    ax.grid(True, alpha=0.2)

    legend_elements = [Patch(facecolor=bucket_colors[b], edgecolor="black", label=b.capitalize())
                      for b in ["small", "medium", "large"]]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower left")

    mean_r = np.mean([q["recall"] for q in per_query])
    ax.text(0.95, 0.05, f"Mean recall: {mean_r:.3f}",
           transform=ax.transAxes, ha="right", fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.suptitle("Per-Query Recall: Baseline k=10 vs HC \u03b3=0.1 (matched mean K\u22488-10)",
            fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "3_per_query_recall_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 3_per_query_recall_scatter.png")


# =========================================================================
# PLOT 4: K selection per query — HC adapts, baseline is fixed
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

hc_configs = [("0.1", "HC \u03b3=0.1 (max K=10)"), ("0.2", "HC \u03b3=0.2 (max K=20)")]

for idx, (gamma, title) in enumerate(hc_configs):
    ax = axes[idx]
    hc_pq = sorted(hc[gamma]["per_query"], key=lambda x: x["true_k"])
    countries = [q["country"] for q in hc_pq]
    true_ks_sorted = [q["true_k"] for q in hc_pq]
    hc_ks_sorted = [q["hc_k"] for q in hc_pq]
    bucket_sorted = [q["size_bucket"] for q in hc_pq]
    colors_sorted = [bucket_colors[b] for b in bucket_sorted]

    y = np.arange(len(countries))

    ax.barh(y, true_ks_sorted, height=0.4, color=colors_sorted,
            alpha=0.3, edgecolor="black", linewidth=0.3, label="True K")
    ax.scatter(hc_ks_sorted, y, s=40, color="crimson", zorder=3,
              edgecolors="black", linewidth=0.5, label=f"HC K (\u03b3={gamma})")
    ax.scatter([10]*len(y), y, s=20, color="gray", marker="|", zorder=2,
              linewidths=1.5, label="Baseline k=10")

    ax.set_yticks(y)
    ax.set_yticklabels(countries, fontsize=6)
    ax.set_xlabel("K (documents selected)", fontsize=12)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")

plt.suptitle("HC Adapts K Per Query (baseline is fixed at k=10)",
            fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "4_k_selection_per_query.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 4_k_selection_per_query.png")


# =========================================================================
# PLOT 5: Recall and Precision by bucket across all K values
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(17, 10))

for row, metric in enumerate(["recall", "precision"]):
    for col, bucket in enumerate(["small", "medium", "large"]):
        ax = axes[row, col]

        b_vals = []
        b_ks_plot = []
        for k in baseline_ks:
            per_q = [q[metric] for q in baseline[k]["per_query"] if q["size_bucket"] == bucket]
            b_vals.append(np.mean(per_q))
            b_ks_plot.append(int(k))

        h_vals = []
        h_ks_plot = []
        for g in hc_gammas:
            per_q = [q[metric] for q in hc[g]["per_query"] if q["size_bucket"] == bucket]
            h_vals.append(np.mean(per_q))
            h_ks_plot.append(hc[g]["aggregate"]["mean_k"])

        ax.plot(b_ks_plot, b_vals, "o-", color="gray", linewidth=2, markersize=6,
                label="Baseline")
        ax.plot(h_ks_plot, h_vals, "s-", color="crimson", linewidth=2, markersize=6,
                label="HC")

        ax.set_xlabel("Mean K", fontsize=10)
        ax.set_ylabel(metric.capitalize(), fontsize=10)
        ax.set_title(f"{bucket.capitalize()} countries ({metric.capitalize()})",
                    fontsize=11, fontweight="bold",
                    color=bucket_colors[bucket])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 1.1)

plt.suptitle("HC vs Baseline: Recall and Precision by Country Size",
            fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "5_metrics_by_bucket.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 5_metrics_by_bucket.png")

print("\nAll comparison plots saved to:", OUT_DIR)
