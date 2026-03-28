"""
Amazon Categories: Full Experiment Plots (matching category-key plot style).

Generates 8 plots (0-7) mirroring the category-key experiment plots:
  0_k_distribution.png       - True K distribution across queries
  1_null_calibration.png     - Global z-score null calibration (p-values)
  2_candidate_separation.png - What HC sees at runtime (top candidates)
  3_recall_precision_tradeoff.png - HC vs all baselines (grouped bar chart)
  4_per_bucket_recall.png    - Per-bucket recall at matched mean K
  5_per_query_recall_scatter.png - Per-query recall vs true K
  6_k_selection_per_query.png - HC adapts K per query (horizontal bars)
  7_hc_outcome.png           - 3-panel summary (K scatter, p-values, recall by bucket)

Uses pre-computed results from JSON files + live computation for plots 1, 2, 7.
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BUCKET_COLORS = {"small": "royalblue", "medium": "darkorange", "large": "forestgreen"}
RESULTS_DIR = PROJECT_ROOT / "results" / "amazon_categories"
DATASET_DIR = PROJECT_ROOT / "datasets" / "amazon_categories"


def load_results():
    """Load pre-computed results."""
    results = {}

    # Baseline
    p = RESULTS_DIR / "baseline" / "baseline_results.json"
    if p.exists():
        with open(p) as f:
            results["baseline"] = json.load(f)

    # HC (prefer combined file with large pool results)
    p = RESULTS_DIR / "hc" / "hc_all_results.json"
    if not p.exists():
        p = RESULTS_DIR / "hc" / "hc_results.json"
    if p.exists():
        with open(p) as f:
            results["hc"] = json.load(f)

    return results


def find_best_hc(hc_results):
    """Find best HC config by F1."""
    best_key, best_f1 = None, -1
    for key, data in hc_results.items():
        r = data["aggregate"]
        f1 = r.get("f1", 0)
        if f1 == 0:
            rec, prec = r["recall"], r["precision"]
            f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
        if f1 > best_f1:
            best_f1, best_key = f1, key
    return best_key, best_f1


def compute_f1(r, p):
    return 2 * r * p / (r + p) if (r + p) > 0 else 0


# =============================================================================
# Plot 0: K Distribution
# =============================================================================
def plot_k_distribution(hc_results, best_hc_key, out_dir):
    """Distribution of true K across queries."""
    per_query = hc_results[best_hc_key]["per_query"]
    true_ks = np.array([q["true_k"] for q in per_query])
    buckets = [q["k_bucket"] for q in per_query]
    subcats = [q["subcategory"].split(" > ")[-1] for q in per_query]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: histogram
    ax = axes[0]
    for bucket in ["small", "medium", "large"]:
        mask = np.array([b == bucket for b in buckets])
        ks = true_ks[mask]
        if len(ks) > 0:
            ax.hist(ks, bins=range(int(ks.min()), int(ks.max()) + 2), alpha=0.7,
                    color=BUCKET_COLORS[bucket], edgecolor="black", linewidth=0.5,
                    label=f"{bucket.capitalize()} (n={mask.sum()}, K={ks.min()}-{ks.max()})")
    ax.set_xlabel("True K (relevant documents per query)", fontsize=12)
    ax.set_ylabel("Number of queries", fontsize=12)
    ax.set_title("Retrieval: Distribution of True K Across Queries", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)

    # Right: sorted bar chart (top 50 by K for readability)
    ax2 = axes[1]
    idx = np.argsort(true_ks)[::-1][:50]  # Top 50 largest K
    sorted_subcats = [subcats[i][:30] for i in idx]  # Truncate names
    sorted_ks = true_ks[idx]
    sorted_colors = [BUCKET_COLORS[buckets[i]] for i in idx]
    y = np.arange(len(sorted_subcats))
    ax2.barh(y, sorted_ks, color=sorted_colors, edgecolor="black", linewidth=0.3)
    ax2.set_yticks(y)
    ax2.set_yticklabels(sorted_subcats, fontsize=6)
    ax2.set_xlabel("True K", fontsize=12)
    ax2.set_title("Retrieval: True K Per Category (top 50, sorted)", fontsize=12, fontweight="bold")
    ax2.invert_yaxis()
    legend_elements = [Patch(facecolor=BUCKET_COLORS[b], edgecolor="black", label=b.capitalize())
                       for b in ["small", "medium", "large"]]
    ax2.legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.suptitle("Retrieval: Amazon Categories Dataset — Variable K Distribution",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "0_k_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  0_k_distribution.png")


# =============================================================================
# Plot 1: Null Calibration (global z-score null)
# =============================================================================
def plot_null_calibration(queries, qrels, vector_db, global_null, model, out_dir):
    """P-value distributions from global z-score null for 3 example queries."""
    from hc.higher_criticism import HigherCriticism

    # Pick one query from each bucket
    by_bucket = {"small": [], "medium": [], "large": []}
    for q in queries:
        by_bucket[q.k_bucket].append(q)

    examples = []
    for bucket in ["small", "medium", "large"]:
        qs = sorted(by_bucket[bucket], key=lambda x: len(qrels.get(x.query_id, set())))
        examples.append(qs[len(qs) // 2])  # Median query

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    bins = np.linspace(0, 1, 21)

    for col, q in enumerate(examples):
        relevant_set = qrels.get(q.query_id, set())
        true_k = len(relevant_set)
        query_emb = model.embed_query(q.text)

        # Get similarities to all docs
        all_candidates = vector_db.search(query_emb, k=vector_db.get_num_documents())
        all_sims = np.array([c.similarity for c in all_candidates])
        all_ids = [c.doc_id for c in all_candidates]
        is_rel = np.array([d in relevant_set for d in all_ids])

        # Compute z-scores then p-values via global null
        mu = np.mean(all_sims)
        sigma = np.std(all_sims)
        z_scores = (all_sims - mu) / sigma if sigma > 0 else np.zeros_like(all_sims)

        hc_mod = HigherCriticism(null_distribution=global_null)
        p_vals = hc_mod.compute_p_values(z_scores)

        p_nonrel = p_vals[~is_rel]
        p_rel = p_vals[is_rel]

        # Top row: non-relevant p-values (should be uniform)
        ax = axes[0, col]
        ax.hist(p_nonrel, bins=bins, density=True, alpha=0.7,
                color="steelblue", edgecolor="black", linewidth=0.5,
                label=f"Non-relevant (n={len(p_nonrel)})")
        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, label="Uniform(0,1)")
        ks_s, ks_p = stats.kstest(p_nonrel, "uniform")
        ax.set_xlabel("p-value", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        subcat_short = q.subcategory.split(" > ")[-1][:25]
        ax.set_title(f'{subcat_short} ({q.k_bucket}, K={true_k})\n'
                     f"KS stat={ks_s:.3f}, p={ks_p:.3f}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 2.0)

        # Bottom row: relevant p-values (should be near 0)
        ax2 = axes[1, col]
        ax2.hist(p_rel, bins=bins, density=True, alpha=0.7,
                 color="crimson", edgecolor="black", linewidth=0.5,
                 label=f"Relevant (n={len(p_rel)})")
        ax2.axvline(x=0.05, color="green", linestyle="--", linewidth=1.5, label="p = 0.05")
        pct = np.mean(p_rel < 0.05) * 100
        ax2.set_xlabel("p-value", fontsize=11)
        ax2.set_ylabel("Density", fontsize=11)
        ax2.set_title(f"Relevant docs: {pct:.0f}% have p < 0.05", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.set_xlim(-0.02, 1.02)

    fig.suptitle("Retrieval: Null Calibration \u2014 P-values (Global Z-Score Null)\n"
                 "(Top: non-relevant should be uniform | Bottom: relevant should be near 0)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_dir / "1_null_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  1_null_calibration.png")


# =============================================================================
# Plot 2: Candidate Separation (what HC sees at runtime)
# =============================================================================
def plot_candidate_separation(queries, qrels, vector_db, global_null, model, out_dir,
                               max_candidates=1000, hc_gamma=0.05):
    """Ranked candidates + similarity distributions for 3 example queries."""
    from hc.higher_criticism import HigherCriticism

    by_bucket = {"small": [], "medium": [], "large": []}
    for q in queries:
        by_bucket[q.k_bucket].append(q)

    examples = []
    for bucket in ["small", "medium", "large"]:
        qs = sorted(by_bucket[bucket], key=lambda x: len(qrels.get(x.query_id, set())))
        examples.append(qs[len(qs) // 2])

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

    for col, q in enumerate(examples):
        relevant_set = qrels.get(q.query_id, set())
        true_k = len(relevant_set)
        query_emb = model.embed_query(q.text)

        candidates = vector_db.search(query_emb, k=max_candidates)
        cand_ids = [c.doc_id for c in candidates]
        cand_sims = np.array([c.similarity for c in candidates])
        is_rel = np.array([d in relevant_set for d in cand_ids])

        # Compute HC threshold
        mu = np.mean(cand_sims)
        sigma = np.std(cand_sims)
        z_scores = (cand_sims - mu) / sigma if sigma > 0 else np.zeros_like(cand_sims)
        hc_mod = HigherCriticism(null_distribution=global_null)
        hc_result = hc_mod.compute_hc_threshold(z_scores, gamma=hc_gamma, min_hc=0.0, allow_empty=False)
        hc_k = hc_result.k

        # Top row: waterfall
        ax = axes[0, col]
        ranks = np.arange(1, len(cand_sims) + 1)
        ax.scatter(ranks[~is_rel], cand_sims[~is_rel], s=8, color="steelblue", alpha=0.4,
                   label="Non-relevant", zorder=2)
        ax.scatter(ranks[is_rel], cand_sims[is_rel], s=30, color="crimson", alpha=0.85,
                   label="Relevant", zorder=3, edgecolors="black", linewidth=0.5)
        if 0 < hc_k < len(cand_sims):
            ax.axvline(x=hc_k + 0.5, color="green", linewidth=2.5,
                       label=f"HC cut (k={hc_k})", zorder=4)
        ax.set_xlabel("Candidate rank", fontsize=11)
        ax.set_ylabel("Cosine similarity", fontsize=11)
        subcat_short = q.subcategory.split(" > ")[-1][:25]
        ax.set_title(f'{subcat_short}: true K={true_k}, HC K={hc_k}',
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")

        # Bottom row: similarity distributions
        ax2 = axes[1, col]
        sim_rel = cand_sims[is_rel]
        sim_nonrel = cand_sims[~is_rel]
        bins_sim = np.linspace(cand_sims.min() - 0.01, cand_sims.max() + 0.01, 30)
        if len(sim_nonrel) > 0:
            ax2.hist(sim_nonrel, bins=bins_sim, alpha=0.5, color="steelblue",
                     edgecolor="black", linewidth=0.5,
                     label=f"Non-relevant (n={len(sim_nonrel)})")
        if len(sim_rel) > 0:
            ax2.hist(sim_rel, bins=bins_sim, alpha=0.6, color="crimson",
                     edgecolor="black", linewidth=0.5,
                     label=f"Relevant (n={len(sim_rel)})")
        if 0 < hc_k < len(cand_sims):
            threshold_sim = cand_sims[hc_k - 1] if hc_k <= len(cand_sims) else cand_sims[-1]
            ax2.axvline(x=threshold_sim, color="green", linewidth=2.5,
                        label=f"HC threshold={threshold_sim:.3f}")
        ax2.set_xlabel("Cosine similarity", fontsize=11)
        ax2.set_ylabel("Count", fontsize=11)
        ax2.set_title("Similarity distributions of candidates", fontsize=10)
        ax2.legend(fontsize=7)

    fig.suptitle(f"Retrieval: What HC Sees at Runtime \u2014 Top-{max_candidates} Candidates\n"
                 "(Top: ranked candidates | Bottom: similarity distributions)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_dir / "2_candidate_separation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  2_candidate_separation.png")


# =============================================================================
# Plot 3: HC vs All Baselines (grouped bar chart)
# =============================================================================
def plot_recall_precision_tradeoff(baseline_results, hc_results, best_hc_key, out_dir):
    """Grouped bar chart: HC vs all baseline K values on key metrics."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    h_agg = hc_results[best_hc_key]["aggregate"]
    h_f1 = h_agg.get("f1", compute_f1(h_agg["recall"], h_agg["precision"]))

    baseline_ks = sorted(baseline_results.keys(), key=int)
    metrics = ["Recall", "Precision", "F1", "MRR", "MAP"]

    methods = []
    for k in baseline_ks:
        a = baseline_results[k]["aggregate"]
        f1 = compute_f1(a["recall"], a["precision"])
        methods.append((f"k={k}", [a["recall"], a["precision"], f1, a["mrr"], a["map"]]))
    methods.append(("HC", [h_agg["recall"], h_agg["precision"], h_f1, h_agg["mrr"], h_agg["map"]]))

    n_methods = len(methods)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    total_width = 0.82
    bar_width = total_width / n_methods

    baseline_colors = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#76b7b2"]
    hc_color = "#1b1b1b"

    for i, (label, vals) in enumerate(methods):
        offset = (i - n_methods / 2 + 0.5) * bar_width
        is_hc = (label == "HC")
        color = hc_color if is_hc else baseline_colors[i % len(baseline_colors)]
        lw = 1.5 if is_hc else 0.6
        hatch = "///" if is_hc else None

        bars = ax.bar(x + offset, vals, bar_width, color=color, edgecolor="black",
                      linewidth=lw, alpha=0.85, label=label, zorder=3 if is_hc else 2, hatch=hatch)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=6,
                    color=color if not is_hc else "black", fontweight="bold", rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, max(0.35, max(v for _, vals in methods for v in vals) + 0.05))
    ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=n_methods, frameon=True, edgecolor="gray", fancybox=True)
    ax.grid(axis="y", alpha=0.2)

    hc_cfg = hc_results[best_hc_key].get("config", {})
    ax.set_title(f"Retrieval: HC (\u03b3={hc_cfg.get('gamma', '?')}, pool={hc_cfg.get('max_candidates', '?')}, "
                 f"adaptive K, mean K={h_agg['mean_k']:.0f}) vs All Baseline Top-K",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "3_recall_precision_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  3_recall_precision_tradeoff.png")


# =============================================================================
# Plot 4: Per-Bucket Recall at Matched Mean K
# =============================================================================
def plot_per_bucket_recall(baseline_results, hc_results, best_hc_key, out_dir):
    """Bar chart of recall per K bucket, HC vs matched baseline."""
    hc_data = hc_results[best_hc_key]
    hc_mean_k = hc_data["aggregate"]["mean_k"]

    # Find closest baseline k
    matched_k = min(baseline_results.keys(), key=lambda k: abs(int(k) - hc_mean_k))

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    bucket_order = ["small", "medium", "large"]
    x = np.arange(len(bucket_order))
    width = 0.35

    # Baseline per-bucket recall
    b_bucket = {}
    for q in baseline_results[matched_k]["per_query"]:
        b_bucket.setdefault(q["k_bucket"], []).append(q["recall"])
    # HC per-bucket recall
    h_bucket = {}
    for q in hc_data["per_query"]:
        h_bucket.setdefault(q["k_bucket"], []).append(q["recall"])

    b_means = [np.mean(b_bucket.get(b, [0])) for b in bucket_order]
    h_means = [np.mean(h_bucket.get(b, [0])) for b in bucket_order]
    b_stds = [np.std(b_bucket.get(b, [0])) for b in bucket_order]
    h_stds = [np.std(h_bucket.get(b, [0])) for b in bucket_order]

    ax.bar(x - width / 2, b_means, width, yerr=b_stds, capsize=4,
           color="lightgray", edgecolor="black", linewidth=0.5,
           label=f"Baseline k={matched_k}")
    ax.bar(x + width / 2, h_means, width, yerr=h_stds, capsize=4,
           color="crimson", edgecolor="black", linewidth=0.5, alpha=0.7,
           label=f"HC {best_hc_key}")

    for i in range(len(bucket_order)):
        ax.text(x[i] - width / 2, b_means[i] + b_stds[i] + 0.005,
                f"{b_means[i]:.3f}", ha="center", fontsize=9, color="gray")
        ax.text(x[i] + width / 2, h_means[i] + h_stds[i] + 0.005,
                f"{h_means[i]:.3f}", ha="center", fontsize=9, color="crimson")

    # Add true K info to x labels
    true_k_means = {}
    for q in hc_data["per_query"]:
        true_k_means.setdefault(q["k_bucket"], []).append(q["true_k"])
    xlabels = []
    for b in bucket_order:
        mean_tk = np.mean(true_k_means.get(b, [0]))
        xlabels.append(f"{b.capitalize()}\n(true K\u2248{mean_tk:.0f})")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title(f"Retrieval: Recall by K Bucket at Matched Mean K\n"
                 f"(Baseline k={matched_k} vs HC {best_hc_key})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    y_max = max(max(b_means), max(h_means)) + 0.05
    ax.set_ylim(0, min(y_max + 0.1, 1.2))
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_dir / "4_per_bucket_recall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  4_per_bucket_recall.png")


# =============================================================================
# Plot 5: Per-Query Recall Scatter
# =============================================================================
def plot_per_query_recall(baseline_results, hc_results, best_hc_key, out_dir):
    """True K vs recall scatter for baseline and HC side by side."""
    hc_data = hc_results[best_hc_key]
    hc_mean_k = hc_data["aggregate"]["mean_k"]
    matched_k = min(baseline_results.keys(), key=lambda k: abs(int(k) - hc_mean_k))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    configs = [
        (baseline_results[matched_k]["per_query"], f"Baseline k={matched_k}"),
        (hc_data["per_query"], f"HC {best_hc_key}"),
    ]

    for idx, (per_query, label) in enumerate(configs):
        ax = axes[idx]
        for q in per_query:
            bucket = q.get("k_bucket", "medium")
            c = BUCKET_COLORS.get(bucket, "gray")
            ax.scatter(q["true_k"], q["recall"], s=40, color=c,
                       edgecolors="black", linewidth=0.4, alpha=0.7, zorder=3)

        ax.set_xlabel("True K (ground truth)", fontsize=12)
        ax.set_ylabel("Recall", fontsize=12)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
        ax.grid(True, alpha=0.2)

        legend_elements = [Patch(facecolor=BUCKET_COLORS[b], edgecolor="black", label=b.capitalize())
                           for b in ["small", "medium", "large"]]
        ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

        mean_r = np.mean([q["recall"] for q in per_query])
        ax.text(0.95, 0.05, f"Mean recall: {mean_r:.4f}",
                transform=ax.transAxes, ha="right", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.suptitle(f"Retrieval: Per-Query Recall — Baseline k={matched_k} vs HC {best_hc_key}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "5_per_query_recall_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  5_per_query_recall_scatter.png")


# =============================================================================
# Plot 6: K Selection Per Query (horizontal bars)
# =============================================================================
def plot_k_selection(hc_results, best_hc_key, out_dir):
    """Horizontal bar chart showing HC adapts K per query."""
    hc_data = hc_results[best_hc_key]
    hc_pq = sorted(hc_data["per_query"], key=lambda x: x["true_k"])

    # With 131 queries, show all but use small font
    subcats = [q["subcategory"].split(" > ")[-1][:25] for q in hc_pq]
    true_ks = [q["true_k"] for q in hc_pq]
    hc_ks = [q["hc_k"] for q in hc_pq]
    bucket_sorted = [q["k_bucket"] for q in hc_pq]
    colors = [BUCKET_COLORS[b] for b in bucket_sorted]

    hc_mean_k = np.mean(hc_ks)
    baseline_k = int(round(hc_mean_k))

    fig, ax = plt.subplots(1, 1, figsize=(10, 18))
    y = np.arange(len(subcats))

    ax.barh(y, true_ks, height=0.4, color=colors, alpha=0.3,
            edgecolor="black", linewidth=0.2, label="True K")
    ax.scatter(hc_ks, y, s=25, color="crimson", zorder=3,
               edgecolors="black", linewidth=0.4, label=f"HC K")
    ax.axvline(x=baseline_k, color="gray", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Baseline k={baseline_k} (matched mean)")

    ax.set_yticks(y)
    ax.set_yticklabels(subcats, fontsize=4)
    ax.set_xlabel("K (documents selected)", fontsize=12)

    hc_cfg = hc_results[best_hc_key].get("config", {})
    ax.set_title(f"Retrieval: HC Adapts K Per Query ({best_hc_key})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_dir / "6_k_selection_per_query.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  6_k_selection_per_query.png")


# =============================================================================
# Plot 7: HC Outcome (3-panel summary)
# =============================================================================
def plot_hc_outcome(baseline_results, hc_results, best_hc_key,
                    queries, qrels, vector_db, global_null, model, out_dir,
                    max_candidates=1000, hc_gamma=0.05):
    """3-panel: True K vs HC K scatter, pooled p-values, recall by bucket."""
    from hc.higher_criticism import HigherCriticism

    hc_data = hc_results[best_hc_key]
    per_query = hc_data["per_query"]

    true_ks = np.array([q["true_k"] for q in per_query])
    hc_ks = np.array([q["hc_k"] for q in per_query])
    buckets = [q["k_bucket"] for q in per_query]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # Left: True K vs HC K
    ax = axes[0]
    for bucket in ["small", "medium", "large"]:
        mask = np.array([b == bucket for b in buckets])
        if mask.any():
            ax.scatter(true_ks[mask], hc_ks[mask], s=50, alpha=0.7,
                       color=BUCKET_COLORS[bucket], label=bucket,
                       edgecolors="black", linewidth=0.5)
    k_max = max(true_ks.max(), hc_ks.max()) + 5
    ax.plot([0, k_max], [0, k_max], "k--", linewidth=1.5, alpha=0.4, label="Perfect")
    ax.set_xlabel("True K", fontsize=12)
    ax.set_ylabel("HC K", fontsize=12)
    corr = np.corrcoef(true_ks, hc_ks)[0, 1] if len(set(hc_ks)) > 1 else 0
    ax.set_title(f"True K vs HC K (r={corr:.3f})", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # Middle: Pooled p-values (sample 10 queries for speed)
    ax = axes[1]
    np.random.seed(42)
    sample_indices = np.random.choice(len(queries), min(10, len(queries)), replace=False)
    all_p_nonrel, all_p_rel = [], []

    for i in sample_indices:
        q = queries[i]
        relevant_set = qrels.get(q.query_id, set())
        query_emb = model.embed_query(q.text)
        candidates = vector_db.search(query_emb, k=max_candidates)
        cand_sims = np.array([c.similarity for c in candidates])
        cand_ids = [c.doc_id for c in candidates]
        is_rel = np.array([d in relevant_set for d in cand_ids])

        mu = np.mean(cand_sims)
        sigma = np.std(cand_sims)
        z_scores = (cand_sims - mu) / sigma if sigma > 0 else np.zeros_like(cand_sims)
        hc_mod = HigherCriticism(null_distribution=global_null)
        p_vals = hc_mod.compute_p_values(z_scores)

        all_p_nonrel.extend(p_vals[~is_rel])
        all_p_rel.extend(p_vals[is_rel])

    all_p_nonrel = np.array(all_p_nonrel)
    all_p_rel = np.array(all_p_rel)

    p_bins = np.linspace(0, 1, 21)
    ks_stat, ks_pvalue = stats.kstest(all_p_nonrel, "uniform")
    ax.hist(all_p_nonrel, bins=p_bins, density=True, alpha=0.5, color="steelblue",
            edgecolor="black", linewidth=0.5,
            label=f"Non-relevant (n={len(all_p_nonrel)})")
    ax.hist(all_p_rel, bins=p_bins, density=True, alpha=0.5, color="crimson",
            edgecolor="black", linewidth=0.5,
            label=f"Relevant (n={len(all_p_rel)})")
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("p-value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Pooled p-values (10 queries)\nKS: stat={ks_stat:.3f}, p={ks_pvalue:.3f}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    if ks_pvalue > 0.05:
        ax.text(0.5, 0.92, "PASS: non-rel p-values are uniform",
                transform=ax.transAxes, ha="center", fontsize=10, color="green",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

    # Right: Recall by bucket
    ax = axes[2]
    hc_mean_k = hc_data["aggregate"]["mean_k"]
    matched_k = min(baseline_results.keys(), key=lambda k: abs(int(k) - hc_mean_k))

    bucket_order = ["small", "medium", "large"]
    x_pos = np.arange(len(bucket_order))
    width = 0.35

    b_bucket = {}
    for q in baseline_results[matched_k]["per_query"]:
        b_bucket.setdefault(q["k_bucket"], []).append(q["recall"])
    h_bucket = {}
    for q in per_query:
        h_bucket.setdefault(q["k_bucket"], []).append(q["recall"])

    b_means = [np.mean(b_bucket.get(b, [0])) for b in bucket_order]
    h_means = [np.mean(h_bucket.get(b, [0])) for b in bucket_order]

    ax.bar(x_pos - width / 2, b_means, width, color="lightgray", edgecolor="black",
           linewidth=0.5, label=f"Baseline k={matched_k}")
    ax.bar(x_pos + width / 2, h_means, width, color="crimson", edgecolor="black",
           linewidth=0.5, alpha=0.7, label=f"HC")

    for i in range(len(bucket_order)):
        ax.text(x_pos[i] - width / 2, b_means[i] + 0.003, f"{b_means[i]:.3f}",
                ha="center", fontsize=8)
        ax.text(x_pos[i] + width / 2, h_means[i] + 0.003, f"{h_means[i]:.3f}",
                ha="center", fontsize=8, color="crimson")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([b.capitalize() for b in bucket_order], fontsize=11)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Recall by K Bucket", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    y_max = max(max(b_means), max(h_means)) + 0.03
    ax.set_ylim(0, min(y_max + 0.05, 1.15))
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle(f"Retrieval: HC Outcome — {best_hc_key}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "7_hc_outcome.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  7_hc_outcome.png")


# =============================================================================
# Main
# =============================================================================
def main():
    from embeddings.embedding_model import EmbeddingModel
    from embeddings.vector_database import VectorDatabase
    from hc.null_distribution import NullDistribution
    from data_loader import load_amazon_dataset

    out_dir = RESULTS_DIR / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Amazon Categories: Full Plot Suite (matching category-key style)")
    print("=" * 70)

    # Load pre-computed results
    print("\nLoading results...")
    results = load_results()
    baseline = results.get("baseline", {})
    hc = results.get("hc", {})

    if not baseline or not hc:
        print("ERROR: Missing baseline or HC results. Run experiments first.")
        return

    best_hc_key, best_hc_f1 = find_best_hc(hc)
    best_hc_cfg = hc[best_hc_key].get("config", {})
    print(f"  Best HC: {best_hc_key} (F1={best_hc_f1:.4f})")
    print(f"  Baseline k values: {sorted(baseline.keys(), key=int)}")

    # Plots 0, 3, 4, 5, 6 only need JSON data
    print("\nGenerating plots from JSON data...")
    plot_k_distribution(hc, best_hc_key, out_dir)
    plot_recall_precision_tradeoff(baseline, hc, best_hc_key, out_dir)
    plot_per_bucket_recall(baseline, hc, best_hc_key, out_dir)
    plot_per_query_recall(baseline, hc, best_hc_key, out_dir)
    plot_k_selection(hc, best_hc_key, out_dir)

    # Plots 1, 2, 7 need live computation (vector DB + null + embeddings)
    print("\nLoading models for live computation (plots 1, 2, 7)...")
    db_path = str(DATASET_DIR / "amazon_categories_vector_db")
    null_path = str(DATASET_DIR / "null_distributions" / "amazon_global_null")

    print("  Loading dataset...")
    queries, qrels = load_amazon_dataset(str(DATASET_DIR))
    print("  Loading vector database...")
    vector_db = VectorDatabase.load(db_path)
    print("  Loading global null distribution...")
    global_null = NullDistribution.load(null_path)
    print("  Loading embedding model...")
    model = EmbeddingModel(model_name=EmbeddingModel.BGE_MODEL, normalize_embeddings=True)

    max_candidates = best_hc_cfg.get("max_candidates", 1000)
    hc_gamma = best_hc_cfg.get("gamma", 0.05)

    print("\nGenerating plots with live computation...")
    plot_null_calibration(queries, qrels, vector_db, global_null, model, out_dir)
    plot_candidate_separation(queries, qrels, vector_db, global_null, model, out_dir,
                               max_candidates=max_candidates, hc_gamma=hc_gamma)
    plot_hc_outcome(baseline, hc, best_hc_key, queries, qrels, vector_db, global_null, model,
                    out_dir, max_candidates=max_candidates, hc_gamma=hc_gamma)

    print(f"\nAll 8 plots saved to: {out_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
