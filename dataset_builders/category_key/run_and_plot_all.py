"""
Run HC and Baseline experiments, generate all plots in one place.

HC gamma is set so that max retrievable K = max true K in the dataset.
All outputs go to results/category_key/.
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embeddings.embedding_model import EmbeddingModel
from embeddings.vector_database import VectorDatabase
from hc.null_distribution import QueryNullDistributions
from hc.higher_criticism import HigherCriticism
from retrieval.hc_retrieval import HCRetrieval
from retrieval.baseline_retrieval import BaselineRetrieval
from evaluation.evaluator import RetrievalEvaluator

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_category_key_dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =============================================================================
# Experiment runners
# =============================================================================

def run_baseline(k, queries, qrels, vector_db, model, evaluator):
    retriever = BaselineRetrieval(vector_db=vector_db, k=k)
    per_query = []
    for q in queries:
        emb = model.embed_query(q.text)
        out = retriever.retrieve(q.query_id, emb)
        rel = qrels.get(q.query_id, set())
        r = evaluator.evaluate_single(
            query_id=q.query_id, retrieved_ids=out.retrieved_ids,
            retrieved_scores=out.retrieved_scores, relevant_ids=rel,
            metadata={"country": q.country, "size_bucket": q.size_bucket, "true_k": len(rel)}
        )
        per_query.append(r)
    agg = evaluator.evaluate_batch(per_query)
    return agg, per_query


def run_hc(gamma, queries, qrels, vector_db, null_dists, model, evaluator, max_candidates=500):
    retriever = HCRetrieval(
        vector_db=vector_db, query_null_distributions=null_dists,
        gamma=gamma, max_candidates=max_candidates, min_hc=0.0,
        allow_empty=False, embedding_model=model
    )
    per_query = []
    for q in queries:
        out = retriever.retrieve_from_text(q.query_id, q.text)
        rel = qrels.get(q.query_id, set())
        r = evaluator.evaluate_single(
            query_id=q.query_id, retrieved_ids=out.retrieved_ids,
            retrieved_scores=out.retrieved_scores, relevant_ids=rel,
            metadata={"country": q.country, "size_bucket": q.size_bucket,
                      "true_k": len(rel), "hc_k": out.k, "threshold": out.threshold}
        )
        per_query.append(r)
    agg = evaluator.evaluate_batch(per_query)
    return agg, per_query


def serialize_results(agg, per_query, method_key):
    pq_data = []
    for r in per_query:
        d = {
            "query_id": r.query_id, "k": r.k, "true_k": len(r.relevant_ids),
            "recall": r.recall_at_k, "precision": r.precision_at_k,
            "mrr": r.reciprocal_rank, "ndcg": r.ndcg_at_k,
            "hit_rate": r.hit_rate, "map": r.average_precision,
            "country": r.metadata.get("country", ""),
            "size_bucket": r.metadata.get("size_bucket", ""),
            "retrieved_ids": r.retrieved_ids,
        }
        if "hc_k" in r.metadata:
            d["hc_k"] = r.metadata["hc_k"]
            d["threshold"] = r.metadata["threshold"]
        pq_data.append(d)
    return {
        "aggregate": {
            "recall": agg.mean_recall_at_k_labeled,
            "precision": agg.mean_precision_at_k_labeled,
            "mrr": agg.mean_reciprocal_rank_labeled,
            "ndcg": agg.mean_ndcg_at_k_labeled,
            "hit_rate": agg.mean_hit_rate_labeled,
            "map": agg.mean_average_precision_labeled,
            "mean_k": agg.mean_k, "min_k": agg.min_k, "max_k": agg.max_k,
        },
        "per_query": pq_data,
    }


# =============================================================================
# Null validation helpers
# =============================================================================

def compute_full_corpus_pvalues(query, vector_db, null_dist, model, qrels, doc_embeddings,
                                max_candidates=500, hc_gamma=0.048):
    """Compute p-values for ALL docs (calibration) and top-N (runtime)."""
    relevant_set = qrels.get(query.query_id, set())
    query_embedding = model.embed_query(query.text)

    # Full corpus
    all_sims = np.dot(doc_embeddings, query_embedding)
    hc_mod = HigherCriticism(null_distribution=null_dist)
    all_pvals = hc_mod.compute_p_values(all_sims)
    is_rel_full = np.array([d in relevant_set for d in vector_db.doc_ids])

    # Top-N candidates
    candidates = vector_db.search(query_embedding, k=max_candidates)
    cand_ids = [c.doc_id for c in candidates]
    cand_sims = np.array([c.similarity for c in candidates])
    cand_pvals = hc_mod.compute_p_values(cand_sims)
    hc_result = hc_mod.compute_hc_threshold(cand_sims, gamma=hc_gamma, min_hc=0.0, allow_empty=False)
    is_rel_cand = np.array([d in relevant_set for d in cand_ids])

    return {
        "query_id": query.query_id, "country": query.country,
        "size_bucket": query.size_bucket, "true_k": len(relevant_set),
        "p_nonrel_full": all_pvals[~is_rel_full], "p_rel_full": all_pvals[is_rel_full],
        "candidate_ids": cand_ids, "candidate_sims": cand_sims,
        "candidate_pvals": cand_pvals, "is_relevant_cand": is_rel_cand,
        "hc_k": hc_result.k, "hc_threshold": hc_result.threshold,
    }


# =============================================================================
# Plotting
# =============================================================================

BUCKET_COLORS = {"small": "royalblue", "medium": "darkorange", "large": "forestgreen"}


def plot_k_distribution(queries, qrels, out_dir):
    """Plot 0: True K distribution."""
    true_ks, buckets, countries = [], [], []
    for q in queries:
        k = len(qrels.get(q.query_id, set()))
        true_ks.append(k)
        buckets.append(q.size_bucket)
        countries.append(q.country)
    true_ks = np.array(true_ks)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
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
    ax.set_title("Distribution of True K Across Queries", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)

    # Sorted bar chart
    ax2 = axes[1]
    idx = np.argsort(true_ks)
    sorted_countries = [countries[i] for i in idx]
    sorted_ks = true_ks[idx]
    sorted_colors = [BUCKET_COLORS[buckets[i]] for i in idx]
    ax2.barh(range(len(sorted_countries)), sorted_ks, color=sorted_colors,
             edgecolor="black", linewidth=0.3)
    ax2.set_yticks(range(len(sorted_countries)))
    ax2.set_yticklabels(sorted_countries, fontsize=7)
    ax2.set_xlabel("True K", fontsize=12)
    ax2.set_title("True K Per Country (sorted)", fontsize=12, fontweight="bold")
    legend_elements = [Patch(facecolor=BUCKET_COLORS[b], edgecolor="black", label=b.capitalize())
                      for b in ["small", "medium", "large"]]
    ax2.legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.suptitle("Category-Key Dataset: Variable K Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "0_k_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  0_k_distribution.png")


def plot_null_calibration(val_data, out_dir):
    """Plot 1: Null calibration — p-values over full corpus."""
    by_bucket = {}
    for qd in val_data:
        by_bucket.setdefault(qd["size_bucket"], []).append(qd)
    examples = []
    for bucket in ["small", "medium", "large"]:
        if bucket in by_bucket:
            qs = sorted(by_bucket[bucket], key=lambda x: x["true_k"])
            examples.append(qs[len(qs) // 2])

    fig, axes = plt.subplots(2, len(examples), figsize=(5.5 * len(examples), 9))
    if len(examples) == 1:
        axes = axes.reshape(-1, 1)
    bins = np.linspace(0, 1, 21)

    for col, qd in enumerate(examples):
        ax = axes[0, col]
        ax.hist(qd["p_nonrel_full"], bins=bins, density=True, alpha=0.7,
                color="steelblue", edgecolor="black", linewidth=0.5,
                label=f"Non-relevant (n={len(qd['p_nonrel_full'])})")
        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, label="Uniform(0,1)")
        ks_s, ks_p = stats.kstest(qd["p_nonrel_full"], "uniform")
        ax.set_xlabel("p-value", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f'{qd["country"]} ({qd["size_bucket"]}, K={qd["true_k"]})\n'
                    f"KS stat={ks_s:.3f}, p={ks_p:.3f}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 2.0)

        ax2 = axes[1, col]
        ax2.hist(qd["p_rel_full"], bins=bins, density=True, alpha=0.7,
                color="crimson", edgecolor="black", linewidth=0.5,
                label=f"Relevant (n={len(qd['p_rel_full'])})")
        ax2.axvline(x=0.05, color="green", linestyle="--", linewidth=1.5, label="p = 0.05")
        pct = np.mean(qd["p_rel_full"] < 0.05) * 100
        ax2.set_xlabel("p-value", fontsize=11)
        ax2.set_ylabel("Density", fontsize=11)
        ax2.set_title(f"Relevant docs: {pct:.0f}% have p < 0.05", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.set_xlim(-0.02, 1.02)

    fig.suptitle("Plot 1: Null Calibration \u2014 P-values Over Full Corpus\n"
                "(Top: non-relevant should be uniform | Bottom: relevant should be near 0)",
                fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_dir / "1_null_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  1_null_calibration.png")


def plot_candidate_separation(val_data, out_dir, max_candidates=500):
    """Plot 2: What HC sees at runtime — top-N candidates."""
    by_bucket = {}
    for qd in val_data:
        by_bucket.setdefault(qd["size_bucket"], []).append(qd)
    examples = []
    for bucket in ["small", "medium", "large"]:
        if bucket in by_bucket:
            qs = sorted(by_bucket[bucket], key=lambda x: x["true_k"])
            examples.append(qs[len(qs) // 2])

    fig, axes = plt.subplots(2, len(examples), figsize=(5.5 * len(examples), 9))
    if len(examples) == 1:
        axes = axes.reshape(-1, 1)

    for col, qd in enumerate(examples):
        sims = qd["candidate_sims"]
        is_rel = qd["is_relevant_cand"]
        hc_k = qd["hc_k"]
        true_k = qd["true_k"]

        # --- Top row: waterfall ---
        ax = axes[0, col]
        ranks = np.arange(1, len(sims) + 1)
        ax.scatter(ranks[~is_rel], sims[~is_rel], s=18, color="steelblue", alpha=0.5, label="Non-relevant", zorder=2)
        ax.scatter(ranks[is_rel], sims[is_rel], s=45, color="crimson", alpha=0.85, label="Relevant", zorder=3,
                  edgecolors="black", linewidth=0.5)

        if 0 < hc_k < len(sims):
            ax.axvline(x=hc_k + 0.5, color="green", linewidth=2.5, label=f"HC cut (k={hc_k})", zorder=4)

        ax.set_xlabel("Candidate rank", fontsize=11)
        ax.set_ylabel("Cosine similarity", fontsize=11)
        ax.set_title(f'{qd["country"]}: true K={true_k}, HC K={hc_k}', fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")

        # --- Bottom row: similarity distributions ---
        ax2 = axes[1, col]
        sim_rel = sims[is_rel]
        sim_nonrel = sims[~is_rel]
        bins_sim = np.linspace(sims.min() - 0.01, sims.max() + 0.01, 25)
        if len(sim_nonrel) > 0:
            ax2.hist(sim_nonrel, bins=bins_sim, alpha=0.5, color="steelblue", edgecolor="black", linewidth=0.5,
                    label=f"Non-relevant (n={len(sim_nonrel)})")
        if len(sim_rel) > 0:
            ax2.hist(sim_rel, bins=bins_sim, alpha=0.6, color="crimson", edgecolor="black", linewidth=0.5,
                    label=f"Relevant (n={len(sim_rel)})")

        if 0 < hc_k < len(sims):
            ax2.axvline(x=qd["hc_threshold"], color="green", linewidth=2.5,
                       label=f'HC threshold={qd["hc_threshold"]:.3f}')
        ax2.set_xlabel("Cosine similarity", fontsize=11)
        ax2.set_ylabel("Count", fontsize=11)
        ax2.set_title("Similarity distributions of candidates", fontsize=10)
        ax2.legend(fontsize=8)

    fig.suptitle(f"Plot 2: What HC Sees at Runtime \u2014 Top-{max_candidates} Candidates\n"
                "(Top: ranked candidates | Bottom: similarity distributions)",
                fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_dir / "2_candidate_separation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  2_candidate_separation.png")


def plot_recall_precision_tradeoff(baseline_results, hc_result, hc_gamma, matched_baseline_k, out_dir):
    """Plot 3: Bar chart comparing HC vs all baseline k values on key metrics."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    h_agg = hc_result["aggregate"]
    h_f1 = 2 * h_agg["recall"] * h_agg["precision"] / (h_agg["recall"] + h_agg["precision"])

    baseline_ks = ["5", "8", "10", "12", "15", "20"]
    metrics = ["Recall", "Precision", "F1", "NDCG", "MAP"]

    # Build data for all methods
    methods = []
    for k in baseline_ks:
        a = baseline_results[k]["aggregate"]
        f1 = 2 * a["recall"] * a["precision"] / (a["recall"] + a["precision"])
        methods.append((f"k={k}", [a["recall"], a["precision"], f1, a["ndcg"], a["map"]]))
    methods.append(("HC", [h_agg["recall"], h_agg["precision"], h_f1, h_agg["ndcg"], h_agg["map"]]))

    n_methods = len(methods)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    total_width = 0.82
    bar_width = total_width / n_methods

    baseline_colors = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#76b7b2", "#b07aa1"]
    hc_color = "#1b1b1b"

    for i, (label, vals) in enumerate(methods):
        offset = (i - n_methods / 2 + 0.5) * bar_width
        is_hc = (label == "HC")
        color = hc_color if is_hc else baseline_colors[i]
        edgecolor = "black"
        lw = 1.5 if is_hc else 0.6
        hatch = "///" if is_hc else None

        bars = ax.bar(x + offset, vals, bar_width, color=color, edgecolor=edgecolor,
                      linewidth=lw, alpha=0.85, label=label, zorder=3 if is_hc else 2, hatch=hatch)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=6.5,
                    color=color if not is_hc else "black",
                    fontweight="bold", rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)
    ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.08),
             ncol=n_methods, frameon=True, edgecolor="gray", fancybox=True)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title(f"HC (\u03b3={hc_gamma}, adaptive K, mean K={h_agg['mean_k']:.0f}) vs All Baseline Top-K",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "3_recall_precision_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  3_recall_precision_tradeoff.png")


def plot_per_bucket_recall(baseline_results, hc_result, hc_gamma, matched_baseline_k, out_dir):
    """Plot 4: Per-bucket recall at matched mean K."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    bucket_order = ["small", "medium", "large"]
    x = np.arange(len(bucket_order))
    width = 0.35

    b_bucket = {}
    for q in baseline_results[matched_baseline_k]["per_query"]:
        b_bucket.setdefault(q["size_bucket"], []).append(q["recall"])
    h_bucket = {}
    for q in hc_result["per_query"]:
        h_bucket.setdefault(q["size_bucket"], []).append(q["recall"])

    b_means = [np.mean(b_bucket.get(b, [0])) for b in bucket_order]
    h_means = [np.mean(h_bucket.get(b, [0])) for b in bucket_order]
    b_stds = [np.std(b_bucket.get(b, [0])) for b in bucket_order]
    h_stds = [np.std(h_bucket.get(b, [0])) for b in bucket_order]

    ax.bar(x - width/2, b_means, width, yerr=b_stds, capsize=4,
           color="lightgray", edgecolor="black", linewidth=0.5,
           label=f"Baseline k={matched_baseline_k}")
    ax.bar(x + width/2, h_means, width, yerr=h_stds, capsize=4,
           color="crimson", edgecolor="black", linewidth=0.5, alpha=0.7,
           label=f"HC \u03b3={hc_gamma}")

    for i in range(len(bucket_order)):
        ax.text(x[i] - width/2, b_means[i] + b_stds[i] + 0.02,
               f"{b_means[i]:.2f}", ha="center", fontsize=9, color="gray")
        ax.text(x[i] + width/2, h_means[i] + h_stds[i] + 0.02,
               f"{h_means[i]:.2f}", ha="center", fontsize=9, color="crimson")

    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in bucket_order], fontsize=12)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title(f"Recall by Country Size at Matched Mean K\n"
                f"(Baseline k={matched_baseline_k} vs HC \u03b3={hc_gamma})",
                fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_dir / "4_per_bucket_recall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  4_per_bucket_recall.png")


def plot_per_query_recall(baseline_results, hc_result, hc_gamma, matched_baseline_k, out_dir):
    """Plot 5: Per-query scatter — true K vs recall."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    configs = [
        (baseline_results[matched_baseline_k]["per_query"], f"Baseline k={matched_baseline_k}"),
        (hc_result["per_query"], f"HC \u03b3={hc_gamma}"),
    ]

    for idx, (per_query, label) in enumerate(configs):
        ax = axes[idx]
        for q in per_query:
            ax.scatter(q["true_k"], q["recall"], s=50, color=BUCKET_COLORS[q["size_bucket"]],
                      edgecolors="black", linewidth=0.5, alpha=0.7, zorder=3)
        ax.set_xlabel("True K (ground truth)", fontsize=12)
        ax.set_ylabel("Recall", fontsize=12)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
        ax.grid(True, alpha=0.2)
        legend_elements = [Patch(facecolor=BUCKET_COLORS[b], edgecolor="black", label=b.capitalize())
                          for b in ["small", "medium", "large"]]
        ax.legend(handles=legend_elements, fontsize=9, loc="lower left")
        mean_r = np.mean([q["recall"] for q in per_query])
        ax.text(0.95, 0.05, f"Mean recall: {mean_r:.3f}", transform=ax.transAxes, ha="right", fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.suptitle(f"Per-Query Recall: Baseline k={matched_baseline_k} vs HC \u03b3={hc_gamma}",
                fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "5_per_query_recall_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  5_per_query_recall_scatter.png")


def plot_k_selection(hc_result, hc_gamma, out_dir, max_candidates=500):
    """Plot 6: K selection per query — HC adapts, baseline is fixed."""
    hc_pq = sorted(hc_result["per_query"], key=lambda x: x["true_k"])
    countries = [q["country"] for q in hc_pq]
    true_ks = [q["true_k"] for q in hc_pq]
    hc_ks = [q["hc_k"] for q in hc_pq]
    bucket_sorted = [q["size_bucket"] for q in hc_pq]
    colors = [BUCKET_COLORS[b] for b in bucket_sorted]
    y = np.arange(len(countries))

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    ax.barh(y, true_ks, height=0.4, color=colors, alpha=0.3, edgecolor="black", linewidth=0.3, label="True K")
    ax.scatter(hc_ks, y, s=50, color="crimson", zorder=3, edgecolors="black", linewidth=0.5,
              label=f"HC K (\u03b3={hc_gamma})")

    # Find closest baseline k to HC mean
    hc_mean_k = np.mean(hc_ks)
    baseline_k = int(round(hc_mean_k))
    ax.axvline(x=baseline_k, color="gray", linestyle="--", linewidth=1.5, alpha=0.7,
              label=f"Baseline k={baseline_k} (matched mean)")

    ax.set_yticks(y)
    ax.set_yticklabels(countries, fontsize=7)
    ax.set_xlabel("K (documents selected)", fontsize=12)
    ax.set_title(f"HC Adapts K Per Query (\u03b3={hc_gamma}, max K={int(np.floor(hc_gamma * max_candidates))})",
                fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_dir / "6_k_selection_per_query.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  6_k_selection_per_query.png")


def plot_hc_outcome(val_data, baseline_results, hc_result, hc_gamma, matched_baseline_k, out_dir,
                    max_candidates=500):
    """Plot 7: HC outcome — true K vs HC K + pooled p-values + recall by bucket."""
    true_ks = np.array([qd["true_k"] for qd in val_data])
    hc_ks = np.array([qd["hc_k"] for qd in val_data])
    buckets = [qd["size_bucket"] for qd in val_data]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # Left: True K vs HC K
    ax = axes[0]
    for bucket in ["small", "medium", "large"]:
        mask = np.array([b == bucket for b in buckets])
        if mask.any():
            ax.scatter(true_ks[mask], hc_ks[mask], s=60, alpha=0.7, color=BUCKET_COLORS[bucket],
                      label=bucket, edgecolors="black", linewidth=0.5)
    k_min = min(true_ks.min(), hc_ks.min()) - 1
    k_max = max(true_ks.max(), hc_ks.max()) + 1
    ax.plot([k_min, k_max], [k_min, k_max], "k--", linewidth=1.5, alpha=0.4, label="Perfect")
    ax.set_xlabel("True K", fontsize=12)
    ax.set_ylabel("HC K", fontsize=12)
    corr = np.corrcoef(true_ks, hc_ks)[0, 1]
    ax.set_title(f"True K vs HC K (r={corr:.3f})", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(k_min, k_max)
    ax.set_ylim(k_min, k_max)

    # Middle: Pooled p-values
    ax = axes[1]
    all_p_nonrel = np.concatenate([qd["p_nonrel_full"] for qd in val_data])
    all_p_rel = np.concatenate([qd["p_rel_full"] for qd in val_data])
    ks_stat, ks_pvalue = stats.kstest(all_p_nonrel, "uniform")
    bins = np.linspace(0, 1, 21)
    ax.hist(all_p_nonrel, bins=bins, density=True, alpha=0.5, color="steelblue", edgecolor="black", linewidth=0.5,
            label=f"Non-relevant (n={len(all_p_nonrel)})")
    ax.hist(all_p_rel, bins=bins, density=True, alpha=0.5, color="crimson", edgecolor="black", linewidth=0.5,
            label=f"Relevant (n={len(all_p_rel)})")
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("p-value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Pooled p-values\nKS: stat={ks_stat:.3f}, p={ks_pvalue:.3f}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    if ks_pvalue > 0.05:
        ax.text(0.5, 0.92, "PASS: non-rel p-values are uniform", transform=ax.transAxes, ha="center", fontsize=10,
               color="green", fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

    # Right: Recall by bucket (HC vs best-matched baseline)
    ax = axes[2]
    bucket_order = ["small", "medium", "large"]
    x_pos = np.arange(len(bucket_order))
    width = 0.35

    b_bucket = {}
    for q in baseline_results[matched_baseline_k]["per_query"]:
        b_bucket.setdefault(q["size_bucket"], []).append(q["recall"])
    h_bucket = {}
    for q in hc_result["per_query"]:
        h_bucket.setdefault(q["size_bucket"], []).append(q["recall"])

    b_means = [np.mean(b_bucket.get(b, [0])) for b in bucket_order]
    h_means = [np.mean(h_bucket.get(b, [0])) for b in bucket_order]

    ax.bar(x_pos - width/2, b_means, width, color="lightgray", edgecolor="black", linewidth=0.5,
           label=f"Baseline k={matched_baseline_k}")
    ax.bar(x_pos + width/2, h_means, width, color="crimson", edgecolor="black", linewidth=0.5, alpha=0.7,
           label=f"HC \u03b3={hc_gamma}")

    for i in range(len(bucket_order)):
        ax.text(x_pos[i] - width/2, b_means[i] + 0.02, f"{b_means[i]:.2f}", ha="center", fontsize=8)
        ax.text(x_pos[i] + width/2, h_means[i] + 0.02, f"{h_means[i]:.2f}", ha="center", fontsize=8, color="crimson")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([b.capitalize() for b in bucket_order], fontsize=11)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Recall by Country Size", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)

    fig.suptitle(f"HC Outcome: \u03b3={hc_gamma} (max K={int(np.floor(hc_gamma*max_candidates))})",
                fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "7_hc_outcome.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  7_hc_outcome.png")


# =============================================================================
# Main
# =============================================================================

def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent.parent / "datasets" / "category_key"
    db_path = str(data_dir / "category_key_vector_db")
    null_path = str(data_dir / "null_distributions" / "category_key_per_query_null")

    # All outputs go here
    out_dir = PROJECT_ROOT / "results" / "category_key"

    # Clean previous results
    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Category-Key: Full Experiment + Plots")
    print("=" * 70)

    # --- Load everything ---
    print("\nLoading dataset...")
    queries, qrels = load_category_key_dataset(str(data_dir))

    print("Loading vector database...")
    vector_db = VectorDatabase.load(db_path)

    print("Loading null distributions...")
    null_dists = QueryNullDistributions.load(null_path)

    print("Loading embedding model...")
    model = EmbeddingModel(model_name=EmbeddingModel.BGE_MODEL, normalize_embeddings=True)

    evaluator = RetrievalEvaluator()

    # --- Determine gamma ---
    max_true_k = max(len(qrels[q.query_id]) for q in queries)
    max_candidates = 500
    hc_gamma = max_true_k / max_candidates  # 24/500 = 0.048
    max_hc_k = int(np.floor(hc_gamma * max_candidates))
    print(f"\nMax true K = {max_true_k}, max_candidates = {max_candidates}")
    print(f"Setting gamma = {hc_gamma} (max HC K = {max_hc_k})")

    # --- Run baseline ---
    baseline_ks = ["5", "8", "10", "12", "15", "20"]
    baseline_results = {}
    print(f"\nRunning baseline at k={baseline_ks}...")
    for k_str in baseline_ks:
        k = int(k_str)
        agg, pq = run_baseline(k, queries, qrels, vector_db, model, evaluator)
        baseline_results[k_str] = serialize_results(agg, pq, f"baseline_k{k}")
        print(f"  k={k}: recall={agg.mean_recall_at_k_labeled:.4f}, precision={agg.mean_precision_at_k_labeled:.4f}")

    # --- Run HC at primary gamma ---
    print(f"\nRunning HC at gamma={hc_gamma}...")
    hc_agg, hc_pq = run_hc(hc_gamma, queries, qrels, vector_db, null_dists, model, evaluator, max_candidates)
    hc_result = serialize_results(hc_agg, hc_pq, f"hc_gamma{hc_gamma}")
    print(f"  HC: recall={hc_agg.mean_recall_at_k_labeled:.4f}, precision={hc_agg.mean_precision_at_k_labeled:.4f}, "
          f"mean_k={hc_agg.mean_k:.1f}, range={hc_agg.min_k}-{hc_agg.max_k}")


    # --- Find matched baseline k ---
    hc_mean_k = hc_result["aggregate"]["mean_k"]
    matched_baseline_k = min(baseline_ks, key=lambda k: abs(int(k) - hc_mean_k))
    print(f"  HC mean K = {hc_mean_k:.1f}, matched baseline k = {matched_baseline_k}")

    # --- Null validation data ---
    print("\nComputing null validation data...")
    doc_embeddings = vector_db.index.reconstruct_n(0, len(vector_db.doc_ids))
    val_data = []
    for i, q in enumerate(queries):
        nd = null_dists.get(q.query_id)
        vd = compute_full_corpus_pvalues(q, vector_db, nd, model, qrels, doc_embeddings,
                                         max_candidates=max_candidates, hc_gamma=hc_gamma)
        val_data.append(vd)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(queries)}] {vd['country']}: true_k={vd['true_k']}, HC_k={vd['hc_k']}")

    # --- Save raw results ---
    print("\nSaving results...")
    with open(out_dir / "baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)
    with open(out_dir / "hc_results.json", "w") as f:
        json.dump({"gamma": hc_gamma, "max_hc_k": max_hc_k, "results": hc_result}, f, indent=2)

    # Validation report
    all_p_nonrel = np.concatenate([qd["p_nonrel_full"] for qd in val_data])
    all_p_rel = np.concatenate([qd["p_rel_full"] for qd in val_data])
    ks_stat, ks_pvalue = stats.kstest(all_p_nonrel, "uniform")
    n_pass = sum(1 for qd in val_data if stats.kstest(qd["p_nonrel_full"], "uniform")[1] > 0.05)
    true_ks_arr = np.array([qd["true_k"] for qd in val_data])
    hc_ks_arr = np.array([qd["hc_k"] for qd in val_data])
    corr = np.corrcoef(true_ks_arr, hc_ks_arr)[0, 1]

    report = {
        "gamma": hc_gamma, "max_hc_k": max_hc_k, "max_true_k": max_true_k,
        "null_calibration": {
            "pooled_ks_stat": float(ks_stat), "pooled_ks_pvalue": float(ks_pvalue),
            "nonrel_mean": float(np.mean(all_p_nonrel)), "nonrel_std": float(np.std(all_p_nonrel)),
            "per_query_pass_rate": float(n_pass / len(val_data)), "n_nonrel": len(all_p_nonrel),
        },
        "signal_detection": {
            "rel_mean": float(np.mean(all_p_rel)), "rel_median": float(np.median(all_p_rel)),
            "pct_below_005": float(np.mean(all_p_rel < 0.05) * 100), "n_rel": len(all_p_rel),
        },
        "hc_accuracy": {
            "mean_true_k": float(np.mean(true_ks_arr)), "mean_hc_k": float(np.mean(hc_ks_arr)),
            "k_correlation": float(corr), "mean_k_error": float(np.mean(hc_ks_arr - true_ks_arr)),
        },
        "comparison": {
            "hc_recall": hc_result["aggregate"]["recall"],
            "hc_precision": hc_result["aggregate"]["precision"],
            "matched_baseline_k": int(matched_baseline_k),
            "baseline_recall": baseline_results[matched_baseline_k]["aggregate"]["recall"],
            "baseline_precision": baseline_results[matched_baseline_k]["aggregate"]["precision"],
        },
    }
    with open(out_dir / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # --- Generate all plots ---
    print("\nGenerating plots...")
    plot_k_distribution(queries, qrels, out_dir)
    plot_null_calibration(val_data, out_dir)
    plot_candidate_separation(val_data, out_dir, max_candidates=max_candidates)
    plot_recall_precision_tradeoff(baseline_results, hc_result, hc_gamma, matched_baseline_k, out_dir)
    plot_per_bucket_recall(baseline_results, hc_result, hc_gamma, matched_baseline_k, out_dir)
    plot_per_query_recall(baseline_results, hc_result, hc_gamma, matched_baseline_k, out_dir)
    plot_k_selection(hc_result, hc_gamma, out_dir, max_candidates=max_candidates)
    plot_hc_outcome(val_data, baseline_results, hc_result, hc_gamma, matched_baseline_k, out_dir,
                    max_candidates=max_candidates)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  HC gamma = {hc_gamma} (max K = {max_hc_k}, matching max true K = {max_true_k})")
    print(f"\n  Null calibration: KS stat={ks_stat:.4f}, p={ks_pvalue:.4f} "
          f"({'PASS' if ks_pvalue > 0.05 else 'FAIL'})")
    print(f"  Per-query pass rate: {n_pass}/{len(val_data)} ({n_pass/len(val_data)*100:.0f}%)")
    print(f"  Signal detection: {np.mean(all_p_rel < 0.05)*100:.1f}% of relevant docs have p < 0.05")
    print(f"\n  HC vs Baseline (matched at k={matched_baseline_k}):")
    print(f"    {'Method':<20} {'Recall':<10} {'Precision':<10} {'Mean K':<10}")
    print(f"    {'-'*50}")
    print(f"    {'Baseline k='+matched_baseline_k:<20} "
          f"{baseline_results[matched_baseline_k]['aggregate']['recall']:<10.4f} "
          f"{baseline_results[matched_baseline_k]['aggregate']['precision']:<10.4f} "
          f"{baseline_results[matched_baseline_k]['aggregate']['mean_k']:<10.1f}")
    print(f"    {'HC g='+str(hc_gamma):<20} "
          f"{hc_result['aggregate']['recall']:<10.4f} "
          f"{hc_result['aggregate']['precision']:<10.4f} "
          f"{hc_result['aggregate']['mean_k']:<10.1f}")
    print(f"\n  HC K correlation with true K: {corr:.4f}")
    print(f"  HC K range: {hc_agg.min_k}-{hc_agg.max_k}")

    print(f"\nAll outputs saved to: {out_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
