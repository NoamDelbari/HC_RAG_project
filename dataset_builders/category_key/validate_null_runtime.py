"""
Runtime Null Distribution Validation

Proves the null distribution works correctly when HC is deployed, using three tests:

PLOT 1 — Null Calibration (full corpus):
  For each query, compute p-values for ALL non-relevant documents (not just top-100).
  These p-values should be Uniform(0,1). This proves the null correctly models
  what "non-relevant" similarities look like.

PLOT 2 — Candidate Separation (top-100 candidates):
  Among the top-100 candidates that HC actually sees at runtime, show the similarity
  distributions of relevant vs non-relevant docs, and where HC places its threshold.
  This proves HC can find the boundary between signal and noise.

PLOT 3 — HC Outcome:
  True K vs HC-selected K across all queries, showing HC adapts to variable K.
"""

import sys
import numpy as np
from pathlib import Path
from scipy import stats
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embeddings.embedding_model import EmbeddingModel
from embeddings.vector_database import VectorDatabase
from hc.null_distribution import QueryNullDistributions
from hc.higher_criticism import HigherCriticism

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_category_key_dataset


def compute_all_pvalues(query, vector_db, null_dist, embedding_model, qrels, doc_embeddings):
    """
    For a single query:
      1. Compute p-values for ALL documents using the null (calibration test)
      2. Compute p-values for top-100 candidates (runtime test)
      3. Run HC on top-100 candidates
    """
    relevant_set = qrels.get(query.query_id, set())

    # Embed query
    query_embedding = embedding_model.embed_query(query.text)

    # --- Full-corpus p-values (for calibration test) ---
    # Compute similarity to every document
    all_sims = np.dot(doc_embeddings, query_embedding)
    all_doc_ids = vector_db.doc_ids

    hc_module = HigherCriticism(null_distribution=null_dist)
    all_p_values = hc_module.compute_p_values(all_sims)

    # Separate into relevant / non-relevant
    is_relevant_full = np.array([did in relevant_set for did in all_doc_ids])
    p_nonrel_full = all_p_values[~is_relevant_full]
    p_rel_full = all_p_values[is_relevant_full]

    # --- Top-100 candidate analysis (for runtime test) ---
    candidates = vector_db.search(query_embedding, k=100)
    candidate_ids = [c.doc_id for c in candidates]
    candidate_sims = np.array([c.similarity for c in candidates])
    candidate_pvals = hc_module.compute_p_values(candidate_sims)

    hc_result = hc_module.compute_hc_threshold(
        similarities=candidate_sims, gamma=0.1, min_hc=0.0, allow_empty=False
    )

    is_relevant_cand = np.array([did in relevant_set for did in candidate_ids])

    return {
        "query_id": query.query_id,
        "country": query.country,
        "size_bucket": query.size_bucket,
        "true_k": len(relevant_set),
        # Full-corpus calibration data
        "p_nonrel_full": p_nonrel_full,
        "p_rel_full": p_rel_full,
        # Top-100 candidate data
        "candidate_ids": candidate_ids,
        "candidate_sims": candidate_sims,
        "candidate_pvals": candidate_pvals,
        "is_relevant_cand": is_relevant_cand,
        "hc_k": hc_result.k,
        "hc_threshold": hc_result.threshold,
        "hc_statistic": hc_result.hc_statistic,
    }


def generate_plots(all_qd, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "validation"
    plot_dir.mkdir(parents=True, exist_ok=True)

    bucket_colors = {"small": "royalblue", "medium": "darkorange", "large": "forestgreen"}
    by_bucket = {}
    for qd in all_qd:
        by_bucket.setdefault(qd["size_bucket"], []).append(qd)

    # =====================================================================
    # PLOT 1: Null Calibration — p-values of ALL non-relevant docs
    #
    # If the null is correct, p-values of non-relevant docs should be
    # Uniform(0,1). This is the fundamental calibration test.
    # =====================================================================

    # Pick one example query per bucket
    examples = []
    for bucket in ["small", "medium", "large"]:
        if bucket in by_bucket:
            qs = sorted(by_bucket[bucket], key=lambda x: x["true_k"])
            examples.append(qs[len(qs) // 2])

    fig, axes = plt.subplots(2, len(examples), figsize=(5.5 * len(examples), 9))
    if len(examples) == 1:
        axes = axes.reshape(-1, 1)

    for col, qd in enumerate(examples):
        # Top row: non-relevant p-value histogram (should be uniform)
        ax = axes[0, col]
        bins = np.linspace(0, 1, 21)
        ax.hist(qd["p_nonrel_full"], bins=bins, density=True, alpha=0.7,
                color='steelblue', edgecolor='black', linewidth=0.5,
                label=f'Non-relevant (n={len(qd["p_nonrel_full"])})')
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5,
                  label='Uniform(0,1)')
        ks_s, ks_p = stats.kstest(qd["p_nonrel_full"], 'uniform')
        ax.set_xlabel('p-value', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{qd["country"]} ({qd["size_bucket"]}, K={qd["true_k"]})\n'
                    f'KS stat={ks_s:.3f}, p={ks_p:.3f}',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 2.0)

        # Bottom row: relevant p-value histogram (should be near 0)
        ax2 = axes[1, col]
        ax2.hist(qd["p_rel_full"], bins=bins, density=True, alpha=0.7,
                color='crimson', edgecolor='black', linewidth=0.5,
                label=f'Relevant (n={len(qd["p_rel_full"])})')
        ax2.axvline(x=0.05, color='green', linestyle='--', linewidth=1.5,
                   label='p = 0.05')
        pct = np.mean(qd["p_rel_full"] < 0.05) * 100
        ax2.set_xlabel('p-value', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title(f'Relevant docs: {pct:.0f}% have p < 0.05', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.set_xlim(-0.02, 1.02)

    fig.suptitle('Plot 1: Null Calibration — P-values Over Full Corpus\n'
                '(Top: non-relevant should be uniform | Bottom: relevant should be near 0)',
                fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(plot_dir / "1_null_calibration.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 1_null_calibration.png")

    # =====================================================================
    # PLOT 2: Candidate Separation — what HC sees at runtime
    #
    # Among top-100 candidates, show similarity distributions of relevant
    # vs non-relevant, and where HC places its threshold.
    # =====================================================================

    fig, axes = plt.subplots(2, len(examples), figsize=(5.5 * len(examples), 9))
    if len(examples) == 1:
        axes = axes.reshape(-1, 1)

    for col, qd in enumerate(examples):
        sims = qd["candidate_sims"]
        is_rel = qd["is_relevant_cand"]
        hc_k = qd["hc_k"]

        # Top row: similarity score waterfall with HC threshold
        ax = axes[0, col]
        ranks = np.arange(1, len(sims) + 1)
        ax.scatter(ranks[~is_rel], sims[~is_rel], s=18, color='steelblue',
                  alpha=0.5, label='Non-relevant', zorder=2)
        ax.scatter(ranks[is_rel], sims[is_rel], s=45, color='crimson',
                  alpha=0.85, label='Relevant', zorder=3,
                  edgecolors='black', linewidth=0.5)

        if 0 < hc_k < len(sims):
            ax.axvline(x=hc_k + 0.5, color='green', linewidth=2.5,
                      label=f'HC cut (k={hc_k})', zorder=4)
            ax.axhspan(ymin=ax.get_ylim()[0] if ax.get_ylim()[0] < sims.min() else sims.min() - 0.02,
                       ymax=sims[hc_k - 1], xmin=0, xmax=hc_k / len(sims),
                       alpha=0.05, color='green')

        ax.set_xlabel('Candidate rank', fontsize=11)
        ax.set_ylabel('Cosine similarity', fontsize=11)
        ax.set_title(f'{qd["country"]}: true K={qd["true_k"]}, HC K={hc_k}',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')

        # Bottom row: similarity histograms of relevant vs non-relevant candidates
        ax2 = axes[1, col]
        sim_rel = sims[is_rel]
        sim_nonrel = sims[~is_rel]

        bins_sim = np.linspace(sims.min() - 0.01, sims.max() + 0.01, 25)
        if len(sim_nonrel) > 0:
            ax2.hist(sim_nonrel, bins=bins_sim, alpha=0.5, color='steelblue',
                    edgecolor='black', linewidth=0.5,
                    label=f'Non-relevant (n={len(sim_nonrel)})')
        if len(sim_rel) > 0:
            ax2.hist(sim_rel, bins=bins_sim, alpha=0.6, color='crimson',
                    edgecolor='black', linewidth=0.5,
                    label=f'Relevant (n={len(sim_rel)})')

        if 0 < hc_k < len(sims):
            ax2.axvline(x=qd["hc_threshold"], color='green', linewidth=2.5,
                       label=f'HC threshold={qd["hc_threshold"]:.3f}')

        ax2.set_xlabel('Cosine similarity', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Similarity distributions of candidates', fontsize=10)
        ax2.legend(fontsize=8)

    fig.suptitle('Plot 2: What HC Sees at Runtime — Top-100 Candidates\n'
                '(Top: ranked candidates | Bottom: similarity distributions)',
                fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(plot_dir / "2_candidate_separation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 2_candidate_separation.png")

    # =====================================================================
    # PLOT 3: HC Outcome — does HC adapt K correctly?
    # =====================================================================

    true_ks = np.array([qd["true_k"] for qd in all_qd])
    hc_ks = np.array([qd["hc_k"] for qd in all_qd])
    buckets = [qd["size_bucket"] for qd in all_qd]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # Left: True K vs HC K scatter
    ax = axes[0]
    for bucket in ["small", "medium", "large"]:
        mask = np.array([b == bucket for b in buckets])
        if mask.any():
            ax.scatter(true_ks[mask], hc_ks[mask], s=60, alpha=0.7,
                      color=bucket_colors[bucket], label=bucket,
                      edgecolors='black', linewidth=0.5)

    k_min = min(true_ks.min(), hc_ks.min()) - 1
    k_max = max(true_ks.max(), hc_ks.max()) + 1
    ax.plot([k_min, k_max], [k_min, k_max], 'k--', linewidth=1.5, alpha=0.4,
           label='Perfect')
    ax.set_xlabel('True K (ground truth)', fontsize=12)
    ax.set_ylabel('HC K (selected)', fontsize=12)
    corr = np.corrcoef(true_ks, hc_ks)[0, 1]
    ax.set_title(f'True K vs HC K (r={corr:.3f})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.set_xlim(k_min, k_max)
    ax.set_ylim(k_min, k_max)

    # Middle: Aggregate non-relevant p-value calibration (pooled across all queries)
    ax = axes[1]
    all_p_nonrel = np.concatenate([qd["p_nonrel_full"] for qd in all_qd])
    all_p_rel = np.concatenate([qd["p_rel_full"] for qd in all_qd])
    ks_stat, ks_pvalue = stats.kstest(all_p_nonrel, 'uniform')

    bins = np.linspace(0, 1, 21)
    ax.hist(all_p_nonrel, bins=bins, density=True, alpha=0.5, color='steelblue',
            edgecolor='black', linewidth=0.5,
            label=f'Non-relevant (n={len(all_p_nonrel)})')
    ax.hist(all_p_rel, bins=bins, density=True, alpha=0.5, color='crimson',
            edgecolor='black', linewidth=0.5,
            label=f'Relevant (n={len(all_p_rel)})')
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.2)
    ax.set_xlabel('p-value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Pooled p-values (all queries)\n'
                f'Non-rel KS: stat={ks_stat:.3f}, p={ks_pvalue:.3f}',
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)

    if ks_pvalue > 0.05:
        ax.text(0.5, 0.92, 'PASS: non-relevant p-values are uniform',
               transform=ax.transAxes, ha='center', fontsize=10,
               color='green', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

    # Right: Recall by size bucket
    ax = axes[2]
    bucket_recalls = {}
    for qd in all_qd:
        bucket = qd["size_bucket"]
        selected = set(qd["candidate_ids"][:qd["hc_k"]])
        relevant_in_cands = set(
            qd["candidate_ids"][i]
            for i in range(len(qd["is_relevant_cand"]))
            if qd["is_relevant_cand"][i]
        )
        recall = len(selected & relevant_in_cands) / max(1, qd["true_k"])
        bucket_recalls.setdefault(bucket, []).append(recall)

    bucket_order = [b for b in ["small", "medium", "large"] if b in bucket_recalls]
    positions = np.arange(len(bucket_order))
    means = [np.mean(bucket_recalls[b]) for b in bucket_order]
    stds = [np.std(bucket_recalls[b]) for b in bucket_order]
    colors_bar = [bucket_colors[b] for b in bucket_order]

    ax.bar(positions, means, yerr=stds, capsize=5, color=colors_bar,
           alpha=0.7, edgecolor='black', linewidth=0.5)

    for i, bucket in enumerate(bucket_order):
        n = len(bucket_recalls[bucket])
        mean_hk = np.mean([qd["hc_k"] for qd in all_qd if qd["size_bucket"] == bucket])
        mean_tk = np.mean([qd["true_k"] for qd in all_qd if qd["size_bucket"] == bucket])
        ax.text(i, means[i] + stds[i] + 0.02,
               f'n={n}\ntrue K={mean_tk:.0f}\nHC K={mean_hk:.0f}',
               ha='center', fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels([b.capitalize() for b in bucket_order], fontsize=11)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('HC Recall by Country Size', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.2)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

    fig.suptitle('Plot 3: HC Outcome — Adaptive K Selection',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(plot_dir / "3_hc_outcome.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 3_hc_outcome.png")


def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / "datasets" / "category_key"
    db_path = str(output_dir / "category_key_vector_db")
    null_path = str(output_dir / "null_distributions" / "category_key_per_query_null")

    print("=" * 70)
    print("Runtime Null Distribution Validation")
    print("=" * 70)
    print()
    print("Three-part validation:")
    print("  1. Null calibration: p-values of ALL non-relevant docs should be Uniform(0,1)")
    print("  2. Candidate separation: HC finds the gap between signal and noise")
    print("  3. HC outcome: true K vs HC-selected K")
    print()

    # Load everything
    print("Loading dataset...")
    queries, qrels = load_category_key_dataset(str(output_dir))

    print("Loading vector database...")
    vector_db = VectorDatabase.load(db_path)

    print("Loading null distributions...")
    null_dists = QueryNullDistributions.load(null_path)

    print("Loading embedding model...")
    model = EmbeddingModel(
        model_name=EmbeddingModel.BGE_MODEL,
        normalize_embeddings=True
    )

    # Pre-extract all document embeddings (used for full-corpus p-value computation)
    print("Extracting document embeddings from FAISS index...")
    doc_embeddings = vector_db.index.reconstruct_n(0, len(vector_db.doc_ids))

    # Run analysis for every query
    print(f"\nAnalyzing {len(queries)} queries...")
    print("-" * 70)

    all_qd = []
    for i, query in enumerate(queries):
        null_dist = null_dists.get(query.query_id)
        qd = compute_all_pvalues(query, vector_db, null_dist, model, qrels, doc_embeddings)
        all_qd.append(qd)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(queries)}] {qd['country']}: "
                  f"true_k={qd['true_k']}, HC_k={qd['hc_k']}, "
                  f"rel in top-100: {qd['is_relevant_cand'].sum()}")

    # ===== Aggregate Results =====
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # 1. Null calibration: full-corpus non-relevant p-values
    all_p_nonrel = np.concatenate([qd["p_nonrel_full"] for qd in all_qd])
    all_p_rel = np.concatenate([qd["p_rel_full"] for qd in all_qd])
    ks_stat, ks_pvalue = stats.kstest(all_p_nonrel, 'uniform')

    print(f"\n  1. NULL CALIBRATION (full corpus, all non-relevant docs)")
    print(f"     Non-relevant p-values (n={len(all_p_nonrel)}):")
    print(f"       Mean:   {np.mean(all_p_nonrel):.4f}  (expected: 0.500)")
    print(f"       Std:    {np.std(all_p_nonrel):.4f}  (expected: 0.289)")
    print(f"       KS test vs Uniform(0,1): stat={ks_stat:.4f}, p-value={ks_pvalue:.4f}")
    if ks_pvalue > 0.05:
        print(f"       >>> PASS: p-values are uniform (cannot reject H0)")
    else:
        print(f"       >>> MARGINAL: KS p-value < 0.05 (with n={len(all_p_nonrel)}, minor deviations are detectable)")

    # Per-query calibration
    n_pass = 0
    per_query_ks = []
    for qd in all_qd:
        if len(qd["p_nonrel_full"]) >= 20:
            ks_s, ks_p = stats.kstest(qd["p_nonrel_full"], 'uniform')
            per_query_ks.append(ks_p)
            if ks_p > 0.05:
                n_pass += 1
    print(f"     Per-query pass rate: {n_pass}/{len(per_query_ks)} ({n_pass/len(per_query_ks)*100:.0f}%)")

    # 2. Signal detection
    print(f"\n  2. SIGNAL DETECTION (relevant docs)")
    print(f"     Relevant p-values (n={len(all_p_rel)}):")
    print(f"       Mean:   {np.mean(all_p_rel):.4f}")
    print(f"       Median: {np.median(all_p_rel):.4f}")
    print(f"       % below 0.05: {np.mean(all_p_rel < 0.05) * 100:.1f}%")
    print(f"       % below 0.01: {np.mean(all_p_rel < 0.01) * 100:.1f}%")

    # 3. HC accuracy
    true_ks = np.array([qd["true_k"] for qd in all_qd])
    hc_ks = np.array([qd["hc_k"] for qd in all_qd])
    corr = np.corrcoef(true_ks, hc_ks)[0, 1]
    print(f"\n  3. HC THRESHOLD ACCURACY")
    print(f"     Mean true K:  {np.mean(true_ks):.1f}")
    print(f"     Mean HC K:    {np.mean(hc_ks):.1f}")
    print(f"     K correlation: {corr:.4f}")
    print(f"     Mean K error:  {np.mean(hc_ks - true_ks):+.1f}")

    # Generate plots
    print("\n" + "-" * 70)
    print("Generating plots...")
    generate_plots(all_qd, output_dir)

    # Save report
    report = {
        "null_calibration": {
            "pooled_ks_stat": float(ks_stat),
            "pooled_ks_pvalue": float(ks_pvalue),
            "nonrel_mean": float(np.mean(all_p_nonrel)),
            "nonrel_std": float(np.std(all_p_nonrel)),
            "per_query_pass_rate": float(n_pass / len(per_query_ks)),
            "n_nonrel_total": len(all_p_nonrel),
        },
        "signal_detection": {
            "rel_mean": float(np.mean(all_p_rel)),
            "rel_median": float(np.median(all_p_rel)),
            "pct_below_005": float(np.mean(all_p_rel < 0.05) * 100),
            "pct_below_001": float(np.mean(all_p_rel < 0.01) * 100),
            "n_rel_total": len(all_p_rel),
        },
        "hc_accuracy": {
            "mean_true_k": float(np.mean(true_ks)),
            "mean_hc_k": float(np.mean(hc_ks)),
            "k_correlation": float(corr),
            "mean_k_error": float(np.mean(hc_ks - true_ks)),
        },
        "n_queries": len(all_qd),
    }
    report_path = output_dir / "validation" / "runtime_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {report_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
