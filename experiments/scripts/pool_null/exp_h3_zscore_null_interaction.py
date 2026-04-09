"""Direction 3: Per-query z-scoring × global null interaction.

H3.1 — Z-score heterogeneity across pool sizes (1K-10K step 1K)
H3.2 — Z-score spread predicts calibration direction (pool=1000)
H3.3 — Isolating the cause: 3 null conditions × 4 z_score_fractions

Usage:
    python -m experiments.scripts.pool_null.exp_h3_zscore_null_interaction \
        --config experiments/configs/amazon_compound.yaml [--force]
"""

import argparse
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

from hc_rag.embeddings.embedding_model import create_embedding_model
from hc_rag.embeddings.vector_database import VectorDatabase
from hc_rag.hc.null_distribution import NullDistribution
from hc_rag.hc.higher_criticism import HigherCriticism
from experiments.lib.config import load_config
from experiments.lib.registry import get_adapter
from experiments.lib.results_io import save_results_json
import experiments.datasets  # noqa: F401


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_RELEVANT = 43
POOL_SIZES = list(range(1000, 10001, 1000))
Z_SCORE_FRACTIONS = [0.5, 0.6, 0.7, 0.8]
DEFAULT_Z_FRAC = 0.8
POOL_SIZE = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def zscore_candidates(candidate_sims, z_score_fraction):
    """Z-score candidates using bottom fraction. Returns (z_scores, mu, sigma) or None."""
    sorted_desc = np.sort(candidate_sims)[::-1]
    n = len(sorted_desc)
    n_null_est = max(int(n * z_score_fraction), 10)
    bottom = sorted_desc[-n_null_est:]
    mu = float(np.mean(bottom))
    sigma = float(np.std(bottom))
    if sigma < 1e-10:
        return None
    z_scores = (candidate_sims - mu) / sigma
    return z_scores, mu, sigma


def build_pool_null(queries, qrels, all_faiss_results, pool_size, z_score_fraction):
    """Build pool-matched null from pre-computed FAISS results."""
    all_z_scores = []
    for i, query in enumerate(queries):
        candidates = all_faiss_results[i][:pool_size]
        if len(candidates) == 0:
            continue
        cids = [c[0] for c in candidates]
        csims = np.array([c[1] for c in candidates])
        result = zscore_candidates(csims, z_score_fraction)
        if result is None:
            continue
        z_scores, _, _ = result
        relevant = qrels.get(query.query_id, set())
        nonrel_mask = np.array([cid not in relevant for cid in cids])
        nonrel_z = z_scores[nonrel_mask]
        if len(nonrel_z) > 0:
            all_z_scores.append(nonrel_z)
    pooled = np.concatenate(all_z_scores)
    return NullDistribution(
        similarities=pooled, mean=float(np.mean(pooled)),
        std=float(np.std(pooled)), min_val=float(np.min(pooled)),
        max_val=float(np.max(pooled)), n_samples=len(pooled),
    )


def build_raw_sim_null(queries, qrels, all_faiss_results, pool_size):
    """Build null from raw similarities (no z-scoring)."""
    all_sims = []
    for i, query in enumerate(queries):
        candidates = all_faiss_results[i][:pool_size]
        if len(candidates) == 0:
            continue
        cids = [c[0] for c in candidates]
        csims = np.array([c[1] for c in candidates])
        relevant = qrels.get(query.query_id, set())
        nonrel_mask = np.array([cid not in relevant for cid in cids])
        nonrel_sims = csims[nonrel_mask]
        if len(nonrel_sims) > 0:
            all_sims.append(nonrel_sims)
    pooled = np.concatenate(all_sims)
    return NullDistribution(
        similarities=pooled, mean=float(np.mean(pooled)),
        std=float(np.std(pooled)), min_val=float(np.min(pooled)),
        max_val=float(np.max(pooled)), n_samples=len(pooled),
    )


def compute_metrics(retrieved_ids, relevant_ids):
    """Recall / precision / F1."""
    if len(relevant_ids) == 0:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    hits = len(set(retrieved_ids) & relevant_ids)
    recall = hits / len(relevant_ids)
    precision = hits / len(retrieved_ids) if len(retrieved_ids) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}


# ---------------------------------------------------------------------------
# H3.1: Z-score heterogeneity across pool sizes
# ---------------------------------------------------------------------------

def h3_1_heterogeneity(queries, qrels, all_faiss_results):
    """Measure cross-query variance of z-score statistics at each pool size."""
    print("\n" + "=" * 80)
    print("H3.1: Z-SCORE HETEROGENEITY ACROSS POOL SIZES")
    print("=" * 80)

    results = {}
    for ps in POOL_SIZES:
        per_query_stats = []
        for i, query in enumerate(queries):
            candidates = all_faiss_results[i][:ps]
            if len(candidates) < 10:
                continue
            csims = np.array([c[1] for c in candidates])
            res = zscore_candidates(csims, DEFAULT_Z_FRAC)
            if res is None:
                continue
            z_scores, mu, sigma = res
            per_query_stats.append({
                "std": float(np.std(z_scores)),
                "skewness": float(sp_stats.skew(z_scores)),
                "kurtosis": float(sp_stats.kurtosis(z_scores)),
                "mu_est": mu,
                "sigma_est": sigma,
            })

        stds = [s["std"] for s in per_query_stats]
        skews = [s["skewness"] for s in per_query_stats]
        kurts = [s["kurtosis"] for s in per_query_stats]
        sigma_ests = [s["sigma_est"] for s in per_query_stats]

        results[str(ps)] = {
            "n_queries": len(per_query_stats),
            "cross_query_variance_of_std": float(np.var(stds)),
            "cross_query_variance_of_skewness": float(np.var(skews)),
            "cross_query_variance_of_kurtosis": float(np.var(kurts)),
            "cross_query_variance_of_sigma_est": float(np.var(sigma_ests)),
            "mean_std": float(np.mean(stds)),
            "mean_skewness": float(np.mean(skews)),
            "mean_kurtosis": float(np.mean(kurts)),
            "mean_sigma_est": float(np.mean(sigma_ests)),
            "std_of_sigma_est": float(np.std(sigma_ests)),
        }

        r = results[str(ps)]
        print(f"  pool={ps}: var(std)={r['cross_query_variance_of_std']:.6f}  "
              f"var(skew)={r['cross_query_variance_of_skewness']:.6f}  "
              f"var(sigma_est)={r['cross_query_variance_of_sigma_est']:.6f}")

    return results


# ---------------------------------------------------------------------------
# H3.2: Z-score spread predicts calibration direction
# ---------------------------------------------------------------------------

def h3_2_spread_prediction(queries, qrels, all_faiss_results, null_dist):
    """Correlate per-query z-score std ratio with HC stat and calibration."""
    print("\n" + "=" * 80)
    print("H3.2: Z-SCORE SPREAD PREDICTS CALIBRATION DIRECTION")
    print("=" * 80)

    hc_module = HigherCriticism(null_distribution=null_dist)
    null_sorted = np.sort(null_dist.similarities)
    N_null = len(null_sorted)
    gamma = MAX_RELEVANT / POOL_SIZE

    per_query = []
    for i, query in enumerate(queries):
        relevant = qrels.get(query.query_id, set())
        candidates = all_faiss_results[i][:POOL_SIZE]
        if len(candidates) < 10:
            continue
        cids = [c[0] for c in candidates]
        csims = np.array([c[1] for c in candidates])
        res = zscore_candidates(csims, DEFAULT_Z_FRAC)
        if res is None:
            continue
        z_scores, mu, sigma = res

        # HC stat
        hc_result = hc_module.compute_hc_threshold(z_scores, gamma=gamma)

        # Per-query z-score statistics
        nonrel_mask = np.array([cid not in relevant for cid in cids])
        nonrel_z = z_scores[nonrel_mask]
        query_nonrel_std = float(np.std(nonrel_z)) if len(nonrel_z) > 0 else None

        # Per-query KS test (uniformity)
        if len(nonrel_z) >= 5:
            pv = np.clip((N_null - np.searchsorted(null_sorted, nonrel_z, side="left") + 1.0) / (N_null + 1.0), 1e-10, 1.0)
            ks_stat, ks_p = sp_stats.kstest(pv, "uniform")
        else:
            ks_stat, ks_p = None, None

        # Signal detection rate
        rel_mask = np.array([cid in relevant for cid in cids])
        rel_z = z_scores[rel_mask]
        if len(rel_z) > 0:
            rel_pv = np.clip((N_null - np.searchsorted(null_sorted, rel_z, side="left") + 1.0) / (N_null + 1.0), 1e-10, 1.0)
            signal_rate = float(np.mean(rel_pv < 0.05))
        else:
            signal_rate = None

        std_ratio = query_nonrel_std / null_dist.std if query_nonrel_std and null_dist.std > 0 else None

        per_query.append({
            "query_id": query.query_id,
            "hc_stat": hc_result.hc_statistic,
            "hc_k": hc_result.k,
            "hc_negative": hc_result.hc_statistic < 0,
            "query_nonrel_std": query_nonrel_std,
            "null_std": null_dist.std,
            "std_ratio": std_ratio,
            "ks_stat": ks_stat,
            "ks_pval": ks_p,
            "signal_rate": signal_rate,
            "sigma_est": sigma,
        })

    # Correlations
    valid = [q for q in per_query if q["std_ratio"] is not None]
    std_ratios = np.array([q["std_ratio"] for q in valid])
    hc_stats = np.array([q["hc_stat"] for q in valid])
    hc_signs = np.array([1 if q["hc_negative"] else 0 for q in valid])

    corr_hc_stat = float(sp_stats.pearsonr(std_ratios, hc_stats)[0]) if len(valid) > 2 else None
    corr_hc_sign = float(sp_stats.pointbiserialr(hc_signs, std_ratios)[0]) if len(valid) > 2 else None

    valid_ks = [q for q in valid if q["ks_stat"] is not None]
    ks_stats_arr = np.array([q["ks_stat"] for q in valid_ks])
    ratios_ks = np.array([q["std_ratio"] for q in valid_ks])
    corr_ks = float(sp_stats.pearsonr(ratios_ks, ks_stats_arr)[0]) if len(valid_ks) > 2 else None

    valid_sig = [q for q in valid if q["signal_rate"] is not None]
    sig_rates = np.array([q["signal_rate"] for q in valid_sig])
    ratios_sig = np.array([q["std_ratio"] for q in valid_sig])
    corr_signal = float(sp_stats.pearsonr(ratios_sig, sig_rates)[0]) if len(valid_sig) > 2 else None

    correlations = {
        "std_ratio_vs_hc_stat": corr_hc_stat,
        "std_ratio_vs_hc_negative": corr_hc_sign,
        "std_ratio_vs_ks_stat": corr_ks,
        "std_ratio_vs_signal_rate": corr_signal,
    }

    print(f"\n  Correlations:")
    for k, v in correlations.items():
        print(f"    {k}: {v:.4f}" if v is not None else f"    {k}: N/A")

    # Summary by std_ratio bins
    print(f"\n  Summary by std_ratio bins:")
    bins = [(0, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, float("inf"))]
    bin_labels = ["<0.8 (conservative)", "0.8-1.0", "1.0-1.2", ">1.2 (anti-conservative)"]
    for (lo, hi), label in zip(bins, bin_labels):
        in_bin = [q for q in valid if lo <= q["std_ratio"] < hi]
        if not in_bin:
            continue
        n_neg = sum(1 for q in in_bin if q["hc_negative"])
        mean_hc = float(np.mean([q["hc_stat"] for q in in_bin]))
        print(f"    {label}: n={len(in_bin)}, neg_hc={n_neg}, mean_hc={mean_hc:.2f}")

    return {
        "correlations": correlations,
        "null_std": null_dist.std,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# H3.3: Isolating the cause — 3 conditions × 4 z_score_fractions
# ---------------------------------------------------------------------------

def h3_3_isolation(queries, qrels, all_faiss_results):
    """Test 3 null conditions × 4 z_score_fractions at pool=1000."""
    print("\n" + "=" * 80)
    print("H3.3: ISOLATING THE CAUSE")
    print("=" * 80)

    gamma = MAX_RELEVANT / POOL_SIZE
    results = {}

    for z_frac in Z_SCORE_FRACTIONS:
        print(f"\n--- z_score_fraction = {z_frac} ---")
        results[str(z_frac)] = {}

        # Condition (a): per-query z-scoring + global null (current pipeline)
        print("  Building condition (a): z-scored + global null...")
        global_null = build_pool_null(queries, qrels, all_faiss_results, POOL_SIZE, z_frac)
        a_result = evaluate_condition(
            queries, qrels, all_faiss_results, global_null,
            gamma, z_frac, use_zscore=True, label="zscore_global_null"
        )
        results[str(z_frac)]["zscore_global_null"] = a_result

        # Condition (b): raw similarities + global null (no z-scoring)
        print("  Building condition (b): raw sims + global null...")
        raw_null = build_raw_sim_null(queries, qrels, all_faiss_results, POOL_SIZE)
        b_result = evaluate_condition(
            queries, qrels, all_faiss_results, raw_null,
            gamma, z_frac, use_zscore=False, label="raw_global_null"
        )
        results[str(z_frac)]["raw_global_null"] = b_result

        # Condition (c): per-query z-scoring + per-query null
        print("  Running condition (c): z-scored + per-query null...")
        c_result = evaluate_per_query_null_condition(
            queries, qrels, all_faiss_results, gamma, z_frac
        )
        results[str(z_frac)]["zscore_per_query_null"] = c_result

    # Print comparison table
    print("\n" + "=" * 80)
    print("H3.3 COMPARISON TABLE")
    print("=" * 80)
    conditions = ["zscore_global_null", "raw_global_null", "zscore_per_query_null"]
    print(f"\n  {'z_frac':>8} {'Condition':<25} {'Unif%':>8} {'SigDet%':>8} {'NegHC':>8} {'MeanHC':>8} {'MeanK':>8} {'F1':>8}")
    print(f"  {'-'*88}")
    for z_frac in Z_SCORE_FRACTIONS:
        for cond in conditions:
            r = results[str(z_frac)][cond]
            print(f"  {z_frac:>8.1f} {cond:<25} {r['uniformity_pass_rate']:>8.3f} "
                  f"{r['signal_detection_rate']:>8.3f} {r['n_negative_hc']:>8} "
                  f"{r['mean_hc_stat']:>8.2f} {r['mean_k']:>8.1f} {r['mean_f1']:>8.4f}")

    return results


def evaluate_condition(queries, qrels, all_faiss_results, null_dist,
                       gamma, z_score_fraction, use_zscore, label):
    """Evaluate a null condition: uniformity, signal detection, HC stats, F1."""
    hc_module = HigherCriticism(null_distribution=null_dist)
    null_sorted = np.sort(null_dist.similarities)
    N_null = len(null_sorted)

    ks_passes = 0
    ks_tested = 0
    all_rel_pvals = []
    hc_stats = []
    hc_ks = []
    f1s = []

    for i, query in enumerate(queries):
        relevant = qrels.get(query.query_id, set())
        candidates = all_faiss_results[i][:POOL_SIZE]
        if len(candidates) < 10:
            continue
        cids = [c[0] for c in candidates]
        csims = np.array([c[1] for c in candidates])

        if use_zscore:
            res = zscore_candidates(csims, z_score_fraction)
            if res is None:
                continue
            values, _, _ = res
        else:
            values = csims

        # HC
        hc_result = hc_module.compute_hc_threshold(values, gamma=gamma)
        hc_stats.append(hc_result.hc_statistic)
        hc_ks.append(hc_result.k)

        # F1
        retrieved_ids = [cids[j] for j in range(min(hc_result.k, len(cids)))]
        m = compute_metrics(retrieved_ids, relevant)
        f1s.append(m["f1"])

        # Uniformity (non-relevant)
        rel_mask = np.array([cid in relevant for cid in cids])
        nonrel_values = values[~rel_mask]
        if len(nonrel_values) >= 5:
            pv = np.clip((N_null - np.searchsorted(null_sorted, nonrel_values, side="left") + 1.0) / (N_null + 1.0), 1e-10, 1.0)
            _, ks_p = sp_stats.kstest(pv, "uniform")
            ks_tested += 1
            if ks_p > 0.05:
                ks_passes += 1

        # Signal detection (relevant)
        rel_values = values[rel_mask]
        if len(rel_values) > 0:
            pv = np.clip((N_null - np.searchsorted(null_sorted, rel_values, side="left") + 1.0) / (N_null + 1.0), 1e-10, 1.0)
            all_rel_pvals.extend(pv.tolist())

    return {
        "label": label,
        "uniformity_pass_rate": ks_passes / ks_tested if ks_tested > 0 else 0.0,
        "signal_detection_rate": float(np.mean(np.array(all_rel_pvals) < 0.05)) if all_rel_pvals else 0.0,
        "mean_hc_stat": float(np.mean(hc_stats)) if hc_stats else 0.0,
        "n_negative_hc": int(sum(1 for s in hc_stats if s < 0)),
        "mean_k": float(np.mean(hc_ks)) if hc_ks else 0.0,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "n_queries": ks_tested,
    }


def evaluate_per_query_null_condition(queries, qrels, all_faiss_results, gamma, z_score_fraction):
    """Condition (c): each query gets its own null built from its own non-relevant z-scores."""
    ks_passes = 0
    ks_tested = 0
    all_rel_pvals = []
    hc_stats = []
    hc_ks = []
    f1s = []

    for i, query in enumerate(queries):
        relevant = qrels.get(query.query_id, set())
        candidates = all_faiss_results[i][:POOL_SIZE]
        if len(candidates) < 10:
            continue
        cids = [c[0] for c in candidates]
        csims = np.array([c[1] for c in candidates])

        res = zscore_candidates(csims, z_score_fraction)
        if res is None:
            continue
        z_scores, _, _ = res

        rel_mask = np.array([cid in relevant for cid in cids])
        nonrel_z = z_scores[~rel_mask]
        rel_z = z_scores[rel_mask]

        if len(nonrel_z) < 10:
            continue

        # Build per-query null from this query's non-relevant z-scores
        per_query_null = NullDistribution(
            similarities=nonrel_z,
            mean=float(np.mean(nonrel_z)),
            std=float(np.std(nonrel_z)),
            min_val=float(np.min(nonrel_z)),
            max_val=float(np.max(nonrel_z)),
            n_samples=len(nonrel_z),
        )

        hc_module = HigherCriticism(null_distribution=per_query_null)
        null_sorted = np.sort(nonrel_z)
        N_null = len(null_sorted)

        # HC
        hc_result = hc_module.compute_hc_threshold(z_scores, gamma=gamma)
        hc_stats.append(hc_result.hc_statistic)
        hc_ks.append(hc_result.k)

        # F1
        retrieved_ids = [cids[j] for j in range(min(hc_result.k, len(cids)))]
        m = compute_metrics(retrieved_ids, relevant)
        f1s.append(m["f1"])

        # Uniformity
        if len(nonrel_z) >= 5:
            pv = np.clip((N_null - np.searchsorted(null_sorted, nonrel_z, side="left") + 1.0) / (N_null + 1.0), 1e-10, 1.0)
            _, ks_p = sp_stats.kstest(pv, "uniform")
            ks_tested += 1
            if ks_p > 0.05:
                ks_passes += 1

        # Signal detection
        if len(rel_z) > 0:
            pv = np.clip((N_null - np.searchsorted(null_sorted, rel_z, side="left") + 1.0) / (N_null + 1.0), 1e-10, 1.0)
            all_rel_pvals.extend(pv.tolist())

    return {
        "label": "zscore_per_query_null",
        "uniformity_pass_rate": ks_passes / ks_tested if ks_tested > 0 else 0.0,
        "signal_detection_rate": float(np.mean(np.array(all_rel_pvals) < 0.05)) if all_rel_pvals else 0.0,
        "mean_hc_stat": float(np.mean(hc_stats)) if hc_stats else 0.0,
        "n_negative_hc": int(sum(1 for s in hc_stats if s < 0)),
        "mean_k": float(np.mean(hc_ks)) if hc_ks else 0.0,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "n_queries": ks_tested,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="H3: Z-score × null interaction")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--force", action="store_true", help="Rerun")
    args = parser.parse_args()

    config = load_config(args.config)
    adapter = get_adapter(config.dataset.name)
    artifacts_dir = Path(config.dataset.artifacts_dir)
    results_dir = Path(config.dataset.results_dir) / "null_research" / "h3_zscore_null_interaction"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_files = {
        "h3_1": results_dir / "h3_heterogeneity.json",
        "h3_2": results_dir / "h3_spread_prediction.json",
        "h3_3": results_dir / "h3_isolation.json",
    }
    if all(f.exists() for f in out_files.values()) and not args.force:
        print(f"Results exist in {results_dir}. Use --force to rerun.")
        return

    # Load data
    queries, qrels, _ = adapter.load_dataset(config.dataset.data_dir)
    print(f"Loaded {len(queries)} queries")

    vector_db = VectorDatabase.load(str(artifacts_dir / "vector_db"))
    print(f"Vector DB: {vector_db.get_num_documents():,} documents")

    model = create_embedding_model(
        config.embedding.model, normalize_embeddings=config.embedding.normalize
    )

    # Pre-compute FAISS at max pool size (10K for H3.1)
    max_pool = max(POOL_SIZES)
    print(f"\nPre-computing FAISS at K={max_pool} for {len(queries)} queries...")
    all_faiss_results = []
    for i, query in enumerate(queries):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Query {i+1}/{len(queries)}")
        query_emb = model.embed_query(query.text)
        candidates = vector_db.search(query_emb, k=max_pool)
        all_faiss_results.append([(c.doc_id, c.similarity) for c in candidates])
    print(f"  Cached {len(all_faiss_results)} queries")

    config_dict = {
        "dataset": config.dataset.name,
        "pool_sizes": POOL_SIZES,
        "z_score_fractions": Z_SCORE_FRACTIONS,
        "max_relevant": MAX_RELEVANT,
        "n_queries": len(queries),
    }

    # H3.1: Heterogeneity
    h3_1_results = h3_1_heterogeneity(queries, qrels, all_faiss_results)
    save_results_json({"config": config_dict, "by_pool_size": h3_1_results}, str(out_files["h3_1"]))
    print(f"\nSaved: {out_files['h3_1']}")

    # H3.2: Spread prediction (needs pool null at pool=1000)
    pool_null_path = artifacts_dir / "pool_null" / "global_pool_null"
    null_dist = NullDistribution.load(str(pool_null_path))
    print(f"\nPool-matched null: {null_dist}")

    h3_2_results = h3_2_spread_prediction(queries, qrels, all_faiss_results, null_dist)
    save_results_json({"config": config_dict, **h3_2_results}, str(out_files["h3_2"]))
    print(f"Saved: {out_files['h3_2']}")

    # H3.3: Isolation
    h3_3_results = h3_3_isolation(queries, qrels, all_faiss_results)
    save_results_json({"config": config_dict, "grid": h3_3_results}, str(out_files["h3_3"]))
    print(f"Saved: {out_files['h3_3']}")

    print("\n" + "=" * 80)
    print("H3 complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
