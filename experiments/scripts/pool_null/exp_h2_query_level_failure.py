"""Direction 2: What causes HC to fail at the query level?

Five layered experiments at pool=1000:
  H2.1 — Query characterization (negative-HC vs positive-HC)
  H2.2 — Similarity landscape shapes
  H2.3 — Rank position and clustering of relevant docs
  H2.4 — Z-scoring contamination analysis
  H2.5 — Per-query null fit

Usage:
    python -m experiments.scripts.pool_null.exp_h2_query_level_failure \
        --config experiments/configs/amazon_compound.yaml [--force]
"""

import argparse
import re
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
POOL_SIZE = 1000
MAX_RELEVANT = 43
Z_SCORE_FRACTIONS = [0.5, 0.6, 0.7, 0.8]
DEFAULT_Z_FRAC = 0.8


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


def compute_metrics(retrieved_ids, relevant_ids):
    """Recall / precision / F1 for a single query."""
    if len(relevant_ids) == 0:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    hits = len(set(retrieved_ids) & relevant_ids)
    recall = hits / len(relevant_ids)
    precision = hits / len(retrieved_ids) if len(retrieved_ids) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}


# ---------------------------------------------------------------------------
# Shared: compute HC stat for each query, partition into neg/pos
# ---------------------------------------------------------------------------

def compute_per_query_hc(queries, qrels, all_faiss_results, null_dist, z_score_fraction, gamma):
    """Compute HC stat + related data for every query. Returns list of per-query dicts."""
    hc_module = HigherCriticism(null_distribution=null_dist)
    null_sorted = np.sort(null_dist.similarities)
    N_null = len(null_sorted)
    per_query = []

    for qi, query in enumerate(queries):
        relevant = qrels.get(query.query_id, set())
        true_k = len(relevant)
        candidates = all_faiss_results[qi][:POOL_SIZE]
        if len(candidates) < 10:
            continue

        cids = [c[0] for c in candidates]
        csims = np.array([c[1] for c in candidates])
        res = zscore_candidates(csims, z_score_fraction)
        if res is None:
            continue
        z_scores, mu_est, sigma_est = res

        # HC computation
        hc_result = hc_module.compute_hc_threshold(z_scores, gamma=gamma)

        # Relevance masks
        rel_mask = np.array([cid in relevant for cid in cids])
        nonrel_mask = ~rel_mask

        # Rank positions of relevant docs (0-indexed)
        rel_positions = np.where(rel_mask)[0].tolist()  # positions in ranked list

        # Recall ceiling
        n_rel_in_pool = int(np.sum(rel_mask))
        recall_ceiling = n_rel_in_pool / true_k if true_k > 0 else 0.0

        per_query.append({
            "qi": qi,
            "query_id": query.query_id,
            "query_text": query.text,
            "true_k": true_k,
            "n_rel_in_pool": n_rel_in_pool,
            "recall_ceiling": recall_ceiling,
            "hc_stat": hc_result.hc_statistic,
            "hc_k": hc_result.k,
            "hc_negative": hc_result.hc_statistic < 0,
            "mu_est": mu_est,
            "sigma_est": sigma_est,
            "cids": cids,
            "csims": csims,
            "z_scores": z_scores,
            "rel_mask": rel_mask,
            "nonrel_mask": nonrel_mask,
            "rel_positions": rel_positions,
        })

    return per_query


# ---------------------------------------------------------------------------
# H2.1: Query characterization
# ---------------------------------------------------------------------------

def h2_1_query_characterization(per_query):
    """Compare negative-HC vs positive-HC query properties."""
    print("\n" + "=" * 80)
    print("H2.1: QUERY CHARACTERIZATION")
    print("=" * 80)

    neg = [q for q in per_query if q["hc_negative"]]
    pos = [q for q in per_query if not q["hc_negative"]]

    def summarize_group(group, label):
        if not group:
            return {"label": label, "n": 0}
        texts = [q["query_text"] for q in group]
        has_numeric = [bool(re.search(r'\d', t)) for t in texts]
        has_price = [bool(re.search(r'\$|price|under|over|less than|more than|cheap|expensive', t, re.I)) for t in texts]
        return {
            "label": label,
            "n": len(group),
            "mean_true_k": float(np.mean([q["true_k"] for q in group])),
            "std_true_k": float(np.std([q["true_k"] for q in group])),
            "mean_n_rel_in_pool": float(np.mean([q["n_rel_in_pool"] for q in group])),
            "mean_recall_ceiling": float(np.mean([q["recall_ceiling"] for q in group])),
            "mean_query_length": float(np.mean([len(t.split()) for t in texts])),
            "frac_has_numeric": float(np.mean(has_numeric)),
            "frac_has_price_term": float(np.mean(has_price)),
            "mean_hc_stat": float(np.mean([q["hc_stat"] for q in group])),
            "mean_hc_k": float(np.mean([q["hc_k"] for q in group])),
        }

    neg_summary = summarize_group(neg, "negative_hc")
    pos_summary = summarize_group(pos, "positive_hc")

    for s in [neg_summary, pos_summary]:
        if s["n"] == 0:
            continue
        print(f"\n  {s['label'].upper()} ({s['n']} queries):")
        for k, v in s.items():
            if k in ("label", "n"):
                continue
            print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")

    # Per-query detail for negative-HC group
    per_query_detail = []
    for q in per_query:
        per_query_detail.append({
            "query_id": q["query_id"],
            "query_text": q["query_text"],
            "true_k": q["true_k"],
            "n_rel_in_pool": q["n_rel_in_pool"],
            "recall_ceiling": q["recall_ceiling"],
            "hc_stat": q["hc_stat"],
            "hc_k": q["hc_k"],
            "hc_negative": q["hc_negative"],
            "query_length": len(q["query_text"].split()),
            "has_numeric": bool(re.search(r'\d', q["query_text"])),
            "has_price_term": bool(re.search(r'\$|price|under|over|less than|more than|cheap|expensive', q["query_text"], re.I)),
        })

    return {
        "negative_hc": neg_summary,
        "positive_hc": pos_summary,
        "per_query": per_query_detail,
    }


# ---------------------------------------------------------------------------
# H2.2: Similarity landscape shapes
# ---------------------------------------------------------------------------

def h2_2_similarity_landscapes(per_query):
    """Analyze similarity distributions for negative-HC vs positive-HC."""
    print("\n" + "=" * 80)
    print("H2.2: SIMILARITY LANDSCAPE SHAPES")
    print("=" * 80)

    per_query_landscapes = []
    for q in per_query:
        csims = q["csims"]
        rel_mask = q["rel_mask"]
        nonrel_mask = q["nonrel_mask"]

        rel_sims = csims[rel_mask]
        nonrel_sims = csims[nonrel_mask]

        # Overlap: fraction of relevant docs within non-relevant IQR
        if len(nonrel_sims) > 0 and len(rel_sims) > 0:
            q25, q75 = np.percentile(nonrel_sims, [25, 75])
            in_iqr = np.sum((rel_sims >= q25) & (rel_sims <= q75))
            overlap_frac = float(in_iqr / len(rel_sims))
        else:
            overlap_frac = None

        entry = {
            "query_id": q["query_id"],
            "hc_negative": q["hc_negative"],
            "n_rel": int(np.sum(rel_mask)),
            "n_nonrel": int(np.sum(nonrel_mask)),
        }
        if len(rel_sims) > 0:
            entry.update({
                "rel_sim_mean": float(np.mean(rel_sims)),
                "rel_sim_std": float(np.std(rel_sims)),
                "rel_sim_min": float(np.min(rel_sims)),
                "rel_sim_max": float(np.max(rel_sims)),
            })
        if len(nonrel_sims) > 0:
            entry.update({
                "nonrel_sim_mean": float(np.mean(nonrel_sims)),
                "nonrel_sim_std": float(np.std(nonrel_sims)),
                "nonrel_sim_min": float(np.min(nonrel_sims)),
                "nonrel_sim_max": float(np.max(nonrel_sims)),
                "nonrel_sim_skew": float(sp_stats.skew(nonrel_sims)),
            })
        if len(rel_sims) > 0 and len(nonrel_sims) > 0:
            entry["sim_gap"] = float(np.mean(rel_sims) - np.mean(nonrel_sims))
            entry["overlap_in_nonrel_iqr"] = overlap_frac
            # Top-100 non-relevant comparison
            nonrel_sorted = np.sort(nonrel_sims)[::-1]
            top100_nonrel_mean = float(np.mean(nonrel_sorted[:min(100, len(nonrel_sorted))]))
            entry["sim_gap_vs_top100"] = float(np.mean(rel_sims) - top100_nonrel_mean)

        per_query_landscapes.append(entry)

    # Aggregate by group
    neg_entries = [e for e in per_query_landscapes if e["hc_negative"]]
    pos_entries = [e for e in per_query_landscapes if not e["hc_negative"]]

    def agg_group(entries, label):
        if not entries:
            return {"label": label, "n": 0}
        def safe_mean(key):
            vals = [e[key] for e in entries if key in e and e[key] is not None]
            return float(np.mean(vals)) if vals else None
        return {
            "label": label, "n": len(entries),
            "mean_rel_sim_mean": safe_mean("rel_sim_mean"),
            "mean_nonrel_sim_mean": safe_mean("nonrel_sim_mean"),
            "mean_sim_gap": safe_mean("sim_gap"),
            "mean_sim_gap_vs_top100": safe_mean("sim_gap_vs_top100"),
            "mean_overlap_in_nonrel_iqr": safe_mean("overlap_in_nonrel_iqr"),
            "mean_rel_sim_std": safe_mean("rel_sim_std"),
            "mean_nonrel_sim_std": safe_mean("nonrel_sim_std"),
        }

    neg_agg = agg_group(neg_entries, "negative_hc")
    pos_agg = agg_group(pos_entries, "positive_hc")

    for agg in [neg_agg, pos_agg]:
        print(f"\n  {agg['label'].upper()} ({agg['n']} queries):")
        for k, v in agg.items():
            if k in ("label", "n") or v is None:
                continue
            print(f"    {k}: {v:.4f}")

    return {
        "aggregate": {"negative_hc": neg_agg, "positive_hc": pos_agg},
        "per_query": per_query_landscapes,
    }


# ---------------------------------------------------------------------------
# H2.3: Rank position and clustering
# ---------------------------------------------------------------------------

def h2_3_rank_positions(per_query):
    """Analyze where relevant docs sit in the ranked pool."""
    print("\n" + "=" * 80)
    print("H2.3: RANK POSITION AND CLUSTERING")
    print("=" * 80)

    per_query_ranks = []
    for q in per_query:
        positions = q["rel_positions"]  # 0-indexed positions in ranked pool
        n_rel = len(positions)

        entry = {
            "query_id": q["query_id"],
            "hc_negative": q["hc_negative"],
            "true_k": q["true_k"],
            "n_rel_in_pool": q["n_rel_in_pool"],
            "positions": positions,
        }

        if n_rel > 0:
            positions_arr = np.array(positions)
            entry["in_top_50"] = int(np.sum(positions_arr < 50))
            entry["in_top_100"] = int(np.sum(positions_arr < 100))
            entry["in_top_200"] = int(np.sum(positions_arr < 200))
            entry["mean_position"] = float(np.mean(positions_arr))
            entry["median_position"] = float(np.median(positions_arr))
            entry["min_position"] = int(np.min(positions_arr))
            entry["max_position"] = int(np.max(positions_arr))
            entry["position_range"] = int(np.max(positions_arr) - np.min(positions_arr))

            # Clustering metric: mean pairwise distance between positions
            if n_rel > 1:
                sorted_pos = np.sort(positions_arr)
                gaps = np.diff(sorted_pos)
                entry["mean_gap"] = float(np.mean(gaps))
                entry["max_gap"] = int(np.max(gaps))
                # Dispersion: range / count — lower = more clustered
                entry["dispersion"] = float(entry["position_range"] / n_rel)
            else:
                entry["mean_gap"] = 0.0
                entry["max_gap"] = 0
                entry["dispersion"] = 0.0
        else:
            entry.update({
                "in_top_50": 0, "in_top_100": 0, "in_top_200": 0,
                "mean_position": None, "median_position": None,
                "min_position": None, "max_position": None,
                "position_range": None, "mean_gap": None,
                "max_gap": None, "dispersion": None,
            })

        per_query_ranks.append(entry)

    # Aggregate
    neg = [e for e in per_query_ranks if e["hc_negative"]]
    pos = [e for e in per_query_ranks if not e["hc_negative"]]

    def agg_group(entries, label):
        if not entries:
            return {"label": label, "n": 0}
        def safe_mean(key):
            vals = [e[key] for e in entries if e.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        return {
            "label": label, "n": len(entries),
            "mean_n_rel_in_pool": safe_mean("n_rel_in_pool"),
            "mean_in_top_50": safe_mean("in_top_50"),
            "mean_in_top_100": safe_mean("in_top_100"),
            "mean_in_top_200": safe_mean("in_top_200"),
            "mean_position": safe_mean("mean_position"),
            "mean_dispersion": safe_mean("dispersion"),
            "mean_mean_gap": safe_mean("mean_gap"),
        }

    neg_agg = agg_group(neg, "negative_hc")
    pos_agg = agg_group(pos, "positive_hc")

    for agg in [neg_agg, pos_agg]:
        print(f"\n  {agg['label'].upper()} ({agg['n']} queries):")
        for k, v in agg.items():
            if k in ("label", "n") or v is None:
                continue
            print(f"    {k}: {v:.2f}" if isinstance(v, float) else f"    {k}: {v}")

    return {
        "aggregate": {"negative_hc": neg_agg, "positive_hc": pos_agg},
        "per_query": per_query_ranks,
    }


# ---------------------------------------------------------------------------
# H2.4: Z-scoring contamination
# ---------------------------------------------------------------------------

def h2_4_zscore_contamination(queries, qrels, all_faiss_results, null_dist, per_query_data):
    """Check if relevant docs in the estimation set contaminate mu/sigma."""
    print("\n" + "=" * 80)
    print("H2.4: Z-SCORING CONTAMINATION")
    print("=" * 80)

    hc_module = HigherCriticism(null_distribution=null_dist)
    gamma = MAX_RELEVANT / POOL_SIZE

    per_query_contam = []
    for q in per_query_data:
        qi = q["qi"]
        cids = q["cids"]
        csims = q["csims"]
        rel_mask = q["rel_mask"]
        relevant = qrels.get(q["query_id"], set())

        entry = {
            "query_id": q["query_id"],
            "hc_negative": q["hc_negative"],
            "true_k": q["true_k"],
        }

        # For each z_score_fraction, compute contamination
        frac_results = {}
        for z_frac in Z_SCORE_FRACTIONS:
            sorted_desc_idx = np.argsort(csims)[::-1]
            n = len(csims)
            n_null_est = max(int(n * z_frac), 10)
            # Bottom fraction indices (lowest similarity)
            bottom_idx = sorted_desc_idx[-n_null_est:]

            # How many relevant docs are in the estimation set?
            n_rel_in_bottom = int(np.sum(rel_mask[bottom_idx]))
            frac_rel_in_bottom = n_rel_in_bottom / n_null_est

            # Compute mu/sigma WITH contamination (current behavior)
            bottom_sims = csims[bottom_idx]
            mu_with = float(np.mean(bottom_sims))
            sigma_with = float(np.std(bottom_sims))

            # Compute mu/sigma WITHOUT relevant docs
            bottom_nonrel_idx = bottom_idx[~rel_mask[bottom_idx]]
            if len(bottom_nonrel_idx) < 5:
                frac_results[str(z_frac)] = {
                    "n_rel_in_estimation": n_rel_in_bottom,
                    "frac_rel_in_estimation": frac_rel_in_bottom,
                    "too_few_nonrel": True,
                }
                continue

            bottom_nonrel_sims = csims[bottom_nonrel_idx]
            mu_without = float(np.mean(bottom_nonrel_sims))
            sigma_without = float(np.std(bottom_nonrel_sims))

            # Compute HC stat with decontaminated z-scores
            if sigma_without > 1e-10:
                z_decontam = (csims - mu_without) / sigma_without
                hc_decontam = hc_module.compute_hc_threshold(z_decontam, gamma=gamma)
                hc_stat_decontam = hc_decontam.hc_statistic
                hc_k_decontam = hc_decontam.k
            else:
                hc_stat_decontam = None
                hc_k_decontam = None

            frac_results[str(z_frac)] = {
                "n_rel_in_estimation": n_rel_in_bottom,
                "frac_rel_in_estimation": frac_rel_in_bottom,
                "n_estimation_set": n_null_est,
                "mu_with": mu_with,
                "sigma_with": sigma_with,
                "mu_without": mu_without,
                "sigma_without": sigma_without,
                "delta_mu": mu_with - mu_without,
                "delta_sigma": sigma_with - sigma_without,
                "hc_stat_original": q["hc_stat"],
                "hc_stat_decontaminated": hc_stat_decontam,
                "hc_k_decontaminated": hc_k_decontam,
                "flipped_to_positive": bool(
                    q["hc_negative"] and hc_stat_decontam is not None and hc_stat_decontam >= 0
                ),
            }

        entry["by_z_frac"] = frac_results
        per_query_contam.append(entry)

    # Aggregate: how many queries flip from negative to positive?
    for z_frac in Z_SCORE_FRACTIONS:
        z_key = str(z_frac)
        neg_queries = [e for e in per_query_contam if e["hc_negative"]]
        flipped = [e for e in neg_queries
                   if z_key in e["by_z_frac"]
                   and e["by_z_frac"][z_key].get("flipped_to_positive", False)]
        print(f"\n  z_frac={z_frac}: {len(flipped)}/{len(neg_queries)} negative-HC queries "
              f"flipped to positive after decontamination")

        # Mean contamination rate
        contam_rates = [e["by_z_frac"][z_key]["frac_rel_in_estimation"]
                        for e in per_query_contam
                        if z_key in e["by_z_frac"] and "frac_rel_in_estimation" in e["by_z_frac"][z_key]]
        if contam_rates:
            print(f"    Mean contamination rate: {np.mean(contam_rates):.4f}")

    return {"per_query": per_query_contam}


# ---------------------------------------------------------------------------
# H2.5: Per-query null fit
# ---------------------------------------------------------------------------

def h2_5_null_fit(per_query, null_dist):
    """Compare each query's non-relevant z-score distribution to the global null."""
    print("\n" + "=" * 80)
    print("H2.5: PER-QUERY NULL FIT")
    print("=" * 80)

    null_sorted = np.sort(null_dist.similarities)
    null_std = null_dist.std
    null_mean = null_dist.mean

    per_query_fit = []
    for q in per_query:
        z_scores = q["z_scores"]
        nonrel_mask = q["nonrel_mask"]
        nonrel_z = z_scores[nonrel_mask]

        entry = {
            "query_id": q["query_id"],
            "hc_negative": q["hc_negative"],
            "n_nonrel": int(len(nonrel_z)),
        }

        if len(nonrel_z) >= 5:
            # KS test against global null
            # Compare this query's non-relevant z-scores to the global null distribution
            ks_stat, ks_p = sp_stats.ks_2samp(nonrel_z, null_dist.similarities)
            entry["ks_stat"] = float(ks_stat)
            entry["ks_pval"] = float(ks_p)
            entry["ks_passes"] = bool(ks_p > 0.05)

            # Shape comparison
            entry["query_nonrel_mean"] = float(np.mean(nonrel_z))
            entry["query_nonrel_std"] = float(np.std(nonrel_z))
            entry["query_nonrel_skew"] = float(sp_stats.skew(nonrel_z))
            entry["query_nonrel_kurtosis"] = float(sp_stats.kurtosis(nonrel_z))

            # Ratio of query std to null std
            entry["std_ratio"] = float(np.std(nonrel_z) / null_std) if null_std > 0 else None

            # Direction of mismatch
            if entry["std_ratio"] is not None:
                if entry["std_ratio"] < 0.8:
                    entry["mismatch_direction"] = "conservative"  # query narrower than null
                elif entry["std_ratio"] > 1.2:
                    entry["mismatch_direction"] = "anti_conservative"  # query wider than null
                else:
                    entry["mismatch_direction"] = "matched"

        per_query_fit.append(entry)

    # Aggregate
    neg = [e for e in per_query_fit if e["hc_negative"] and "ks_stat" in e]
    pos = [e for e in per_query_fit if not e["hc_negative"] and "ks_stat" in e]

    def agg_group(entries, label):
        if not entries:
            return {"label": label, "n": 0}
        return {
            "label": label, "n": len(entries),
            "mean_ks_stat": float(np.mean([e["ks_stat"] for e in entries])),
            "mean_ks_pval": float(np.mean([e["ks_pval"] for e in entries])),
            "frac_ks_passes": float(np.mean([e["ks_passes"] for e in entries])),
            "mean_std_ratio": float(np.mean([e["std_ratio"] for e in entries if e.get("std_ratio") is not None])),
            "mean_nonrel_std": float(np.mean([e["query_nonrel_std"] for e in entries])),
            "mean_nonrel_skew": float(np.mean([e["query_nonrel_skew"] for e in entries])),
            "frac_conservative": float(np.mean([e.get("mismatch_direction") == "conservative" for e in entries])),
            "frac_anti_conservative": float(np.mean([e.get("mismatch_direction") == "anti_conservative" for e in entries])),
            "frac_matched": float(np.mean([e.get("mismatch_direction") == "matched" for e in entries])),
        }

    neg_agg = agg_group(neg, "negative_hc")
    pos_agg = agg_group(pos, "positive_hc")

    for agg in [neg_agg, pos_agg]:
        print(f"\n  {agg['label'].upper()} ({agg['n']} queries):")
        for k, v in agg.items():
            if k in ("label", "n"):
                continue
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    return {
        "global_null_stats": {"mean": null_mean, "std": null_std},
        "aggregate": {"negative_hc": neg_agg, "positive_hc": pos_agg},
        "per_query": per_query_fit,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="H2: Query-level failure analysis")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--force", action="store_true", help="Rerun")
    args = parser.parse_args()

    config = load_config(args.config)
    adapter = get_adapter(config.dataset.name)
    artifacts_dir = Path(config.dataset.artifacts_dir)
    results_dir = Path(config.dataset.results_dir) / "null_research" / "h2_query_failure"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_files = {
        "h2_1": results_dir / "h2_query_characterization.json",
        "h2_2": results_dir / "h2_similarity_landscapes.json",
        "h2_3": results_dir / "h2_rank_positions.json",
        "h2_4": results_dir / "h2_zscore_contamination.json",
        "h2_5": results_dir / "h2_null_fit.json",
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

    # Load pool-matched null at pool=1000
    pool_null_path = artifacts_dir / "pool_null" / "global_pool_null"
    null_dist = NullDistribution.load(str(pool_null_path))
    print(f"Pool-matched null: {null_dist}")

    # Pre-compute FAISS at pool_size=1000
    print(f"\nPre-computing FAISS at K={POOL_SIZE} for {len(queries)} queries...")
    all_faiss_results = []
    for i, query in enumerate(queries):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Query {i+1}/{len(queries)}")
        query_emb = model.embed_query(query.text)
        candidates = vector_db.search(query_emb, k=POOL_SIZE)
        all_faiss_results.append([(c.doc_id, c.similarity) for c in candidates])
    print(f"  Cached {len(all_faiss_results)} queries")

    # Compute per-query HC data
    gamma = MAX_RELEVANT / POOL_SIZE
    print(f"\nComputing HC for all queries (gamma={gamma:.4f})...")
    per_query = compute_per_query_hc(
        queries, qrels, all_faiss_results, null_dist, DEFAULT_Z_FRAC, gamma
    )
    n_neg = sum(1 for q in per_query if q["hc_negative"])
    n_pos = sum(1 for q in per_query if not q["hc_negative"])
    print(f"  {n_neg} negative-HC, {n_pos} positive-HC queries")

    config_dict = {
        "dataset": config.dataset.name,
        "pool_size": POOL_SIZE,
        "z_score_fraction": DEFAULT_Z_FRAC,
        "gamma": gamma,
        "max_relevant": MAX_RELEVANT,
        "n_queries": len(queries),
        "n_queries_analyzed": len(per_query),
        "n_negative_hc": n_neg,
        "n_positive_hc": n_pos,
    }

    # H2.1
    h2_1_results = h2_1_query_characterization(per_query)
    save_results_json({"config": config_dict, **h2_1_results}, str(out_files["h2_1"]))
    print(f"\nSaved: {out_files['h2_1']}")

    # H2.2
    h2_2_results = h2_2_similarity_landscapes(per_query)
    save_results_json({"config": config_dict, **h2_2_results}, str(out_files["h2_2"]))
    print(f"Saved: {out_files['h2_2']}")

    # H2.3
    h2_3_results = h2_3_rank_positions(per_query)
    save_results_json({"config": config_dict, **h2_3_results}, str(out_files["h2_3"]))
    print(f"Saved: {out_files['h2_3']}")

    # H2.4
    h2_4_results = h2_4_zscore_contamination(queries, qrels, all_faiss_results, null_dist, per_query)
    save_results_json({"config": config_dict, **h2_4_results}, str(out_files["h2_4"]))
    print(f"Saved: {out_files['h2_4']}")

    # H2.5
    h2_5_results = h2_5_null_fit(per_query, null_dist)
    save_results_json({"config": config_dict, **h2_5_results}, str(out_files["h2_5"]))
    print(f"Saved: {out_files['h2_5']}")

    print("\n" + "=" * 80)
    print("H2 complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
