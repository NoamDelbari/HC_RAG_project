"""Direction 4: Per-query pipeline trace and failure mode classification.

Four analyses from a single FAISS retrieval loop:
  H4.1 — Full pipeline trace (similarity → z-score → p-value → HC curve)
  H4.2 — Relevant document p-value analysis
  H4.3 — Failure mode classification of irreducible-failure queries
  H4.4 — Targeted fix tests (contamination removal + detection boundary)

Usage:
    python -m experiments.scripts.pool_null.exp_h4_pipeline_trace \
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
POOL_SIZE = 1000
MAX_RELEVANT = 43
GAMMA = MAX_RELEVANT / POOL_SIZE  # 0.043
Z_SCORE_FRACTION = 0.8
TOP_N_STORE = 100  # Store full trace for top-100 candidates per query


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


def compute_pvalues_empirical(values, null_dist):
    """Compute p-values of values against a NullDistribution (empirical CDF)."""
    null_sorted = np.sort(null_dist.similarities)
    N_null = len(null_sorted)
    indices = np.searchsorted(null_sorted, values, side="left")
    n_greater_equal = N_null - indices
    p_values = (n_greater_equal + 1.0) / (N_null + 1.0)
    eps = 1.0 / (N_null + 1.0)
    p_values = np.clip(p_values, eps, 1.0 - eps)
    return p_values


def compute_hc_curve(p_values_sorted_asc, n_gamma, n):
    """Compute HC_i at each position i=1..n_gamma.

    Args:
        p_values_sorted_asc: p-values sorted ascending (smallest first).
        n_gamma: Number of positions to evaluate.
        n: Total number of candidates.

    Returns:
        Array of HC_i values (length n_gamma).
    """
    hc_values = np.zeros(n_gamma)
    for i in range(1, n_gamma + 1):
        p_i = p_values_sorted_asc[i - 1]
        expected = i / n
        denom = np.sqrt(p_i * (1 - p_i) / n)
        if denom > 0:
            hc_values[i - 1] = (expected - p_i) / denom
    return hc_values


def compute_metrics(retrieved_ids, relevant_ids):
    """Recall / precision / F1 for a single query."""
    if len(relevant_ids) == 0:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    hits = len(retrieved_set & relevant_set)
    recall = hits / len(relevant_set) if relevant_set else 0.0
    precision = hits / len(retrieved_set) if retrieved_set else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}


# ---------------------------------------------------------------------------
# H4.1 — Full Pipeline Trace
# ---------------------------------------------------------------------------

def h4_1_pipeline_traces(queries, qrels, all_faiss_results, global_null):
    """Trace the full HC pipeline for every query.

    Returns list of per-query trace dicts (one per query).
    """
    n_gamma = int(np.floor(GAMMA * POOL_SIZE))  # 43
    traces = []

    for qi, query in enumerate(queries):
        relevant_ids = qrels.get(query.query_id, set())
        faiss_results = all_faiss_results[qi][:POOL_SIZE]

        cids = [cid for cid, _ in faiss_results]
        csims = np.array([sim for _, sim in faiss_results])

        # Sort descending by similarity
        sort_idx = np.argsort(csims)[::-1]
        csims = csims[sort_idx]
        cids = [cids[j] for j in sort_idx]

        # Label relevant/non-relevant
        rel_mask = np.array([cid in relevant_ids for cid in cids])
        n_rel_in_pool = int(rel_mask.sum())

        # Z-score
        zs_result = zscore_candidates(csims, Z_SCORE_FRACTION)
        if zs_result is None:
            continue
        z_scores, mu_est, sigma_est = zs_result

        # P-values against global null
        pvals_global = compute_pvalues_empirical(z_scores, global_null)

        # Build per-query null from this query's non-relevant z-scores
        nonrel_z = z_scores[~rel_mask]
        if len(nonrel_z) < 10:
            continue
        per_query_null = NullDistribution(
            similarities=nonrel_z,
            mean=float(np.mean(nonrel_z)),
            std=float(np.std(nonrel_z)),
            min_val=float(np.min(nonrel_z)),
            max_val=float(np.max(nonrel_z)),
            n_samples=len(nonrel_z),
        )
        pvals_perquery = compute_pvalues_empirical(z_scores, per_query_null)

        # HC curves — need p-values sorted ascending (most significant first)
        pval_order_global = np.argsort(pvals_global)
        pvals_sorted_global = pvals_global[pval_order_global]
        hc_curve_global = compute_hc_curve(pvals_sorted_global, n_gamma, POOL_SIZE)
        hc_stat_global = float(np.max(hc_curve_global))
        k_global = int(np.argmax(hc_curve_global) + 1) if hc_stat_global > 0 else 0

        pval_order_pq = np.argsort(pvals_perquery)
        pvals_sorted_pq = pvals_perquery[pval_order_pq]
        hc_curve_pq = compute_hc_curve(pvals_sorted_pq, n_gamma, POOL_SIZE)
        hc_stat_pq = float(np.max(hc_curve_pq))
        k_pq = int(np.argmax(hc_curve_pq) + 1) if hc_stat_pq > 0 else 0

        # Which candidates are relevant in the p-value ordering?
        rel_in_pval_order_global = rel_mask[pval_order_global]
        rel_in_pval_order_pq = rel_mask[pval_order_pq]

        trace = {
            "query_id": query.query_id,
            "query_text": query.text,
            "true_k": len(relevant_ids),
            "n_relevant_in_pool": n_rel_in_pool,
            "mu_est": mu_est,
            "sigma_est": sigma_est,
            # Top-100 candidate details (manageable size)
            "top100_sims": csims[:TOP_N_STORE].tolist(),
            "top100_is_relevant": rel_mask[:TOP_N_STORE].tolist(),
            "top100_zscores": z_scores[:TOP_N_STORE].tolist(),
            "top100_pvals_global": pvals_global[:TOP_N_STORE].tolist(),
            "top100_pvals_perquery": pvals_perquery[:TOP_N_STORE].tolist(),
            # Full HC curves (43 values each)
            "hc_curve_global": hc_curve_global.tolist(),
            "hc_curve_perquery": hc_curve_pq.tolist(),
            "hc_stat_global": hc_stat_global,
            "hc_stat_perquery": hc_stat_pq,
            "k_global": k_global,
            "k_perquery": k_pq,
            # Relevant doc p-values (variable length, all relevant docs)
            "relevant_pvals_global": pvals_global[rel_mask].tolist(),
            "relevant_pvals_perquery": pvals_perquery[rel_mask].tolist(),
            "relevant_zscores": z_scores[rel_mask].tolist(),
            "relevant_sims": csims[rel_mask].tolist(),
            # Relevant doc positions in p-value ordering (what HC sees)
            "relevant_pval_ranks_global": np.where(rel_in_pval_order_global)[0].tolist(),
            "relevant_pval_ranks_perquery": np.where(rel_in_pval_order_pq)[0].tolist(),
            # Summary stats for all 1000 candidates
            "all_sims_mean": float(np.mean(csims)),
            "all_sims_std": float(np.std(csims)),
            "all_zscores_mean": float(np.mean(z_scores)),
            "all_zscores_std": float(np.std(z_scores)),
            "nonrel_zscores_std": float(np.std(nonrel_z)),
            "std_ratio": float(np.std(nonrel_z) / global_null.std) if global_null.std > 0 else 0.0,
        }
        traces.append(trace)

    return traces


# ---------------------------------------------------------------------------
# H4.2 — Relevant Document P-Value Analysis
# ---------------------------------------------------------------------------

def h4_2_relevant_pvalues(traces):
    """Analyze p-values of relevant documents across three query groups.

    Groups:
      - positive_hc: HC stat > 0 under global null (80 queries expected)
      - fixable: HC negative under global, positive under per-query (10 expected)
      - irreducible: HC negative under both (45 expected)
    """
    n_gamma = int(np.floor(GAMMA * POOL_SIZE))

    groups = {"positive_hc": [], "fixable": [], "irreducible": []}
    for t in traces:
        if t["hc_stat_global"] > 0:
            groups["positive_hc"].append(t)
        elif t["hc_stat_perquery"] > 0:
            groups["fixable"].append(t)
        else:
            groups["irreducible"].append(t)

    result = {"groups": {}}
    for group_name, group_traces in groups.items():
        if not group_traces:
            result["groups"][group_name] = {"n": 0}
            continue

        per_query = []
        all_best_pvals_global = []
        all_best_pvals_pq = []
        all_frac_sig_global = []
        all_frac_sig_pq = []
        all_n_rel_in_gamma_global = []
        all_n_rel_in_gamma_pq = []

        for t in group_traces:
            rpg = np.array(t["relevant_pvals_global"])
            rpp = np.array(t["relevant_pvals_perquery"])
            ranks_global = np.array(t["relevant_pval_ranks_global"])
            ranks_pq = np.array(t["relevant_pval_ranks_perquery"])

            n_rel = len(rpg)
            if n_rel == 0:
                continue

            best_pval_global = float(np.min(rpg))
            best_pval_pq = float(np.min(rpp))
            frac_sig_global = float(np.mean(rpg < 0.05))
            frac_sig_pq = float(np.mean(rpp < 0.05))
            frac_sig01_global = float(np.mean(rpg < 0.01))
            frac_sig01_pq = float(np.mean(rpp < 0.01))
            n_in_gamma_global = int(np.sum(ranks_global < n_gamma))
            n_in_gamma_pq = int(np.sum(ranks_pq < n_gamma))

            all_best_pvals_global.append(best_pval_global)
            all_best_pvals_pq.append(best_pval_pq)
            all_frac_sig_global.append(frac_sig_global)
            all_frac_sig_pq.append(frac_sig_pq)
            all_n_rel_in_gamma_global.append(n_in_gamma_global)
            all_n_rel_in_gamma_pq.append(n_in_gamma_pq)

            per_query.append({
                "query_id": t["query_id"],
                "n_relevant_in_pool": t["n_relevant_in_pool"],
                "best_pval_global": best_pval_global,
                "best_pval_perquery": best_pval_pq,
                "frac_significant_global": frac_sig_global,
                "frac_significant_perquery": frac_sig_pq,
                "frac_sig01_global": frac_sig01_global,
                "frac_sig01_perquery": frac_sig01_pq,
                "n_rel_in_gamma_global": n_in_gamma_global,
                "n_rel_in_gamma_perquery": n_in_gamma_pq,
                "median_pval_global": float(np.median(rpg)),
                "median_pval_perquery": float(np.median(rpp)),
            })

        result["groups"][group_name] = {
            "n": len(group_traces),
            "aggregate": {
                "mean_best_pval_global": float(np.mean(all_best_pvals_global)) if all_best_pvals_global else None,
                "mean_best_pval_perquery": float(np.mean(all_best_pvals_pq)) if all_best_pvals_pq else None,
                "mean_frac_significant_global": float(np.mean(all_frac_sig_global)) if all_frac_sig_global else None,
                "mean_frac_significant_perquery": float(np.mean(all_frac_sig_pq)) if all_frac_sig_pq else None,
                "mean_n_rel_in_gamma_global": float(np.mean(all_n_rel_in_gamma_global)) if all_n_rel_in_gamma_global else None,
                "mean_n_rel_in_gamma_perquery": float(np.mean(all_n_rel_in_gamma_pq)) if all_n_rel_in_gamma_pq else None,
            },
            "per_query": per_query,
        }

    return result


# ---------------------------------------------------------------------------
# H4.3 — Failure Mode Classification
# ---------------------------------------------------------------------------

def h4_3_failure_modes(traces, global_null):
    """Classify each irreducible-failure query into a failure mode.

    Modes (applied in order — first match wins):
      1. Weak signal — best relevant p-value (per-query null) > 0.05
      2. Too sparse — relevant docs in gamma window < 3
      3. Z-score distortion — clean z-scoring flips HC positive
      4. Borderline — HC stat (per-query null) > -1.0
      5. Too scattered — remaining cases
    """
    n_gamma = int(np.floor(GAMMA * POOL_SIZE))

    # Identify irreducible queries: negative HC under BOTH global and per-query null
    irreducible = [t for t in traces
                   if t["hc_stat_global"] <= 0 and t["hc_stat_perquery"] <= 0]

    classifications = []
    mode_counts = {"weak_signal": 0, "too_sparse": 0, "zscore_distortion": 0,
                   "borderline": 0, "too_scattered": 0}

    for t in irreducible:
        rpq = np.array(t["relevant_pvals_perquery"])
        ranks_pq = np.array(t["relevant_pval_ranks_perquery"])
        n_rel = len(rpq)

        best_pval_pq = float(np.min(rpq)) if n_rel > 0 else 1.0
        n_rel_in_gamma = int(np.sum(ranks_pq < n_gamma)) if n_rel > 0 else 0
        n_sig = int(np.sum(rpq < 0.05)) if n_rel > 0 else 0

        # clean_flips_hc will be patched by main() after H4.4 runs
        clean_flips_hc = False

        # Classification (order matters)
        if n_rel == 0 or best_pval_pq > 0.05:
            mode = "weak_signal"
        elif n_rel_in_gamma < 3:
            mode = "too_sparse"
        elif clean_flips_hc:
            mode = "zscore_distortion"
        elif t["hc_stat_perquery"] > -1.0:
            mode = "borderline"
        else:
            mode = "too_scattered"

        # Concentration ratio: fraction of top-gamma positions that are relevant
        concentration = n_rel_in_gamma / n_gamma if n_gamma > 0 else 0.0

        mode_counts[mode] += 1
        classifications.append({
            "query_id": t["query_id"],
            "query_text": t["query_text"],
            "mode": mode,
            "hc_stat_global": t["hc_stat_global"],
            "hc_stat_perquery": t["hc_stat_perquery"],
            "best_pval_perquery": best_pval_pq,
            "n_relevant_in_pool": t["n_relevant_in_pool"],
            "n_rel_in_gamma": n_rel_in_gamma,
            "n_significant_005": n_sig,
            "concentration_ratio": concentration,
            "clean_flips_hc": clean_flips_hc,
        })

    return {
        "n_irreducible": len(irreducible),
        "mode_counts": mode_counts,
        "classifications": classifications,
    }


# ---------------------------------------------------------------------------
# H4.4 — Targeted Fix Tests
# ---------------------------------------------------------------------------

def h4_4_targeted_fixes(queries, qrels, all_faiss_results, global_null, traces):
    """Test whether specific interventions rescue failing queries.

    Fix 1: Z-score contamination removal (oracle clean z-scoring)
    Fix 2: Detection boundary analysis (Donoho-Jin phase diagram)
    """
    n_gamma = int(np.floor(GAMMA * POOL_SIZE))

    # Build query_id -> trace index mapping
    trace_by_qid = {t["query_id"]: t for t in traces}

    # Identify irreducible queries
    irreducible_qids = {t["query_id"] for t in traces
                        if t["hc_stat_global"] <= 0 and t["hc_stat_perquery"] <= 0}

    # ------ Fix 1: Contamination removal ------
    fix1_results = []
    fix1_flips = 0

    for qi, query in enumerate(queries):
        if query.query_id not in irreducible_qids:
            continue

        relevant_ids = qrels.get(query.query_id, set())
        faiss_results = all_faiss_results[qi][:POOL_SIZE]

        cids = [cid for cid, _ in faiss_results]
        csims = np.array([sim for _, sim in faiss_results])
        sort_idx = np.argsort(csims)[::-1]
        csims = csims[sort_idx]
        cids = [cids[j] for j in sort_idx]
        rel_mask = np.array([cid in relevant_ids for cid in cids])

        # Standard z-scoring (contaminated)
        zs_result = zscore_candidates(csims, Z_SCORE_FRACTION)
        if zs_result is None:
            continue
        z_scores_dirty, mu_dirty, sigma_dirty = zs_result

        # Clean z-scoring (exclude relevant docs from estimation set)
        sorted_desc = np.sort(csims)[::-1]
        n = len(sorted_desc)
        n_null_est = max(int(n * Z_SCORE_FRACTION), 10)
        bottom_indices = np.arange(n - n_null_est, n)
        bottom_rel = rel_mask[bottom_indices]
        bottom_clean = sorted_desc[bottom_indices][~bottom_rel]

        if len(bottom_clean) < 10:
            continue

        mu_clean = float(np.mean(bottom_clean))
        sigma_clean = float(np.std(bottom_clean))
        if sigma_clean < 1e-10:
            continue

        z_scores_clean = (csims - mu_clean) / sigma_clean

        # Build per-query null from clean non-relevant z-scores
        nonrel_z_clean = z_scores_clean[~rel_mask]
        if len(nonrel_z_clean) < 10:
            continue
        clean_pq_null = NullDistribution(
            similarities=nonrel_z_clean,
            mean=float(np.mean(nonrel_z_clean)),
            std=float(np.std(nonrel_z_clean)),
            min_val=float(np.min(nonrel_z_clean)),
            max_val=float(np.max(nonrel_z_clean)),
            n_samples=len(nonrel_z_clean),
        )

        # HC with clean z-scores + clean per-query null
        pvals_clean = compute_pvalues_empirical(z_scores_clean, clean_pq_null)
        pvals_sorted = np.sort(pvals_clean)
        hc_curve_clean = compute_hc_curve(pvals_sorted, n_gamma, POOL_SIZE)
        hc_stat_clean = float(np.max(hc_curve_clean))

        t = trace_by_qid[query.query_id]
        hc_stat_original = t["hc_stat_perquery"]
        flipped = hc_stat_original <= 0 and hc_stat_clean > 0
        if flipped:
            fix1_flips += 1

        fix1_results.append({
            "query_id": query.query_id,
            "hc_stat_original_pq": hc_stat_original,
            "hc_stat_clean_pq": hc_stat_clean,
            "delta": hc_stat_clean - hc_stat_original,
            "flipped": flipped,
            "mu_dirty": mu_dirty,
            "mu_clean": mu_clean,
            "sigma_dirty": sigma_dirty,
            "sigma_clean": sigma_clean,
            "n_relevant_in_estimation": int(bottom_rel.sum()),
        })

    # ------ Fix 2: Detection boundary analysis ------
    boundary_results = []

    for t in traces:
        rpq = np.array(t["relevant_pvals_perquery"])
        ranks_pq = np.array(t["relevant_pval_ranks_perquery"])
        n_rel = len(rpq)

        n_rel_in_gamma = int(np.sum(ranks_pq < n_gamma)) if n_rel > 0 else 0
        epsilon = n_rel_in_gamma / n_gamma if n_gamma > 0 else 0.0

        rel_z = np.array(t["relevant_zscores"])

        if n_rel > 0 and n_rel_in_gamma > 0:
            in_gamma_mask = ranks_pq < n_gamma
            rel_z_in_gamma = rel_z[in_gamma_mask] if len(in_gamma_mask) > 0 else np.array([])

            if len(rel_z_in_gamma) > 0:
                nonrel_std = t["nonrel_zscores_std"]
                mu_signal = float(np.mean(rel_z_in_gamma)) / nonrel_std if nonrel_std > 0 else 0.0
            else:
                mu_signal = 0.0
        else:
            mu_signal = 0.0

        if epsilon > 0:
            boundary_threshold = np.sqrt(2 * np.log(1.0 / epsilon))
            above_boundary = mu_signal > boundary_threshold
        else:
            boundary_threshold = float("inf")
            above_boundary = False

        if t["hc_stat_global"] > 0:
            group = "positive_hc"
        elif t["hc_stat_perquery"] > 0:
            group = "fixable"
        else:
            group = "irreducible"

        boundary_results.append({
            "query_id": t["query_id"],
            "group": group,
            "epsilon": epsilon,
            "mu_signal": mu_signal,
            "boundary_threshold": boundary_threshold,
            "above_boundary": above_boundary,
            "n_rel_in_gamma": n_rel_in_gamma,
            "hc_stat_perquery": t["hc_stat_perquery"],
        })

    # Aggregate boundary stats by group
    boundary_summary = {}
    for group_name in ["positive_hc", "fixable", "irreducible"]:
        group_items = [b for b in boundary_results if b["group"] == group_name]
        if group_items:
            boundary_summary[group_name] = {
                "n": len(group_items),
                "n_above_boundary": sum(1 for b in group_items if b["above_boundary"]),
                "n_below_boundary": sum(1 for b in group_items if not b["above_boundary"]),
                "mean_epsilon": float(np.mean([b["epsilon"] for b in group_items])),
                "mean_mu_signal": float(np.mean([b["mu_signal"] for b in group_items])),
            }

    return {
        "fix1_contamination_removal": {
            "n_tested": len(fix1_results),
            "n_flipped": fix1_flips,
            "per_query": fix1_results,
        },
        "fix2_detection_boundary": {
            "boundary_formula": "mu > sqrt(2 * log(1/epsilon))",
            "summary_by_group": boundary_summary,
            "per_query": boundary_results,
        },
    }
