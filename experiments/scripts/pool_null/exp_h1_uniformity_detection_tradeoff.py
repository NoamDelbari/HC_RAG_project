"""Direction 1: Uniformity–detection tradeoff characterization.

H1.1 — Monotonicity test: Run 4-test validation at pool sizes 1K–10K (step 1K)
       and z_score_fractions [0.5, 0.6, 0.7, 0.8]. Produces a 2D grid of
       uniformity pass rate and signal detection rate.

H1.2 — Artifact test: At each (pool_size, z_score_fraction), split queries into
       "uniform" vs "non-uniform" by per-query KS test, compute signal detection
       rate for each group separately.

Usage:
    python -m experiments.scripts.pool_null.exp_h1_uniformity_detection_tradeoff \
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
MAX_RELEVANT = 43  # global max relevant docs across all queries
POOL_SIZES = list(range(1000, 10001, 1000))
Z_SCORE_FRACTIONS = [0.5, 0.6, 0.7, 0.8]


# ---------------------------------------------------------------------------
# Helpers (inlined from exp_null_deep_analysis.py pattern)
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


def build_pool_null_at_size(queries, qrels, all_faiss_results, pool_size, z_score_fraction):
    """Build pool-matched null at a given pool_size from pre-computed FAISS results."""
    all_z_scores = []
    for i, query in enumerate(queries):
        candidates = all_faiss_results[i][:pool_size]
        if len(candidates) == 0:
            continue
        candidate_ids = [c[0] for c in candidates]
        candidate_sims = np.array([c[1] for c in candidates])
        result = zscore_candidates(candidate_sims, z_score_fraction)
        if result is None:
            continue
        z_scores, _, _ = result
        relevant = qrels.get(query.query_id, set())
        nonrel_mask = np.array([cid not in relevant for cid in candidate_ids])
        nonrel_z = z_scores[nonrel_mask]
        if len(nonrel_z) > 0:
            all_z_scores.append(nonrel_z)
    pooled = np.concatenate(all_z_scores)
    return NullDistribution(
        similarities=pooled,
        mean=float(np.mean(pooled)),
        std=float(np.std(pooled)),
        min_val=float(np.min(pooled)),
        max_val=float(np.max(pooled)),
        n_samples=len(pooled),
    )


# ---------------------------------------------------------------------------
# H1.1: Monotonicity test — 4-test validation at each (pool_size, z_frac)
# ---------------------------------------------------------------------------

def run_validation(null_dist, queries, qrels, all_faiss_results,
                   pool_size, z_score_fraction, gamma):
    """Run the 4-test validation gate. Returns dict with per-test results."""
    null_sorted = np.sort(null_dist.similarities)
    N_null = len(null_sorted)
    hc_module = HigherCriticism(null_distribution=null_dist)

    # --- Test 1: Split-half KS ---
    sims = null_dist.similarities
    ks_stats, ks_pvals = [], []
    for seed in range(20):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(sims))
        half = len(idx) // 2
        ref_sorted = np.sort(sims[idx[:half]])
        test_half = sims[idx[half:]]
        pv = np.clip(1.0 - np.searchsorted(ref_sorted, test_half) / len(ref_sorted), 1e-10, 1.0)
        ks_stat, ks_p = sp_stats.kstest(pv, "uniform")
        ks_stats.append(ks_stat)
        ks_pvals.append(ks_p)
    t1_passed = bool(np.median(ks_pvals) > 0.05 and np.median(ks_stats) < 0.1)
    t1 = {"test": "split_half_ks", "passed": t1_passed,
           "median_ks_stat": float(np.median(ks_stats)),
           "median_ks_pval": float(np.median(ks_pvals))}

    # --- Test 2: Per-query uniformity (all queries, not sampled) ---
    # Also collect per-query KS results for H1.2
    per_query_ks = []  # list of (query_index, ks_pval, passed_bool)
    for qi, query in enumerate(queries):
        relevant = qrels.get(query.query_id, set())
        candidates = all_faiss_results[qi][:pool_size]
        if len(candidates) < 10:
            continue
        cids = [c[0] for c in candidates]
        csims = np.array([c[1] for c in candidates])
        res = zscore_candidates(csims, z_score_fraction)
        if res is None:
            continue
        z_scores, _, _ = res
        nonrel_mask = np.array([cid not in relevant for cid in cids])
        nonrel_z = z_scores[nonrel_mask]
        if len(nonrel_z) < 5:
            continue
        pv = np.clip((N_null - np.searchsorted(null_sorted, nonrel_z, side="left") + 1.0) / (N_null + 1.0), 1e-10, 1.0)
        _, ks_p = sp_stats.kstest(pv, "uniform")
        per_query_ks.append((qi, float(ks_p), bool(ks_p > 0.05)))

    n_tested = len(per_query_ks)
    n_passed = sum(1 for _, _, p in per_query_ks if p)
    pass_rate = n_passed / n_tested if n_tested > 0 else 0.0
    t2 = {"test": "per_query_uniformity", "passed": bool(pass_rate >= 0.70),
           "pass_rate": float(pass_rate), "n_queries_tested": n_tested}

    # --- Test 3: Signal detection (all queries) ---
    all_rel_pvals = []
    per_query_signal = []  # for H1.2: (query_index, signal_rate)
    for qi, query in enumerate(queries):
        relevant = qrels.get(query.query_id, set())
        if not relevant:
            continue
        candidates = all_faiss_results[qi][:pool_size]
        if len(candidates) < 10:
            continue
        cids = [c[0] for c in candidates]
        csims = np.array([c[1] for c in candidates])
        res = zscore_candidates(csims, z_score_fraction)
        if res is None:
            continue
        z_scores, _, _ = res
        rel_mask = np.array([cid in relevant for cid in cids])
        rel_z = z_scores[rel_mask]
        if len(rel_z) == 0:
            per_query_signal.append((qi, 0.0))
            continue
        pv = np.clip((N_null - np.searchsorted(null_sorted, rel_z, side="left") + 1.0) / (N_null + 1.0), 1e-10, 1.0)
        rate = float(np.mean(pv < 0.05))
        per_query_signal.append((qi, rate))
        all_rel_pvals.extend(pv.tolist())

    signal_rate = float(np.mean(np.array(all_rel_pvals) < 0.05)) if all_rel_pvals else 0.0
    t3 = {"test": "signal_detection", "passed": bool(signal_rate >= 0.60),
           "signal_rate": signal_rate, "n_relevant_docs": len(all_rel_pvals)}

    # --- Test 4: HC K correlation ---
    hc_stats, hc_ks = [], []
    for qi, query in enumerate(queries):
        candidates = all_faiss_results[qi][:pool_size]
        if len(candidates) < 10:
            continue
        cids = [c[0] for c in candidates]
        csims = np.array([c[1] for c in candidates])
        res = zscore_candidates(csims, z_score_fraction)
        if res is None:
            continue
        z_scores, _, _ = res
        r = hc_module.compute_hc_threshold(z_scores, gamma=gamma)
        hc_stats.append(r.hc_statistic)
        hc_ks.append(r.k)
    mean_hc = float(np.mean(hc_stats)) if hc_stats else 0.0
    unique_ks = len(set(hc_ks))
    t4 = {"test": "hc_k_correlation", "passed": bool(mean_hc > 1.0 and unique_ks >= 3),
           "mean_hc_statistic": mean_hc, "unique_k_values": unique_ks,
           "n_negative_hc": int(sum(1 for s in hc_stats if s < 0))}

    all_passed = all(t["passed"] for t in [t1, t2, t3, t4])

    return {
        "all_passed": all_passed,
        "tests": [t1, t2, t3, t4],
        "per_query_ks": per_query_ks,
        "per_query_signal": per_query_signal,
    }


# ---------------------------------------------------------------------------
# H1.2: Artifact test — signal detection split by uniformity group
# ---------------------------------------------------------------------------

def compute_artifact_test(validation_result):
    """Split signal detection by uniform vs non-uniform queries."""
    per_query_ks = validation_result["per_query_ks"]
    per_query_signal = validation_result["per_query_signal"]

    # Build lookup: query_index -> ks_passed
    ks_map = {qi: passed for qi, _, passed in per_query_ks}
    # Build lookup: query_index -> signal_rate
    sig_map = {qi: rate for qi, rate in per_query_signal}

    # Only include queries that appear in both
    common_qi = set(ks_map.keys()) & set(sig_map.keys())

    uniform_rates = [sig_map[qi] for qi in common_qi if ks_map.get(qi, False)]
    non_uniform_rates = [sig_map[qi] for qi in common_qi if not ks_map.get(qi, False)]

    return {
        "uniform_group": {
            "n_queries": len(uniform_rates),
            "mean_signal_rate": float(np.mean(uniform_rates)) if uniform_rates else 0.0,
            "median_signal_rate": float(np.median(uniform_rates)) if uniform_rates else 0.0,
        },
        "non_uniform_group": {
            "n_queries": len(non_uniform_rates),
            "mean_signal_rate": float(np.mean(non_uniform_rates)) if non_uniform_rates else 0.0,
            "median_signal_rate": float(np.median(non_uniform_rates)) if non_uniform_rates else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="H1: Uniformity-detection tradeoff")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--force", action="store_true", help="Rerun even if output exists")
    args = parser.parse_args()

    config = load_config(args.config)
    adapter = get_adapter(config.dataset.name)
    artifacts_dir = Path(config.dataset.artifacts_dir)
    results_dir = Path(config.dataset.results_dir) / "null_research" / "h1_uniformity_detection"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_mono = results_dir / "h1_monotonicity.json"
    out_artifact = results_dir / "h1_artifact_test.json"

    if out_mono.exists() and out_artifact.exists() and not args.force:
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

    # Pre-compute FAISS at max pool size
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

    # ===== H1.1: Monotonicity — 2D grid (pool_size x z_score_fraction) =====
    print("\n" + "=" * 80)
    print("H1.1: MONOTONICITY TEST")
    print("=" * 80)

    mono_results = {}
    artifact_results = {}

    for z_frac in Z_SCORE_FRACTIONS:
        print(f"\n--- z_score_fraction = {z_frac} ---")
        mono_results[str(z_frac)] = {}
        artifact_results[str(z_frac)] = {}

        for ps in POOL_SIZES:
            gamma = MAX_RELEVANT / ps
            print(f"  pool={ps}, gamma={gamma:.4f} ... ", end="", flush=True)

            # Build pool-matched null at this (pool_size, z_frac)
            null_dist = build_pool_null_at_size(
                queries, qrels, all_faiss_results, ps, z_frac
            )

            # Run full validation
            val = run_validation(
                null_dist, queries, qrels, all_faiss_results,
                ps, z_frac, gamma
            )

            # Store H1.1 results (strip per-query data for JSON)
            mono_results[str(z_frac)][str(ps)] = {
                "all_passed": val["all_passed"],
                "tests": val["tests"],
            }

            # Store H1.2 artifact test
            artifact = compute_artifact_test(val)
            artifact_results[str(z_frac)][str(ps)] = artifact

            # Print summary
            t2_rate = val["tests"][1]["pass_rate"]
            t3_rate = val["tests"][2]["signal_rate"]
            n_neg = val["tests"][3]["n_negative_hc"]
            u_n = artifact["uniform_group"]["n_queries"]
            u_sig = artifact["uniform_group"]["mean_signal_rate"]
            nu_n = artifact["non_uniform_group"]["n_queries"]
            nu_sig = artifact["non_uniform_group"]["mean_signal_rate"]
            print(f"unif={t2_rate:.2f} sig={t3_rate:.2f} neg_hc={n_neg} | "
                  f"uniform({u_n}): sig={u_sig:.3f}, non-uniform({nu_n}): sig={nu_sig:.3f}")

    # Save
    save_results_json({
        "config": {
            "dataset": config.dataset.name,
            "pool_sizes": POOL_SIZES,
            "z_score_fractions": Z_SCORE_FRACTIONS,
            "max_relevant": MAX_RELEVANT,
            "n_queries": len(queries),
        },
        "grid": mono_results,
    }, str(out_mono))
    print(f"\nSaved: {out_mono}")

    save_results_json({
        "config": {
            "dataset": config.dataset.name,
            "pool_sizes": POOL_SIZES,
            "z_score_fractions": Z_SCORE_FRACTIONS,
            "max_relevant": MAX_RELEVANT,
            "n_queries": len(queries),
        },
        "grid": artifact_results,
    }, str(out_artifact))
    print(f"Saved: {out_artifact}")

    # ===== Summary table =====
    print("\n" + "=" * 80)
    print("SUMMARY: Uniformity pass rate / Signal detection rate")
    print("=" * 80)
    for z_frac in Z_SCORE_FRACTIONS:
        print(f"\n  z_score_fraction = {z_frac}")
        print(f"  {'Pool':>6} {'Uniformity':>12} {'SignalDet':>12} {'SplitHalf':>12} {'HC-K':>12} {'NegHC':>8}")
        print(f"  {'-'*62}")
        for ps in POOL_SIZES:
            r = mono_results[str(z_frac)][str(ps)]
            t1, t2, t3, t4 = r["tests"]
            print(f"  {ps:>6} {t2['pass_rate']:>12.3f} {t3['signal_rate']:>12.3f} "
                  f"{'PASS' if t1['passed'] else 'FAIL':>12} "
                  f"{'PASS' if t4['passed'] else 'FAIL':>12} "
                  f"{t4['n_negative_hc']:>8}")

    print("\n" + "=" * 80)
    print("H1 complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
