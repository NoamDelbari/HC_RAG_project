# HC Retrieval Investigation — Complete Summary

**Date:** 2026-04-01
**Dataset:** amazon_compound (135 queries, 80K docs, compound category+price queries)
**Project:** Higher Criticism for adaptive document retrieval in RAG

---

## 1. What HC Does

Higher Criticism (HC) is used to adaptively determine how many documents to retrieve per query, replacing fixed top-k. The pipeline:

1. FAISS search returns top `pool_size` candidates (default 1000) ranked by cosine similarity
2. Z-score normalize: sort candidates descending, use bottom 80% to estimate mu/sigma, z-score all candidates
3. Compute p-values: look up z-scores in a pre-built null distribution (empirical CDF)
4. Compute HC statistic: `HC_i = sqrt(n) * (i/n - p_(i)) / sqrt(p_(i) * (1-p_(i)))` for positions i=1..gamma*n
5. Select k = argmax(HC_i) + 1 documents
6. If max(HC) < 0 (min_hc=0.0 gate), return k=0 (no documents)

**Key code:** `src/hc_rag/hc/higher_criticism.py` (HC computation), `src/hc_rag/retrieval/hc_retrieval.py` (retrieval wrapper), `experiments/scripts/pool_null/` (all experimental scripts)

---

## 2. Two Null Distributions

### Original Null (`artifacts/amazon_compound/global_null.pkl`)
- **Build:** For each query, compute similarity to ALL 80K docs. Use ground-truth labels to identify non-relevant. Z-score using mean/std of ALL non-relevant docs. Sample 5000 z-scores per query. Pool across queries.
- **Stats:** mean=0.0002, std=1.0004, 675K samples. Near-standard-normal.
- **Built by:** `experiments/scripts/build_null.py`

### Pool-Matched Null (`artifacts/amazon_compound/pool_null/global_pool_null.pkl`)
- **Build:** For each query, FAISS top-1000. Z-score using bottom 80% of pool (same as inference). Add only non-relevant z-scores to pool.
- **Stats:** mean=0.79, std=2.06, 132K samples. Right-skewed, heavy-tailed.
- **Built by:** `experiments/scripts/pool_null/build_pool_null.py`
- **Rationale:** Matches inference procedure exactly — build and inference use identical z-scoring.

### The Key Difference
At inference, FAISS returns top-1000 candidates (already high similarity). Z-scoring from bottom 800 produces z-scores centered around 0.8 with std ~2. The original null (centered at 0, std 1) has never seen z-scores this high — everything looks like an extreme outlier. The pool-matched null represents the actual inference z-score distribution.

---

## 3. Starting Point — HC Worked Well

The original report (`results/HC_Adaptive_Retrieval_Report.md`) showed HC beating all baselines using the ORIGINAL null:

| Method | Recall | Precision | F1 | Mean K |
|---|---|---|---|---|
| Top-20 | 0.180 | 0.192 | 0.186 | 20 |
| Top-29 (avg relevant) | 0.220 | 0.171 | 0.192 | 29 |
| Top-50 | 0.287 | 0.139 | 0.187 | 50 |
| **HC (original null)** | **0.228** | **0.185** | **0.204** | **34.1** |

HC config: gamma=0.1, pool_size=1000, z_score_fraction=0.8.

---

## 4. The Investigation — What We Found

### 4.1 Pool-Matched Null Has Better Calibration But Worse Retrieval

| Metric | Original Null | Pool-Matched Null |
|---|---|---|
| P-value calibration (KS stat) | 0.21 (poor) | **0.04** (good) |
| Non-relevant p<0.05 rate | 24% (should be 5%) | **5.0%** (correct) |
| Relevant p<0.05 rate | 71.6% | 38.5% |
| Mean HC stat | 417.9 | 4.16 |
| Mean K | 34.1 | 40.1 |
| Mean F1 | **0.176** | 0.160 |

**The "correct" null made results worse.** (Experiments E, compare_nulls.py)

### 4.2 Recall Ceiling by Pool Size

| K (pool size) | Mean Recall Ceiling |
|---|---|
| 1000 | 0.728 |
| 2000 | 0.796 |
| 4000 | 0.861 |
| 6000 | 0.892 |
| 10000 | 0.918 |

At pool=1000, 27% of relevant docs aren't even in the pool. (Experiment C)

### 4.3 F1 Drops With Larger Pool Sizes

Using pool-matched null at each pool size, gamma=0.1:

| Pool | F1 | Mean K |
|---|---|---|
| 2000 | 0.150 | 79 |
| 4000 | 0.126 | 155 |
| 6000 | 0.111 | 219 |
| 10000 | 0.097 | 311 |

K grows linearly with pool size (~0.03 * pool_size). (Experiment D)

### 4.4 Fixed Gamma (gamma=43/pool_size) Doesn't Fix It

Even fixing the HC search window to exactly 43 positions (max relevant docs):

| Pool | Gamma | F1 | Mean K | HC Stat |
|---|---|---|---|---|
| 1000 | 0.043 | 0.144 | 16.9 | 3.05 |
| 4000 | 0.011 | 0.122 | 16.0 | 1.43 |
| 10000 | 0.004 | 0.100 | 14.2 | 0.43 |

F1 still drops. HC stat drops from 3.05 to 0.43 — signal becomes less detectable at larger pools. (Experiment F)

### 4.5 Query Signal Strength

| Group | Queries | % | Signal Rate | Sim Gap |
|---|---|---|---|---|
| Strong | 26 | 19% | 0.953 | +0.054 |
| Medium | 30 | 22% | 0.593 | -0.008 |
| Weak | 79 | **59%** | 0.090 | **-0.065** |

59% of queries have weak signal — relevant docs score LOWER than top non-relevant docs. The embedding model (text-embedding-3-small) can't capture price constraints. (Signal analysis experiment)

### 4.6 No Null Passes All 4 Validation Tests

| Null | Split-Half | Uniformity | Signal Detection | HC-K |
|---|---|---|---|---|
| Original | PASS | FAIL (0%) | PASS (76%) | PASS |
| Pool 1000 | PASS | FAIL (68%) | FAIL (47%) | PASS |
| Pool 4000 | PASS | FAIL (52%) | PASS (62%) | PASS |

Fundamental tradeoff: original null passes signal detection but fails uniformity. Pool nulls have better uniformity but worse signal detection. (Null deep analysis)

### 4.7 The Root Cause of F1 Discrepancy

**41 of 135 queries (30.4%) have negative HC stat** with the pool-matched null. The HigherCriticism class has a `min_hc=0.0` gate: when HC stat < 0, it returns k=0 (retrieve nothing). With the original null, HC stat is always 400+ (never triggers the gate).

When the gate is removed (manual HC computation), the pool null actually **outperforms** the original null (F1 0.186 vs 0.176).

The gate is theoretically correct (negative HC = no signal detected = don't retrieve), but returning k=0 for 30% of queries is catastrophic for mean F1.

---

## 5. Why the Original Null "Works"

The original null (mean=0, std=1) is on a completely different scale than inference z-scores (which reach z=10-20). When these z-scores are looked up against the original null, ALL top candidates get p-values at the floor (p=0.0000015). This causes HC_i to grow monotonically (no natural peak), and the argmax lands wherever the first non-floor p-value appears — typically in a reasonable range (20-50) by accident.

The original null works because:
1. Its miscalibration makes ALL candidates look significant → HC always finds signal → k>0 always
2. The p-value floor creates a monotonically rising HC curve → argmax is at a late position → reasonable k
3. It avoids the k=0 gate entirely (HC stat is always 400+)

---

## 6. Why HC Degrades at Larger Pools

Two mechanisms:

**A. Z-score dilution:** At pool=10000, the bottom 80% (8000 docs) spans a much wider similarity range than at pool=1000 (800 docs). The wider sigma compresses ALL z-scores, making relevant and non-relevant less distinguishable. HC stat drops from 3.05 to 0.43.

**B. HC argmax under null concentrates at gamma*n boundary (Ferger 2025).** When HC can't detect signal, argmax drifts to the edge of the gamma window. With gamma=0.1, pool=10000, this edge is position 1000 → k=1000. With fixed gamma, this is avoided but signal dilution still occurs.

---

## 7. HC Theory Insights

From literature review (Donoho & Jin 2004, 2008, 2015; Ferger 2025):

1. **HC was designed for detection (is there signal?), NOT thresholding (where is the boundary?).** The argmax position was meant to define a p-value threshold, not directly count features.

2. **HCT (HC Thresholding) assumes signal fraction is fixed as n grows.** In retrieval, relevant docs are fixed (~30) while pool grows → signal fraction shrinks → crosses detection boundary.

3. **Under H0, argmax concentrates at the gamma window boundary.** This is proven mathematically (Ferger 2025). k grows linearly with n when gamma is fixed.

4. **Three promising alternatives from literature:**
   - HC detection + local FDR selection (lfdr < threshold) — doesn't depend on n
   - Fixed absolute gamma window — gamma = max_relevant/pool_size
   - HC value threshold — select last position where HC_i > critical_value

---

## 8. Experiments Run

| Experiment | Script | What it tests |
|---|---|---|
| B: Code paths | exp_b_codepath.py | Manual vs class HC — identical |
| C: Recall ceiling | exp_c_recall_ceiling.py | Max recall at each pool size |
| D: Pool x gamma grid | exp_d_pool_gamma_grid.py | F1 across (pool, gamma) combinations |
| D2: Diagnostics | exp_d2_pool_diagnostics.py | P-value calibration, k tracking, HC curves |
| E: Null methods | exp_e_null_methods.py | Original vs pool-matched vs hybrid null |
| F: Fixed gamma | exp_f_fixed_gamma.py | gamma = 43/pool_size across pool sizes |
| G: Selection methods | exp_g_selection_methods.py | HC argmax vs FDR vs p-threshold vs gap |
| Signal analysis | exp_signal_analysis.py | Strong/medium/weak query categorization |
| Null deep analysis | exp_null_deep_analysis.py | Null shape, validation, p-value comparison |
| HC trace | exp_hc_trace.py | Per-query HC curve tracing, both nulls |
| Discrepancy test | exp_hc_discrepancy_test.py | min_hc=0 gate as root cause |

All scripts in: `experiments/scripts/pool_null/`
All results in: `results/amazon_compound/null_research/`

---

## 9. Current Open Questions

### Q1: Why do 41/135 queries (30%) have negative HC stat with pool-matched null?
A negative HC stat means the observed p-values are LESS extreme than expected under uniformity — the data looks "more null than null." This is the most pressing question. Possible causes:
- The pool-matched null is too wide (std=2.06), making most z-scores look unremarkable
- The z-scoring from bottom 80% of pool overestimates sigma for queries with many relevant docs in the bottom 80%
- The embedding model simply can't distinguish relevant from non-relevant for these queries (59% have negative sim gap)

### Q2: What is the right selection rule for retrieval?
HC argmax is theoretically meant for detection. Alternatives: FDR-based selection (too conservative at 0.05), p-threshold (grows with n), gap-based (too aggressive, k=4). None beat fixed top-k in experiments so far.

### Q3: Is the original null's "beneficial miscalibration" a viable approach for the paper?
It works empirically but for theoretically wrong reasons. A reviewer would question it.

### Q4: Should we explore different datasets?
The amazon_compound dataset has 59% weak-signal queries (embedding can't capture price constraints). A simpler dataset (category-only) might show HC's true advantage.

---

## 10. Key Files

```
src/hc_rag/hc/higher_criticism.py    — HC statistic computation
src/hc_rag/hc/null_distribution.py   — NullDistribution class
src/hc_rag/retrieval/hc_retrieval.py — HCRetrieval wrapper
experiments/scripts/pool_null/       — All experimental scripts
experiments/configs/amazon_compound.yaml — Config (pool=1000, gamma=0.1, z_frac=0.8)
artifacts/amazon_compound/           — Vector DB + null distributions
results/amazon_compound/             — All results
docs/superpowers/specs/              — Design specs for experiments
```

## 11. Config Parameters

```yaml
hc:
  gamma: 0.1        # HC searches positions 1..gamma*pool_size
  pool_size: 1000    # FAISS top-K candidates
null:
  z_score_fraction: 0.8  # Bottom 80% of pool for mu/sigma estimation
```
