# HC Pipeline Trace — Research Design Spec

**Date:** 2026-04-11
**Dataset:** amazon_compound (135 queries, 80K docs, compound category+price queries)
**Goal:** Trace the full HC pipeline per query to determine WHY each failing query produces negative HC, and test whether targeted interventions can rescue them.

---

## Motivation

Previous research ([hc_failure_mechanisms_results.md](../../research/hc_failure_mechanisms_results.md)) identified the root cause of HC failure as a per-query z-scoring x global null mismatch (std_ratio predicts HC sign at r=0.83). Per-query null reduces negative-HC queries from 55 to 45 and improves F1 from 0.144 to 0.217.

However, the research has a critical gap: **the full pipeline was never traced for failing queries.** We know the aggregate statistics (sim_gap, std_ratio, rank positions), but we never looked at:
- What p-values relevant docs actually receive
- Where relevant docs land in the p-value ordering (which is what HC sees)
- What the HC curve looks like at each position for a failing query
- Whether the 45 "irreducible" failures are genuinely undetectable or fail for specific, fixable reasons

The claim that 45 queries are "genuine no-signal cases" was inferred from "per-query null still negative," but this conflates several possible failure modes: weak signal, sparse signal, scattered signal, z-score distortion, and detection boundary violations. This research separates them.

## Connection to Previous Research

- **H1–H3 research** ([plan](../plans/2026-04-09-hc-failure-mechanisms.md), [spec](2026-04-08-hc-failure-mechanisms-design.md)): Established the std_ratio mechanism and per-query null as theoretical best case
- **This research (H4)**: Traces the actual pipeline to verify WHY each query fails, filling the gap between aggregate statistics and per-query causation
- **Next step**: After H4, combine H1–H4 findings into actionable research (method improvements, dataset selection)

---

## Global Research Question

For each query that produces negative HC stat: at which pipeline step does the signal break down? Is it the embedding (no separation), the z-scoring (distortion/contamination), the null (miscalibration), or the HC statistic itself (insufficient detection power for this signal regime)?

## Global Parameters

- **Pool size:** 1000 (fixed)
- **Gamma:** 43/1000 = 0.043 (adaptive, MAX_RELEVANT / pool_size)
- **z_score_fraction:** 0.8 (default)
- **Null:** Both global pool-matched null and per-query null

---

## Experiment H4.1 — Full Pipeline Trace

### Research Question

What are the intermediate values at each pipeline step for every query?

### What it does

For each of the 135 queries:
1. FAISS top-1000 candidates with cosine similarities
2. Label each candidate as relevant/non-relevant using ground truth
3. Z-score all candidates (bottom 80% for mu/sigma estimation)
4. Compute p-values against the global pool-matched null
5. Compute p-values against a per-query null (built from this query's non-relevant z-scores)
6. Compute the full HC curve (HC_i at each position i=1..43) under both nulls
7. Record the max HC stat, argmax position, and resulting k

### What it records per query

- `query_id`, `query_text`, `true_k` (number of relevant docs in ground truth)
- `n_relevant_in_pool` (recall ceiling)
- `mu_est`, `sigma_est` (z-scoring parameters)
- `candidate_sims` (array of 1000 raw cosine similarities, sorted descending)
- `candidate_is_relevant` (boolean array matching candidate_sims)
- `candidate_zscores` (array of z-scores matching candidate_sims)
- `candidate_pvals_global` (p-values against global null)
- `candidate_pvals_perquery` (p-values against per-query null)
- `hc_curve_global` (HC_i values at positions 1..43, global null)
- `hc_curve_perquery` (HC_i values at positions 1..43, per-query null)
- `hc_stat_global`, `hc_stat_perquery` (max HC under each null)
- `k_global`, `k_perquery` (selected k under each null)

### What it validates

This is the raw data layer. All subsequent analyses (H4.2, H4.3, H4.4) are computed from this trace. Having per-query intermediate values allows us to inspect individual queries and verify any aggregate claim.

### Output

`h4_pipeline_traces.json` — Per-query trace data for all 135 queries.

Note: To keep file size manageable, store only the top-100 candidates' full trace (sims, labels, z-scores, p-values) per query, plus summary statistics for the full 1000. The HC curve (43 values) is stored in full.

---

## Experiment H4.2 — Relevant Document P-Value Analysis

### Research Question

What p-values do relevant documents actually receive? Do they look significant under the null, or are they indistinguishable from non-relevant candidates?

### What it does

From the H4.1 traces, extract for each query:
- P-values of all relevant docs in the pool (under both nulls)
- Ranks of relevant docs in the **p-value ordering** (ascending p-value = most significant first). This is what HC actually sees — not the similarity ordering.
- Count of relevant docs in the gamma window (positions 1–43 by p-value rank)
- Best (smallest) p-value achieved by any relevant doc
- Fraction of relevant docs with p < 0.05 (significant)
- Fraction of relevant docs with p < 0.01 (highly significant)

### What it validates

- Whether relevant docs get small p-values at all (if not → signal is genuinely absent)
- Whether relevant docs appear in the gamma window (if not → HC can't see them regardless of signal)
- Whether per-query null gives better p-values to relevant docs than global null

### Analysis Groups

Compare across three groups:
1. **Positive-HC (80 queries):** HC works, baseline for comparison
2. **Negative-HC fixable (10 queries):** Negative under global null, positive under per-query null — the std_ratio mismatch cases
3. **Negative-HC irreducible (45 queries):** Negative under both nulls — the ones we need to explain

### Output

`h4_relevant_pvalues.json` — Per-group aggregate statistics and per-query relevant doc p-value data.

---

## Experiment H4.3 — Failure Mode Classification

### Research Question

For each of the 45 irreducible-failure queries, which specific failure mode explains the negative HC?

### Failure Modes

Classify each query into one or more of these modes based on its H4.1 trace data:

**Mode 1 — Weak signal:** The best relevant doc p-value (under per-query null) is > 0.05. Even with perfect calibration, no relevant doc is statistically distinguishable from noise. This is a genuine embedding failure.

**Mode 2 — Too sparse:** At least one relevant doc has p < 0.05, but fewer than 3 relevant docs fall in the gamma window (top 43 by p-value). HC needs a sufficient fraction of signal to detect deviation from uniformity — a single significant doc among 43 isn't enough.

**Mode 3 — Too scattered:** 3+ relevant docs have p < 0.05, but they're interspersed with non-relevant docs at similar p-values. The ordered p-value sequence shows no clear departure from uniformity because relevant and non-relevant p-values are interleaved.

Operationalize: compute the "concentration ratio" — among positions 1–43 (by p-value), what fraction are relevant? If signal_fraction < 0.1 despite having relevant docs with good p-values, classify as scattered.

**Mode 4 — Z-score distortion:** Compare HC stat using "clean" z-scores (mu/sigma estimated excluding all relevant docs from the estimation set) vs standard z-scores. If clean z-scoring flips HC from negative to positive (under per-query null), the contamination in the estimation set is the cause.

**Mode 5 — Borderline:** HC stat (per-query null) is in range [-1.0, 0). The query is near the detection boundary — signal exists but is at the edge of HC's detection power.

### Classification Rules

Apply in order (a query gets classified into the FIRST matching mode):
1. If best relevant p-value > 0.05 → Mode 1
2. If relevant docs in gamma window < 3 → Mode 2
3. If clean z-scoring flips HC positive → Mode 4
4. If HC stat > -1.0 → Mode 5
5. Otherwise → Mode 3

### What it validates

Produces a distribution: "X queries fail because of A, Y because of B." This replaces the blanket assumption that all 45 are "genuine no-signal" and identifies which failure modes are dominant and potentially addressable.

### Output

`h4_failure_modes.json` — Per-query classification with supporting evidence, aggregate mode distribution.

---

## Experiment H4.4 — Targeted Fix Tests

### Research Question

Can specific interventions rescue failing queries, and which failure modes do they address?

### Fix 1 — Z-Score Contamination Removal

**What it does:**
- For each of the 45 irreducible queries, re-estimate mu/sigma using ONLY candidates confirmed non-relevant (oracle clean z-scoring)
- Recompute z-scores, p-values (per-query null rebuilt from clean z-scores), and HC stat
- Compare: original HC stat vs clean HC stat, delta, how many queries flip from negative to positive

**What it validates:**
- Whether the z-scoring step is self-sabotaging — relevant docs in the bottom 80% inflate sigma and suppress z-scores of the top candidates
- If significant flips occur, the contamination is a fixable cause (e.g., by using a smaller z_score_fraction or contamination-robust estimation)

**Output:** Per-query delta in HC stat, number of flips, aggregate summary.

### Fix 2 — Detection Boundary Analysis

**What it does:**
- For each of the 135 queries, compute:
  - **Signal fraction** (epsilon): number of relevant docs in gamma window / (gamma * n). This is the fraction of "non-null" observations HC is trying to detect.
  - **Signal strength** (mu): mean z-score of relevant docs in gamma window minus mean z-score of non-relevant docs in gamma window, divided by std of non-relevant z-scores. This is the effect size.
- Plot all 135 queries on the Donoho-Jin detection phase diagram (epsilon vs mu plane)
- The detection boundary is: mu > sqrt(2 * log(1/epsilon)) (simplified). Queries below this boundary are theoretically undetectable.

**What it validates:**
- Whether the 45 irreducible failures are below the detection boundary (HC CANNOT detect them — theoretical limit) vs above it (HC SHOULD detect them — implementation issue)
- If many are above the boundary, something in the pipeline is losing signal that theoretically should be detectable
- Connects our empirical findings to the Donoho-Jin theory — strong for the paper

**Output:** Per-query (epsilon, mu) coordinates, boundary classification (above/below), aggregate counts.

---

## Experiment Organization

### Script

One script: `exp_h4_pipeline_trace.py`

All four experiments share the same per-query FAISS retrieval loop. H4.1 collects the traces, H4.2–H4.4 analyze them in the same script.

### Dependencies

- H4.1 produces the trace data
- H4.2, H4.3, H4.4 all consume H4.1 traces
- H4.3 Mode 4 (clean z-scoring) computes its own z-scores but uses H4.1's candidates
- No dependency on H1–H3 results (fully self-contained)

### Output Structure

```
results/amazon_compound/null_research/h4_pipeline_trace/
  h4_pipeline_traces.json     # Per-query traces (top-100 candidates + HC curves)
  h4_relevant_pvalues.json    # Relevant doc p-value analysis by group
  h4_failure_modes.json       # Classification of 45 irreducible failures
  h4_targeted_fixes.json      # Contamination removal + detection boundary results
```

### Existing Code Reuse

- Pool null loading: `artifacts/amazon_compound/pool_null/global_pool_null.pkl`
- Z-scoring helper: `zscore_candidates()` from H2 script (copy pattern)
- HC computation: `src/hc_rag/hc/higher_criticism.py` — use `_compute_hc_values` for HC curve
- FAISS retrieval: `src/hc_rag/embeddings/vector_database.py`
- Config/adapter: `experiments/lib/config.py`, `experiments/lib/registry.py`
- Results I/O: `experiments/lib/results_io.py`

### Config

- Base config: `experiments/configs/amazon_compound.yaml`
- Gamma: 43/1000 = 0.043 (hardcoded as MAX_RELEVANT / POOL_SIZE)
- z_score_fraction: 0.8
- Pool size: 1000
