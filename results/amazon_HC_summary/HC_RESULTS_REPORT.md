# Higher Criticism (HC) Retrieval — Results Report

**Project:** Adaptive-k Retrieval pipeline
**Scope of this report:** all results in the repository for the HC retrieval algorithm and its comparison against other retrieval methods.
**Date compiled:** 2026-06-13

---

## 1. Overview

Higher Criticism (HC) retrieval is a statistical method for choosing how many documents to
retrieve (`k`) per query. It compares each observed query–context similarity against a
precomputed **null distribution** of random query–context similarities, converts them to
p-values, and selects the cutoff `k` that **maximizes the Higher Criticism statistic**. A
threshold gate lets the retriever abstain (return 0 documents) when no cutoff is significant.

HC is positioned as an alternative to the existing **adaptive-k** method and to fixed-context
baselines (**full-context**, **self-route**).

### Algorithm
1. **Null distribution** — random query↔context similarities, precomputed per sample.
2. **P-values** — for each observed similarity `s`, `p = P(null ≥ s)` with `(n≥ + 1)/(N + 1)` smoothing.
3. **HC statistic** — `HC_i = (i/n − p_i) / √(p_i (1 − p_i) / n)` over similarities sorted descending.
4. **Threshold** — `k = argmax(HC) + 1` searched over the top `γ` fraction of similarities.
5. **Gate** — if `max(HC) < hc_threshold`, return 0 documents.
6. **Optional** — `retrieve_more` adds a safety margin beyond `k`.

### Implementation map
| File | Role |
|---|---|
| `adaptive-k-retrieval/retriever.py` (L670–754) | `_compute_hc_p_values`, `_compute_hc_statistic`, `hc_retrieve` |
| `adaptive-k-retrieval/compute_null_distributions.py` | Precomputes null distributions (`.npy`) |
| `HC_RETRIEVAL_README.md` | Design / usage doc |
| `visualize_hc_diagnostics.py` + `hc_analysis/` | Diagnostic plots |
| `run_hotpotqa_test.sh`, `run_holobench_test.sh` | Experiment runners |

> **Parameter inconsistencies to reconcile before publishing**
> - `γ` = **0.2** in production `retriever.py`, but **0.1** in the README and `visualize_hc_diagnostics.py`.
> - `hc_threshold` = **0.2** for HotpotQA, **0.01** for HoloBench.

---

## 2. Experimental setup

- **Embedding model:** `BAAI/bge-large-en-v1.5` (all experiments)
- **Generation model:** `gpt-4o-mini-2024-07-18` (all experiments)
- **Datasets with results:** HotpotQA (HELMET) and HoloBench. No results exist for NQ,
  TriviaQA, gpt-4o, or any other model.

### Dataset statistics (`dataset_statistics.csv`)
| Dataset | Queries | Unique docs | Total chunks | Avg true (gold) docs |
|---|---|---|---|---|
| HotpotQA | 987 | 1,991 | 987,000 | 2.04 |
| NaturalQuestions | 993 | 986 | 993,000 | 2.96 |
| TriviaQA | 976 | 966 | 976,000 | 2.39 |
| HoloBench (All) | 90 | 260 | 13,647 | 151.63 |

### Methods compared
| Method | Configuration |
|---|---|
| **HC** | null-distribution HC; `hc_threshold` 0.2 (HotpotQA) / 0.01 (HoloBench); γ = 0.2 |
| **adaptive-k** | `largest_gap`, `ignore_extreme 0.05`, `ignore_extreme_tail 0.1`, `retrieve_more 5` |
| **full-context** | all 1000 chunks passed to the LLM |
| **self-route** | fixed 5000-token retrieval + routing prompt template |

### Conditions actually present
| Dataset | Conditions | Samples |
|---|---|---|
| HotpotQA | HC, adaptive-k | 987 each |
| HotpotQA | full-context, self-route | 300 each |
| HoloBench info10k | HC, adaptive-k | 90 each (evaluated) |
| HoloBench info5k | adaptive-k | 10 |
| HoloBench info5k | **HC — retrieval only, never evaluated** | 10 |

---

## 3. HotpotQA results (main comparison)

| Method | N | **F1** | **EM** | Substr-EM | Avg input tokens | Avg #retrieved | Ctx precision | Ctx recall | Ctx-F1 | Reduction ratio | Gen time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **HC** | 987 | **0.332** | **0.150** | 0.624 | **1,320** | **10.9** | **0.221** | 0.624 | **0.299** | **0.989** | 1.06 |
| adaptive-k | 987 | 0.306 | 0.122 | 0.655 | 25,250 | 227.6 | 0.020 | 0.892 | 0.039 | 0.772 | 2.63 |
| full-context | 300 | 0.254 | 0.087 | 0.660 | 109,806 | 1000 | 0.002 | 1.000 | 0.004 | 0.000 | 10.45 |
| self-route | 300 | 0.107 | 0.010 | **0.727** | 28,783 | 35.6 | 0.048 | 0.836 | 0.090 | 0.964 | 5.50 |

**Findings**
- **HC achieves the best F1 (0.332) and EM (0.150) while using ~19× fewer input tokens than
  adaptive-k (1.3k vs 25k) and ~83× fewer than full-context.**
- HC retrieves only **~11 of 1000** chunks (98.9% reduction) yet has **~11× higher precision**
  (0.221 vs 0.020) and the **best context-F1** (0.299 vs 0.039).
- adaptive-k trades precision for recall (0.892 recall, 0.020 precision); full-context has
  perfect recall but near-zero precision; self-route collapses on F1/EM.
- HC also has the fastest generation step (1.06 s) because it feeds the least context.

> **Caveat:** HC vs adaptive-k is a clean comparison (both 987 samples). full-context and
> self-route were only run on **300** samples, so those two rows are not strictly
> apples-to-apples with the other two.

---

## 4. HoloBench info10k results (LLM-as-a-judge)

| Method | N | **Judge score** | Avg input tokens | Avg #retrieved | Ctx precision | Ctx recall | Ctx-F1 | Reduction ratio | Retr time (s) |
|---|---|---|---|---|---|---|---|---|---|
| **HC** | 90 | 0.321 | **5,849** | **179** | **1.000** | 0.606 | **0.706** | 0.394 | **1.46** |
| adaptive-k | 90 | 0.326 | 20,234 | 638 | 0.531 | 0.685 | 0.497 | 0.792 | 11.49 |

**Findings**
- Generation quality is **statistically tied** (0.321 vs 0.326).
- HC reaches that quality with **~3.5× fewer input tokens**, **~3.5× fewer documents**,
  **perfect precision (1.00)**, higher context-F1 (0.706 vs 0.497), and **~8× faster
  retrieval** (1.46 s vs 11.49 s). adaptive-k has marginally higher recall (0.685 vs 0.606).

> **Data-integrity caveat:** the info10k **adaptive-k** eval files were computed at N=90, but
> its `generation_results.json` on disk now contains only **10** samples — it was partially
> overwritten after evaluation. The reported metrics are valid (N=90) but **not reproducible
> from the current results file**. The HC counterpart is intact at 90.

---

## 5. HoloBench info5k results (incomplete)

| Method | N | Judge score | Avg input tokens | Avg #retrieved | Ctx precision | Ctx recall | Ctx-F1 | Reduction |
|---|---|---|---|---|---|---|---|---|
| adaptive-k | 10 | 0.335 | 25,197 | 904.5 | 0.115 | 0.859 | 0.183 | 0.678 |
| **HC** | 10 | **— not evaluated —** | — | 68.6 | — | — | — | 0.364 |

- HC at info5k has **retrieval results only**; no generation or judge evaluation was ever run.
- HC retrieval stats: avg 68.6 docs retrieved (range 6–150), reduction 0.364, avg gold
  `true_k` = 103.2.
- **Not usable for a head-to-head** until HC is evaluated; only 10 samples regardless.

---

## 6. Diagnostic figures

Located in `hc_analysis/` (HotpotQA, sample 9):
- `plot_a_similarities_sample_9.png` — histogram overlay of all similarities vs the null
  distribution vs positive-document similarities.
- `plot_b_pvalues_sample_9.png` — p-value vs rank, positive docs highlighted, p=0.05/0.01 lines.
- `plot_c_hc_curve_sample_9.png` — HC statistic vs `k`, marking the optimal `k` and threshold.

These are single-sample, qualitative illustrations of the method (suitable as a
methods-section figure).

---

## 7. Bottom line

> **HC retrieval matches or beats adaptive-k on answer quality while retrieving an order of
> magnitude less context, with far higher precision.**
> - HotpotQA: HC is the top method on F1/EM at ~19× fewer input tokens than adaptive-k.
> - HoloBench info10k: HC ties adaptive-k on the judge score at ~3.5× fewer tokens and ~8×
>   faster retrieval.

### Gaps to close before publishing
1. Evaluate HoloBench **info5k HC** (currently retrieval-only).
2. Regenerate the info10k **adaptive-k** results file (currently 10 of 90 samples on disk).
3. Reconcile the **γ = 0.1 vs 0.2** discrepancy across code and docs.
4. Optionally broaden scope: add **NQ / TriviaQA** and a **second generator** — every result
   here is gpt-4o-mini on just two datasets.

---

## Appendix A — Source files for every number

| Result | File |
|---|---|
| HotpotQA HC, quality | `RAG_results/hotpotqa/bge-large-en-v1.5/gpt-4o-mini/full/hc-retrieval/generation_eval.json` |
| HotpotQA HC, retrieval | `RAG_results/hotpotqa/bge-large-en-v1.5/gpt-4o-mini/full/hc-retrieval/retrieval_eval.json` |
| HotpotQA adaptive-k, quality | `RAG_results/hotpotqa/bge-large-en-v1.5/gpt-4o-mini/full/adaptive-k-noclass-ignore5p/generation_eval.json` |
| HotpotQA adaptive-k, retrieval | `RAG_results/hotpotqa/bge-large-en-v1.5/gpt-4o-mini/full/adaptive-k-noclass-ignore5p/retrieval_eval.json` |
| HotpotQA full-context | `RAG_results/hotpotqa/bge-large-en-v1.5/gpt-4o-mini/300/full-context/{generation,retrieval}_eval.json` |
| HotpotQA self-route | `RAG_results/hotpotqa/bge-large-en-v1.5/gpt-4o-mini/300/self-route/{generation,retrieval}_eval.json` |
| HoloBench info10k (both methods, summary) | `RAG_results/holobench/bge-large-en-v1.5/gpt-4o-mini/info10k/eval_results_holobench.yaml` |
| HoloBench info10k HC | `RAG_results/holobench/.../info10k/hc-retrieval/{generation,retrieval,holobench}_eval.json` |
| HoloBench info10k adaptive-k | `RAG_results/holobench/.../info10k/adaptive-k-noclass-ignore5p/{generation,retrieval,holobench}_eval.json` |
| HoloBench info5k adaptive-k | `RAG_results/holobench/.../info5k/adaptive-k-noclass-ignore5p/{generation,retrieval}_eval.json` |
| HoloBench info5k HC (retrieval only) | `RAG_results/holobench/.../info5k/hc-retrieval/retrieval_results.json` |

## Appendix B — Experiment provenance (logs)

- `hotpotqa_experiment_log.txt`: HC run on Sun Dec 14 2025 (gpt-4o-mini).
- `holobench_experiment_log.txt`: HC info10k run on Sat Nov 29 2025 (gpt-4o-mini).
