# HC for Adaptive Document Retrieval — Research Paper Design Spec

**Date**: 2026-03-29
**Goal**: Design the experiments, baselines, datasets, and paper structure for a research paper on Higher Criticism as a principled adaptive retrieval method.

---

## 1. Paper Thesis

**Working title:** *"Higher Criticism for Adaptive Document Retrieval: A Statistically Principled Alternative to Fixed Top-k"*

**One-line thesis:** Document retrieval is a sparse signal detection problem — Higher Criticism, which is provably optimal for this regime, provides a training-free, per-query adaptive method that replaces fixed top-k with a statistically principled threshold.

**Narrative arc:**

1. **The problem:** Fixed top-k retrieval is a fundamental limitation. No single k works for all queries — easy queries need few documents, hard queries need many. Over-retrieval wastes tokens and introduces distraction (Lost in the Middle, Liu et al. TACL 2024); under-retrieval misses relevant information.

2. **Why existing solutions fall short:** Adaptive-k (EMNLP 2025) uses a max-gap heuristic with no theoretical foundation. Surprise (SIGIR 2023) models only the score tail via EVT. Learned methods (Choppy, DynamicRAG) require training data. Conformal methods guarantee coverage but not detection optimality.

3. **The key insight:** Retrieval from an embedding space is structurally identical to sparse signal detection — a few relevant documents (sparse signals) embedded among many non-relevant ones (noise). HC is provably optimal for exactly this regime (Donoho & Jin, 2004).

4. **The method:** Convert cosine similarity scores to p-values against a calibrated null distribution, then apply HC to find the threshold where the number of significant scores deviates most from chance. No training, no hyperparameter k, works with any embedding model.

5. **Results on retrieval benchmarks:** HC outperforms fixed top-k and matches or beats Adaptive-k on datasets with variable relevance set sizes, while retrieving 30-60% fewer documents.

6. **RAG as demonstrative application:** The practical payoff — fewer, more relevant documents mean lower LLM cost, less distraction, and comparable or better generation quality across multiple LLM tiers.

**Framing:** The innovation is the retrieval technique. RAG demonstrates practical value (per advisor's directive).

---

## 2. Competitive Landscape

### Direct Competitors (training-free adaptive methods)

| Method | How it chooses k | Statistical rigor | Source |
|--------|-----------------|-------------------|--------|
| Fixed top-k | Hyperparameter | None | Standard |
| **Adaptive-k** | Max gap in sorted similarity scores: k = argmax(s_i - s_{i+1}) | None (heuristic) | Taguchi et al., EMNLP 2025 |
| **Surprise (EVT)** | Fits Generalized Pareto Distribution to score tails | Tail modeling only | Bahri et al., SIGIR 2023 |
| **CAR (clustering)** | Clustering transition point in similarity scores | None (heuristic) | Xu et al., arXiv 2025 |
| **Kneedle** | Knee/elbow detection in similarity curve | None (geometric) | Satopaa et al., 2011 |
| **HC (ours)** | P-values against calibrated null → HC statistic | Detection optimality (Donoho-Jin boundary) | Novel |

### Adjacent Methods (require training — noted but not primary baselines)

- Cosine Adapter (CIKM 2024) — learned score calibration
- Choppy/BiCut/AttnCut (SIGIR/AAAI 2020-21) — neural ranked list truncation
- DynamicRAG (NeurIPS 2025) — RL-optimized reranking and selection
- Conformal methods: TRAQ (NAACL 2024), Principled Context Engineering (2025) — coverage guarantees, not detection optimality

### Adaptive "When to Retrieve" (complementary, not competitors)

- Self-RAG (ICLR 2024) — decides whether to retrieve at all
- FLARE (EMNLP 2023) — iterative retrieval during generation
- Adaptive-RAG (NAACL 2024) — routes queries to retrieval strategies
- CRAG/Corrective RAG (2024) — evaluates and corrects after retrieval

HC is complementary to all of these — they decide *whether/when* to retrieve, HC decides *how many*.

---

## 3. HC Theoretical Foundation

### Core Papers

- **Donoho & Jin (2004)**, "Higher criticism for detecting sparse heterogeneous mixtures", *Annals of Statistics*. Foundational. HC statistic, detection boundary, optimality proof.
- **Donoho & Jin (2008)**, "Higher criticism thresholding: Optimal feature selection when useful features are rare and weak", *PNAS*. HC as feature selection threshold — direct analogy to retrieval.
- **Donoho & Jin (2015)**, "Higher Criticism for Large-Scale Inference", *Statistical Science*. Comprehensive review.

### Advisor's HC Papers

- **Donoho & Kipnis (2022)**, "HC for frequency table comparison", *Annals of Statistics*. HC adapted for two-sample testing.
- **Kipnis (2022)**, "HC for authorship attribution", *Annals of Applied Statistics*. HC for text similarity — closest application to retrieval.
- **Kashtan & Kipnis (2024)**, "Detecting edits in AI-generated text", *Harvard Data Science Review*. HC identifies sparse edited sentences — analogous to identifying sparse relevant documents.
- **Donoho & Kipnis (2024)**, "The impossibility region for HC", *Annals of Applied Probability*. When HC fails — important for Discussion section.
- **Gong, Kipnis, Xie (2024)**, "HC for sparse multi-stream change-point detection", arXiv. HC thresholding identifies affected streams — parallels identifying relevant documents.

### Theoretical Concerns to Address

- Arias-Castro & Ying (2019): HC may not achieve detection boundary for power-law tailed distributions. Cosine similarity distributions may be non-Gaussian (Player 2025 shows gamma mixtures). Discuss in limitations.
- Hall & Jin (2010): Innovated HC handles correlated noise — embeddings are correlated. Potential extension.

---

## 4. Baselines

### To implement

| Method | Implementation | Effort |
|--------|---------------|--------|
| Fixed top-k (k=5,10,20,50) | Already implemented | None |
| Oracle-k (true k per query) | Use ground truth relevance count | Low |
| Adaptive-k (max-gap) | Port from `megagonlabs/adaptive-k-retrieval` GitHub | Medium |
| Surprise (EVT/GPD) | Implement GPD fitting per Bahri et al. SIGIR 2023 | Medium |
| Kneedle (knee detection) | Use `kneed` Python library on sorted scores | Low |

### Not implementing (noted in paper)

- Learned methods (Choppy, DynamicRAG, Cosine Adapter) — require training, different class of methods
- Conformal methods — require calibration data, guarantee coverage not detection

---

## 5. Datasets

### Existing (Amazon product retrieval)

| Dataset | Queries | Corpus | Avg K | K Range | Query Type |
|---------|---------|--------|-------|---------|------------|
| amazon_categories_max50 | 92 | 18K | ~37 | 5-50 | Category only |
| amazon_compound | 135 | 80K | ~29 | 5-43 | Category + price |
| amazon_compound_electronics_100k | 44 | 100K | — | — | Category + price |

### New benchmarks to add

| Benchmark | Queries | Corpus | Avg Rel/Query | Why good for HC | Source |
|-----------|---------|--------|---------------|-----------------|--------|
| **BEIR: NFCorpus** | 323 | 3.6K | 38.2 | High, variable K | `beir` Python package |
| **BEIR: TREC-COVID** | 50 | 171K | 493.5 | Extreme K variance | `beir` Python package |
| **FRAMES** | 824 | Wikipedia | 2-15 | Explicit variable K | HuggingFace `google/frames-benchmark` |
| **NQ** (control) | 3,452 | 2.68M | ~1.2 | Control — K≈1 | Standard. Note: 2.68M corpus is expensive to embed. Consider using a pre-built NQ subset or a smaller control dataset (e.g., SciFact from BEIR: 300 queries, 5K docs, avg 1.1 rel/query). |

### Dataset Selection Rationale

HC's advantage is per-query adaptation. It should shine when:
- True K varies significantly across queries (NFCorpus, TREC-COVID)
- Queries have compound constraints producing different relevance set sizes (Amazon compound)
- Multi-hop questions require variable evidence (FRAMES)

HC should match (not beat) fixed top-k when K≈1 (NQ) — showing no downside.

---

## 6. Metrics

### Retrieval Metrics (primary — this is the contribution)

- **Recall@k** — fraction of relevant docs retrieved
- **Precision@k** — fraction of retrieved docs that are relevant
- **F1@k** — harmonic mean of Recall and Precision
- **NDCG** — ranking quality
- **Mean retrieved k** — how many documents each method retrieves on average

### Efficiency Metrics (Approach B elements)

- **Total input tokens** — sum of tokens sent to LLM across all queries
- **Token reduction %** — vs fixed top-k baselines
- **Estimated API cost** — projected cost savings (optional)

### E2E RAG Metrics (demonstrative, not primary)

- **Token F1** — token-level F1 between generated and gold answers
- **Exact Match** — strict answer match
- **Answer Faithfulness** — LLM judge score (1-5)
- **Context Precision / Recall** — LLM judge scores

### LLM Tiers for E2E (per advisor's suggestion — 3 models)

- gpt-4o-mini (weaker)
- gpt-4o (medium)
- gpt-5-mini (stronger)

---

## 7. Experiment Structure

### Experiment 1: Retrieval Quality (core contribution)

All methods x all datasets. Retrieval metrics only. This is the main results table.

- HC vs fixed top-k (5, 10, 20, 50) vs Oracle-k vs Adaptive-k vs Surprise vs Kneedle
- Across all datasets (Amazon x3, BEIR x2, FRAMES, NQ)
- Report: Recall, Precision, F1, NDCG, Mean K

### Experiment 2: Per-Query Adaptivity Analysis

Show that HC's adaptive k correlates with true k better than competing methods.

- Scatter plot: method-selected k vs oracle k (for HC, Adaptive-k, Surprise, Kneedle)
- Pearson/Spearman correlation between selected k and oracle k
- Per-bucket analysis: how each method performs on small-K vs medium-K vs large-K queries

### Experiment 3: Efficiency Analysis

Token savings across methods.

- Bar chart: total tokens by method for each dataset
- HC retrieves X% fewer tokens than top-50 while achieving Y% of retrieval quality
- Tokens per query distribution (box plot)

### Experiment 4: E2E RAG Demonstration (supporting)

Select 2 datasets (one Amazon, one BEIR/FRAMES). Run across 3 LLM tiers.

- HC vs top-20 vs top-50 vs Adaptive-k
- Metrics: Token F1, Answer Faithfulness
- Show: HC's advantage varies with LLM capability (per existing results — stronger LLMs exploit HC's cleaner context better)

### Experiment 5: Ablation Study

- **Gamma sensitivity**: 0.05, 0.1, 0.25
- **Pool size**: 200, 500, 1000
- **Null distribution**: global z-score vs per-query
- **Z-score fraction**: 0.7, 0.8, 0.9
- **Embedding model**: test with 2-3 different embedding models (text-embedding-3-small, all-MiniLM-L6-v2, one other)

---

## 8. Paper Structure

| Section | Pages | Content |
|---------|-------|---------|
| Abstract | ~200 words | Problem, insight, method, key results |
| 1. Introduction | 1.5 | Fixed-k limitation, landscape, contribution, results summary |
| 2. Related Work | 1.5 | 2.1 Adaptive retrieval for RAG, 2.2 Ranked list truncation, 2.3 Higher Criticism theory |
| 3. Method | 2 | 3.1 Problem formulation, 3.2 HC background + detection boundary, 3.3 Null distribution construction, 3.4 HC-based retrieval algorithm, 3.5 Practical considerations |
| 4. Experimental Setup | 1.5 | Datasets, baselines, metrics, implementation details |
| 5. Results | 3 | 5.1 Retrieval quality, 5.2 Adaptivity analysis, 5.3 Efficiency, 5.4 E2E RAG, 5.5 Ablation |
| 6. Discussion | 1 | When HC works best/worst, limitations, connection to impossibility region |
| 7. Conclusion | 0.5 | Summary, future work (reranker integration, learned nulls, multi-stage) |
| References | ~2 | — |
| Appendix | ~2 | Full per-dataset tables, additional plots, null validation details |

**Total:** ~13 pages + appendix. Fits NeurIPS (9 pages + unlimited appendix), EMNLP (8 pages + unlimited appendix), or SIGIR (9 pages) format with minor trimming.

---

## 9. Research Plan and Timeline

### Phase 1: Baseline Implementation (Weeks 1-3)

- Implement Adaptive-k (port from GitHub)
- Implement Surprise (EVT/GPD fitting)
- Implement Kneedle (use `kneed` library)
- Add Oracle-k baseline
- All produce `RetrievalOutput` compatible with existing pipeline

### Phase 2: New Benchmarks (Weeks 3-6)

- Add BEIR: NFCorpus dataset adapter + config
- Add BEIR: TREC-COVID dataset adapter + config
- Add FRAMES dataset adapter + config
- Add NQ (control) dataset adapter + config
- For each: embed corpus, build FAISS index, build null distribution, validate null

### Phase 3: Core Experiments (Weeks 6-10)

- Experiment 1: Retrieval quality (all methods x all datasets)
- Experiment 2: Adaptivity analysis (k correlation, per-bucket)
- Experiment 3: Efficiency (token counts)
- Experiment 4: E2E RAG (2 datasets x 3 LLM tiers)
- Experiment 5: Ablation (gamma, pool size, null type, z-score fraction)

### Phase 4: Analysis and Writing (Weeks 10-16)

- Analyze results, identify story
- Draft paper sections
- Create figures and tables
- Iterate with advisor

### Phase 5: Polish and Submit (Weeks 16-20)

- Finalize paper
- Choose venue based on results:
  - Strong across many datasets → NeurIPS / ICML / ICLR
  - Solid on retrieval → SIGIR / ECIR / CIKM
  - Good with RAG focus → EMNLP / ACL / NAACL
- Submit

### Key Risks

| Risk | Mitigation |
|------|-----------|
| HC doesn't beat Adaptive-k on standard benchmarks | Story shifts to "comparable quality + theoretical guarantees" — still publishable |
| Null distribution doesn't generalize to new datasets | Test early in Phase 2; may need dataset-specific calibration |
| BEIR datasets have k≈1 for most queries | Choose subsets with high k variance (NFCorpus, TREC-COVID); include k≈1 as controls |
| E2E results muddied by LLM noise | Keep E2E as supporting evidence, not the main claim |
| Cosine similarity distributions are non-Gaussian | Discuss limitation per Arias-Castro & Ying (2019); test empirically |

---

## 10. Key References

### HC Theory
- Donoho & Jin (2004). Higher criticism for detecting sparse heterogeneous mixtures. *Annals of Statistics*.
- Donoho & Jin (2008). HC thresholding: Optimal feature selection. *PNAS*.
- Donoho & Jin (2015). HC for large-scale inference. *Statistical Science*.
- Donoho & Kipnis (2022). HC for frequency table comparison. *Annals of Statistics*.
- Donoho & Kipnis (2024). Impossibility region for HC. *Annals of Applied Probability*.
- Kipnis (2022). HC for authorship attribution. *Annals of Applied Statistics*.

### Adaptive Retrieval (Competitors)
- Taguchi et al. (2025). Adaptive-k: Efficient context selection. *EMNLP 2025*.
- Bahri et al. (2023). Surprise: Result list truncation via EVT. *SIGIR 2023*.
- Rossi et al. (2024). Relevance filtering for embedding-based retrieval. *CIKM 2024*.
- Xu et al. (2025). CAR: Cluster-based adaptive retrieval. *arXiv*.

### RAG and Context Quality
- Liu et al. (2024). Lost in the middle. *TACL*.
- Cuconasu et al. (2025). The distracting effect. *ACL 2025*.
- Asai et al. (2024). Self-RAG. *ICLR 2024*.
- Jeong et al. (2024). Adaptive-RAG. *NAACL 2024*.
- Yu et al. (2024). RankRAG. *NeurIPS 2024*.

### Benchmarks
- Thakur et al. (2021). BEIR. *NeurIPS 2021 D&B*.
- Yang et al. (2024). CRAG benchmark. *NeurIPS 2024 D&B*.
- Google/Harvard (2025). FRAMES. *NAACL 2025*.

### Score Distribution Theory
- Manmatha et al. (2001). Modeling score distributions. *SIGIR 2001*.
- Smith et al. (2023). Distribution of cosine similarity. *arXiv*.
- Player (2025). Gamma mixture modeling for cosine similarity. *arXiv*.
