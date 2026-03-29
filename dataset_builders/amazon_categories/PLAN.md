# Implementation Plan: Amazon Categories Dataset + Global Null

Based on advisor feedback and scaling research analysis.

## Two Deliverables

1. **Global null distribution** — replace per-query null with a single pooled null
2. **Amazon categories dataset** — 100k+ document corpus with variable K

---

## Phase 1: Global Null on Category-Key (validate concept first)

**Goal**: Prove global null works on existing 726-doc dataset before scaling.

### 1a. Build global null

- Pool all (query, non-relevant doc) cosine similarities across all 44 queries into one array (~31k pairs)
- Store as a single `NullDistribution` object (same dataclass, just one instead of 44)
- New script: `dataset_builders/category_key/build_global_null.py`

### 1b. Validate global null (MANDATORY GATE)

Must pass ALL existing validation criteria before proceeding:

| Test | Criterion | Method |
|------|-----------|--------|
| Split-half KS | Median KS p-value > 0.05 | 20-seed split-half on pooled null |
| Uniformity | Per-query non-relevant p-values ~ Uniform(0,1) | KS test per query against the global null |
| Signal detection | Relevant docs have small p-values (p < 0.05 for >80% of relevant) | Full-corpus p-value computation |
| HC accuracy | True K vs HC K correlation > 0.8 | Run HC with global null on all 44 queries |

New script: `dataset_builders/category_key/validate_global_null.py`

### 1c. Compare global vs per-query null

- Run HC experiments with global null: `run_hc.py` (modified to accept null type flag)
- Compare F1, recall, precision, MRR against per-query results
- Document degradation (expected: slight, acceptable if <5% F1 drop)

### 1d. Core HC module changes

Modify `src/hc/higher_criticism.py` and `src/retrieval/hc_retrieval.py`:
- `HCRetrieval` currently does `null_dist = self.query_null_distributions.get(query_id)` per query
- Add support for a single global `NullDistribution` (if no per-query found, fall back to global; or explicit mode flag)
- `NullDistribution` dataclass unchanged — the global null is just one instance used for all queries

---

## Phase 2: Amazon Categories Dataset Builder

**Goal**: Build a 100k-200k document corpus from Amazon Books metadata with variable K.

### 2a. Data acquisition

- Source: `McAuley-Lab/Amazon-Reviews-2023`, config `raw_meta_Books`
- 4.4M products, hierarchical `categories` field (up to 7 levels deep)
- Load via HuggingFace `datasets` library with streaming
- NOTE: Advisor linked RelBench rel-amazon, which wraps this same underlying data but as a graph ML benchmark — we use the raw source directly

### 2b. Category hierarchy analysis

Explore and profile the `categories` field:
- Count products per subcategory at each hierarchy level
- Identify the right depth level for queries (likely level 2 or 3)
- Map the K distribution: which subcategories have 5-15 / 20-40 / 50-100 products?
- Detect multi-label products (products appearing in multiple subcategories)

Decision point: choose hierarchy level that gives best K distribution coverage.

### 2c. Dataset construction

| Parameter | Target |
|-----------|--------|
| Total documents | 100k-200k |
| Queries | 200-500 (one per selected subcategory) |
| K range | 5-100 relevant docs per query |
| K buckets | Small (5-15), Medium (20-40), Large (50-100) |
| Sparsity | 0.005%-0.1% per query |

Steps:
1. Filter products: require `title` + (`description` or `features`) with >= 30 words combined
2. Select subcategories spanning K range, balanced across Small/Medium/Large buckets
3. For multi-label products: assign to most specific (leaf) category only
4. Cap oversized categories at K=100 via random subsampling
5. Include remaining products as distractors (no category match to any query)
6. Construct document text: `title + ". " + description + " " + features` (concatenated)
7. Construct queries: "What are [subcategory] products?" (template-varied)

### 2d. Output format

Same CSV format as category-key:
- `corpus.csv`: doc_id, text, type, subcategory, main_category
- `queries.csv`: query_id, text, subcategory, k_bucket
- `qrels.csv`: query_id, doc_id, relevance
- `metadata.json`: statistics

Output to `datasets/amazon_categories/`

### 2e. Builder location

`dataset_builders/amazon_categories/` with:
- `explore_categories.py` — profile hierarchy, output category stats
- `generate_dataset.py` — build corpus/queries/qrels
- `data_loader.py` — load dataset (same interface as category_key)
- `build_vector_db.py` — embed with BGE, build FAISS index

---

## Phase 3: Global Null on Amazon Dataset

### 3a. Build global null at scale

- With 200-500 queries × 100k+ docs, full pairwise = ~50M+ pairs
- Use **sampled null**: for each query, sample 10k random non-relevant docs, compute cosine sims
- Pool across all queries → ~2-5M similarity samples
- Store as single `NullDistribution`

### 3b. Validate global null (MANDATORY GATE)

Same validation suite as Phase 1b, adapted for scale:

| Test | Criterion | Adaptation for scale |
|------|-----------|---------------------|
| Split-half KS | Median KS p-value > 0.05 | On the 2-5M pooled sample |
| Per-query uniformity | >90% queries pass KS test | Sample 50 queries, full non-relevant p-values |
| Signal detection | >80% relevant docs have p < 0.05 | Sample 50 queries across K buckets |
| Anderson-Darling | AD p-value > 0.05 | Additional uniformity test on pooled p-values |

**This gate MUST pass before any HC experiments run.**

If it fails:
- Diagnose: check if certain query types or K buckets cause non-uniformity
- Potential fixes: stratified sampling (by main_category), outlier query removal, larger sample size
- If fundamentally broken: fall back to per-query null (but document why)

### 3c. HC experiments

- Run HC with global null on all queries
- Sweep: gamma ∈ {0.05, 0.1, 0.15, 0.2}, candidate pool ∈ {50, 100, 200, 500}
- Baselines: Top-K for K ∈ {5, 10, 20, 50, 100}
- Metrics: F1, Recall, Precision, MRR, NDCG, MAP
- Analysis by K bucket (Small/Medium/Large)

---

## Phase 4: Analysis & Reporting

- Compare HC vs Top-K at 100k+ scale
- Compare global null performance: category-key (726 docs) vs Amazon (100k+)
- Document where HC wins/loses as a function of K and corpus size
- Generate PDF report + plots
- Prepare summary for advisor

---

## File Structure (final)

```
dataset_builders/
├── category_key/           # existing (726 docs, synthetic)
│   ├── build_global_null.py        # NEW — Phase 1a
│   ├── validate_global_null.py     # NEW — Phase 1b
│   └── ... (existing scripts)
├── amazon_categories/      # NEW — Phase 2
│   ├── explore_categories.py
│   ├── generate_dataset.py
│   ├── data_loader.py
│   ├── build_vector_db.py
│   ├── build_global_null.py
│   ├── validate_global_null.py
│   ├── run_baseline.py
│   ├── run_hc.py
│   └── config/
datasets/
├── category_key/           # existing output
├── amazon_categories/      # NEW output
src/
├── hc/
│   ├── higher_criticism.py         # MODIFY — support global null mode
│   └── null_distribution.py        # MODIFY — add GlobalNullDistribution or mode flag
├── retrieval/
│   └── hc_retrieval.py             # MODIFY — accept global null
```

---

## Order of Operations

1. Phase 1a-1d: Global null on category-key (~1-2 days)
2. Phase 2a-2b: Download + explore Amazon data (~1 day)
3. Phase 2c-2e: Build Amazon dataset (~2 days)
4. Phase 3a-3b: Build + validate global null at scale (~2 days)
5. Phase 3c: HC experiments (~2 days)
6. Phase 4: Analysis + report (~2 days)

**Critical path**: Phase 1 (global null validation) blocks everything. If the global null fails on the 726-doc dataset, the entire approach needs rethinking before scaling.
