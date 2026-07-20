# Category-Key Dataset

Synthetic dataset for evaluating adaptive retrieval (HC vs fixed top-k). Each query asks "What are the major cities in [country]?" with a controlled number of relevant documents (K) that varies across countries.

## Dataset overview

- **726 documents**: 506 city descriptions + 220 distractors (landmarks, food, history)
- **44 queries**: one per country, template-varied
- **K range**: 4–24 relevant docs per query (small=4–5, medium=9–11, large=18–24)
- **Deterministic**: no API calls, fully reproducible from config files

Output goes to `datasets/category_key/`:

| File | Description |
|------|-------------|
| `corpus.csv` | All documents (doc_id, text, type, country, city, size_bucket, distractor_category) |
| `queries.csv` | All queries (query_id, text, country, size_bucket) |
| `qrels.csv` | Relevance judgments (query_id, doc_id, relevance) |
| `metadata.json` | Dataset statistics |

## Usage

```bash
# 1. Generate dataset
python dataset_builders/category_key/generate_dataset.py

# 2. Build FAISS vector DB (BGE embeddings)
python dataset_builders/category_key/build_vector_db.py

# 3. Build per-query null distributions for HC
python dataset_builders/category_key/build_null.py

# 4. Validate null distributions (must pass before experiments)
python dataset_builders/category_key/validate_null.py

# 5. Run experiments
python dataset_builders/category_key/run_baseline.py
python dataset_builders/category_key/run_hc.py

# 6. Compare results
python dataset_builders/category_key/compare_results.py

# Run everything end-to-end (steps 4–6)
python dataset_builders/category_key/run_and_plot_all.py
```

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_dataset.py` | Generate corpus, queries, qrels from config |
| `build_vector_db.py` | Embed corpus with BGE and build FAISS index |
| `build_null.py` | Build per-query null similarity distributions |
| `validate_null.py` | Statistical validation of null distributions |
| `validate_null_runtime.py` | Runtime validation with actual retrieval |
| `run_baseline.py` | Run fixed top-k retrieval experiments |
| `run_hc.py` | Run HC adaptive retrieval experiments |
| `run_pool_size_sweep.py` | Sweep over candidate pool sizes |
| `run_and_plot_all.py` | Full pipeline: validate + run + plot |
| `compare_results.py` | Compare HC vs baseline metrics |
| `plot_comparison.py` | Generate comparison plots |
| `generate_report.py` | Generate PDF report |
| `data_loader.py` | Load dataset into Python (used by all scripts) |

## Loading in Python

```python
from dataset_builders.category_key.data_loader import load_category_key_dataset, load_corpus

queries, qrels = load_category_key_dataset()  # List[CategoryKeyQuery], Dict[str, Set[str]]
corpus = load_corpus()                         # Dict[str, dict]
```

## Config

Templates and country/city definitions live in `config/`:
- `countries_cities.json` — countries grouped by size bucket with their cities
- `templates.json` — query and document text templates
