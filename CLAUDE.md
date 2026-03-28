# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project implementing **adaptive document retrieval using Higher Criticism (HC) statistics**, evaluated on the CRAG benchmark. HC replaces fixed top-k retrieval by statistically determining how many documents to retrieve per query.

## Key Commands

```bash
# Install package
pip install -e .

# Run full pipeline for a dataset
python -m experiments.scripts.run_all --config experiments/configs/amazon_compound.yaml

# Run individual steps
python -m experiments.scripts.build_db --config experiments/configs/amazon_compound.yaml
python -m experiments.scripts.build_null --config experiments/configs/amazon_compound.yaml
python -m experiments.scripts.validate_null --config experiments/configs/amazon_compound.yaml
python -m experiments.scripts.run_retrieval --config experiments/configs/amazon_compound.yaml
python -m experiments.scripts.run_eval --config experiments/configs/amazon_compound.yaml

# Force rerun
python -m experiments.scripts.run_all --config ... --force

# Start from a specific step
python -m experiments.scripts.run_all --config ... --from run_retrieval

# Run unit tests
python -m pytest experiments/tests/ -v
python -m pytest src/hc_rag/tests/ -v
```

## Architecture

### Core Pipeline: Query → Embed → Search → Select → Evaluate

1. **Data** (`src/data/`): `CRAGLoader` loads the CRAG benchmark (~2,700 queries, ~12,000 docs). Document IDs are MD5 hashes of URLs.

2. **Embeddings** (`src/embeddings/`): `EmbeddingModel` (sentence-transformers, GPU-accelerated) and `GeminiEmbeddingModel`. Three supported models: `all-MiniLM-L6-v2` (384d, fast), `all-mpnet-base-v2` (768d, quality), `BAAI/bge-base-en-v1.5` (768d, SOTA). `DocumentChunker` subclasses handle fixed/sentence/recursive chunking. `VectorDatabase` wraps FAISS (`IndexFlatIP`, cosine similarity).

3. **Retrieval** (`src/retrieval/`): `BaselineRetrieval` (fixed top-k) and `HCRetrieval` (adaptive via HC statistics) share a common interface. `ChunkedRetrievalMixin` provides chunk-to-document score aggregation (max/mean/sum strategies).

4. **Higher Criticism** (`src/hc/`): The core statistical module. `HigherCriticism` converts cosine similarities → p-values → HC statistic to find the optimal threshold adaptively. `NegativePairingNull` builds per-query null distributions from negative document pairs.

5. **Evaluation** (`src/evaluation/`): `RetrievalEvaluator` computes IR metrics (Recall@k, Precision@k, MRR, NDCG, MAP, Hit Rate). Results use `RetrievalResult` and `AggregateMetrics` dataclasses.

6. **LLM** (`src/llm/`): Scaffolding for end-to-end RAG (Phase 4, not yet implemented). `OpenRouterLLM` for API calls, modular prompt templates.

7. **Database Builder** (`src/database/`): `build_vector_db.py` supports full-document and chunked modes with resumable builds. Pre-built DB available in `src/database/crag_vector_db.7z`.

8. **Datasets** (`datasets/`): Each subdirectory is a self-contained dataset with its own generator code and config. `datasets/crag/` holds the CRAG benchmark data. `datasets/cross_entity_qa/` generates cross-entity QA datasets using Wikidata SPARQL, Wikipedia passage mapping, and LLM-based question generation with constraint-based sub-clustering for variable K.

### Key Design Patterns

- **Factory pattern**: `create_embedding_model()` for model instantiation
- **Mixin pattern**: `ChunkedRetrievalMixin` shared between retrieval classes
- **Dataclass results**: `RetrievalOutput`, `RetrievalResult`, `AggregateMetrics` for structured outputs
- **Unified retrieval API**: Both retrieval strategies share the same interface

## Environment

- Python 3.11+, PyTorch with CUDA 12.1
- API keys configured via `.env` (see `.env.example` for template: `GEMINI_API_KEY`)
- Formatting: `black`
- Testing: `pytest`
