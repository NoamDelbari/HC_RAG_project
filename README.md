# HC-RAG: Higher Criticism Retrieval-Augmented Generation

Implementation of Higher Criticism (HC) based adaptive retrieval for RAG systems, evaluated on the CRAG benchmark.

## Overview

This project implements and compares two retrieval strategies:
- **Baseline Top-k**: Fixed number of retrieved documents
- **HC-RAG**: Adaptive retrieval using statistical significance testing

**Phase 1 Status:** Baseline retrieval evaluation with IR metrics (no LLM generation yet)

## Setup

### 1. Create Conda Environment

```bash
conda create -n hcrag python=3.11
conda activate hcrag
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support (for GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Install FAISS (GPU version for faster indexing)
conda install -c pytorch faiss-gpu

# Install 7z for extracting the pre-built database (if not already installed)
# Windows: Download from https://www.7-zip.org/
# Linux: sudo apt-get install p7zip-full
# macOS: brew install p7zip
```

### 3. Download CRAG Dataset

Download the CRAG benchmark dataset from [CRAG GitHub](https://github.com/facebookresearch/CRAG):

```bash
# Create dataset directory
mkdir -p datasets/crag

# Download Task 1&2 (required)
cd datasets/crag
wget https://raw.githubusercontent.com/facebookresearch/CRAG/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2

# Download Task 3 (optional)
wget https://raw.githubusercontent.com/facebookresearch/CRAG/main/data/crag_task_3_dev_v4.tar.bz2
```

**Dataset Statistics:**
- Tasks 1&2: ~2,700 queries, ~12,000 documents
- Task 3: More queries, up to 50 documents per query

## Quick Start

### 1. Setup Vector Database

**Option A: Use Pre-built Database (Recommended)**

Extract the provided database (built with `sentence-transformers/all-mpnet-base-v2`):

```bash
# Extract the .7z file in src/database/
cd src/database
7z x crag_vector_db.7z
```

This will create:
- `crag_vector_db.faiss` - FAISS index with 11,980 documents
- `crag_vector_db.metadata.pkl` - Document metadata

**Option B: Build Your Own Database**

```bash
# Build database from Tasks 1&2 with quality model
python src/database/build_vector_db.py --model quality

# Or with fast model (384d embeddings)
python src/database/build_vector_db.py --model fast
```

### 2. Run Baseline Experiments

```bash
# Test with 100 queries
python src/tests/run_baseline_experiments.py --max-queries 100

# Run full experiments (all 2,700 queries)
python src/tests/run_baseline_experiments.py
```

**Output:** Results saved to `results/baseline_experiments/`

### 3. Analyze Document Lengths

```bash
# Check for truncation issues
python analyze_document_lengths.py
```

## Project Structure

```
HC_RAG/
├── src/
│   ├── data/              # CRAG dataset loading
│   ├── embeddings/        # Embedding models and vector database
│   ├── retrieval/         # Baseline and HC retrieval
│   ├── evaluation/        # IR metrics (Recall@k, NDCG, MRR, etc.)
│   ├── hc/                # Higher Criticism statistics
│   └── tests/             # Experiment scripts
├── datasets/crag/         # CRAG dataset files
├── results/               # Experiment results
└── requirements.txt
```

## Current Features

✅ CRAG dataset loader (Tasks 1&2 and Task 3)
✅ GPU-accelerated embedding generation
✅ FAISS vector database with persistence
✅ **Pre-built vector database** (11,980 docs, mpnet-base-v2 768d embeddings)
✅ Baseline top-k retrieval
✅ IR metrics evaluation (Recall@k, Precision@k, MRR, NDCG, MAP)
✅ Document length analysis

🚧 **In Progress:**
- Higher Criticism statistic calculation
- HC-based adaptive retrieval
- Document chunking strategy

## Usage Examples

### Load CRAG Data

```python
from src.data.crag_loader import CRAGLoader

loader = CRAGLoader(use_full_html=False)
queries, documents = loader.load_by_tasks(["1_2"])

print(f"Loaded {len(queries)} queries, {len(documents)} documents")
```

### Generate Embeddings

```python
from src.embeddings.embedding_model import EmbeddingModel

model = EmbeddingModel(model_name=EmbeddingModel.QUALITY_MODEL)
query_emb = model.embed_query("What is the capital of France?")
doc_embs = model.embed_documents(doc_texts, show_progress=True)
```

### Retrieve Documents

```python
from src.embeddings.vector_database import VectorDatabase
from src.retrieval.baseline_retrieval import BaselineRetrieval

# Load database
vector_db = VectorDatabase.load("src/database/crag_vector_db")

# Create retriever
retriever = BaselineRetrieval(vector_db=vector_db, k=10)

# Retrieve
result = retriever.retrieve(query_id="q1", query_embedding=query_emb)
print(f"Retrieved {result.k} documents")
```

### Evaluate Retrieval

```python
from src.evaluation.evaluator import RetrievalEvaluator

evaluator = RetrievalEvaluator()

# Evaluate single query
eval_result = evaluator.evaluate_single(
    query_id="q1",
    retrieved_ids=["doc1", "doc2", "doc3"],
    retrieved_scores=[0.9, 0.8, 0.7],
    relevant_ids={"doc1", "doc3"}
)

print(f"Recall@3: {eval_result.recall_at_k:.3f}")
print(f"NDCG@3: {eval_result.ndcg_at_k:.3f}")
```

## GPU Support

For ~10-20x speedup, GPU is highly recommended:

```bash
# Check GPU availability
python check_gpu.py
```
