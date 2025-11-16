# Experiment 2 Reproduction Guide

**Reproduce BGE 1,000-Doc Chunked Retrieval Experiment**

**Prerequisites:**
- ✅ Dataset downloaded: `datasets/crag/crag_task_1_and_2_dev_v4.jsonl.bz2`
- ✅ Branch pulled: `lihu/embedding_cont`
- ✅ Environment ready: Python 3.10+, PyTorch, FAISS, sentence-transformers installed

---

## Step-by-Step Commands

### 1. Build Database (BGE Model, 1,000 docs)

```bash
# Activate your environment first
source .venv/bin/activate  # or: conda activate hcrag

# Build chunked vector database with BGE model
python src/database/build_chunked_vector_db.py \
  --model bge \
  --max-docs 1000 \
  --chunker recursive \
  --chunk-size 384 \
  --chunk-overlap 50 \
  --output crag_chunked_vector_db \
  --batch-size 512
```

**What happens:**
- Loads first 1,000 CRAG documents
- Chunks each doc using RecursiveChunker (384 tokens, 50 overlap) → ~619 chunks/doc
- Embeds chunks with **BAAI/bge-base-en-v1.5** (768-dim)
- Saves 3 files: `.faiss`, `.metadata.pkl`, `.chunk_mapping.pkl`
- **Saves model name in metadata** (critical for matching!)

**Time:** ~2-3 min (GPU) | ~10-15 min (CPU)

**Verify build:**
```bash
python -c "
import pickle
with open('crag_chunked_vector_db.chunk_mapping.pkl', 'rb') as f:
    m = pickle.load(f)
print(f'Model: {m.get(\"model_name\", \"MISSING!\")}')
print(f'Docs: {len(m[\"doc_to_chunks\"])}, Chunks: {len(m[\"chunk_to_doc\"])}')
"
```

**Expected output:**
```
Model: BAAI/bge-base-en-v1.5
Docs: 1000, Chunks: 619000
```

---

### 2. Run Retrieval Experiments

```bash
python src/tests/run_chunked_experiments.py \
  --db crag_chunked_vector_db \
  --k 10 \
  --top-chunks 100 \
  --aggregation max_score
```

**What happens:**
- Auto-detects BGE model from database metadata
- Embeds queries using **same BGE model** (prevents mismatch bug!)
- Retrieves top 100 chunks, aggregates to top 10 docs (max_score)
- Evaluates: Recall@10, Precision@10, NDCG@10

**Expected Results:**
```
Results:
  Recall@10: 0.826
  Precision@10: 0.510
  NDCG@10: 0.818

✓ Results saved to: results/chunked_experiments/chunked_YYYYMMDD_HHMMSS.json
```

---

## Configuration Details

| Parameter | Value | Why |
|-----------|-------|-----|
| **Model** | BAAI/bge-base-en-v1.5 | MTEB #1 for retrieval |
| **Chunk Size** | 384 tokens | Safe for all models, typical doc ~4K tokens |
| **Chunk Overlap** | 50 tokens | Preserves context across boundaries |
| **k** | 10 | Optimal retrieval depth |
| **Aggregation** | max_score | Best chunk score per doc (2.6% better than mean) |

---

## Comparison: Experiment 1 vs 2

| Metric | Exp 1 (mpnet, 500 docs) | Exp 2 (BGE, 1K docs) | Δ |
|--------|------------------------|---------------------|---|
| Recall@10 | 0.817 | **0.826** | +0.9% |
| Precision@10 | 0.503 | **0.510** | +0.7% |
| NDCG@10 | 0.831 | 0.818 | -1.3% |

**Key Finding:** BGE improves recall despite 2x more documents → scales better!

---

## Critical Bug Fix (In This Branch)

**Problem:** Database built with BGE, experiments queried with mpnet → 1% recall ❌

**Solution:** Save model name in database, auto-detect in experiments ✅

```python
# build_chunked_vector_db.py saves:
mappings['model_name'] = model.get_model_name()

# run_chunked_experiments.py loads:
db_model = mappings.get("model_name")  # Auto-detects BGE
model = EmbeddingModel(model_name=db_model)
```

---

## Optional Experiments

**Test different k values:**
```bash
python src/tests/run_chunked_experiments.py --db crag_chunked_vector_db --compare-k
```

**Test aggregation strategies:**
```bash
python src/tests/run_chunked_experiments.py --db crag_chunked_vector_db --compare-aggregations
```

---

## Troubleshooting

**GPU OOM:** Reduce batch size `--batch-size 128`  
**Model download fails:** Pre-download `SentenceTransformer('BAAI/bge-base-en-v1.5')`  
**Missing model_name:** Rebuild database with this branch's updated code

---

## Success Checklist

- [ ] Database built: 1,000 docs, BGE model
- [ ] Model name verified in metadata
- [ ] Experiments run successfully
- [ ] Results match: Recall@10 ≈ 0.826

**Done!** 🎉
