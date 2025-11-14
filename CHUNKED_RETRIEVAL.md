# Chunked Retrieval Integration

This directory contains the complete chunked retrieval system for handling long documents in the CRAG benchmark.

## Quick Start

After `quick_chunking_demo.py` finishes, follow these steps:

### 1. Build Chunked Vector Database

```bash
# Build with default settings (recursive chunking, 256 tokens, 50 overlap)
conda activate hcrag && python src/database/build_chunked_vector_db.py --model quality

# Or with custom settings
conda activate hcrag && python src/database/build_chunked_vector_db.py \
    --model quality \
    --chunker recursive \
    --chunk-size 384 \
    --chunk-overlap 64
```

**Expected output:**
- `crag_chunked_vector_db.faiss` - FAISS index with chunk embeddings
- `crag_chunked_vector_db.metadata.pkl` - Chunk metadata
- `crag_chunked_vector_db.chunk_mapping.pkl` - Chunk-to-document mappings

**Build time estimate:** ~30-45 minutes on CPU for full dataset (12,949 docs → ~80,000 chunks)

### 2. Run Chunked Experiments

```bash
# Quick test on 100 queries
conda activate hcrag && python src/tests/run_chunked_experiments.py --max-queries 100

# Full evaluation
conda activate hcrag && python src/tests/run_chunked_experiments.py

# Compare aggregation strategies
conda activate hcrag && python src/tests/run_chunked_experiments.py \
    --max-queries 100 \
    --compare-aggregations

# Compare different k values
conda activate hcrag && python src/tests/run_chunked_experiments.py \
    --max-queries 100 \
    --compare-k
```

**Expected improvements over baseline:**
- **Baseline (truncated):** Recall@10: 0.180, NDCG@10: 0.172
- **Chunked (expected):** Recall@10: 0.220-0.240, NDCG@10: 0.210-0.230

## Files Created

### `src/database/build_chunked_vector_db.py`
Builds a vector database from chunked documents:
- Loads CRAG dataset
- Chunks documents using selected strategy
- Embeds all chunks
- Creates chunk-to-document mappings
- Saves everything for retrieval

**Options:**
- `--chunker`: fixed, sentence, recursive (default: recursive)
- `--chunk-size`: tokens per chunk (default: 256)
- `--chunk-overlap`: overlap between chunks (default: 50)
- `--model`: fast or quality (default: quality)

### `src/retrieval/chunked_retrieval.py`
Retrieval system for chunked documents:
- Retrieves top-N chunks
- Aggregates chunk scores to document scores
- Returns top-k documents

**Aggregation strategies:**
- `max_score`: Use highest chunk score (default, works best)
- `mean_score`: Average all chunk scores
- `sum_score`: Sum chunk scores (biases toward longer docs)

### `src/tests/run_chunked_experiments.py`
Comprehensive experiment runner:
- Evaluates chunked retrieval on CRAG
- Compares aggregation strategies
- Compares different k values
- Saves results to JSON

## Workflow

```
1. quick_chunking_demo.py finishes
   ↓
2. Build chunked vector DB (30-45 min)
   python src/database/build_chunked_vector_db.py --model quality
   ↓
3. Run experiments (5-10 min for 100 queries)
   python src/tests/run_chunked_experiments.py --max-queries 100
   ↓
4. Compare with baseline
   Baseline: Recall@10=0.180 (from results/baseline_experiments/)
   Chunked: Recall@10=??? (from results/chunked_experiments/)
```

## Performance Notes

**CPU timing estimates:**
- Chunking 12,949 docs: ~10-15 minutes
- Embedding ~80,000 chunks: ~20-30 minutes
- Retrieval (100 queries): ~2-3 minutes
- Retrieval (2,706 queries): ~30-40 minutes

**Memory requirements:**
- Vector DB: ~250MB (80K chunks × 768 dim × 4 bytes)
- Chunk mappings: ~5MB
- Model: ~500MB in memory

## Expected Results

Based on the analysis showing 100% document truncation:

**Before chunking (baseline):**
- Documents truncated to first 384 tokens
- 99.7% content loss on average
- Recall@10: 0.180
- NDCG@10: 0.172

**After chunking (expected):**
- Full document coverage via chunks
- No content loss
- Recall@10: 0.220-0.240 (+22-33% improvement)
- NDCG@10: 0.210-0.230 (+22-34% improvement)

## Troubleshooting

**Issue:** Database file not found
- **Solution:** Make sure `build_chunked_vector_db.py` completed successfully

**Issue:** Out of memory during embedding
- **Solution:** Reduce batch size in `embedding_model.py` or process in smaller batches

**Issue:** Slow performance
- **Solution:** Use `--max-queries 100` for testing, or reduce `--top-chunks`

## Next Steps

After validating chunking improves baseline:

1. **Tune hyperparameters:**
   - Try different chunk sizes (128, 256, 384, 512)
   - Adjust chunk overlap (25, 50, 100)
   - Test aggregation strategies

2. **Integrate with Higher Criticism:**
   - Apply HC threshold to chunk retrieval
   - Adaptive retrieval based on HC statistics

3. **Compare strategies:**
   - Fixed vs Sentence vs Recursive chunking
   - Different embedding models

## Files You Can Now Delete

After building the chunked database, you can remove:
- `crag_vector_db.faiss` (old truncated database)
- `crag_vector_db.metadata.pkl` (old metadata)

Keep the baseline experiment results for comparison!
