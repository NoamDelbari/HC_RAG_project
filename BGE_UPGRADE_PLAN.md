# BGE Model Upgrade Plan

## Summary

Upgrading from `sentence-transformers/all-mpnet-base-v2` to `BAAI/bge-base-en-v1.5` for better retrieval performance on CRAG benchmark.

## Why BGE-base-en-v1.5?

- 🥇 **#1 on MTEB leaderboard** for retrieval tasks
- **768d embeddings** (same as mpnet-base-v2, no storage increase)
- **512 token max_seq_length** (vs 384 for mpnet) = better for long chunks
- **Simple HuggingFace setup** (drop-in replacement via sentence-transformers)
- **Industry standard** for RAG systems

## Changes Made

### 1. EmbeddingModel Class
```python
# Added new model constant in src/embeddings/embedding_model.py
BGE_MODEL = "BAAI/bge-base-en-v1.5"  # 768 dim, SOTA for retrieval (MTEB #1)
```

### 2. Chunking Configuration
```python
# Updated default chunk size in src/database/build_chunked_vector_db.py
--chunk-size 384  # (was 256) - safe for all models, maximizes context
```

### 3. Build Scripts

**Test Script: `test_bge_small.sh`**
- Tests with 10 documents
- Verifies chunking + embedding pipeline
- Checks mapping correctness (should show 10 docs, no fix_mapping.py needed)
- Runtime: 1-2 minutes

**Full Build Script: `build_full_database_bge.sh`**
- Builds database with ALL 12,949 CRAG documents
- Uses BGE model with recursive chunking
- Chunk size: 384 tokens, overlap: 50 tokens
- Estimated: ~11M chunks, 6-8 hours on Tesla T4 GPU
- Output files: `crag_chunked_bge_full.*`

### 4. Verification Script
```bash
# Updated verify_mapping.py to accept --db argument
python verify_mapping.py --db crag_chunked_bge_full
```

## Execution Plan

### Step 1: Local Test (1-2 minutes)
```bash
# Test on small sample to verify everything works
./test_bge_small.sh
```

**Expected output:**
- ✅ Mapping shows exactly 10 documents
- ✅ No `_sub*` artifacts
- ✅ Experiment runs successfully
- ✅ No need for `fix_mapping.py`

### Step 2: Full Build on Lightning.ai (6-8 hours)
```bash
# Copy these files to Lightning.ai:
# - build_full_database_bge.sh
# - src/ directory (with updated code)
# - datasets/ directory

# Run full build
./build_full_database_bge.sh
```

**What happens:**
1. Updates build script to use `EmbeddingModel.BGE_MODEL`
2. Chunks all 12,949 documents (recursive, 384 tokens, 50 overlap)
3. Embeds ~11M chunks on GPU (batch_size=512)
4. Saves database files:
   - `crag_chunked_bge_full.faiss` (FAISS index)
   - `crag_chunked_bge_full.metadata.pkl` (chunk metadata)
   - `crag_chunked_bge_full.chunk_mapping.pkl` (chunk↔doc mapping)
   - `crag_chunked_bge_full.chunks.pkl` (raw chunks for re-embedding)

### Step 3: Verify & Test (5 minutes)
```bash
# Verify mapping correctness
python verify_mapping.py --db crag_chunked_bge_full

# Quick test with 10 queries
python src/tests/run_chunked_experiments.py \
    --db crag_chunked_bge_full \
    --max-queries 10
```

**Expected:**
- ✅ 12,949 unique documents
- ✅ ~11M chunks (avg ~870 per doc)
- ✅ No mapping issues
- ✅ Experiment runs successfully

### Step 4: Full Experiments (10-15 minutes)
```bash
# Compare k values
python src/tests/run_chunked_experiments.py \
    --db crag_chunked_bge_full \
    --compare-k

# Compare aggregation strategies
python src/tests/run_chunked_experiments.py \
    --db crag_chunked_bge_full \
    --compare-aggregations
```

**Expected results:**
- Recall@10: 0.82-0.85 (vs 0.817 with mpnet on 500 docs)
- NDCG@10: 0.83-0.86 (vs 0.831 with mpnet)
- BGE should match or slightly exceed mpnet performance

## Files to Save (After Full Build)

These 4 files contain all your work and enable full reproduction:

```
crag_chunked_bge_full.faiss           # ~4-5 GB (FAISS index with ~11M embeddings)
crag_chunked_bge_full.metadata.pkl    # ~2-3 GB (chunk metadata)
crag_chunked_bge_full.chunk_mapping.pkl  # ~100 MB (chunk↔doc mappings)
crag_chunked_bge_full.chunks.pkl      # ~2-3 GB (raw chunks for re-embedding)
```

**Total: ~10 GB** - Copy these to safe storage before continuing with other experiments.

## Comparison with Current Setup

| Aspect | Current (mpnet, 500 docs) | New (BGE, 12,949 docs) |
|--------|---------------------------|------------------------|
| Model | all-mpnet-base-v2 | BAAI/bge-base-en-v1.5 |
| Dimensions | 768 | 768 |
| Max tokens | 384 | 512 |
| Documents | 500 | 12,949 (26x more) |
| Chunks | 435,030 | ~11M (25x more) |
| Chunk size | 256 tokens | 384 tokens (50% larger) |
| Recall@10 | 0.817 | TBD (expect 0.82-0.85) |
| Build time | 2-3 hours | 6-8 hours |
| MTEB Rank | Good | #1 for retrieval |

## Next Steps After BGE

Once BGE database is built and validated:

1. **Compare Models**: Create comparison chart (mpnet vs BGE on same 500 docs)
2. **HC Implementation**: Start Higher Criticism retrieval algorithm
3. **Additional Models**: Try other embedding models if needed
4. **LLM Integration**: Build end-to-end RAG pipeline

## Troubleshooting

### If chunking creates artifacts:
- Check line 393 in `src/embeddings/chunking.py` has: `sub_chunk.parent_doc_id = doc_id`
- Run `verify_mapping.py` to diagnose
- Should NOT need `fix_mapping.py` anymore

### If GPU runs out of memory:
- Reduce batch_size: `--batch-size 256` (default is 512)
- Or use CPU (much slower): build script auto-detects

### If you need to resume:
```bash
# Step 1: Chunk only (fast)
python src/database/build_chunked_vector_db.py \
    --tasks 1_2 \
    --use-full-html \
    --save-chunks-only \
    --output crag_chunked_bge_full

# Step 2: Embed later (slow)
python src/database/build_chunked_vector_db.py \
    --load-chunks crag_chunked_bge_full.chunks.pkl \
    --output crag_chunked_bge_full
```

## Success Criteria

✅ **Small test passes** (10 docs, 1-2 min)
✅ **Full build completes** (12,949 docs, 6-8 hours)
✅ **Mapping verified** (12,949 unique docs, no artifacts)
✅ **Experiments run** (recall ≥ 0.82, NDCG ≥ 0.83)
✅ **No manual fixes needed** (no fix_mapping.py required)

---

**Status**: Ready for Step 1 (local test)
**Next Action**: Run `./test_bge_small.sh` locally to verify setup
