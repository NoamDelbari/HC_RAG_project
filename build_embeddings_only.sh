#!/bin/bash

# Phase 2: Embed pre-chunked documents (GPU-required, run on L40S)
# This is the expensive part but FAST with GPU (2-3 hours)

set -e

echo "=================================="
echo "PHASE 2: EMBEDDING ONLY (GPU)"
echo "=================================="
echo ""
echo "This will:"
echo "  - Load pre-chunked data from crag_chunked_bge_full.chunks.pkl"
echo "  - Embed using BAAI/bge-base-en-v1.5 on GPU"
echo "  - Build FAISS index"
echo ""
echo "Runtime: ~2-3 hours on L40S GPU"
echo "Cost: ~$6-9 (instead of $24-27!)"
echo ""

# Check if chunks file exists
if [ ! -f "crag_chunked_bge_full.chunks.pkl" ]; then
    echo "❌ ERROR: crag_chunked_bge_full.chunks.pkl not found!"
    echo ""
    echo "You need to:"
    echo "  1. Run build_chunks_only.sh on a CPU instance first"
    echo "  2. Transfer the .chunks.pkl file here"
    echo ""
    exit 1
fi

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Embedding chunks on GPU (this will take 2-3 hours)..."
echo ""

python3 src/database/build_chunked_vector_db.py \
    --model bge \
    --output crag_chunked_bge_full \
    --load-chunks crag_chunked_bge_full.chunks.pkl

# Verify the mapping
echo ""
echo "Verifying chunk-to-document mapping..."
python3 verify_mapping.py --db crag_chunked_bge_full

echo ""
echo "=================================="
echo "FULL DATABASE BUILD COMPLETE!"
echo "=================================="
echo ""
echo "Files created:"
echo "  crag_chunked_bge_full.faiss"
echo "  crag_chunked_bge_full.metadata.pkl"
echo "  crag_chunked_bge_full.chunk_mapping.pkl"
echo "  crag_chunked_bge_full.chunks.pkl"
echo ""
echo "Next step:"
echo "  Run experiments: python3 src/tests/run_chunked_experiments.py --db crag_chunked_bge_full --compare-k"
echo ""
