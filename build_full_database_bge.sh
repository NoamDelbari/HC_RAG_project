#!/bin/bash

# Build Full CRAG Chunked Vector Database with BGE Model
# This script embeds ALL 12,949 CRAG documents using BAAI/bge-base-en-v1.5
# Expected runtime: 6-8 hours on Tesla T4 GPU

set -e  # Exit on error

echo "=================================="
echo "FULL DATABASE BUILD WITH BGE MODEL"
echo "=================================="
echo ""
echo "This will:"
echo "  - Load ALL 12,949 CRAG documents (Tasks 1&2)"
echo "  - Chunk with recursive strategy (384 tokens, 50 overlap)"
echo "  - Embed using BAAI/bge-base-en-v1.5"
echo "  - Estimated: ~11M chunks, 6-8 hours on GPU"
echo ""
echo "Files created:"
echo "  crag_chunked_bge_full.faiss"
echo "  crag_chunked_bge_full.metadata.pkl"
echo "  crag_chunked_bge_full.chunk_mapping.pkl"
echo "  crag_chunked_bge_full.chunks.pkl"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Building chunked vector database with BGE model..."
echo "   This will take 6-8 hours on GPU..."
echo ""

python3 src/database/build_chunked_vector_db.py \
    --model bge \
    --tasks 1_2 \
    --chunker recursive \
    --chunk-size 384 \
    --chunk-overlap 50 \
    --use-full-html \
    --output crag_chunked_bge_full

# Verify the mapping
echo ""
echo "Verifying chunk-to-document mapping..."
python3 verify_mapping.py --db crag_chunked_bge_full

echo ""
echo "=================================="
echo "FULL DATABASE BUILD COMPLETE!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Run experiments: python src/tests/run_chunked_experiments.py --db crag_chunked_bge_full --compare-k"
echo "2. Compare with baseline: Check results/chunked_experiments/"
echo ""
