#!/bin/bash

# Build 1K-doc BGE database for quick testing
# Fast validation before committing to full 12,949-doc build

set -e

echo "=================================="
echo "1K-DOC BGE DATABASE (QUICK TEST)"
echo "=================================="
echo ""
echo "This will:"
echo "  - Load ONLY 1,000 CRAG documents"
echo "  - Chunk with recursive strategy (384 tokens)"
echo "  - Embed using BAAI/bge-base-en-v1.5"
echo ""
echo "Expected time: ~30 minutes on L4 GPU"
echo ""
echo "Files created:"
echo "  crag_chunked_bge_1k.faiss"
echo "  crag_chunked_bge_1k.metadata.pkl"
echo "  crag_chunked_bge_1k.chunk_mapping.pkl"
echo "  crag_chunked_bge_1k.chunks.pkl"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Building 1K-doc database with BGE model..."
echo ""

python3 src/database/build_chunked_vector_db.py \
    --model bge \
    --tasks 1_2 \
    --max-docs 1000 \
    --chunker recursive \
    --chunk-size 384 \
    --chunk-overlap 50 \
    --use-full-html \
    --output crag_chunked_bge_1k

# Verify the mapping
echo ""
echo "Verifying chunk-to-document mapping..."
python3 verify_mapping.py --db crag_chunked_bge_1k

echo ""
echo "=================================="
echo "1K DATABASE BUILD COMPLETE!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Run quick test: python3 src/tests/run_chunked_experiments.py --db crag_chunked_bge_1k --compare-k"
echo "2. If results look good, decide on full 12,949-doc build"
echo ""
