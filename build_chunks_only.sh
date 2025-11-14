#!/bin/bash

# Phase 1: Chunk documents (CPU-only, run on FREE/CHEAP instance)
# This is the slow part (6 hours) but doesn't need GPU

set -e

echo "=================================="
echo "PHASE 1: CHUNKING ONLY (CPU)"
echo "=================================="
echo ""
echo "This will:"
echo "  - Load ALL 12,949 CRAG documents"
echo "  - Chunk with recursive strategy (384 tokens)"
echo "  - Save chunks to disk (NO EMBEDDING YET)"
echo ""
echo "Runtime: ~6 hours on CPU"
echo "Cost: FREE on CPU instance (or very cheap)"
echo ""
echo "After this completes, run build_embeddings_only.sh on GPU"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Chunking documents (this will take several hours)..."
echo ""

python3 src/database/build_chunked_vector_db.py \
    --model bge \
    --tasks 1_2 \
    --chunker recursive \
    --chunk-size 384 \
    --chunk-overlap 50 \
    --use-full-html \
    --output crag_chunked_bge_full \
    --save-chunks-only

echo ""
echo "=================================="
echo "PHASE 1 COMPLETE!"
echo "=================================="
echo ""
echo "Files created:"
echo "  crag_chunked_bge_full.chunks.pkl (~2-3GB)"
echo ""
echo "Next step:"
echo "  1. Transfer crag_chunked_bge_full.chunks.pkl to Lightning.ai with L40S GPU"
echo "  2. Run: ./build_embeddings_only.sh"
echo ""
echo "This saves you ~$18 by chunking on CPU!"
echo ""
