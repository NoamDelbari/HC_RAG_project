#!/bin/bash

# Test BGE Model on Small Sample (10 documents)
# Quick test to verify chunking and embedding pipeline works correctly
# Expected runtime: 1-2 minutes

set -e

echo "=================================="
echo "TESTING BGE MODEL (10 documents)"
echo "=================================="
echo ""

# Run with conda run to ensure environment is active
echo "Building database with 10 documents..."
conda run -n hcrag python3 src/database/build_chunked_vector_db.py \
    --model bge \
    --tasks 1_2 \
    --max-docs 10 \
    --chunker recursive \
    --chunk-size 384 \
    --chunk-overlap 50 \
    --use-full-html \
    --output crag_chunked_bge_test

# Verify mapping
echo ""
echo "Verifying mapping..."
conda run -n hcrag python3 verify_mapping.py --db crag_chunked_bge_test

# Run quick experiment
echo ""
echo "Running quick experiment..."
conda run -n hcrag python3 src/tests/run_chunked_experiments.py \
    --db crag_chunked_bge_test \
    --max-queries 5

echo ""
echo "=================================="
echo "TEST COMPLETE!"
echo "=================================="
echo ""
echo "If this worked correctly:"
echo "  - Mapping shows exactly 10 documents"
echo "  - No need for fix_mapping.py"
echo "  - Experiment ran successfully"
echo ""
echo "Next: Run full build with build_full_database_bge.sh"
echo ""
