#!/bin/bash
# Simple wrapper to run BGE test with correct Python

/opt/homebrew/Caskroom/miniconda/base/envs/hcrag/bin/python src/database/build_chunked_vector_db.py \
    --model bge \
    --tasks 1_2 \
    --max-docs 10 \
    --chunker recursive \
    --chunk-size 384 \
    --chunk-overlap 50 \
    --use-full-html \
    --output crag_chunked_bge_test
