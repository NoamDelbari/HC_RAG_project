"""Build Per-Query Null Distributions."""

import argparse
from pathlib import Path

from hc_rag.data.crag_loader import CRAGLoader
from hc_rag.embeddings.embedding_model import EmbeddingModel
from hc_rag.embeddings.vector_database import VectorDatabase
from hc_rag.hc.null_distribution import NegativePairingNull


def main():
    parser = argparse.ArgumentParser(description="Build per-query null distributions")
    parser.add_argument("--dataset-path", type=str,
                        default="datasets/crag/crag_task_1_and_2_dev_v4.jsonl.bz2")
    parser.add_argument("--vector-db-path", type=str,
                        default="src/database/crag_snippet_chunked_vector_db")
    parser.add_argument("--embedding-model", type=str,
                        default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--output-path", type=str,
                        default="data/null_distributions/crag_snippet_chunk_per_query_null.pkl")
    parser.add_argument("--negatives-per-query", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    print(f"\n=== Building Per-Query Null Distributions ===")
    print(f"Negatives per query: {args.negatives_per_query}")
    print(f"Seed: {args.seed}\n")

    # Load data
    print("[1/3] Loading data...")
    dataset_path = Path(args.dataset_path)
    loader = CRAGLoader(dataset_dir=str(dataset_path.parent), use_full_html=False)
    queries = loader.load_queries(filename=dataset_path.name)
    if args.max_queries:
        queries = queries[:args.max_queries]
    print(f"  Loaded {len(queries)} queries")

    vector_db = VectorDatabase.load(args.vector_db_path)
    print(f"  Loaded vector DB: {len(vector_db.doc_ids)} docs")

    embedding_model = EmbeddingModel(args.embedding_model)
    print(f"  Loaded embedding model: {args.embedding_model}")

    # Build per-query null distributions
    print("\n[2/3] Building per-query null distributions...")
    builder = NegativePairingNull(vector_db, embedding_model, seed=args.seed)
    query_null_dists = builder.build(queries, negatives_per_query=args.negatives_per_query)
    print(f"  {query_null_dists}")

    # Save
    print("\n[3/3] Saving...")
    output_path = Path(args.output_path)
    query_null_dists.save(output_path)
    print(f"  Saved to: {output_path}")

    print(f"\n=== Complete ===")
    print(f"Use: python src/tests/run_hc_experiments.py --query-null-dist-path {output_path}")


if __name__ == "__main__":
    main()
