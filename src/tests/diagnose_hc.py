"""
Diagnose HC behavior: Why is it always picking near max k?

This script visualizes:
1. Similarity distributions for top candidates
2. P-value distributions 
3. HC statistic curve (where is the actual maximum?)
"""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.vector_database import VectorDatabase
from hc.higher_criticism import HigherCriticism
from hc.null_distribution import NullDistribution, QueryNullDistributions


def diagnose_query(query_idx: int, db, queries, query_embeddings, per_query_nulls):
    """Diagnose HC behavior for a single query."""
    query = queries[query_idx]
    query_id = query['query_id']
    query_emb = query_embeddings[query_idx]
    n_relevant = len(query['relevant_docs'])
    
    print(f"\n{'='*70}")
    print(f"Query {query_idx}: {query_id}")
    print(f"Relevant docs: {n_relevant}")
    print(f"{'='*70}")
    
    # Get candidates
    results = db.search(query_emb, k=100)
    similarities = np.array([r.similarity for r in results])
    retrieved_ids = [r.doc_id for r in results]
    
    # Check which are relevant
    relevant_set = set(query['relevant_docs'])
    is_relevant = [doc_id in relevant_set for doc_id in retrieved_ids]
    relevant_positions = [i for i, rel in enumerate(is_relevant) if rel]
    
    print(f"\nSimilarity distribution (top 100):")
    print(f"  Max: {similarities.max():.4f}")
    print(f"  Min: {similarities.min():.4f}")
    print(f"  Mean: {similarities.mean():.4f}")
    print(f"  Std: {similarities.std():.4f}")
    print(f"  Relevant docs found at positions: {relevant_positions[:20]}...")
    
    # Get per-query null
    null_dist = per_query_nulls.get(query_id)
    print(f"\nNull distribution for this query:")
    print(f"  Mean: {null_dist.mean:.4f}")
    print(f"  Std: {null_dist.std:.4f}")
    print(f"  N samples: {null_dist.n_samples}")
    
    # Compute p-values
    hc = HigherCriticism()
    hc.set_null_distribution(null_dist)
    
    sorted_sims = np.sort(similarities)[::-1]
    p_values = hc.compute_p_values(sorted_sims)
    
    print(f"\nP-value distribution:")
    print(f"  Min: {p_values.min():.6f}")
    print(f"  Max: {p_values.max():.6f}")
    print(f"  Mean: {p_values.mean():.4f}")
    print(f"  % below 0.05: {100 * np.mean(p_values < 0.05):.1f}%")
    print(f"  % below 0.01: {100 * np.mean(p_values < 0.01):.1f}%")
    
    # Compute HC at each position
    print(f"\nHC statistic curve (position -> HC value):")
    n = len(p_values)
    hc_values = []
    for i in range(1, n + 1):
        expected = i / n
        p_i = p_values[i - 1]
        numerator = expected - p_i
        denominator = np.sqrt(p_i * (1 - p_i) / n)
        if denominator > 0:
            hc_i = numerator / denominator
        else:
            hc_i = 0.0
        hc_values.append(hc_i)
    
    hc_values = np.array(hc_values)
    
    # Show HC at key positions
    positions = [1, 5, 10, 20, 30, 50, 70, 100]
    print(f"  {'Pos':<6} {'Sim':>8} {'P-value':>10} {'HC':>10} {'Relevant':>10}")
    print(f"  {'-'*50}")
    for pos in positions:
        if pos <= n:
            is_rel = "YES" if (pos-1) in relevant_positions else ""
            print(f"  {pos:<6} {sorted_sims[pos-1]:>8.4f} {p_values[pos-1]:>10.6f} "
                  f"{hc_values[pos-1]:>10.2f} {is_rel:>10}")
    
    # Where is the actual maximum?
    max_idx = np.argmax(hc_values)
    max_hc = hc_values[max_idx]
    print(f"\n  HC Maximum: {max_hc:.2f} at position {max_idx + 1}")
    print(f"  (Gamma=0.7 would limit search to position 70)")
    
    # Issue diagnosis
    if max_idx >= 50:
        print(f"\n  ⚠️  ISSUE: HC max is near the end of candidates!")
        print(f"      This means p-values stay small throughout.")
        print(f"      Either embedding is too 'hot' or null is miscalibrated.")
    
    return {
        'query_id': query_id,
        'n_relevant': n_relevant,
        'hc_max_position': max_idx + 1,
        'hc_max_value': max_hc,
        'pct_pvalues_below_05': 100 * np.mean(p_values < 0.05)
    }


def main():
    print("="*70)
    print("HC DIAGNOSTIC: Why is k always near maximum?")
    print("="*70)
    
    # Load data
    db_path = Path("datasets/beir/fiqa_vector_db")
    db = VectorDatabase.load(str(db_path))
    print(f"Loaded database: {db.get_num_documents()} documents")
    
    with open(db_path.with_suffix(".query_data.pkl"), "rb") as f:
        query_data = pickle.load(f)
    queries = query_data["queries"]
    query_embeddings = query_data["query_embeddings"]
    
    per_query_nulls = QueryNullDistributions.load(
        str(db_path.with_suffix(".per_query_nulls.pkl"))
    )
    print(f"Loaded {len(queries)} queries with per-query nulls")
    
    # Diagnose a few queries
    results = []
    for idx in [0, 10, 25, 50, 75]:
        result = diagnose_query(idx, db, queries, query_embeddings, per_query_nulls)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    avg_max_pos = np.mean([r['hc_max_position'] for r in results])
    avg_pct_sig = np.mean([r['pct_pvalues_below_05'] for r in results])
    print(f"Average HC max position: {avg_max_pos:.1f} (out of 100)")
    print(f"Average % p-values < 0.05: {avg_pct_sig:.1f}%")
    
    if avg_pct_sig > 50:
        print("\n⚠️  DIAGNOSIS: Over half of top-100 candidates have p < 0.05")
        print("   This explains why HC picks large k — the signal extends deep!")
        print("   Possible causes:")
        print("   1. Embedding model is very effective (most candidates are 'relevant')")
        print("   2. Null distribution was built from docs that are MORE irrelevant")
        print("      than the test set (train/test distribution shift)")
        print("   3. The top-100 candidates are genuinely similar to query (even if not labeled relevant)")


if __name__ == "__main__":
    main()
