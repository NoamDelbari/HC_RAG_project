"""Test if non-relevant test candidates get uniform p-values."""
import pickle
import numpy as np
from scipy import stats
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from embeddings.vector_database import VectorDatabase

# Load database
print("Loading database...")
db = VectorDatabase.load("datasets/beir/fiqa_vector_db")
print(f"Database: {db.index.ntotal} documents")

# Load query data (includes queries and qrels)
print("Loading queries...")
with open("datasets/beir/fiqa_vector_db.query_data.pkl", "rb") as f:
    query_data = pickle.load(f)

queries = query_data["queries"]
query_embeddings = query_data["query_embeddings"]
print(f"Queries: {len(queries)}")

# Load per-query nulls
print("Loading per-query nulls...")
with open('datasets/beir/fiqa_vector_db.per_query_nulls.pkl', 'rb') as f:
    null_data = pickle.load(f)

# Get shared null samples (z-scores from training)
# Aggregate all per-query nulls - sample to make it faster
all_null_zscores = []
for qid, null_obj in null_data.distributions.items():
    all_null_zscores.extend(null_obj.similarities)
all_null = np.array(all_null_zscores)

# Sample for speed (1M samples is enough for p-value estimation)
if len(all_null) > 1000000:
    print(f"Sampling 1M from {len(all_null)} null z-scores for speed...")
    np.random.seed(42)
    all_null = np.random.choice(all_null, size=1000000, replace=False)

null_zscores = np.sort(all_null)
N_null = len(null_zscores)
print(f"Null z-scores: {N_null}")

# Compute p-values for non-relevant test candidates
print("\nComputing p-values for non-relevant test candidates...")
non_rel_pvalues = []
n_queries = min(100, len(queries))

for i in range(n_queries):
    if i % 20 == 0:
        print(f"  Processing query {i+1}/{n_queries}...")
    q = queries[i]
    query_emb = query_embeddings[i]
    relevant_ids = set(q['relevant_docs'])
    
    # Get top-100 candidates
    results = db.search(query_emb, k=100)
    
    # Compute ROBUST z-scores for candidates (median/IQR)
    sims = np.array([r.similarity for r in results])
    median_sim = np.median(sims)
    q25, q75 = np.percentile(sims, [25, 75])
    iqr = q75 - q25
    scale = iqr * 0.7413 if iqr > 1e-6 else 1.0
    if scale < 1e-8:
        continue
    zscores = (sims - median_sim) / scale
    
    # Get p-values for NON-RELEVANT candidates only
    for j, r in enumerate(results):
        if r.doc_id not in relevant_ids:
            z = zscores[j]
            # P-value: proportion of null z-scores >= observed
            idx = np.searchsorted(null_zscores, z, side='left')
            n_greater = N_null - idx
            pval = (n_greater + 1) / (N_null + 1)
            non_rel_pvalues.append(pval)

non_rel_pvalues = np.array(non_rel_pvalues)
print(f"Non-relevant candidates: {len(non_rel_pvalues)}")

# P-value histogram
hist, edges = np.histogram(non_rel_pvalues, bins=10, range=(0, 1))
print('\nP-value histogram for NON-RELEVANT candidates (should be ~uniform):')
expected = len(non_rel_pvalues) / 10
for i in range(10):
    bar = '#' * int(hist[i] / expected * 20)
    diff = (hist[i] - expected) / expected * 100
    print(f'  [{edges[i]:.1f}-{edges[i+1]:.1f}]: {hist[i]:>5} ({diff:+.1f}%) {bar}')

# KS test
ks_stat, ks_pval = stats.kstest(non_rel_pvalues, 'uniform')
print(f'\nKS test: stat={ks_stat:.4f}, p-value={ks_pval:.2e}')

# Check smallest p-values specifically  
print(f'\nSmallest p-values (should be rare for non-relevant):')
print(f'  p < 0.01: {np.mean(non_rel_pvalues < 0.01)*100:.2f}% (expected: 1%)')
print(f'  p < 0.05: {np.mean(non_rel_pvalues < 0.05)*100:.2f}% (expected: 5%)')
print(f'  p < 0.10: {np.mean(non_rel_pvalues < 0.10)*100:.2f}% (expected: 10%)')

if ks_pval < 0.01:
    print("\n⚠️  P-values are NOT uniform - this hurts HC performance!")
else:
    print("\n✓ P-values are approximately uniform - HC should work correctly")
