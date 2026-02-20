"""Check null distribution p-value uniformity in detail."""
import pickle
import numpy as np
from scipy import stats

# Need to make hc module importable for unpickling
import sys
sys.path.insert(0, 'src')

# Load the null distribution
with open('datasets/beir/fiqa_vector_db.per_query_nulls.pkl', 'rb') as f:
    data = pickle.load(f)

# It's a QueryNullDistributions object with 'distributions' dict 
# and optionally a 'shared' key
print(f"Data type: {type(data)}")
print(f"Distributions keys (sample): {list(data.distributions.keys())[:5]}")

# Check for shared null - it should be under 'shared' key or we need to aggregate
if 'shared' in data.distributions:
    null_obj = data.distributions['shared']
    null_samples = null_obj.similarities
    print(f"Found shared null")
else:
    # Aggregate from all per-query nulls
    print("No shared null, aggregating from per-query distributions...")
    all_samples = []
    for qid, null_obj in data.distributions.items():
        all_samples.extend(null_obj.similarities)
    null_samples = np.array(all_samples)

print(f'Null samples: {len(null_samples)}')

# Compute p-values for null samples against themselves (should be uniform)
sorted_null = np.sort(null_samples)
N = len(sorted_null)

# For each null sample, compute its p-value
indices = np.searchsorted(sorted_null, null_samples, side='left')
n_greater_equal = N - indices
p_values = (n_greater_equal + 1.0) / (N + 1.0)

# Histogram of p-values (10 bins)
hist, edges = np.histogram(p_values, bins=10, range=(0, 1))
print('\nP-value histogram (should be ~equal counts):')
expected = len(p_values) / 10
for i in range(10):
    bar = '#' * int(hist[i] / expected * 20)
    diff = (hist[i] - expected) / expected * 100
    print(f'  [{edges[i]:.1f}-{edges[i+1]:.1f}]: {hist[i]:>5} ({diff:+.1f}%) {bar}')

# KS test
ks_stat, ks_pval = stats.kstest(p_values, 'uniform')
print(f'\nKS test: stat={ks_stat:.4f}, p-value={ks_pval:.2e}')

# Check smallest p-values specifically
print(f'\nSmallest p-values (should be rare):')
print(f'  p < 0.01: {np.mean(p_values < 0.01)*100:.2f}% (expected: 1%)')
print(f'  p < 0.05: {np.mean(p_values < 0.05)*100:.2f}% (expected: 5%)')
print(f'  p < 0.10: {np.mean(p_values < 0.10)*100:.2f}% (expected: 10%)')
