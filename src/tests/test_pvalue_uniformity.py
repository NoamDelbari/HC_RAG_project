"""
Test P-value Uniformity Under Null Hypothesis

For Higher Criticism to work correctly, p-values under H0 must be uniformly distributed.
This script tests whether our null distribution produces uniform p-values when applied
to non-relevant (null) query-document pairs.

Key insight: If we compute p-values for NON-RELEVANT documents (true H0 samples),
they should be uniform on [0, 1]. If they're not, HC will give misleading results.
"""

import sys
import pickle
import logging
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hc.higher_criticism import HigherCriticism
from hc.null_distribution import NullDistribution

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_fiqa_data(db_path: str = "datasets/beir/fiqa_vector_db"):
    """Load FiQA vector database and query data."""
    import faiss
    
    # Load FAISS index
    index = faiss.read_index(f"{db_path}.faiss")
    
    # Load metadata
    with open(f"{db_path}.metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    # Load query data
    with open(f"{db_path}.query_data.pkl", "rb") as f:
        raw_query_data = pickle.load(f)
    
    # Normalize to expected format
    # Handle both old format (null_distribution) and new format (per_query_null)
    query_data = {
        'embeddings': raw_query_data['query_embeddings'],
        'query_ids': [q['query_id'] for q in raw_query_data['queries']],
        'relevance': {q['query_id']: q['relevant_docs'] for q in raw_query_data['queries']},
    }
    
    # Check for per-query nulls (new format) vs global null (old format)
    if 'per_query_null' in raw_query_data:
        query_data['per_query_null'] = raw_query_data['per_query_null']
        query_data['has_per_query_null'] = True
    else:
        query_data['has_per_query_null'] = False
    
    if 'global_null' in raw_query_data:
        query_data['global_null'] = raw_query_data['global_null']
    elif 'null_distribution' in raw_query_data:
        query_data['global_null'] = raw_query_data['null_distribution']
    
    # Load info
    with open(f"{db_path}.info.pkl", "rb") as f:
        info = pickle.load(f)
    
    return index, metadata, query_data, info


def build_global_null_from_db(index, metadata, query_data, n_samples: int = 50000):
    """
    Build global null distribution from random query-document pairs.
    
    This is what the DB builder does - sample random (query, doc) pairs
    and compute their similarities.
    """
    query_embeddings = query_data['embeddings']
    n_queries = len(query_embeddings)
    n_docs = index.ntotal
    
    # Sample random pairs
    np.random.seed(42)
    query_indices = np.random.randint(0, n_queries, size=n_samples)
    doc_indices = np.random.randint(0, n_docs, size=n_samples)
    
    # Compute similarities
    similarities = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        q_emb = query_embeddings[query_indices[i]]
        d_emb = index.reconstruct(int(doc_indices[i]))
        similarities[i] = np.dot(q_emb, d_emb)
    
    null_dist = NullDistribution(
        similarities=similarities,
        mean=float(np.mean(similarities)),
        std=float(np.std(similarities)),
        min_val=float(np.min(similarities)),
        max_val=float(np.max(similarities)),
        n_samples=n_samples
    )
    
    return null_dist


def test_global_null_uniformity(index, metadata, query_data, null_dist, n_test_queries: int = 100):
    """
    Test if p-values for NON-RELEVANT documents are uniform under global null.
    
    This is the critical test: under H0, p-values should be uniform.
    If they're not, HC will fail.
    """
    print("\n" + "="*70)
    print("Testing Global Null Distribution - P-value Uniformity")
    print("="*70)
    
    hc = HigherCriticism(null_distribution=null_dist)
    
    query_embeddings = query_data['embeddings']
    query_ids = query_data['query_ids']
    relevance = query_data['relevance']  # Dict: query_id -> list of relevant doc_ids
    doc_ids = metadata['doc_ids']
    doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    
    # Collect p-values for non-relevant documents
    all_null_pvalues = []
    all_null_similarities = []
    
    np.random.seed(42)
    test_query_indices = np.random.choice(len(query_ids), size=min(n_test_queries, len(query_ids)), replace=False)
    
    print(f"\nTesting {len(test_query_indices)} queries...")
    print(f"For each query, computing p-values for NON-RELEVANT documents\n")
    
    for qi in test_query_indices:
        q_id = query_ids[qi]
        q_emb = query_embeddings[qi]
        relevant_docs = set(relevance.get(q_id, []))
        
        # Get non-relevant doc indices
        non_relevant_indices = [
            doc_id_to_idx[doc_id] 
            for doc_id in doc_ids 
            if doc_id not in relevant_docs and doc_id in doc_id_to_idx
        ]
        
        # Sample some non-relevant docs (not all 57K)
        sample_size = min(500, len(non_relevant_indices))
        sampled_indices = np.random.choice(non_relevant_indices, size=sample_size, replace=False)
        
        # Compute similarities to non-relevant docs
        for d_idx in sampled_indices:
            d_emb = index.reconstruct(int(d_idx))
            sim = np.dot(q_emb, d_emb)
            all_null_similarities.append(sim)
    
    # Convert to numpy and compute p-values
    all_null_similarities = np.array(all_null_similarities)
    all_null_pvalues = hc.compute_p_values(all_null_similarities)
    
    print(f"Collected {len(all_null_pvalues)} p-values from non-relevant (null) pairs")
    print(f"Similarity stats: mean={all_null_similarities.mean():.4f}, std={all_null_similarities.std():.4f}")
    
    # Test uniformity with KS test
    ks_stat, ks_pvalue = stats.kstest(all_null_pvalues, 'uniform')
    
    print(f"\n{'='*70}")
    print("Kolmogorov-Smirnov Test for Uniformity")
    print(f"{'='*70}")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  KS p-value:   {ks_pvalue:.4e}")
    
    if ks_pvalue < 0.01:
        print(f"\n  ❌ FAIL: P-values are NOT uniform (KS p-value < 0.01)")
        print(f"     This means HC will give MISLEADING results!")
    elif ks_pvalue < 0.05:
        print(f"\n  ⚠️  WARNING: P-values may not be uniform (KS p-value < 0.05)")
    else:
        print(f"\n  ✓ PASS: P-values appear uniform (KS p-value >= 0.05)")
    
    # Additional diagnostics
    print(f"\nP-value distribution diagnostics:")
    print(f"  Min:    {all_null_pvalues.min():.4f}")
    print(f"  Max:    {all_null_pvalues.max():.4f}")
    print(f"  Mean:   {all_null_pvalues.mean():.4f} (expected: 0.5)")
    print(f"  Std:    {all_null_pvalues.std():.4f} (expected: {1/np.sqrt(12):.4f})")
    print(f"  Median: {np.median(all_null_pvalues):.4f} (expected: 0.5)")
    
    # Decile analysis
    print(f"\nP-value deciles (each should have ~10% of samples):")
    deciles = np.linspace(0, 1, 11)
    for i in range(10):
        count = np.sum((all_null_pvalues >= deciles[i]) & (all_null_pvalues < deciles[i+1]))
        pct = 100 * count / len(all_null_pvalues)
        expected = 10.0
        deviation = pct - expected
        marker = "✓" if abs(deviation) < 2 else ("⚠️" if abs(deviation) < 5 else "❌")
        print(f"  [{deciles[i]:.1f}, {deciles[i+1]:.1f}): {pct:5.1f}% ({deviation:+.1f}%) {marker}")
    
    return all_null_pvalues, all_null_similarities, ks_stat, ks_pvalue


def test_per_query_null_uniformity(index, metadata, query_data, n_test_queries: int = 50):
    """
    Test if per-query null distributions produce uniform p-values.
    
    For each query, we build a null distribution from its non-relevant documents,
    then test if p-values for other non-relevant documents are uniform.
    """
    print("\n" + "="*70)
    print("Testing Per-Query Null Distribution - P-value Uniformity")
    print("="*70)
    
    query_embeddings = query_data['embeddings']
    query_ids = query_data['query_ids']
    relevance = query_data['relevance']
    doc_ids = metadata['doc_ids']
    doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    
    all_null_pvalues = []
    
    np.random.seed(42)
    test_query_indices = np.random.choice(len(query_ids), size=min(n_test_queries, len(query_ids)), replace=False)
    
    print(f"\nTesting {len(test_query_indices)} queries...")
    print(f"For each query: build per-query null from 500 docs, test on 500 other docs\n")
    
    for qi in test_query_indices:
        q_id = query_ids[qi]
        q_emb = query_embeddings[qi]
        relevant_docs = set(relevance.get(q_id, []))
        
        # Get non-relevant doc indices
        non_relevant_indices = [
            doc_id_to_idx[doc_id] 
            for doc_id in doc_ids 
            if doc_id not in relevant_docs and doc_id in doc_id_to_idx
        ]
        
        # Split into train (for null) and test (for p-values)
        np.random.shuffle(non_relevant_indices)
        null_train_indices = non_relevant_indices[:500]
        null_test_indices = non_relevant_indices[500:1000]
        
        if len(null_test_indices) < 100:
            continue  # Skip if not enough test samples
        
        # Build per-query null from train set
        train_sims = np.array([
            np.dot(q_emb, index.reconstruct(int(d_idx)))
            for d_idx in null_train_indices
        ])
        
        per_query_null = NullDistribution(
            similarities=train_sims,
            mean=float(np.mean(train_sims)),
            std=float(np.std(train_sims)),
            min_val=float(np.min(train_sims)),
            max_val=float(np.max(train_sims)),
            n_samples=len(train_sims)
        )
        
        # Compute p-values on test set
        hc = HigherCriticism(null_distribution=per_query_null)
        
        test_sims = np.array([
            np.dot(q_emb, index.reconstruct(int(d_idx)))
            for d_idx in null_test_indices
        ])
        
        pvalues = hc.compute_p_values(test_sims)
        all_null_pvalues.extend(pvalues)
    
    all_null_pvalues = np.array(all_null_pvalues)
    
    print(f"Collected {len(all_null_pvalues)} p-values from non-relevant (null) pairs")
    
    # Test uniformity
    ks_stat, ks_pvalue = stats.kstest(all_null_pvalues, 'uniform')
    
    print(f"\n{'='*70}")
    print("Kolmogorov-Smirnov Test for Uniformity (Per-Query Null)")
    print(f"{'='*70}")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  KS p-value:   {ks_pvalue:.4e}")
    
    if ks_pvalue < 0.01:
        print(f"\n  ❌ FAIL: P-values are NOT uniform (KS p-value < 0.01)")
    elif ks_pvalue < 0.05:
        print(f"\n  ⚠️  WARNING: P-values may not be uniform (KS p-value < 0.05)")
    else:
        print(f"\n  ✓ PASS: P-values appear uniform (KS p-value >= 0.05)")
    
    # Diagnostics
    print(f"\nP-value distribution diagnostics:")
    print(f"  Mean:   {all_null_pvalues.mean():.4f} (expected: 0.5)")
    print(f"  Std:    {all_null_pvalues.std():.4f} (expected: {1/np.sqrt(12):.4f})")
    
    return all_null_pvalues, ks_stat, ks_pvalue


def test_stored_per_query_null_uniformity(index, metadata, query_data, n_test_queries: int = 50):
    """
    Test if the STORED per-query null distributions produce uniform p-values.
    
    This tests the pre-computed nulls from the database build process.
    This is the most important test - it validates the end-to-end workflow.
    """
    print("\n" + "="*70)
    print("Testing STORED Per-Query Null Distribution - P-value Uniformity")
    print("="*70)
    
    if not query_data.get('has_per_query_null', False):
        print("\n  ⚠️  No per-query null distributions found in data file!")
        print("     Rebuild the database with the updated build script.")
        return None, None, None
    
    per_query_null = query_data['per_query_null']
    query_embeddings = query_data['embeddings']
    query_ids = query_data['query_ids']
    relevance = query_data['relevance']
    doc_ids = metadata['doc_ids']
    doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    
    all_null_pvalues = []
    
    np.random.seed(42)
    test_query_indices = np.random.choice(len(query_ids), size=min(n_test_queries, len(query_ids)), replace=False)
    
    print(f"\nTesting {len(test_query_indices)} queries using STORED per-query nulls...")
    print(f"Validating against held-out non-relevant documents\n")
    
    for qi in test_query_indices:
        q_id = query_ids[qi]
        q_emb = query_embeddings[qi]
        relevant_docs = set(relevance.get(q_id, []))
        
        # Get the stored null distribution for this query
        if q_id not in per_query_null:
            continue
        
        stored_null = per_query_null[q_id]
        null_dist = NullDistribution(
            similarities=stored_null['similarities'],
            mean=stored_null['mean'],
            std=stored_null['std'],
            min_val=float(np.min(stored_null['similarities'])),
            max_val=float(np.max(stored_null['similarities'])),
            n_samples=stored_null['n_samples']
        )
        
        # Get non-relevant doc indices (excluding those used in null)
        non_relevant_indices = [
            doc_id_to_idx[doc_id] 
            for doc_id in doc_ids 
            if doc_id not in relevant_docs and doc_id in doc_id_to_idx
        ]
        
        # Sample held-out non-relevant docs for testing
        np.random.shuffle(non_relevant_indices)
        test_indices = non_relevant_indices[:500]  # Use 500 held-out docs
        
        if len(test_indices) < 100:
            continue
        
        # Compute p-values using the STORED null
        hc = HigherCriticism(null_distribution=null_dist)
        
        test_sims = np.array([
            np.dot(q_emb, index.reconstruct(int(d_idx)))
            for d_idx in test_indices
        ])
        
        pvalues = hc.compute_p_values(test_sims)
        all_null_pvalues.extend(pvalues)
    
    all_null_pvalues = np.array(all_null_pvalues)
    
    print(f"Collected {len(all_null_pvalues)} p-values from non-relevant (null) pairs")
    
    # Test uniformity
    ks_stat, ks_pvalue = stats.kstest(all_null_pvalues, 'uniform')
    
    print(f"\n{'='*70}")
    print("Kolmogorov-Smirnov Test for Uniformity (STORED Per-Query Null)")
    print(f"{'='*70}")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  KS p-value:   {ks_pvalue:.4e}")
    
    if ks_pvalue < 0.01:
        print(f"\n  ❌ FAIL: P-values are NOT uniform (KS p-value < 0.01)")
    elif ks_pvalue < 0.05:
        print(f"\n  ⚠️  WARNING: P-values may not be uniform (KS p-value < 0.05)")
    else:
        print(f"\n  ✓ PASS: P-values appear uniform (KS p-value >= 0.05)")
    
    # Diagnostics
    print(f"\nP-value distribution diagnostics:")
    print(f"  Mean:   {all_null_pvalues.mean():.4f} (expected: 0.5)")
    print(f"  Std:    {all_null_pvalues.std():.4f} (expected: {1/np.sqrt(12):.4f})")
    
    return all_null_pvalues, ks_stat, ks_pvalue


def plot_pvalue_diagnostics(global_pvalues, per_query_pvalues=None, save_path: str = None):
    """Plot p-value distributions for visual inspection."""
    
    n_plots = 2 if per_query_pvalues is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    # Global null histogram
    ax = axes[0]
    ax.hist(global_pvalues, bins=50, density=True, alpha=0.7, edgecolor='black', label='Observed')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Uniform (expected)')
    ax.set_xlabel('P-value')
    ax.set_ylabel('Density')
    ax.set_title('Global Null Distribution\nP-values for Non-Relevant Docs')
    ax.legend()
    ax.set_xlim(0, 1)
    
    # Per-query null histogram
    if per_query_pvalues is not None:
        ax = axes[1]
        ax.hist(per_query_pvalues, bins=50, density=True, alpha=0.7, edgecolor='black', label='Observed')
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Uniform (expected)')
        ax.set_xlabel('P-value')
        ax.set_ylabel('Density')
        ax.set_title('Per-Query Null Distribution\nP-values for Non-Relevant Docs')
        ax.legend()
        ax.set_xlim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to {save_path}")
    
    plt.show()


def main():
    """Run p-value uniformity tests."""
    print("="*70)
    print("P-VALUE UNIFORMITY TEST FOR HIGHER CRITICISM")
    print("="*70)
    print("\nFor HC to work correctly, p-values under H0 must be UNIFORM [0,1]")
    print("We test this by computing p-values for NON-RELEVANT documents\n")
    
    # Load data
    print("Loading FiQA vector database...")
    index, metadata, query_data, info = load_fiqa_data()
    print(f"  Documents: {index.ntotal}")
    print(f"  Queries: {len(query_data['query_ids'])}")
    
    # Build global null (as the DB builder does)
    print("\nBuilding global null distribution (50K random pairs)...")
    global_null = build_global_null_from_db(index, metadata, query_data, n_samples=50000)
    print(f"  Null dist: mean={global_null.mean:.4f}, std={global_null.std:.4f}")
    
    # Test 1: Global null uniformity
    global_pvalues, global_sims, global_ks_stat, global_ks_pvalue = \
        test_global_null_uniformity(index, metadata, query_data, global_null, n_test_queries=100)
    
    # Test 2: Per-query null uniformity (built on-the-fly)
    per_query_pvalues, pq_ks_stat, pq_ks_pvalue = \
        test_per_query_null_uniformity(index, metadata, query_data, n_test_queries=50)
    
    # Test 3: Stored per-query null uniformity (from data file)
    stored_pvalues, stored_ks_stat, stored_ks_pvalue = \
        test_stored_per_query_null_uniformity(index, metadata, query_data, n_test_queries=50)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nGlobal Null Distribution:")
    print(f"  KS statistic: {global_ks_stat:.4f}")
    print(f"  KS p-value:   {global_ks_pvalue:.4e}")
    print(f"  Result: {'PASS ✓' if global_ks_pvalue >= 0.05 else 'FAIL ❌'}")
    
    print(f"\nPer-Query Null Distribution (built on-the-fly):")  
    print(f"  KS statistic: {pq_ks_stat:.4f}")
    print(f"  KS p-value:   {pq_ks_pvalue:.4e}")
    print(f"  Result: {'PASS ✓' if pq_ks_pvalue >= 0.05 else 'FAIL ❌'}")
    
    if stored_ks_stat is not None:
        print(f"\nStored Per-Query Null Distribution (from database):")  
        print(f"  KS statistic: {stored_ks_stat:.4f}")
        print(f"  KS p-value:   {stored_ks_pvalue:.4e}")
        print(f"  Result: {'PASS ✓' if stored_ks_pvalue >= 0.05 else 'FAIL ❌'}")
    else:
        print(f"\nStored Per-Query Null Distribution: NOT AVAILABLE")
        print(f"  Rebuild database with updated build script to generate per-query nulls")
    
    if global_ks_pvalue < 0.05 and pq_ks_pvalue >= 0.05:
        print("\n" + "="*70)
        print("CONCLUSION: Must use PER-QUERY null distributions!")
        print("="*70)
        print("Global null fails uniformity test, but per-query null passes.")
        print("This means each query has its own similarity baseline that")
        print("differs from the global average. HC will only work correctly")
        print("with per-query null distributions.")
        
        if stored_ks_pvalue is not None and stored_ks_pvalue >= 0.05:
            print("\n✓ Stored per-query nulls PASS - ready for HC experiments!")
        elif stored_ks_pvalue is not None:
            print("\n❌ Stored per-query nulls FAIL - check null distribution builder!")
    elif global_ks_pvalue >= 0.05:
        print("\n" + "="*70)
        print("CONCLUSION: Global null distribution is sufficient")
        print("="*70)
        print("Both global and per-query nulls produce uniform p-values.")
    else:
        print("\n" + "="*70)
        print("CONCLUSION: Neither null distribution produces uniform p-values!")
        print("="*70)
        print("This is a serious problem - HC may not work on this dataset.")
    
    # Plot
    plot_pvalue_diagnostics(
        global_pvalues, 
        per_query_pvalues,
        save_path="results/pvalue_uniformity_test.png"
    )
    
    return {
        'global': (global_ks_stat, global_ks_pvalue),
        'per_query': (pq_ks_stat, pq_ks_pvalue),
        'stored': (stored_ks_stat, stored_ks_pvalue)
    }


if __name__ == "__main__":
    main()
