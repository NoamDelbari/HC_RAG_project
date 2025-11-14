"""
Analyze Document Lengths in CRAG Dataset

Analyzes the length distribution of documents to determine if chunking is needed
and what strategy would be most effective.

Usage:
    python src/tests/analyze_doc_lengths.py
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.crag_loader import CRAGLoader
from embeddings.embedding_model import EmbeddingModel

# Optional matplotlib import
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not installed - skipping visualizations")


def analyze_tokenization(texts, model_name):
    """Analyze token counts for texts."""
    from transformers import AutoTokenizer
    from tqdm import tqdm
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    token_counts = []
    for text in tqdm(texts, desc="Tokenizing documents"):
        try:
            # Tokenize with truncation to avoid errors, but don't actually truncate
            tokens = tokenizer.encode(
                text, 
                add_special_tokens=True,
                truncation=False,
                max_length=None
            )
            token_counts.append(len(tokens))
        except Exception as e:
            # If tokenization fails, estimate from character length
            estimated_tokens = len(text) // 4
            token_counts.append(estimated_tokens)
    
    return np.array(token_counts)


def main():
    print("\n" + "="*80)
    print("CRAG DOCUMENT LENGTH ANALYSIS")
    print("="*80 + "\n")
    
    # Load data
    print("Loading CRAG dataset...")
    loader = CRAGLoader(use_full_html=True)
    queries, documents = loader.load_by_tasks(["1_2"])
    
    print(f"✓ Loaded {len(queries)} queries")
    print(f"✓ Loaded {len(documents)} documents\n")
    
    # Analyze character lengths
    print("="*80)
    print("CHARACTER LENGTH ANALYSIS")
    print("="*80)
    
    char_lengths = [len(doc.text) for doc in documents]
    char_lengths = np.array(char_lengths)
    
    print(f"Statistics:")
    print(f"  Mean: {char_lengths.mean():.0f} chars")
    print(f"  Median: {np.median(char_lengths):.0f} chars")
    print(f"  Std: {char_lengths.std():.0f} chars")
    print(f"  Min: {char_lengths.min():.0f} chars")
    print(f"  Max: {char_lengths.max():.0f} chars")
    print(f"\nPercentiles:")
    for p in [50, 75, 90, 95, 99]:
        print(f"  {p}th: {np.percentile(char_lengths, p):.0f} chars")
    
    # Analyze token lengths
    print(f"\n" + "="*80)
    print("TOKEN LENGTH ANALYSIS (all-mpnet-base-v2)")
    print("="*80)
    
    model_name = "sentence-transformers/all-mpnet-base-v2"
    print(f"Tokenizing {len(documents)} documents...")
    print("(This may take a few minutes...)")
    
    doc_texts = [doc.text for doc in documents]
    token_counts = analyze_tokenization(doc_texts, model_name)
    
    print(f"\nStatistics:")
    print(f"  Mean: {token_counts.mean():.0f} tokens")
    print(f"  Median: {np.median(token_counts):.0f} tokens")
    print(f"  Std: {token_counts.std():.0f} tokens")
    print(f"  Min: {token_counts.min():.0f} tokens")
    print(f"  Max: {token_counts.max():.0f} tokens")
    print(f"\nPercentiles:")
    for p in [50, 75, 90, 95, 99]:
        print(f"  {p}th: {np.percentile(token_counts, p):.0f} tokens")
    
    # Check model max length
    max_seq_length = 384  # all-mpnet-base-v2 default
    print(f"\n⚠️  Model max sequence length: {max_seq_length} tokens")
    
    truncated = np.sum(token_counts > max_seq_length)
    truncated_pct = (truncated / len(token_counts)) * 100
    
    print(f"\nDocuments exceeding max length:")
    print(f"  Count: {truncated} / {len(token_counts)} ({truncated_pct:.1f}%)")
    
    if truncated > 0:
        print(f"\n⚠️  WARNING: {truncated_pct:.1f}% of documents will be truncated!")
        print(f"  This may significantly impact retrieval quality.")
        print(f"\n💡 Recommendation: Implement chunking strategy")
    else:
        print(f"\n✓ All documents fit within model's max length")
        print(f"  No chunking needed for current dataset")
    
    # Analyze very long documents
    print(f"\n" + "="*80)
    print("VERY LONG DOCUMENTS (>1000 tokens)")
    print("="*80)
    
    long_docs = token_counts > 1000
    n_long = np.sum(long_docs)
    
    if n_long > 0:
        print(f"Found {n_long} very long documents ({n_long/len(token_counts)*100:.1f}%)")
        print(f"\nExample long documents:")
        
        long_indices = np.where(long_docs)[0][:5]  # Show first 5
        for idx in long_indices:
            doc = documents[idx]
            print(f"\n  Doc ID: {doc.doc_id}")
            print(f"  Title: {doc.title[:60]}...")
            print(f"  Tokens: {token_counts[idx]}")
            print(f"  Chars: {len(doc.text)}")
            print(f"  Preview: {doc.text[:100]}...")
    else:
        print("No documents exceed 1000 tokens")
    
    # Create visualizations
    if HAS_MATPLOTLIB:
        print(f"\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Character length distribution
        axes[0, 0].hist(char_lengths, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.median(char_lengths), color='red', linestyle='--', 
                           label=f'Median: {np.median(char_lengths):.0f}')
        axes[0, 0].set_xlabel('Character Length')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Document Character Length Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Token length distribution
        axes[0, 1].hist(token_counts, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(max_seq_length, color='red', linestyle='--', 
                           label=f'Model Max: {max_seq_length}')
        axes[0, 1].axvline(np.median(token_counts), color='green', linestyle='--', 
                           label=f'Median: {np.median(token_counts):.0f}')
        axes[0, 1].set_xlabel('Token Count')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Document Token Count Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Cumulative distribution
        sorted_tokens = np.sort(token_counts)
        cumulative = np.arange(1, len(sorted_tokens) + 1) / len(sorted_tokens) * 100
        axes[1, 0].plot(sorted_tokens, cumulative, linewidth=2)
        axes[1, 0].axvline(max_seq_length, color='red', linestyle='--', 
                           label=f'Model Max: {max_seq_length}')
        axes[1, 0].set_xlabel('Token Count')
        axes[1, 0].set_ylabel('Cumulative %')
        axes[1, 0].set_title('Cumulative Token Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Box plot
        axes[1, 1].boxplot([token_counts], vert=False)
        axes[1, 1].axvline(max_seq_length, color='red', linestyle='--', 
                           label=f'Model Max: {max_seq_length}')
        axes[1, 1].set_xlabel('Token Count')
        axes[1, 1].set_title('Token Count Box Plot')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        output_file = 'results/document_length_analysis.png'
        Path('results').mkdir(exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {output_file}")
    else:
        print(f"\n" + "="*80)
        print("VISUALIZATIONS SKIPPED")
        print("="*80)
        print("Install matplotlib to generate plots:")
        print("  pip install matplotlib")
    
    # Summary recommendations
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if truncated_pct > 10:
        print("\n🔴 HIGH PRIORITY: Implement chunking strategy")
        print(f"   {truncated_pct:.1f}% of documents exceed model's max length")
        print("\n   Suggested strategies:")
        print("   1. Fixed-size chunks with overlap (e.g., 256 tokens, 50 token overlap)")
        print("   2. Sentence-based chunks (preserve sentence boundaries)")
        print("   3. Semantic chunks (use TextSplitter)")
        print("\n   Benefits:")
        print("   - Better coverage of long documents")
        print("   - More granular retrieval")
        print("   - Potentially improved recall")
    elif truncated_pct > 5:
        print("\n🟡 MEDIUM PRIORITY: Consider chunking")
        print(f"   {truncated_pct:.1f}% of documents exceed max length")
        print("   Chunking may improve retrieval for these documents")
    else:
        print("\n🟢 LOW PRIORITY: Chunking optional")
        print(f"   Only {truncated_pct:.1f}% of documents truncated")
        print("   Current approach may be sufficient")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
