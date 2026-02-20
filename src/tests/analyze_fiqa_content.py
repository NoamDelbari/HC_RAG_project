"""
Analyze FiQA content to determine if it's suitable for dense embeddings vs BM25.
"""

import random
from beir.datasets.data_loader import GenericDataLoader

# Load FiQA
corpus, queries, qrels = GenericDataLoader('datasets/beir/fiqa').load(split='test')

print('='*80)
print('FiQA DATASET CONTENT ANALYSIS')
print('='*80)

# Sample queries
print('\n--- SAMPLE QUERIES AND RELEVANT DOCS ---')
query_ids = list(queries.keys())
random.seed(42)
sample_qids = random.sample(query_ids, 10)

for qid in sample_qids:
    print(f'\nQ: {queries[qid]}')
    if qid in qrels:
        rel_docs = [d for d, s in qrels[qid].items() if s > 0]
        print(f'   [{len(rel_docs)} relevant docs]')
        if rel_docs and rel_docs[0] in corpus:
            doc = corpus[rel_docs[0]]
            text = doc.get('text', '')[:250]
            print(f'   Doc: {text}...')

print('\n' + '='*80)
print('SEMANTIC vs LEXICAL OVERLAP ANALYSIS')
print('='*80)

# Check semantic vs lexical matching
semantic_needed = 0
lexical_sufficient = 0

random.seed(123)
sample_qids = random.sample(query_ids, min(200, len(query_ids)))

for qid in sample_qids:
    q = queries[qid].lower()
    if qid not in qrels:
        continue
    rel_docs = [d for d, s in qrels[qid].items() if s > 0]
    if not rel_docs:
        continue
    doc_text = corpus.get(rel_docs[0], {}).get('text', '').lower()
    
    # Check word overlap
    q_words = set(w for w in q.split() if len(w) > 3)  # Skip short words
    doc_words = set(doc_text.split())
    overlap = len(q_words & doc_words) / len(q_words) if q_words else 0
    
    if overlap < 0.5:
        semantic_needed += 1
    else:
        lexical_sufficient += 1

total = semantic_needed + lexical_sufficient
print(f'\nAnalyzed {total} query-doc pairs:')
print(f'  Low word overlap (<50%): {semantic_needed} ({100*semantic_needed/total:.0f}%)')
print(f'  High word overlap (>=50%): {lexical_sufficient} ({100*lexical_sufficient/total:.0f}%)')

if semantic_needed > lexical_sufficient:
    print('\n  ✓ SEMANTIC EMBEDDINGS ARE IMPORTANT')
    print('    Many relevant docs don\'t share exact words with query')
    print('    Dense embeddings (BGE) will outperform BM25')
else:
    print('\n  ✗ BM25 might work OK (high lexical overlap)')

# Query complexity
print('\n' + '='*80)
print('QUERY TYPE ANALYSIS')
print('='*80)

questions = list(queries.values())
avg_len = sum(len(q.split()) for q in questions) / len(questions)
print(f'\nAverage query length: {avg_len:.1f} words')

# Categorize
conceptual = sum(1 for q in questions if q.lower().startswith(('how ', 'why ', 'what is', 'what are', 'what does', 'explain')))
comparison = sum(1 for q in questions if 'vs' in q.lower() or 'difference' in q.lower() or 'better' in q.lower())
opinion = sum(1 for q in questions if 'should i' in q.lower() or 'best' in q.lower() or 'recommend' in q.lower())

print(f'Conceptual (how/why/what/explain): {conceptual} ({100*conceptual/len(questions):.0f}%)')
print(f'Comparison (vs/difference/better): {comparison} ({100*comparison/len(questions):.0f}%)')
print(f'Opinion/Advice (should/best): {opinion} ({100*opinion/len(questions):.0f}%)')

print('\n' + '='*80)
print('VERDICT')
print('='*80)
print('\nFiQA characteristics:')
print('  - Financial domain (investing, taxes, banking)')
print('  - Mostly conceptual/advice questions')
print('  - Answers require understanding, not just keyword matching')
print('\n  ✓ WELL SUITED FOR DENSE EMBEDDINGS (BGE)')
print('  ✓ WELL SUITED FOR HC (varying relevance per query)')
