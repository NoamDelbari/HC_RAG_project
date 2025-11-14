"""
Fix the parent_doc_id mapping in chunk_mapping.pkl

The recursive chunker created sub-IDs like "doc_0_sub1_sub2" but they should 
all map back to the original parent document "doc_0".
"""

import pickle
import re

# Load the broken mapping
with open('crag_chunked_vector_db.chunk_mapping.pkl', 'rb') as f:
    mappings = pickle.load(f)

chunk_to_doc = mappings['chunk_to_doc']
chunk_count_per_doc = mappings['chunk_count_per_doc']
chunker_config = mappings['chunker_config']

print(f"Original unique parent docs: {len(set(chunk_to_doc.values()))}")

# Fix: Extract the real parent doc ID (remove all _sub* suffixes)
def get_real_parent(doc_id):
    """Extract real parent doc ID by removing _sub* recursion artifacts"""
    # Pattern: query_id_doc_0_sub1_sub2... -> query_id_doc_0
    match = re.match(r'^(.+?_doc_\d+)', doc_id)
    if match:
        return match.group(1)
    return doc_id

# Fix chunk_to_doc mapping
fixed_chunk_to_doc = {}
for chunk_id, parent_id in chunk_to_doc.items():
    real_parent = get_real_parent(parent_id)
    fixed_chunk_to_doc[chunk_id] = real_parent

# Rebuild doc_to_chunks mapping
fixed_doc_to_chunks = {}
for chunk_id, parent_id in fixed_chunk_to_doc.items():
    if parent_id not in fixed_doc_to_chunks:
        fixed_doc_to_chunks[parent_id] = []
    fixed_doc_to_chunks[parent_id].append(chunk_id)

# Rebuild chunk_count_per_doc
fixed_chunk_count = {}
for doc_id, chunks in fixed_doc_to_chunks.items():
    fixed_chunk_count[doc_id] = len(chunks)

print(f"Fixed unique parent docs: {len(fixed_doc_to_chunks)}")
print(f"Average chunks per doc: {sum(fixed_chunk_count.values()) / len(fixed_chunk_count):.1f}")

# Verify the fix worked
total_chunks_before = len(chunk_to_doc)
total_chunks_after = len(fixed_chunk_to_doc)
assert total_chunks_before == total_chunks_after, "ERROR: Lost chunks during fix!"
print(f"✓ Verification: All {total_chunks_after} chunks preserved")

# Save fixed mapping
fixed_mappings = {
    'chunk_to_doc': fixed_chunk_to_doc,
    'doc_to_chunks': fixed_doc_to_chunks,
    'chunk_count_per_doc': fixed_chunk_count,
    'chunker_config': chunker_config
}

with open('crag_chunked_vector_db.chunk_mapping.pkl', 'wb') as f:
    pickle.dump(fixed_mappings, f)

print("✓ Fixed mapping saved!")
print("\nSample fixed doc IDs:")
for doc_id in list(fixed_doc_to_chunks.keys())[:5]:
    print(f"  {doc_id}: {len(fixed_doc_to_chunks[doc_id])} chunks")
