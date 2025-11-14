"""
Verify the chunk-to-document mapping is correct
"""

import pickle
import argparse

parser = argparse.ArgumentParser(description="Verify chunk-to-document mapping")
parser.add_argument(
    "--db",
    type=str,
    default="crag_chunked_vector_db",
    help="Database path (default: crag_chunked_vector_db)"
)
args = parser.parse_args()

print("="*80)
print(f"VERIFYING MAPPING: {args.db}")
print("="*80)

# Load mapping
mapping_file = f"{args.db}.chunk_mapping.pkl"
with open(mapping_file, 'rb') as f:
    mappings = pickle.load(f)

chunk_to_doc = mappings['chunk_to_doc']
doc_to_chunks = mappings['doc_to_chunks']
chunk_count_per_doc = mappings['chunk_count_per_doc']

# Basic counts
print(f"\n✓ Total chunks: {len(chunk_to_doc)}")
print(f"✓ Unique documents: {len(doc_to_chunks)}")
print(f"✓ Average chunks per doc: {sum(chunk_count_per_doc.values()) / len(chunk_count_per_doc):.1f}")
print(f"✓ Min chunks per doc: {min(chunk_count_per_doc.values())}")
print(f"✓ Max chunks per doc: {max(chunk_count_per_doc.values())}")

# Verify no sub IDs remain
print("\n" + "="*80)
print("CHECKING FOR REMAINING _sub* ARTIFACTS")
print("="*80)

bad_parents = [doc_id for doc_id in doc_to_chunks.keys() if '_sub' in doc_id]
if bad_parents:
    print(f"❌ ERROR: Found {len(bad_parents)} documents with _sub in ID!")
    for bad_id in bad_parents[:5]:
        print(f"  {bad_id}")
else:
    print("✅ PERFECT: No _sub artifacts in parent doc IDs!")

# Verify consistency
print("\n" + "="*80)
print("VERIFYING MAPPING CONSISTENCY")
print("="*80)

# Check chunk_to_doc matches doc_to_chunks
all_chunks_from_doc_mapping = set()
for doc_id, chunks in doc_to_chunks.items():
    all_chunks_from_doc_mapping.update(chunks)

all_chunks_from_chunk_mapping = set(chunk_to_doc.keys())

if all_chunks_from_doc_mapping == all_chunks_from_chunk_mapping:
    print(f"✅ CONSISTENT: Both mappings contain same {len(all_chunks_from_chunk_mapping)} chunks")
else:
    print(f"❌ ERROR: Mapping mismatch!")
    print(f"  doc_to_chunks has {len(all_chunks_from_doc_mapping)} chunks")
    print(f"  chunk_to_doc has {len(all_chunks_from_chunk_mapping)} chunks")

# Verify each chunk points to a document that exists
print("\n" + "="*80)
print("VERIFYING CHUNK → DOC REFERENCES")
print("="*80)

orphan_chunks = []
for chunk_id, parent_id in chunk_to_doc.items():
    if parent_id not in doc_to_chunks:
        orphan_chunks.append((chunk_id, parent_id))

if orphan_chunks:
    print(f"❌ ERROR: Found {len(orphan_chunks)} orphan chunks!")
    for chunk_id, parent_id in orphan_chunks[:5]:
        print(f"  {chunk_id} → {parent_id} (doesn't exist in doc_to_chunks)")
else:
    print(f"✅ PERFECT: All chunks reference valid parent documents!")

# Sample some mappings
print("\n" + "="*80)
print("SAMPLE MAPPINGS")
print("="*80)

sample_docs = list(doc_to_chunks.keys())[:3]
for doc_id in sample_docs:
    chunks = doc_to_chunks[doc_id]
    print(f"\nDocument: {doc_id}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Sample chunk IDs:")
    for chunk_id in chunks[:3]:
        parent = chunk_to_doc[chunk_id]
        print(f"    {chunk_id}")
        print(f"      → parent: {parent} {'✅' if parent == doc_id else '❌ MISMATCH'}")

print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

if (not bad_parents and 
    all_chunks_from_doc_mapping == all_chunks_from_chunk_mapping and
    not orphan_chunks):
    print("✅✅✅ MAPPING IS PERFECT! ✅✅✅")
    print(f"✅ {len(doc_to_chunks)} unique documents")
    print(f"✅ {len(chunk_to_doc)} chunks")
    print(f"✅ Average {sum(chunk_count_per_doc.values()) / len(chunk_count_per_doc):.1f} chunks/doc")
    print(f"✅ No artifacts, no orphans, fully consistent")
    print("\n🚀 READY TO RUN EXPERIMENTS! 🚀")
else:
    print("❌ ISSUES FOUND - SEE ABOVE")

print("="*80)
