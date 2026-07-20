# Scaling the HC Dataset: Research & Implementation Plan

## Alon's Feedback — Two Action Items

### 1. Single Global Null Distribution (instead of per-query)

Your current approach builds a **per-query null distribution** from cosine similarities between each query and all its non-relevant documents. Alon wants a **single pooled null distribution** shared across all queries.

**Why this matters for the research:**
- A single null is more realistic for production RAG systems — you can't construct a per-query null at inference time without knowing which documents are relevant (which is the very thing you're trying to determine).
- It tests whether HC is robust to a "one-size-fits-all" null, which is a stronger and more publishable result.
- If performance barely changes, it validates that the similarity score structure is consistent across queries — a key theoretical claim.

**How to implement on the current dataset first:**
1. Pool all non-relevant (query, document) cosine similarity scores across all 44 queries into one array (~31,438 pairs from your PDF).
2. For each query's candidate documents, compute p-values against this single pooled empirical CDF.
3. Run HC as before and compare F1, recall, precision to your per-query results.
4. Validate the single null with a KS test for uniformity (the pooled KS stat of 0.0025 you already have is promising).

**Expected outcome:** Slight degradation is acceptable. If the global null works well, it means HC is fundamentally sound and not relying on per-query tuning — that's a much stronger story for the paper.

---

### 2. Scale to 100k+ Documents — Dataset Options

Alon's core requirement: a dataset that is **analogous** to your category-key dataset (categories → queries, items in categories → relevant documents, variable K) but at **100k+ document scale**, so that retrieval is more meaningful and each query retrieves from a much larger haystack.

Below I evaluate three main approaches, ordered by how well they fit your needs.

---

## Option A: Amazon Product Categories (Recommended)

### The Idea
Use the **Amazon Reviews 2023 dataset** (McAuley Lab, HuggingFace) or the **Amazon ESCI Shopping Queries Dataset** to build a category-key style benchmark at scale. Products are documents, product subcategories define relevance, and queries ask about categories.

### Dataset Sources

**Amazon Reviews 2023** (McAuley Lab):
- 571M reviews, 48M products, **33 main categories** with hierarchical subcategories
- Each product has: `main_category`, `title`, `description`, `features`, `categories` (hierarchical list), `store`, `details`
- Available on HuggingFace: `McAuley-Lab/Amazon-Reviews-2023`
- Per-category subsets downloadable (e.g., `raw_meta_Books`, `raw_meta_Electronics`)
- The `categories` field provides hierarchical paths like `["Electronics", "Computers", "Laptops"]`

**Amazon ESCI Shopping Queries** (Amazon Science):
- 130K unique queries, 2.6M query-product relevance judgments
- Products have: `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color`
- Labels: Exact / Substitute / Complement / Irrelevant
- Already a retrieval benchmark — but queries are natural language shopping queries, not category-enumeration queries

### How to Build the Dataset (Amazon Reviews 2023)

The construction mirrors your country-cities approach exactly:

| Country-Cities Dataset | Amazon Category Dataset |
|---|---|
| Country = category | Product subcategory = category |
| Cities = relevant documents | Products in subcategory = relevant documents |
| "What are the largest cities in [country]?" | "What are [subcategory] products?" |
| 44 queries, 726 documents | 200-500 queries, 100k-500k documents |
| K = 4-24 | K = 5-100+ (controllable via subcategory selection) |

**Step-by-step construction:**

1. **Select a domain subset** (e.g., Books, Electronics, or Grocery) to keep it manageable. The Books subset alone has millions of products.

2. **Extract subcategory structure** from the `categories` hierarchical field. For example, within "Books": Science Fiction, Romance, Biography, Cooking, etc. Each subcategory becomes a query.

3. **Create documents** from product metadata: concatenate `title + description + features` into a single text passage per product. This gives you text documents analogous to your city passages.

4. **Define relevance**: A product is relevant to a query if it belongs to that subcategory. Products from other subcategories are distractors.

5. **Control variable K**: Select subcategories with different sizes. Some subcategories have 5-10 products, others have 50-100+. Filter to achieve a target K distribution similar to your Small/Medium/Large buckets.

6. **Sample to target size**: From the full category, sample ~100k-200k products total, ensuring the relevant documents for each query remain sparse (key requirement #1).

**Concrete example:**
- Query: "What are Science Fiction & Fantasy books on Amazon?"  
- Relevant docs: All products with `categories` containing "Science Fiction & Fantasy" (K might be ~30)
- Non-relevant docs: All products from other subcategories (100k+ distractors)
- Sparsity: 30 / 100,000 = 0.03% — very sparse, exactly what you need

### Why This Works for HC

This satisfies all three of Alon's rules:

1. **Sparse**: With 100k+ documents and K=5-100 per query, relevant documents are 0.005%-0.1% of the corpus — much sparser than your current 726-document setup.

2. **Variable K**: Product subcategories naturally have different sizes. By selecting subcategories at different levels of the hierarchy, you get wide K variation. You can also combine subcategories (like you combined countries with different numbers of cities).

3. **Topic separation in embedding space**: Products in "Science Fiction" have very different text from products in "Cooking" or "Baby Products". This gives you the encoding quality difference HC needs — relevant documents will cluster near the query while distant categories will have low similarity.

### Practical Considerations

**Advantages:**
- Real-world data that Alon specifically suggested
- Natural category structure — no synthetic construction needed
- Trivially scalable: 100k, 500k, or 1M+ documents by adjusting sampling
- Citeable dataset (McAuley Lab is well-known in the community)
- Text-rich metadata enables meaningful embeddings
- Multiple domains available (Books, Electronics, Grocery) for robustness testing

**Risks & Mitigations:**
- *Some products may belong to multiple subcategories* → Use the most specific (leaf) category as the primary label, or explicitly handle multi-label relevance
- *Product descriptions vary in quality* → Filter out products with empty or very short descriptions
- *Category hierarchy depth varies* → Standardize at a fixed depth level (e.g., 2nd or 3rd level)
- *Very large categories may dominate* → Cap maximum K per query and subsample large categories

### Data Access

```python
from datasets import load_dataset

# Load product metadata for Books
meta = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_meta_Books",
    split="full",
    trust_remote_code=True
)

# Example record has: main_category, title, description, 
# features, categories, store, details, price
print(meta[0])
```

Each record contains `categories` (hierarchical list), `title`, `description`, `features` — everything needed to construct documents and define relevance.

---

## Option B: Wikipedia Multi-Entity Scale-Up

### The Idea
Scale your exact country-cities approach to multiple entity types: countries/cities, genres/movies, brands/products, authors/books, sports teams/players, etc. Each entity type provides a set of categories with variable K.

### Construction

Use Wikidata SPARQL queries to enumerate entity sets:
- "What movies are in the [genre] genre?" (K varies by genre)
- "What players play for [team]?" (K varies by team)
- "What universities are in [country]?" (K varies by country)
- "What songs did [artist] release?" (K varies by artist)

For each entity, create document passages from Wikipedia article lead paragraphs.

### Scale
- 50 entity types × 20-50 categories each × 5-50 items per category = easily 100k+ documents
- Add distractor documents from unrelated entity types

### Assessment

**Advantages:**
- Direct extension of your proven approach
- Complete control over K distribution
- SPARQL gives you ground truth completeness
- Wikipedia text is high-quality for embeddings

**Disadvantages:**
- More "synthetic" than real-world data — a reviewer might question ecological validity
- Construction is more complex (many SPARQL queries, entity linking)
- Wikipedia paragraphs may leak information (LLM contamination risk)
- Doesn't align with Alon's specific suggestion of Amazon-style data

**Verdict:** Good fallback if Amazon approach doesn't work, but less aligned with Alon's direction.

---

## Option C: Amazon ESCI Shopping Queries (Pre-built)

### The Idea
Use the Amazon ESCI dataset directly — it already has query-product relevance labels with 130K queries and 2.6M judgments.

### Assessment

**Advantages:**
- Already annotated with relevance (Exact/Substitute/Complement/Irrelevant)
- Natural shopping queries, not synthetic category-enumeration
- 1.8M unique products — well beyond 100k
- Published and peer-reviewed (KDD Cup 2022)

**Disadvantages:**
- The relevance structure is different: each query has up to 40 judged products, but this doesn't mean exactly K products are relevant — it's a partial judgment set
- You don't know the *complete* set of relevant products (unlike your category-key dataset where ground truth is exhaustive)
- Variable K is harder to control — the dataset wasn't designed for this
- The "Substitute" and "Complement" labels create ambiguity about what counts as relevant for HC purposes
- Queries are natural language, not category-enumeration style — HC's task becomes harder to evaluate cleanly

**Verdict:** Interesting for a future follow-up study, but less suitable for the controlled proof-of-concept Alon is asking for. The lack of exhaustive ground truth is a dealbreaker for precisely measuring HC's recall/precision.

---

## Recommendation: Hybrid Amazon Category Approach

I recommend **Option A** (Amazon Product Categories) as the primary approach, with the following specific plan:

### Target Dataset Specifications

| Parameter | Value |
|---|---|
| Total documents | 100k-200k product passages |
| Number of queries | 200-500 |
| K range | 5-100 relevant docs per query |
| K distribution | Small (5-15), Medium (20-40), Large (50-100) |
| Sparsity | 0.005%-0.1% relevant per query |
| Domain | Amazon Books (richest text metadata) |
| Document text | title + description + features concatenated |
| Embedding model | BAAI/bge-base-en-v1.5 (same as current) |

### Implementation Phases

**Phase 1 (Days 1-3): Data Acquisition & Exploration**
- Download Amazon Reviews 2023 Books metadata from HuggingFace
- Explore the category hierarchy and subcategory distribution
- Identify subcategories with appropriate sizes for variable K
- Filter products with sufficient text content (title + description ≥ 50 words)

**Phase 2 (Days 3-5): Dataset Construction**
- Select 200-500 subcategories spanning the K range
- Sample products to achieve target corpus size (100k-200k)
- Construct queries: "What are [subcategory] products on Amazon?"
- Define binary relevance: product in subcategory = relevant, else = irrelevant
- Validate K distribution and sparsity

**Phase 3 (Days 5-7): Single Global Null Distribution**
- Embed all documents + queries using BGE-base-en-v1.5
- Build FAISS index
- Construct a **single global null distribution** by pooling cosine similarities from all (query, non-relevant document) pairs
- Validate uniformity with KS test, Anderson-Darling test
- Compare against per-query nulls to verify robustness (Alon's request #1)

**Phase 4 (Days 7-10): HC Evaluation**
- Run HC with the global null on all queries
- Sweep gamma and candidate pool sizes
- Compare against Top-K baselines (K=5, 10, 20, 50)
- Compute F1, recall, precision, NDCG, MAP
- Analyze performance by K-bucket (Small/Medium/Large)

**Phase 5 (Days 10-12): Analysis & Documentation**
- Create PDF report analogous to your current one
- Analyze where HC wins/loses vs Top-K at this scale
- Document any issues with the global null at 100k+ scale
- Prepare email summary for Alon

### Key Design Decisions

**Why Books specifically?**
- Richest text descriptions among Amazon categories
- Deep category hierarchy (5+ levels)
- Wide range of subcategory sizes
- Familiar domain — easy for reviewers to understand

**How to handle the global null at 100k+ scale?**
With 100k+ documents and 200-500 queries, the global null pool will be enormous (~50M-100M similarity pairs). You have two options:
1. **Full computation**: Compute all pairs (feasible with FAISS, takes ~minutes)
2. **Sampled null**: Random sample of 1-5M non-relevant similarity scores (faster, statistically equivalent for the empirical CDF)

Both should give nearly identical p-values. Option 2 is recommended for computational efficiency.

**What about cross-category similarity?**
Some subcategories will be semantically close (e.g., "Science Fiction" vs "Fantasy"). This is actually **good for your research** — it creates harder retrieval scenarios where HC's adaptive thresholding should shine compared to fixed Top-K. You can analyze HC performance as a function of inter-category semantic distance.

---

## Summary Table

| Criterion | Option A: Amazon Categories | Option B: Wiki Multi-Entity | Option C: Amazon ESCI |
|---|---|---|---|
| Scale (100k+) | ✅ Easily scalable | ✅ Easily scalable | ✅ Already 1.8M |
| Variable K | ✅ Natural from categories | ✅ Controlled via SPARQL | ⚠️ Partial judgments |
| Exhaustive ground truth | ✅ Category = relevance | ✅ SPARQL = completeness | ❌ Only judged subset |
| Real-world data | ✅ Actual products | ⚠️ Wikipedia (synthetic queries) | ✅ Real queries |
| Alon's suggestion | ✅ Directly aligned | ❌ Not mentioned | ⚠️ Partially aligned |
| Sparse | ✅ At 100k+ scale | ✅ Controllable | ✅ Natural |
| Implementation effort | Medium (data wrangling) | High (SPARQL + Wikipedia) | Low (pre-built) |
| Publication strength | Strong (real-world) | Medium (synthetic feel) | Weak (wrong structure) |

**Bottom line:** Go with Option A. It directly implements Alon's suggestion, maintains all three dataset rules, provides exhaustive ground truth, and scales naturally to 100k+.
