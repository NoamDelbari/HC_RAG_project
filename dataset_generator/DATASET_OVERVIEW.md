# Cross-Entity QA Dataset

## What is This Dataset?

A question-answering benchmark where each query requires retrieving **multiple passages** (2-20) to answer completely. Designed for evaluating RAG systems that need to aggregate information across documents.

**Example:**
- Query: *"Who won the Nobel Prize in Physics in 2010?"*
- Requires: 2 passages (Andre Geim, Konstantin Novoselov)

---

## Pipeline Flow

```
Phase A: Data Preparation
─────────────────────────
  Wikidata ──> Entity Clusters ──> Wikipedia ──> Passages
  (SPARQL)    (awards, casts)      (articles)   (chunks)

Phase B/C: Query Generation
───────────────────────────
  Clusters + Passages ──> LLM ──> Queries ──> Filter ──> Dataset
  (with constraints)     (GPT/Claude)        (validate)
```

---

## Clusters Used

| Type | Cluster ID | Description | Entities |
|------|-----------|-------------|----------|
| **Awards** | award_Q38104_2010_2023 | Nobel Prize in Physics (2010-2023) | 38 |
| | award_Q80061_all_all | Turing Award (all years) | 233 |
| | award_Q103360_2000_2023 | Academy Award Best Picture (2000-2023) | 50 |
| **Movie Casts** | cast_Q47703 | The Godfather cast | 40 |
| | cast_Q25188 | Inception cast | 22 |
| | cast_Q44578 | Harry Potter (Deathly Hallows) cast | 51 |
| **Filmographies** | filmography_Q2001 | Stanley Kubrick films | 19 |
| | filmography_Q8877 | Steven Spielberg films | 62 |
| **Political Positions** | position_Q11696_all_all | US Presidents | 65 |
| | position_Q14211_all_all | UK Prime Ministers | 84 |
| **TV Series** | series_Q642878 | Game of Thrones cast | 145 |
| | series_Q8337 | Harry Potter film series cast | 10 |
| | series_Q2484680 | The Simpsons cast | 118 |
| **Geographic** | capitals_Q46 | European capitals | 51 |
| | capitals_Q15 | African capitals | 58 |
| **Founder Relations** | founded_by_Q317521 | Companies founded by Elon Musk | 16 |

**Total: 16 valid clusters, 1,025 entities, 50,069 passages**

---

## Output Files

| File | Contents |
|------|----------|
| `corpus.jsonl` | All passages (one per line, for indexing) |
| `queries.jsonl` | All queries with metadata |
| `qrels.tsv` | Query-passage relevance judgments |

---

## Query Distribution

- **K Range**: 2-5 (25%), 6-10 (31%), 11-15 (25%), 16-20 (19%)
- **Difficulty**: Easy (90%), Medium (7%), Hard (3%)
- **Sparsity**: 2% relevant passages in corpus

---

## Quick Commands

```bash
# Export passages for indexing (no API key needed)
python generate_dataset.py --export-corpus

# Generate queries (test mode)
python generate_dataset.py --test --queries 12

# Full generation
python generate_dataset.py --provider openai
```
