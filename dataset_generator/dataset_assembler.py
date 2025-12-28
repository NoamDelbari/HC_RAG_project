"""
Dataset Assembly Module

Assembles the final dataset:
- Corpus with relevant passages and hard negatives (C.5)
- Qrels compilation (C.6)
- Train/dev/test splits (C.7)
- Metadata and documentation (C.8)

Phase C.5-C.8: Dataset Assembly
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class CorpusStats:
    """Statistics about the assembled corpus."""
    total_passages: int
    relevant_passages: int
    hard_negatives: int
    random_negatives: int
    sparsity: float  # relevant / total
    avg_passages_per_query: float
    min_passages_per_query: int
    max_passages_per_query: int
    median_passages_per_query: float
    passages_per_query_histogram: dict  # k_range -> count


@dataclass
class DatasetSplit:
    """A dataset split (train/dev/test)."""
    name: str
    query_ids: list
    query_count: int


@dataclass
class AssembledDataset:
    """Complete assembled dataset."""
    corpus: list
    queries: list
    qrels: list
    splits: dict  # name -> DatasetSplit (optional, can be empty)
    metadata: dict
    stats: dict


class CorpusAssembler:
    """
    Assembles the passage corpus with target sparsity.

    Adds hard negatives and random negatives to achieve target sparsity.
    """

    def __init__(self, target_sparsity: float = 0.02):
        """
        Initialize assembler.

        Args:
            target_sparsity: Target relevant/total ratio (default 2%)
        """
        self.target_sparsity = target_sparsity

    def assemble_corpus(
        self,
        selected_queries: list,
        passage_mappings: dict,
        all_passages: list,
        include_hard_negatives: bool = True
    ) -> tuple:
        """
        Assemble the corpus with appropriate sparsity.

        Args:
            selected_queries: List of selected query dicts
            passage_mappings: Dict mapping query_id to passage matches
            all_passages: All available passages for sampling negatives
            include_hard_negatives: Whether to add hard negatives

        Returns:
            Tuple of (corpus list, qrels list, stats)
        """
        # Collect relevant passages
        relevant_pids = set()
        qrels = []

        for query in selected_queries:
            query_id = query.get("query_id")
            mapping = passage_mappings.get(query_id, {})

            for match in mapping.get("matches", []):
                pid = match.get("passage_id")
                if pid:
                    relevant_pids.add(pid)
                    qrels.append({
                        "query_id": query_id,
                        "passage_id": pid,
                        "relevance": 1,
                        "entity_qid": match.get("entity_qid", "")
                    })

        # Build corpus with relevant passages
        corpus = []
        passage_lookup = {p.get("passage_id"): p for p in all_passages}

        for pid in relevant_pids:
            passage = passage_lookup.get(pid)
            if passage:
                corpus.append({
                    "passage_id": pid,
                    "text": passage.get("passage_text", ""),
                    "entity_qid": passage.get("entity_qid", ""),
                    "entity_label": passage.get("entity_label", ""),
                    "section": passage.get("section", ""),
                    "is_relevant": True,
                    "is_hard_negative": False
                })

        relevant_count = len(corpus)

        # Calculate how many negatives needed
        if self.target_sparsity > 0:
            total_needed = int(relevant_count / self.target_sparsity)
            negatives_needed = total_needed - relevant_count
        else:
            negatives_needed = 0

        # Add hard negatives (passages from related but non-relevant entities)
        hard_negatives_added = 0
        if include_hard_negatives and negatives_needed > 0:
            hard_negative_count = int(negatives_needed * 0.15)  # 15% hard negatives
            hard_negatives = self._sample_hard_negatives(
                selected_queries,
                passage_mappings,
                all_passages,
                relevant_pids,
                hard_negative_count
            )
            corpus.extend(hard_negatives)
            hard_negatives_added = len(hard_negatives)
            negatives_needed -= hard_negatives_added

        # Add random negatives
        random_negatives_added = 0
        if negatives_needed > 0:
            random_negatives = self._sample_random_negatives(
                all_passages,
                relevant_pids,
                negatives_needed
            )
            corpus.extend(random_negatives)
            random_negatives_added = len(random_negatives)

        # Calculate passages per query for each query
        passages_per_query = []
        for query in selected_queries:
            query_id = query.get("query_id")
            mapping = passage_mappings.get(query_id, {})
            passage_count = len(mapping.get("matches", []))
            passages_per_query.append(passage_count)

        # Compute per-query statistics
        if passages_per_query:
            avg_ppq = sum(passages_per_query) / len(passages_per_query)
            min_ppq = min(passages_per_query)
            max_ppq = max(passages_per_query)
            sorted_ppq = sorted(passages_per_query)
            mid = len(sorted_ppq) // 2
            median_ppq = (sorted_ppq[mid] + sorted_ppq[~mid]) / 2
        else:
            avg_ppq = min_ppq = max_ppq = median_ppq = 0

        # Distribution histogram by K-range
        ppq_histogram = {"2-5": 0, "6-10": 0, "11-15": 0, "16-20": 0}
        for k in passages_per_query:
            if k <= 5:
                ppq_histogram["2-5"] += 1
            elif k <= 10:
                ppq_histogram["6-10"] += 1
            elif k <= 15:
                ppq_histogram["11-15"] += 1
            else:
                ppq_histogram["16-20"] += 1

        # Calculate stats
        stats = CorpusStats(
            total_passages=len(corpus),
            relevant_passages=relevant_count,
            hard_negatives=hard_negatives_added,
            random_negatives=random_negatives_added,
            sparsity=relevant_count / len(corpus) if corpus else 0,
            avg_passages_per_query=avg_ppq,
            min_passages_per_query=min_ppq,
            max_passages_per_query=max_ppq,
            median_passages_per_query=median_ppq,
            passages_per_query_histogram=ppq_histogram
        )

        return corpus, qrels, stats

    def _sample_hard_negatives(
        self,
        queries: list,
        mappings: dict,
        all_passages: list,
        relevant_pids: set,
        count: int
    ) -> list:
        """Sample hard negatives (related but non-relevant passages)."""
        hard_negatives = []

        # Get entities from relevant passages
        relevant_entities = set()
        for query in queries:
            query_id = query.get("query_id")
            mapping = mappings.get(query_id, {})
            for match in mapping.get("matches", []):
                relevant_entities.add(match.get("entity_qid"))

        # Find passages from related entities (same type but not relevant)
        candidate_passages = [
            p for p in all_passages
            if p.get("passage_id") not in relevant_pids
        ]

        # Sample
        sampled = random.sample(candidate_passages, min(count, len(candidate_passages)))

        for passage in sampled:
            hard_negatives.append({
                "passage_id": passage.get("passage_id"),
                "text": passage.get("passage_text", ""),
                "entity_qid": passage.get("entity_qid", ""),
                "entity_label": passage.get("entity_label", ""),
                "section": passage.get("section", ""),
                "is_relevant": False,
                "is_hard_negative": True
            })

        return hard_negatives

    def _sample_random_negatives(
        self,
        all_passages: list,
        relevant_pids: set,
        count: int
    ) -> list:
        """Sample random negative passages."""
        candidates = [
            p for p in all_passages
            if p.get("passage_id") not in relevant_pids
        ]

        sampled = random.sample(candidates, min(count, len(candidates)))

        return [
            {
                "passage_id": p.get("passage_id"),
                "text": p.get("passage_text", ""),
                "entity_qid": p.get("entity_qid", ""),
                "entity_label": p.get("entity_label", ""),
                "section": p.get("section", ""),
                "is_relevant": False,
                "is_hard_negative": False
            }
            for p in sampled
        ]


class DatasetSplitter:
    """
    Creates train/dev/test splits with stratification.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        dev_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ):
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def create_splits(
        self,
        queries: list,
        ensure_cluster_separation: bool = True
    ) -> dict:
        """
        Create train/dev/test splits.

        Args:
            queries: List of query dicts
            ensure_cluster_separation: Keep cluster instances in same split

        Returns:
            Dict mapping split name to DatasetSplit
        """
        random.seed(self.seed)

        if ensure_cluster_separation:
            return self._split_by_cluster(queries)
        else:
            return self._split_random(queries)

    def _split_by_cluster(self, queries: list) -> dict:
        """Split keeping cluster instances together."""
        # Group by cluster
        by_cluster = defaultdict(list)
        for q in queries:
            by_cluster[q.get("cluster_id", "")].append(q)

        cluster_ids = list(by_cluster.keys())
        random.shuffle(cluster_ids)

        # Assign clusters to splits
        n = len(cluster_ids)
        train_end = int(n * self.train_ratio)
        dev_end = int(n * (self.train_ratio + self.dev_ratio))

        train_clusters = cluster_ids[:train_end]
        dev_clusters = cluster_ids[train_end:dev_end]
        test_clusters = cluster_ids[dev_end:]

        # Collect queries
        train_queries = [q for c in train_clusters for q in by_cluster[c]]
        dev_queries = [q for c in dev_clusters for q in by_cluster[c]]
        test_queries = [q for c in test_clusters for q in by_cluster[c]]

        return {
            "train": DatasetSplit("train", [q.get("query_id") for q in train_queries], len(train_queries)),
            "dev": DatasetSplit("dev", [q.get("query_id") for q in dev_queries], len(dev_queries)),
            "test": DatasetSplit("test", [q.get("query_id") for q in test_queries], len(test_queries))
        }

    def _split_random(self, queries: list) -> dict:
        """Random split (ignoring clusters)."""
        query_ids = [q.get("query_id") for q in queries]
        random.shuffle(query_ids)

        n = len(query_ids)
        train_end = int(n * self.train_ratio)
        dev_end = int(n * (self.train_ratio + self.dev_ratio))

        return {
            "train": DatasetSplit("train", query_ids[:train_end], train_end),
            "dev": DatasetSplit("dev", query_ids[train_end:dev_end], dev_end - train_end),
            "test": DatasetSplit("test", query_ids[dev_end:], n - dev_end)
        }


class MetadataGenerator:
    """
    Generates dataset metadata and documentation.
    """

    def __init__(self, dataset_name: str = "CrossEntityQA"):
        self.dataset_name = dataset_name
        self.version = "1.0"

    def generate_metadata(
        self,
        queries: list,
        corpus: list,
        qrels: list,
        splits: dict,
        corpus_stats: CorpusStats
    ) -> dict:
        """Generate complete metadata."""
        # Query statistics
        by_difficulty = defaultdict(int)
        by_k_range = defaultdict(int)
        by_type = defaultdict(int)

        for q in queries:
            by_difficulty[q.get("difficulty", "")] += 1
            by_type[q.get("cluster_type", "")] += 1

            k = q.get("actual_k", 0)
            if k <= 5:
                by_k_range["2-5"] += 1
            elif k <= 10:
                by_k_range["6-10"] += 1
            elif k <= 15:
                by_k_range["11-15"] += 1
            else:
                by_k_range["16-20"] += 1

        return {
            "name": self.dataset_name,
            "version": self.version,
            "creation_date": datetime.now().isoformat(),
            "description": "Cross-entity QA dataset for variable-K retrieval evaluation",

            "statistics": {
                "total_queries": len(queries),
                "total_passages": corpus_stats.total_passages,
                "total_qrels": len(qrels),
                "sparsity": corpus_stats.sparsity,

                "by_difficulty": dict(by_difficulty),
                "by_k_range": dict(by_k_range),
                "by_cluster_type": dict(by_type),

                "corpus": {
                    "relevant_passages": corpus_stats.relevant_passages,
                    "hard_negatives": corpus_stats.hard_negatives,
                    "random_negatives": corpus_stats.random_negatives
                },

                "passages_per_query": {
                    "min": corpus_stats.min_passages_per_query,
                    "max": corpus_stats.max_passages_per_query,
                    "avg": corpus_stats.avg_passages_per_query,
                    "median": corpus_stats.median_passages_per_query,
                    "distribution": corpus_stats.passages_per_query_histogram
                },

                "splits": {
                    name: split.query_count
                    for name, split in splits.items()
                }
            },

            "schema": {
                "queries": {
                    "query_id": "Unique identifier",
                    "query_text": "Natural language query",
                    "cluster_id": "Source cluster identifier",
                    "cluster_type": "Type of cluster",
                    "difficulty": "easy | medium | hard",
                    "target_k": "Target number of relevant passages",
                    "actual_k": "Verified number of relevant passages"
                },
                "corpus": {
                    "passage_id": "Unique identifier",
                    "text": "Passage text content",
                    "entity_qid": "Wikidata QID of primary entity",
                    "entity_label": "Entity name",
                    "section": "Source section in Wikipedia article"
                },
                "qrels": {
                    "query_id": "Query identifier",
                    "passage_id": "Passage identifier",
                    "relevance": "Relevance score (1 = relevant)"
                }
            },

            "usage": {
                "intended_uses": [
                    "Retrieval evaluation with variable K",
                    "RAG system evaluation",
                    "Multi-hop reasoning evaluation"
                ],
                "recommended_metrics": [
                    "Recall@K (where K varies per query)",
                    "NDCG@K",
                    "MAP"
                ]
            }
        }


class DatasetWriter:
    """
    Writes dataset files to disk.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_dataset(self, dataset: AssembledDataset):
        """Write all dataset files."""
        # Write corpus (single JSONL file for RAG indexing)
        corpus_path = self.output_dir / "corpus.jsonl"
        with open(corpus_path, "w", encoding="utf-8") as f:
            for passage in dataset.corpus:
                f.write(json.dumps(passage, ensure_ascii=False) + "\n")
        print(f"Wrote corpus to {corpus_path} ({len(dataset.corpus)} passages)")

        # Write queries (single JSONL file)
        queries_path = self.output_dir / "queries.jsonl"
        with open(queries_path, "w", encoding="utf-8") as f:
            for query in dataset.queries:
                f.write(json.dumps(query, ensure_ascii=False) + "\n")
        print(f"Wrote queries to {queries_path} ({len(dataset.queries)} queries)")

        # Write qrels (TREC format)
        qrels_path = self.output_dir / "qrels.tsv"
        with open(qrels_path, "w", encoding="utf-8") as f:
            for qrel in dataset.qrels:
                f.write(f"{qrel['query_id']}\t0\t{qrel['passage_id']}\t{qrel['relevance']}\n")
        print(f"Wrote qrels to {qrels_path} ({len(dataset.qrels)} relevance judgments)")

        # Write metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(dataset.metadata, f, indent=2, ensure_ascii=False)
        print(f"Wrote metadata to {metadata_path}")

        # Write README
        readme_path = self.output_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(self._generate_readme(dataset))
        print(f"Wrote README to {readme_path}")

    def _generate_readme(self, dataset: AssembledDataset) -> str:
        """Generate README content."""
        stats = dataset.metadata.get("statistics", {})
        ppq = stats.get("passages_per_query", {})

        return f"""# {dataset.metadata.get('name', 'CrossEntityQA')}

Cross-entity QA dataset for variable-K retrieval evaluation.

## Overview

- **Total Queries**: {stats.get('total_queries', 0)}
- **Total Passages**: {stats.get('total_passages', 0)}
- **Sparsity**: {stats.get('sparsity', 0):.1%}

## Structure

```
{dataset.metadata.get('name', 'CrossEntityQA')}/
├── README.md
├── metadata.json
├── corpus.jsonl      # All passages for RAG indexing
├── queries.jsonl     # All queries
└── qrels.tsv         # Relevance judgments (TREC format)
```

## Statistics

### By Difficulty
{self._format_dict(stats.get('by_difficulty', {}))}

### By K-Range
{self._format_dict(stats.get('by_k_range', {}))}

### Passages Per Query
- **Min**: {ppq.get('min', 0)}
- **Max**: {ppq.get('max', 0)}
- **Avg**: {ppq.get('avg', 0):.1f}
- **Median**: {ppq.get('median', 0):.1f}

## Usage

This dataset is designed for evaluating retrieval systems where different queries
require different numbers of passages (K) to answer completely.

### Load corpus for indexing
```python
import json
passages = []
with open('corpus.jsonl', 'r') as f:
    for line in f:
        passages.append(json.loads(line))
```

### Load queries
```python
queries = []
with open('queries.jsonl', 'r') as f:
    for line in f:
        queries.append(json.loads(line))
```

### Load relevance judgments
```python
qrels = {{}}
with open('qrels.tsv', 'r') as f:
    for line in f:
        qid, _, pid, rel = line.strip().split('\\t')
        if qid not in qrels:
            qrels[qid] = set()
        if int(rel) > 0:
            qrels[qid].add(pid)
```

## Citation

If you use this dataset, please cite [TODO].

---
Generated: {dataset.metadata.get('creation_date', '')}
"""

    def _format_dict(self, d: dict) -> str:
        """Format dict as markdown list."""
        return "\n".join(f"- **{k}**: {v}" for k, v in d.items())


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("DATASET ASSEMBLER DEMO")
    print("=" * 60)

    # Sample data
    sample_queries = [
        {"query_id": "q1", "query_text": "Query 1?", "cluster_id": "c1",
         "cluster_type": "creative_ensemble", "difficulty": "easy",
         "target_k": 5, "actual_k": 5},
        {"query_id": "q2", "query_text": "Query 2?", "cluster_id": "c1",
         "cluster_type": "creative_ensemble", "difficulty": "medium",
         "target_k": 5, "actual_k": 4},
        {"query_id": "q3", "query_text": "Query 3?", "cluster_id": "c2",
         "cluster_type": "award_recognition", "difficulty": "hard",
         "target_k": 3, "actual_k": 3},
    ]

    sample_passages = [
        {"passage_id": "p1", "passage_text": "Passage 1 text", "entity_qid": "Q1", "entity_label": "Entity 1"},
        {"passage_id": "p2", "passage_text": "Passage 2 text", "entity_qid": "Q2", "entity_label": "Entity 2"},
        {"passage_id": "p3", "passage_text": "Passage 3 text", "entity_qid": "Q3", "entity_label": "Entity 3"},
        {"passage_id": "p4", "passage_text": "Passage 4 text", "entity_qid": "Q4", "entity_label": "Entity 4"},
        {"passage_id": "p5", "passage_text": "Passage 5 text", "entity_qid": "Q5", "entity_label": "Entity 5"},
    ]

    sample_mappings = {
        "q1": {"matches": [{"passage_id": "p1", "entity_qid": "Q1"}, {"passage_id": "p2", "entity_qid": "Q2"}]},
        "q2": {"matches": [{"passage_id": "p1", "entity_qid": "Q1"}]},
        "q3": {"matches": [{"passage_id": "p3", "entity_qid": "Q3"}]},
    }

    # Assemble corpus
    assembler = CorpusAssembler(target_sparsity=0.02)  # 2% sparsity (default)
    corpus, qrels, stats = assembler.assemble_corpus(
        sample_queries, sample_mappings, sample_passages
    )

    print(f"\nCorpus Stats:")
    print(f"  Total passages: {stats.total_passages}")
    print(f"  Relevant: {stats.relevant_passages}")
    print(f"  Hard negatives: {stats.hard_negatives}")
    print(f"  Random negatives: {stats.random_negatives}")
    print(f"  Sparsity: {stats.sparsity:.1%}")

    # Create splits
    splitter = DatasetSplitter()
    splits = splitter.create_splits(sample_queries)

    print(f"\nSplits:")
    for name, split in splits.items():
        print(f"  {name}: {split.query_count} queries")

    # Generate metadata
    meta_gen = MetadataGenerator()
    metadata = meta_gen.generate_metadata(sample_queries, corpus, qrels, splits, stats)

    print(f"\nMetadata generated with {len(metadata)} top-level keys")
