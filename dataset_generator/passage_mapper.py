"""
Entity-to-Passage Mapping Module

Maps verified entities to their Wikipedia passages for ground truth.

Phase C.2.2: Entity-to-Passage Mapping
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PassageMatch:
    """A matched passage for a query."""
    passage_id: str
    entity_qid: str
    entity_label: str
    passage_text: str
    section: str
    relevance_score: float  # 0-1, higher = more relevant
    match_reason: str


@dataclass
class QueryPassages:
    """All passages matched for a query."""
    query_id: str
    cluster_id: str
    entity_count: int
    passage_count: int
    matches: list  # List of PassageMatch
    missing_entities: list  # Entities without passages
    coverage_rate: float


class PassageMapper:
    """
    Maps entities to passages from the passage corpus.

    Uses the passage corpus from Phase A to find relevant passages
    for each entity in the ground truth.
    """

    def __init__(self, corpus_path: str = None):
        """
        Initialize mapper.

        Args:
            corpus_path: Path to passage corpus JSON/JSONL
        """
        self.passages = []
        self.by_entity = defaultdict(list)  # qid -> [passage_ids]
        self.by_id = {}  # passage_id -> passage data

        if corpus_path:
            self.load_corpus(corpus_path)

    def load_corpus(self, corpus_path: str):
        """Load passage corpus from file."""
        path = Path(corpus_path)

        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        passage = json.loads(line)
                        self._index_passage(passage)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle both list and dict with "passages" key
                passages = data if isinstance(data, list) else data.get("passages", [])
                for passage in passages:
                    self._index_passage(passage)

        print(f"Loaded {len(self.passages)} passages")
        print(f"Indexed {len(self.by_entity)} entities")

    def _index_passage(self, passage: dict):
        """Index a passage by entity."""
        self.passages.append(passage)

        passage_id = passage.get("passage_id", f"p_{len(self.passages)}")
        entity_qid = passage.get("entity_qid")

        self.by_id[passage_id] = passage

        if entity_qid:
            self.by_entity[entity_qid].append(passage_id)

    def map_query(
        self,
        query_id: str,
        cluster_id: str,
        entity_qids: list,
        max_passages_per_entity: int = 3
    ) -> QueryPassages:
        """
        Map a query to its relevant passages.

        Args:
            query_id: Query identifier
            cluster_id: Cluster identifier
            entity_qids: List of entity QIDs to map
            max_passages_per_entity: Max passages to return per entity

        Returns:
            QueryPassages with all matches
        """
        matches = []
        missing = []

        for qid in entity_qids:
            passage_ids = self.by_entity.get(qid, [])

            if not passage_ids:
                missing.append(qid)
                continue

            # Get passages (limited)
            for pid in passage_ids[:max_passages_per_entity]:
                passage = self.by_id.get(pid, {})

                # Compute simple relevance score based on position
                # First passage (lead section) gets highest score
                idx = passage_ids.index(pid)
                relevance = 1.0 - (idx * 0.2)  # 1.0, 0.8, 0.6, ...

                matches.append(PassageMatch(
                    passage_id=pid,
                    entity_qid=qid,
                    entity_label=passage.get("entity_label", ""),
                    passage_text=passage.get("passage_text", "")[:200] + "...",
                    section=passage.get("section", "lead"),
                    relevance_score=max(0.1, relevance),
                    match_reason="entity_match"
                ))

        coverage = (len(entity_qids) - len(missing)) / len(entity_qids) if entity_qids else 0

        return QueryPassages(
            query_id=query_id,
            cluster_id=cluster_id,
            entity_count=len(entity_qids),
            passage_count=len(matches),
            matches=matches,
            missing_entities=missing,
            coverage_rate=coverage
        )

    def map_batch(
        self,
        ground_truths: list,
        max_passages_per_entity: int = 3
    ) -> list:
        """
        Map a batch of ground truths to passages.

        Args:
            ground_truths: List of GroundTruth objects or dicts
            max_passages_per_entity: Max passages per entity

        Returns:
            List of QueryPassages objects
        """
        results = []

        for gt in ground_truths:
            # Handle both dict and dataclass
            if isinstance(gt, dict):
                query_id = gt.get("query_id", "")
                cluster_id = gt.get("cluster_id", "")
                entities = gt.get("verified_entities", [])
            else:
                query_id = gt.query_id
                cluster_id = gt.cluster_id
                entities = gt.verified_entities

            qp = self.map_query(
                query_id=query_id,
                cluster_id=cluster_id,
                entity_qids=entities,
                max_passages_per_entity=max_passages_per_entity
            )
            results.append(qp)

        return results

    def get_corpus_stats(self) -> dict:
        """Get statistics about the loaded corpus."""
        if not self.passages:
            return {"loaded": False}

        passage_counts = [len(pids) for pids in self.by_entity.values()]

        return {
            "loaded": True,
            "total_passages": len(self.passages),
            "total_entities": len(self.by_entity),
            "avg_passages_per_entity": sum(passage_counts) / len(passage_counts) if passage_counts else 0,
            "max_passages_per_entity": max(passage_counts) if passage_counts else 0,
            "min_passages_per_entity": min(passage_counts) if passage_counts else 0,
        }


def query_passages_to_dict(qp: QueryPassages) -> dict:
    """Convert QueryPassages to dict."""
    return {
        "query_id": qp.query_id,
        "cluster_id": qp.cluster_id,
        "entity_count": qp.entity_count,
        "passage_count": qp.passage_count,
        "coverage_rate": qp.coverage_rate,
        "missing_entities": qp.missing_entities,
        "matches": [
            {
                "passage_id": m.passage_id,
                "entity_qid": m.entity_qid,
                "entity_label": m.entity_label,
                "section": m.section,
                "relevance_score": m.relevance_score,
                "match_reason": m.match_reason
            }
            for m in qp.matches
        ]
    }


def save_passage_mappings(mappings: list, output_path: str):
    """Save passage mappings to JSON."""
    data = {
        "count": len(mappings),
        "mappings": [query_passages_to_dict(qp) for qp in mappings]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(mappings)} passage mappings to {output_path}")


if __name__ == "__main__":
    # Demo with synthetic passages
    print("=" * 60)
    print("PASSAGE MAPPER DEMO")
    print("=" * 60)

    # Create sample corpus
    sample_corpus = [
        {
            "passage_id": "Q134773_p0",
            "entity_qid": "Q134773",
            "entity_label": "Leonardo DiCaprio",
            "passage_text": "Leonardo Wilhelm DiCaprio is an American actor and film producer...",
            "section": "lead"
        },
        {
            "passage_id": "Q134773_p1",
            "entity_qid": "Q134773",
            "entity_label": "Leonardo DiCaprio",
            "passage_text": "DiCaprio began his career by appearing in television commercials...",
            "section": "Career"
        },
        {
            "passage_id": "Q188018_p0",
            "entity_qid": "Q188018",
            "entity_label": "Tom Hardy",
            "passage_text": "Edward Thomas Hardy is an English actor and producer...",
            "section": "lead"
        },
        {
            "passage_id": "Q232163_p0",
            "entity_qid": "Q232163",
            "entity_label": "Joseph Gordon-Levitt",
            "passage_text": "Joseph Leonard Gordon-Levitt is an American actor and filmmaker...",
            "section": "lead"
        },
    ]

    # Initialize mapper with sample data
    mapper = PassageMapper()
    for p in sample_corpus:
        mapper._index_passage(p)

    print(f"\nCorpus Stats: {mapper.get_corpus_stats()}")

    # Test mapping
    entities = ["Q134773", "Q188018", "Q232163", "Q999999"]  # Last one missing
    qp = mapper.map_query(
        query_id="test_001",
        cluster_id="creative_inception",
        entity_qids=entities
    )

    print(f"\nQuery Mapping:")
    print(f"  Query ID: {qp.query_id}")
    print(f"  Entity count: {qp.entity_count}")
    print(f"  Passage count: {qp.passage_count}")
    print(f"  Coverage rate: {qp.coverage_rate:.1%}")
    print(f"  Missing entities: {qp.missing_entities}")
    print(f"\n  Matches:")
    for m in qp.matches:
        print(f"    - {m.passage_id}: {m.entity_label} ({m.section}) [score: {m.relevance_score:.2f}]")
