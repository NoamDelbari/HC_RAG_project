"""
K-Targeting Index Builder for Cross-Entity Clusters

This module builds an inverted index of clusters by K-range, query type,
difficulty level, and domain to enable targeted query generation.

Phase A.5: K-Targeting Index Construction
- A.5.1: Calculate effective K ranges for each cluster
- A.5.2: Score cluster suitability for query types and difficulties
- A.5.3: Build inverted index by K-range, query type, difficulty, domain
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ClusterKProfile:
    """K-value profile for a cluster."""
    cluster_id: str
    entity_count: int
    min_k: int
    max_k: int
    optimal_k_values: list
    avg_passages_per_entity: float
    effective_k_ceiling: int  # Limited by passage availability


@dataclass
class ClusterSuitability:
    """Suitability scores for different query types and difficulties."""
    cluster_id: str
    query_type_scores: dict  # {query_type: score}
    difficulty_scores: dict  # {difficulty: score}
    domain_tags: list
    overall_suitability: float


@dataclass
class IndexEntry:
    """Entry in the K-targeting index."""
    cluster_id: str
    cluster_name: str
    cluster_type: str
    k_range: tuple
    query_types: list
    difficulties: list
    domains: list
    entity_count: int
    suitability_score: float
    metadata: dict = field(default_factory=dict)


class KRangeCalculator:
    """
    Calculates effective K ranges for clusters based on entity count
    and passage availability.
    """

    # K-value buckets
    K_BUCKETS = [
        (3, 5, "tiny"),
        (6, 10, "small"),
        (11, 20, "medium"),
        (21, 50, "large"),
        (51, 100, "xlarge")
    ]

    def calculate_k_profile(
        self,
        cluster_data: dict,
        coverage_data: Optional[dict] = None
    ) -> ClusterKProfile:
        """
        Calculate K profile for a cluster.

        Args:
            cluster_data: Cluster dictionary with entities
            coverage_data: Optional Wikipedia coverage data

        Returns:
            ClusterKProfile with K range information
        """
        entity_count = cluster_data.get("entity_count", len(cluster_data.get("entities", [])))

        # Determine K range based on entity count
        min_k = min(3, entity_count)
        max_k = entity_count

        # Calculate optimal K values
        optimal_k = []
        for bucket_min, bucket_max, _ in self.K_BUCKETS:
            if bucket_min <= entity_count:
                k = min(bucket_max, entity_count)
                if k >= bucket_min:
                    optimal_k.append(k)

        # Adjust for passage availability if coverage data provided
        avg_passages = 0
        effective_ceiling = entity_count

        if coverage_data:
            avg_passages = coverage_data.get("avg_passages_per_entity", 0)
            coverage_rate = coverage_data.get("coverage_rate", 1.0)

            # Effective ceiling is limited by entities with Wikipedia
            effective_ceiling = int(entity_count * coverage_rate)

            # Further limit if passage count is very low
            if avg_passages < 3:
                effective_ceiling = min(effective_ceiling, int(entity_count * 0.5))

        return ClusterKProfile(
            cluster_id=cluster_data.get("cluster_id", "unknown"),
            entity_count=entity_count,
            min_k=min_k,
            max_k=max_k,
            optimal_k_values=optimal_k,
            avg_passages_per_entity=avg_passages,
            effective_k_ceiling=effective_ceiling
        )

    def get_k_bucket(self, k: int) -> str:
        """Get the bucket name for a K value."""
        for bucket_min, bucket_max, name in self.K_BUCKETS:
            if bucket_min <= k <= bucket_max:
                return name
        if k > 100:
            return "xlarge"
        return "tiny"


class SuitabilityScorer:
    """
    Scores cluster suitability for different query types and difficulty levels.
    """

    # Query type definitions
    QUERY_TYPES = {
        "enumeration": {
            "description": "List all entities matching criteria",
            "ideal_k_range": (5, 30),
            "cluster_type_affinity": {
                "award_recognition": 1.0,
                "creative_ensemble": 0.9,
                "organizational": 0.8,
                "geographic_spatial": 0.9,
                "event_temporal": 0.8,
                "relational": 0.7,
                "categorical": 0.6
            }
        },
        "attribute_collection": {
            "description": "Gather specific attributes across entities",
            "ideal_k_range": (3, 20),
            "cluster_type_affinity": {
                "award_recognition": 0.8,
                "creative_ensemble": 0.9,
                "organizational": 0.7,
                "geographic_spatial": 0.9,
                "event_temporal": 0.7,
                "relational": 0.8,
                "categorical": 0.6
            }
        },
        "comparison": {
            "description": "Compare attributes across entities",
            "ideal_k_range": (2, 10),
            "cluster_type_affinity": {
                "award_recognition": 0.7,
                "creative_ensemble": 0.8,
                "organizational": 0.6,
                "geographic_spatial": 0.9,
                "event_temporal": 0.6,
                "relational": 0.7,
                "categorical": 0.8
            }
        },
        "aggregation": {
            "description": "Calculate aggregate statistics",
            "ideal_k_range": (5, 50),
            "cluster_type_affinity": {
                "award_recognition": 0.6,
                "creative_ensemble": 0.5,
                "organizational": 0.7,
                "geographic_spatial": 0.9,
                "event_temporal": 0.7,
                "relational": 0.5,
                "categorical": 0.8
            }
        },
        "subset_identification": {
            "description": "Find subset matching additional criteria",
            "ideal_k_range": (10, 50),
            "cluster_type_affinity": {
                "award_recognition": 0.8,
                "creative_ensemble": 0.6,
                "organizational": 0.8,
                "geographic_spatial": 0.7,
                "event_temporal": 0.7,
                "relational": 0.6,
                "categorical": 0.9
            }
        }
    }

    # Difficulty level definitions
    DIFFICULTY_LEVELS = {
        "easy": {
            "description": "High lexical overlap, direct entity mentions",
            "k_range_modifier": 0.8,  # Prefer smaller K
            "passage_quality_threshold": 0.5
        },
        "medium": {
            "description": "Paraphrased properties, indirect references",
            "k_range_modifier": 1.0,
            "passage_quality_threshold": 0.7
        },
        "hard": {
            "description": "Multi-step reasoning, minimal lexical overlap",
            "k_range_modifier": 1.2,  # May need more passages
            "passage_quality_threshold": 0.9
        }
    }

    # Domain tags based on cluster characteristics
    DOMAIN_TAGS = {
        "science": ["physicist", "chemist", "mathematician", "scientist", "research"],
        "entertainment": ["film", "actor", "director", "music", "television"],
        "politics": ["president", "minister", "politician", "government"],
        "sports": ["athlete", "olympic", "championship", "team"],
        "business": ["company", "founder", "ceo", "business"],
        "geography": ["country", "city", "capital", "continent"],
        "history": ["historical", "ancient", "war", "century"],
        "arts": ["artist", "writer", "author", "literary"]
    }

    def score_cluster(
        self,
        cluster_data: dict,
        k_profile: ClusterKProfile,
        coverage_data: Optional[dict] = None
    ) -> ClusterSuitability:
        """
        Score cluster suitability for query generation.

        Returns:
            ClusterSuitability with scores for each query type and difficulty
        """
        cluster_type = cluster_data.get("cluster_type", "categorical")
        entity_count = k_profile.entity_count

        # Score query types
        query_scores = {}
        for query_type, config in self.QUERY_TYPES.items():
            # Base score from cluster type affinity
            base_score = config["cluster_type_affinity"].get(cluster_type, 0.5)

            # Adjust for K range fit
            ideal_min, ideal_max = config["ideal_k_range"]
            if ideal_min <= entity_count <= ideal_max:
                k_fit = 1.0
            elif entity_count < ideal_min:
                k_fit = entity_count / ideal_min
            else:
                k_fit = ideal_max / entity_count
                k_fit = max(k_fit, 0.5)  # Don't penalize too heavily

            query_scores[query_type] = round(base_score * k_fit, 2)

        # Score difficulty levels
        difficulty_scores = {}
        for difficulty, config in self.DIFFICULTY_LEVELS.items():
            # Base score
            base_score = 0.8

            # Adjust for passage availability
            if coverage_data:
                avg_passages = coverage_data.get("avg_passages_per_entity", 5)
                if avg_passages >= 8:
                    passage_fit = 1.0
                elif avg_passages >= 5:
                    passage_fit = 0.9
                elif avg_passages >= 3:
                    passage_fit = 0.7
                else:
                    passage_fit = 0.5

                base_score *= passage_fit

            # Difficulty-specific adjustments
            if difficulty == "hard" and entity_count < 5:
                base_score *= 0.7  # Hard queries need more context
            elif difficulty == "easy" and entity_count > 30:
                base_score *= 0.9  # Easy queries on large sets may be too simple

            difficulty_scores[difficulty] = round(base_score, 2)

        # Extract domain tags
        domains = self._extract_domains(cluster_data)

        # Calculate overall suitability
        overall = (
            sum(query_scores.values()) / len(query_scores) * 0.4 +
            sum(difficulty_scores.values()) / len(difficulty_scores) * 0.4 +
            (k_profile.effective_k_ceiling / k_profile.entity_count if k_profile.entity_count > 0 else 0) * 0.2
        )

        return ClusterSuitability(
            cluster_id=cluster_data.get("cluster_id", "unknown"),
            query_type_scores=query_scores,
            difficulty_scores=difficulty_scores,
            domain_tags=domains,
            overall_suitability=round(overall, 2)
        )

    def _extract_domains(self, cluster_data: dict) -> list:
        """Extract domain tags from cluster metadata."""
        domains = []
        cluster_name = cluster_data.get("cluster_name", "").lower()
        description = cluster_data.get("description", "").lower()
        cluster_type = cluster_data.get("cluster_type", "")

        text = f"{cluster_name} {description} {cluster_type}"

        for domain, keywords in self.DOMAIN_TAGS.items():
            if any(kw in text for kw in keywords):
                domains.append(domain)

        # Default domain based on cluster type
        if not domains:
            type_domain_map = {
                "award_recognition": "awards",
                "creative_ensemble": "entertainment",
                "organizational": "organizations",
                "geographic_spatial": "geography",
                "event_temporal": "events",
                "relational": "relationships",
                "categorical": "general"
            }
            domains.append(type_domain_map.get(cluster_type, "general"))

        return domains


class KTargetingIndex:
    """
    Inverted index of clusters by K-range, query type, difficulty, and domain.
    """

    def __init__(self):
        self.entries = []
        self.by_k_bucket = defaultdict(list)
        self.by_query_type = defaultdict(list)
        self.by_difficulty = defaultdict(list)
        self.by_domain = defaultdict(list)
        self.by_cluster_type = defaultdict(list)

    def add_cluster(
        self,
        cluster_data: dict,
        k_profile: ClusterKProfile,
        suitability: ClusterSuitability,
        coverage_data: Optional[dict] = None
    ):
        """Add a cluster to the index."""
        k_calc = KRangeCalculator()

        entry = IndexEntry(
            cluster_id=cluster_data.get("cluster_id", "unknown"),
            cluster_name=cluster_data.get("cluster_name", ""),
            cluster_type=cluster_data.get("cluster_type", ""),
            k_range=(k_profile.min_k, k_profile.effective_k_ceiling),
            query_types=[qt for qt, score in suitability.query_type_scores.items() if score >= 0.6],
            difficulties=[d for d, score in suitability.difficulty_scores.items() if score >= 0.6],
            domains=suitability.domain_tags,
            entity_count=k_profile.entity_count,
            suitability_score=suitability.overall_suitability,
            metadata={
                "optimal_k_values": k_profile.optimal_k_values,
                "query_type_scores": suitability.query_type_scores,
                "difficulty_scores": suitability.difficulty_scores,
                "coverage_rate": coverage_data.get("coverage_rate") if coverage_data else None
            }
        )

        self.entries.append(entry)

        # Index by K bucket
        for k in k_profile.optimal_k_values:
            bucket = k_calc.get_k_bucket(k)
            self.by_k_bucket[bucket].append(entry)

        # Index by query type
        for qt in entry.query_types:
            self.by_query_type[qt].append(entry)

        # Index by difficulty
        for diff in entry.difficulties:
            self.by_difficulty[diff].append(entry)

        # Index by domain
        for domain in entry.domains:
            self.by_domain[domain].append(entry)

        # Index by cluster type
        self.by_cluster_type[entry.cluster_type].append(entry)

    def query(
        self,
        k_bucket: Optional[str] = None,
        query_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        domain: Optional[str] = None,
        cluster_type: Optional[str] = None,
        min_suitability: float = 0.5
    ) -> list:
        """
        Query the index with optional filters.

        Returns:
            List of matching IndexEntry objects
        """
        candidates = set(range(len(self.entries)))

        if k_bucket:
            bucket_ids = {self.entries.index(e) for e in self.by_k_bucket.get(k_bucket, [])}
            candidates &= bucket_ids

        if query_type:
            qt_ids = {self.entries.index(e) for e in self.by_query_type.get(query_type, [])}
            candidates &= qt_ids

        if difficulty:
            diff_ids = {self.entries.index(e) for e in self.by_difficulty.get(difficulty, [])}
            candidates &= diff_ids

        if domain:
            domain_ids = {self.entries.index(e) for e in self.by_domain.get(domain, [])}
            candidates &= domain_ids

        if cluster_type:
            type_ids = {self.entries.index(e) for e in self.by_cluster_type.get(cluster_type, [])}
            candidates &= type_ids

        # Filter by suitability and return
        results = [
            self.entries[i] for i in candidates
            if self.entries[i].suitability_score >= min_suitability
        ]

        # Sort by suitability
        return sorted(results, key=lambda e: e.suitability_score, reverse=True)

    def get_statistics(self) -> dict:
        """Get index statistics."""
        return {
            "total_clusters": len(self.entries),
            "by_k_bucket": {k: len(v) for k, v in self.by_k_bucket.items()},
            "by_query_type": {k: len(v) for k, v in self.by_query_type.items()},
            "by_difficulty": {k: len(v) for k, v in self.by_difficulty.items()},
            "by_domain": {k: len(v) for k, v in self.by_domain.items()},
            "by_cluster_type": {k: len(v) for k, v in self.by_cluster_type.items()},
            "avg_suitability": sum(e.suitability_score for e in self.entries) / len(self.entries) if self.entries else 0
        }

    def to_dict(self) -> dict:
        """Serialize index to dictionary."""
        return {
            "entries": [
                {
                    "cluster_id": e.cluster_id,
                    "cluster_name": e.cluster_name,
                    "cluster_type": e.cluster_type,
                    "k_range": e.k_range,
                    "query_types": e.query_types,
                    "difficulties": e.difficulties,
                    "domains": e.domains,
                    "entity_count": e.entity_count,
                    "suitability_score": e.suitability_score,
                    "metadata": e.metadata
                }
                for e in self.entries
            ],
            "statistics": self.get_statistics()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KTargetingIndex":
        """Deserialize index from dictionary."""
        index = cls()

        for entry_data in data.get("entries", []):
            entry = IndexEntry(
                cluster_id=entry_data["cluster_id"],
                cluster_name=entry_data["cluster_name"],
                cluster_type=entry_data["cluster_type"],
                k_range=tuple(entry_data["k_range"]),
                query_types=entry_data["query_types"],
                difficulties=entry_data["difficulties"],
                domains=entry_data["domains"],
                entity_count=entry_data["entity_count"],
                suitability_score=entry_data["suitability_score"],
                metadata=entry_data.get("metadata", {})
            )
            index.entries.append(entry)

            # Rebuild indices
            k_calc = KRangeCalculator()
            for k in entry.metadata.get("optimal_k_values", []):
                bucket = k_calc.get_k_bucket(k)
                index.by_k_bucket[bucket].append(entry)

            for qt in entry.query_types:
                index.by_query_type[qt].append(entry)
            for diff in entry.difficulties:
                index.by_difficulty[diff].append(entry)
            for domain in entry.domains:
                index.by_domain[domain].append(entry)
            index.by_cluster_type[entry.cluster_type].append(entry)

        return index


def build_index_from_clusters(cluster_dir: str, coverage_dir: Optional[str] = None) -> KTargetingIndex:
    """
    Build K-targeting index from cluster files.

    Args:
        cluster_dir: Directory containing cluster JSON files
        coverage_dir: Optional directory containing coverage JSON files

    Returns:
        KTargetingIndex
    """
    index = KTargetingIndex()
    k_calc = KRangeCalculator()
    scorer = SuitabilityScorer()

    cluster_path = Path(cluster_dir)

    for filepath in cluster_path.glob("*.json"):
        if "_coverage" in filepath.name or "_passages" in filepath.name:
            continue

        print(f"Indexing: {filepath.name}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                cluster_data = json.load(f)

            # Load coverage data if available
            coverage_data = None
            if coverage_dir:
                coverage_file = Path(coverage_dir) / f"{cluster_data.get('cluster_id')}_coverage.json"
                if coverage_file.exists():
                    with open(coverage_file, "r", encoding="utf-8") as f:
                        coverage_data = json.load(f)

            # Calculate K profile and suitability
            k_profile = k_calc.calculate_k_profile(cluster_data, coverage_data)
            suitability = scorer.score_cluster(cluster_data, k_profile, coverage_data)

            # Add to index
            index.add_cluster(cluster_data, k_profile, suitability, coverage_data)

        except Exception as e:
            print(f"  Error: {e}")

    return index


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build K-targeting index")
    parser.add_argument("--cluster-dir", type=str, required=True, help="Directory with cluster files")
    parser.add_argument("--coverage-dir", type=str, help="Directory with coverage files")
    parser.add_argument("--output", type=str, default="data/k_index.json", help="Output index file")
    parser.add_argument("--query-k", type=str, help="Query by K bucket (tiny/small/medium/large/xlarge)")
    parser.add_argument("--query-type", type=str, help="Query by query type")
    parser.add_argument("--query-difficulty", type=str, help="Query by difficulty")
    parser.add_argument("--query-domain", type=str, help="Query by domain")

    args = parser.parse_args()

    # Build index
    print("Building K-targeting index...")
    index = build_index_from_clusters(args.cluster_dir, args.coverage_dir)

    # Print statistics
    stats = index.get_statistics()
    print("\n" + "=" * 60)
    print("INDEX STATISTICS")
    print("=" * 60)
    print(f"Total clusters: {stats['total_clusters']}")
    print(f"Average suitability: {stats['avg_suitability']:.2f}")

    print("\nBy K bucket:")
    for bucket, count in sorted(stats["by_k_bucket"].items()):
        print(f"  {bucket}: {count}")

    print("\nBy query type:")
    for qt, count in sorted(stats["by_query_type"].items()):
        print(f"  {qt}: {count}")

    print("\nBy difficulty:")
    for diff, count in sorted(stats["by_difficulty"].items()):
        print(f"  {diff}: {count}")

    print("\nBy domain:")
    for domain, count in sorted(stats["by_domain"].items()):
        print(f"  {domain}: {count}")

    # Execute query if specified
    if any([args.query_k, args.query_type, args.query_difficulty, args.query_domain]):
        print("\n" + "=" * 60)
        print("QUERY RESULTS")
        print("=" * 60)

        results = index.query(
            k_bucket=args.query_k,
            query_type=args.query_type,
            difficulty=args.query_difficulty,
            domain=args.query_domain
        )

        print(f"Found {len(results)} matching clusters:")
        for entry in results[:10]:
            print(f"  {entry.cluster_name} ({entry.cluster_id})")
            print(f"    K range: {entry.k_range}, Suitability: {entry.suitability_score}")
            print(f"    Types: {entry.query_types}, Domains: {entry.domains}")

    # Save index
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\nIndex saved to {args.output}")
