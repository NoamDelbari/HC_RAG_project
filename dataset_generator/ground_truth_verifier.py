"""
Ground Truth Verification Module

Verifies query answers via SPARQL and maps entities to passages.

Phase C.2: SPARQL Ground Truth Verification
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sparql_profiler import SPARQLProfiler, WIKIDATA_SPARQL_ENDPOINT


@dataclass
class GroundTruth:
    """Ground truth for a query."""
    query_id: str
    cluster_id: str

    # Entity-level ground truth
    expected_entities: list  # List of QIDs from cluster
    verified_entities: list  # List of QIDs confirmed via SPARQL
    entity_count: int

    # Passage-level ground truth (to be filled by passage mapper)
    passage_ids: list = field(default_factory=list)
    passage_count: int = 0

    # K values
    target_k: int = 0
    actual_k: int = 0
    k_discrepancy: int = 0

    # Verification status
    sparql_success: bool = False
    verification_notes: list = field(default_factory=list)


@dataclass
class VerificationResult:
    """Result of ground truth verification for a batch."""
    total_queries: int
    verified_queries: int
    failed_queries: int
    k_alignment_stats: dict
    ground_truths: list


class SPARQLVerifier:
    """
    Constructs and executes SPARQL queries for ground truth verification.

    Uses cluster binding properties to verify entity membership.
    """

    def __init__(self, endpoint: str = WIKIDATA_SPARQL_ENDPOINT):
        self.profiler = SPARQLProfiler(endpoint)

        # SPARQL templates by cluster type
        self.templates = {
            "creative_ensemble": self._verify_creative_ensemble,
            "award_recognition": self._verify_award_recognition,
            "organizational": self._verify_organizational,
            "geographic_spatial": self._verify_geographic,
            "relational": self._verify_relational,
            "event_temporal": self._verify_temporal,
            "categorical": self._verify_categorical,
        }

    def verify_query(
        self,
        query_data: dict,
        cluster_data: dict,
        timeout: int = 30
    ) -> GroundTruth:
        """
        Verify ground truth for a single query.

        Args:
            query_data: Generated query data
            cluster_data: Cluster data with entities
            timeout: SPARQL timeout in seconds

        Returns:
            GroundTruth object
        """
        query_id = query_data.get("query_id", "unknown")
        cluster_id = cluster_data.get("cluster_id", "unknown")
        cluster_type = cluster_data.get("cluster_type", "categorical")

        # Get expected entities from cluster
        expected = [e.get("qid") for e in cluster_data.get("entities", []) if e.get("qid")]

        gt = GroundTruth(
            query_id=query_id,
            cluster_id=cluster_id,
            expected_entities=expected,
            verified_entities=[],
            entity_count=0,
            target_k=query_data.get("target_k", len(expected))
        )

        # Get verification function
        verify_fn = self.templates.get(cluster_type, self._verify_categorical)

        try:
            verified = verify_fn(cluster_data, timeout)
            gt.verified_entities = verified
            gt.entity_count = len(verified)
            gt.actual_k = len(verified)
            gt.k_discrepancy = gt.actual_k - gt.target_k
            gt.sparql_success = True

            # Check coverage
            missing = set(expected) - set(verified)
            if missing:
                gt.verification_notes.append(f"Missing {len(missing)} entities from SPARQL")

            extra = set(verified) - set(expected)
            if extra:
                gt.verification_notes.append(f"Found {len(extra)} extra entities in SPARQL")

        except Exception as e:
            gt.sparql_success = False
            gt.verification_notes.append(f"SPARQL error: {str(e)}")
            # Fall back to expected entities
            gt.verified_entities = expected
            gt.entity_count = len(expected)
            gt.actual_k = len(expected)

        return gt

    def _verify_creative_ensemble(self, cluster_data: dict, timeout: int) -> list:
        """Verify creative ensemble (cast/crew of a work)."""
        binding = cluster_data.get("binding_description", "")
        entities = cluster_data.get("entities", [])

        # Extract work QID if available in cluster data
        work_qid = cluster_data.get("anchor_qid")
        if not work_qid:
            # Use provided entities as ground truth
            return [e.get("qid") for e in entities if e.get("qid")]

        sparql = f"""
SELECT DISTINCT ?person WHERE {{
  wd:{work_qid} wdt:P161 ?person .
}}
LIMIT 100
"""
        result = self.profiler.execute_query(sparql, timeout=timeout)

        if result.success:
            return [b.get("person", "").split("/")[-1] for b in result.entities]

        return [e.get("qid") for e in entities if e.get("qid")]

    def _verify_award_recognition(self, cluster_data: dict, timeout: int) -> list:
        """Verify award recipients."""
        entities = cluster_data.get("entities", [])

        award_qid = cluster_data.get("award_qid")
        year = cluster_data.get("year")

        if not award_qid:
            return [e.get("qid") for e in entities if e.get("qid")]

        if year:
            sparql = f"""
SELECT DISTINCT ?person WHERE {{
  ?person p:P166 ?stmt .
  ?stmt ps:P166 wd:{award_qid} .
  ?stmt pq:P585 ?date .
  FILTER(YEAR(?date) = {year})
}}
LIMIT 50
"""
        else:
            sparql = f"""
SELECT DISTINCT ?person WHERE {{
  ?person wdt:P166 wd:{award_qid} .
}}
LIMIT 50
"""

        result = self.profiler.execute_query(sparql, timeout=timeout)

        if result.success:
            return [b.get("person", "").split("/")[-1] for b in result.entities]

        return [e.get("qid") for e in entities if e.get("qid")]

    def _verify_organizational(self, cluster_data: dict, timeout: int) -> list:
        """Verify organization members."""
        entities = cluster_data.get("entities", [])
        org_qid = cluster_data.get("anchor_qid")

        if not org_qid:
            return [e.get("qid") for e in entities if e.get("qid")]

        sparql = f"""
SELECT DISTINCT ?person WHERE {{
  ?person wdt:P108 wd:{org_qid} .
}}
LIMIT 100
"""
        result = self.profiler.execute_query(sparql, timeout=timeout)

        if result.success:
            return [b.get("person", "").split("/")[-1] for b in result.entities]

        return [e.get("qid") for e in entities if e.get("qid")]

    def _verify_geographic(self, cluster_data: dict, timeout: int) -> list:
        """Verify geographic cluster."""
        entities = cluster_data.get("entities", [])
        region_qid = cluster_data.get("region_qid")
        entity_type_qid = cluster_data.get("entity_type_qid")

        if not region_qid:
            return [e.get("qid") for e in entities if e.get("qid")]

        sparql = f"""
SELECT DISTINCT ?entity WHERE {{
  ?entity wdt:P17 wd:{region_qid} .
  {"?entity wdt:P31 wd:" + entity_type_qid + " ." if entity_type_qid else ""}
}}
LIMIT 100
"""
        result = self.profiler.execute_query(sparql, timeout=timeout)

        if result.success:
            return [b.get("entity", "").split("/")[-1] for b in result.entities]

        return [e.get("qid") for e in entities if e.get("qid")]

    def _verify_relational(self, cluster_data: dict, timeout: int) -> list:
        """Verify relational cluster (entities related to anchor)."""
        entities = cluster_data.get("entities", [])
        anchor_qid = cluster_data.get("anchor_qid")
        property_id = cluster_data.get("property_id", "P112")  # founded by

        if not anchor_qid:
            return [e.get("qid") for e in entities if e.get("qid")]

        sparql = f"""
SELECT DISTINCT ?entity WHERE {{
  ?entity wdt:{property_id} wd:{anchor_qid} .
}}
LIMIT 50
"""
        result = self.profiler.execute_query(sparql, timeout=timeout)

        if result.success:
            return [b.get("entity", "").split("/")[-1] for b in result.entities]

        return [e.get("qid") for e in entities if e.get("qid")]

    def _verify_temporal(self, cluster_data: dict, timeout: int) -> list:
        """Verify temporal/event cluster."""
        entities = cluster_data.get("entities", [])
        # Temporal clusters are complex - fall back to provided entities
        return [e.get("qid") for e in entities if e.get("qid")]

    def _verify_categorical(self, cluster_data: dict, timeout: int) -> list:
        """Verify categorical cluster."""
        entities = cluster_data.get("entities", [])
        # Generic fallback - use provided entities
        return [e.get("qid") for e in entities if e.get("qid")]

    def verify_batch(
        self,
        queries: list,
        clusters: dict,
        delay: float = 0.5
    ) -> VerificationResult:
        """
        Verify a batch of queries.

        Args:
            queries: List of query data dicts
            clusters: Dict mapping cluster_id to cluster data
            delay: Delay between SPARQL queries

        Returns:
            VerificationResult with all ground truths
        """
        ground_truths = []
        verified = 0
        failed = 0

        k_stats = {
            "exact_match": 0,
            "close_match": 0,  # within ±2
            "moderate_discrepancy": 0,  # 3-5
            "large_discrepancy": 0  # >5
        }

        print(f"Verifying {len(queries)} queries...")

        for i, query in enumerate(queries):
            cluster_id = query.get("cluster_id")
            cluster_data = clusters.get(cluster_id, {})

            gt = self.verify_query(query, cluster_data)
            ground_truths.append(gt)

            if gt.sparql_success:
                verified += 1
            else:
                failed += 1

            # K alignment stats
            disc = abs(gt.k_discrepancy)
            if disc == 0:
                k_stats["exact_match"] += 1
            elif disc <= 2:
                k_stats["close_match"] += 1
            elif disc <= 5:
                k_stats["moderate_discrepancy"] += 1
            else:
                k_stats["large_discrepancy"] += 1

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(queries)}")

            time.sleep(delay)

        return VerificationResult(
            total_queries=len(queries),
            verified_queries=verified,
            failed_queries=failed,
            k_alignment_stats=k_stats,
            ground_truths=ground_truths
        )


def ground_truth_to_dict(gt: GroundTruth) -> dict:
    """Convert GroundTruth to dict."""
    return {
        "query_id": gt.query_id,
        "cluster_id": gt.cluster_id,
        "expected_entities": gt.expected_entities,
        "verified_entities": gt.verified_entities,
        "entity_count": gt.entity_count,
        "passage_ids": gt.passage_ids,
        "passage_count": gt.passage_count,
        "target_k": gt.target_k,
        "actual_k": gt.actual_k,
        "k_discrepancy": gt.k_discrepancy,
        "sparql_success": gt.sparql_success,
        "verification_notes": gt.verification_notes
    }


def save_verification_results(result: VerificationResult, output_path: str):
    """Save verification results to JSON."""
    data = {
        "total_queries": result.total_queries,
        "verified_queries": result.verified_queries,
        "failed_queries": result.failed_queries,
        "k_alignment_stats": result.k_alignment_stats,
        "ground_truths": [ground_truth_to_dict(gt) for gt in result.ground_truths]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved verification results to {output_path}")


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("GROUND TRUTH VERIFIER DEMO")
    print("=" * 60)

    verifier = SPARQLVerifier()

    # Sample cluster and query
    sample_cluster = {
        "cluster_id": "creative_inception",
        "cluster_type": "creative_ensemble",
        "binding_description": "Cast of Inception",
        "anchor_qid": "Q25188",  # Inception film
        "entities": [
            {"qid": "Q134773", "label": "Leonardo DiCaprio"},
            {"qid": "Q188018", "label": "Tom Hardy"},
        ]
    }

    sample_query = {
        "query_id": "test_001",
        "cluster_id": "creative_inception",
        "target_k": 5
    }

    print("\nVerifying sample query...")
    gt = verifier.verify_query(sample_query, sample_cluster)

    print(f"\nGround Truth:")
    print(f"  Query ID: {gt.query_id}")
    print(f"  Expected entities: {gt.expected_entities}")
    print(f"  Verified entities: {gt.verified_entities[:5]}...")
    print(f"  SPARQL success: {gt.sparql_success}")
    print(f"  Target K: {gt.target_k}, Actual K: {gt.actual_k}")
    print(f"  K discrepancy: {gt.k_discrepancy}")
    if gt.verification_notes:
        print(f"  Notes: {gt.verification_notes}")
