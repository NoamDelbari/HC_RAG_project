"""
Entity Type Validator for Cross-Entity Clusters

This module validates entity type consistency within clusters,
ensuring all entities share expected types and identifying outliers.

Phase A.3.3: Validate entity type consistency within clusters
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from sparql_profiler import SPARQLProfiler, WIKIDATA_SPARQL_ENDPOINT


@dataclass
class TypeValidationResult:
    """Results of type validation for a cluster."""
    cluster_id: str
    total_entities: int
    validated_entities: int
    type_distribution: dict
    dominant_type: str
    dominant_type_qid: str
    consistency_score: float
    outliers: list
    warnings: list
    is_valid: bool


@dataclass
class EntityTypeInfo:
    """Type information for an entity."""
    qid: str
    label: str
    types: list  # List of (type_qid, type_label) tuples
    primary_type: Optional[str] = None
    primary_type_qid: Optional[str] = None


class EntityTypeValidator:
    """
    Validates entity type consistency within clusters.
    """

    def __init__(self, endpoint: str = WIKIDATA_SPARQL_ENDPOINT):
        self.profiler = SPARQLProfiler(endpoint)

        # Expected types for different cluster types
        self.expected_types = {
            "award_recognition": {
                "primary": ["Q5"],  # human
                "acceptable": ["Q5", "Q43229", "Q4830453"],  # human, organization, business
                "description": "Award recipients should be humans or organizations"
            },
            "creative_ensemble": {
                "primary": ["Q5"],  # human
                "acceptable": ["Q5"],
                "description": "Cast/crew should be humans"
            },
            "relational": {
                "primary": ["Q5", "Q11424", "Q7725634"],  # human, film, literary work
                "acceptable": ["Q5", "Q11424", "Q7725634", "Q4830453", "Q43229"],
                "description": "Varies by relationship type"
            },
            "organizational": {
                "primary": ["Q5"],  # human for members
                "acceptable": ["Q5", "Q43229", "Q4830453"],
                "description": "Organization members or the organizations themselves"
            },
            "geographic_spatial": {
                "primary": ["Q515", "Q6256"],  # city, country
                "acceptable": ["Q515", "Q6256", "Q82794", "Q3024240", "Q33506"],  # city, country, region, park, museum
                "description": "Geographic entities like cities, countries, places"
            },
            "event_temporal": {
                "primary": ["Q5"],  # human
                "acceptable": ["Q5", "Q43229", "Q6256"],  # human, organization, country
                "description": "Event participants"
            },
            "categorical": {
                "primary": [],  # Varies
                "acceptable": [],
                "description": "Type depends on the category"
            }
        }

    def get_entity_types(self, qids: list) -> dict:
        """
        Get instance-of types for a batch of entities.

        Args:
            qids: List of Wikidata QIDs

        Returns:
            Dictionary mapping QID to EntityTypeInfo
        """
        # Process in batches to avoid query timeout
        batch_size = 50
        all_results = {}

        for i in range(0, len(qids), batch_size):
            batch = qids[i:i + batch_size]
            batch_values = " ".join(f"wd:{qid}" for qid in batch)

            sparql = f"""
SELECT ?entity ?entityLabel ?type ?typeLabel WHERE {{
  VALUES ?entity {{ {batch_values} }}
  ?entity wdt:P31 ?type .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""

            result = self.profiler.execute_query(sparql, timeout=60)

            if result.success:
                # Group by entity
                entity_types = defaultdict(list)
                entity_labels = {}

                for binding in result.entities:
                    qid = binding.get("entity", "").split("/")[-1]
                    type_qid = binding.get("type", "").split("/")[-1]
                    type_label = binding.get("typeLabel", "")
                    entity_label = binding.get("entityLabel", "")

                    entity_types[qid].append((type_qid, type_label))
                    entity_labels[qid] = entity_label

                # Create EntityTypeInfo objects
                for qid in batch:
                    types = entity_types.get(qid, [])
                    primary = types[0] if types else (None, None)

                    all_results[qid] = EntityTypeInfo(
                        qid=qid,
                        label=entity_labels.get(qid, ""),
                        types=types,
                        primary_type=primary[1],
                        primary_type_qid=primary[0]
                    )

        return all_results

    def validate_cluster(
        self,
        cluster_data: dict,
        cluster_type: Optional[str] = None
    ) -> TypeValidationResult:
        """
        Validate type consistency for a cluster.

        Args:
            cluster_data: Cluster data dictionary with entities
            cluster_type: Expected cluster type (for type expectations)

        Returns:
            TypeValidationResult with validation details
        """
        cluster_id = cluster_data.get("cluster_id", "unknown")
        entities = cluster_data.get("entities", [])
        cluster_type = cluster_type or cluster_data.get("cluster_type", "categorical")

        qids = [e.get("qid") for e in entities if e.get("qid")]

        if not qids:
            return TypeValidationResult(
                cluster_id=cluster_id,
                total_entities=0,
                validated_entities=0,
                type_distribution={},
                dominant_type="",
                dominant_type_qid="",
                consistency_score=0.0,
                outliers=[],
                warnings=["No entities to validate"],
                is_valid=False
            )

        # Get types for all entities
        print(f"Fetching types for {len(qids)} entities...")
        type_info = self.get_entity_types(qids)

        # Analyze type distribution
        type_counter = Counter()
        type_qid_map = {}
        outliers = []
        warnings = []

        for qid in qids:
            info = type_info.get(qid)
            if info and info.types:
                for type_qid, type_label in info.types:
                    type_counter[type_label] += 1
                    type_qid_map[type_label] = type_qid
            else:
                warnings.append(f"No type found for {qid}")

        # Calculate consistency
        validated = len([q for q in qids if q in type_info and type_info[q].types])
        total = len(qids)

        if type_counter:
            dominant_type, dominant_count = type_counter.most_common(1)[0]
            dominant_type_qid = type_qid_map.get(dominant_type, "")
            consistency_score = dominant_count / validated if validated > 0 else 0.0
        else:
            dominant_type = ""
            dominant_type_qid = ""
            consistency_score = 0.0

        # Check against expected types
        expected = self.expected_types.get(cluster_type, {})
        expected_qids = expected.get("acceptable", [])

        if expected_qids:
            for qid in qids:
                info = type_info.get(qid)
                if info and info.types:
                    entity_type_qids = [t[0] for t in info.types]
                    if not any(t in expected_qids for t in entity_type_qids):
                        outliers.append({
                            "qid": qid,
                            "label": info.label,
                            "types": info.types,
                            "reason": f"Type not in expected list for {cluster_type}"
                        })

        # Determine validity
        is_valid = consistency_score >= 0.7 and len(outliers) <= len(qids) * 0.1

        if consistency_score < 0.7:
            warnings.append(f"Low type consistency: {consistency_score:.1%}")
        if len(outliers) > len(qids) * 0.1:
            warnings.append(f"High outlier count: {len(outliers)}/{len(qids)}")

        return TypeValidationResult(
            cluster_id=cluster_id,
            total_entities=total,
            validated_entities=validated,
            type_distribution=dict(type_counter.most_common(10)),
            dominant_type=dominant_type,
            dominant_type_qid=dominant_type_qid,
            consistency_score=consistency_score,
            outliers=outliers[:20],  # Limit outliers in result
            warnings=warnings,
            is_valid=is_valid
        )

    def validate_cluster_file(self, filepath: str) -> TypeValidationResult:
        """Load and validate a cluster from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            cluster_data = json.load(f)

        return self.validate_cluster(cluster_data)

    def batch_validate_clusters(self, cluster_dir: str) -> list:
        """
        Validate all clusters in a directory.

        Returns:
            List of TypeValidationResult objects
        """
        cluster_path = Path(cluster_dir)
        results = []

        for filepath in cluster_path.glob("*.json"):
            print(f"\nValidating {filepath.name}...")

            try:
                result = self.validate_cluster_file(str(filepath))
                results.append(result)

                status = "VALID" if result.is_valid else "INVALID"
                print(f"  Status: {status}")
                print(f"  Dominant type: {result.dominant_type} ({result.consistency_score:.1%})")
                print(f"  Outliers: {len(result.outliers)}/{result.total_entities}")

            except Exception as e:
                print(f"  Error: {e}")

        return results


def validation_result_to_dict(result: TypeValidationResult) -> dict:
    """Convert TypeValidationResult to dictionary."""
    return {
        "cluster_id": result.cluster_id,
        "total_entities": result.total_entities,
        "validated_entities": result.validated_entities,
        "type_distribution": result.type_distribution,
        "dominant_type": result.dominant_type,
        "dominant_type_qid": result.dominant_type_qid,
        "consistency_score": result.consistency_score,
        "outliers": result.outliers,
        "warnings": result.warnings,
        "is_valid": result.is_valid
    }


def generate_validation_report(results: list) -> dict:
    """Generate a summary report from validation results."""
    total_clusters = len(results)
    valid_clusters = sum(1 for r in results if r.is_valid)
    avg_consistency = sum(r.consistency_score for r in results) / total_clusters if total_clusters > 0 else 0

    type_distribution = Counter()
    for r in results:
        if r.dominant_type:
            type_distribution[r.dominant_type] += 1

    problematic = [
        {
            "cluster_id": r.cluster_id,
            "consistency": r.consistency_score,
            "outliers": len(r.outliers),
            "warnings": r.warnings
        }
        for r in results if not r.is_valid
    ]

    return {
        "summary": {
            "total_clusters": total_clusters,
            "valid_clusters": valid_clusters,
            "validity_rate": valid_clusters / total_clusters if total_clusters > 0 else 0,
            "average_consistency": avg_consistency
        },
        "dominant_type_distribution": dict(type_distribution.most_common()),
        "problematic_clusters": problematic,
        "detailed_results": [validation_result_to_dict(r) for r in results]
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate entity type consistency")
    parser.add_argument("--cluster", type=str, help="Path to cluster JSON file")
    parser.add_argument("--cluster-dir", type=str, help="Directory of cluster files to validate")
    parser.add_argument("--output", type=str, help="Output file for validation report")

    args = parser.parse_args()

    validator = EntityTypeValidator()

    if args.cluster:
        # Validate single cluster
        result = validator.validate_cluster_file(args.cluster)

        print("\n" + "=" * 60)
        print("VALIDATION RESULT")
        print("=" * 60)
        print(f"Cluster: {result.cluster_id}")
        print(f"Status: {'VALID' if result.is_valid else 'INVALID'}")
        print(f"Entities: {result.validated_entities}/{result.total_entities}")
        print(f"Consistency: {result.consistency_score:.1%}")
        print(f"Dominant type: {result.dominant_type} ({result.dominant_type_qid})")

        if result.type_distribution:
            print("\nType Distribution:")
            for type_name, count in list(result.type_distribution.items())[:5]:
                print(f"  {type_name}: {count}")

        if result.outliers:
            print(f"\nOutliers ({len(result.outliers)}):")
            for outlier in result.outliers[:5]:
                print(f"  {outlier['qid']}: {outlier['label']}")
                print(f"    Types: {outlier['types'][:3]}")

        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  - {warning}")

    elif args.cluster_dir:
        # Validate all clusters in directory
        results = validator.batch_validate_clusters(args.cluster_dir)
        report = generate_validation_report(results)

        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total clusters: {report['summary']['total_clusters']}")
        print(f"Valid clusters: {report['summary']['valid_clusters']}")
        print(f"Validity rate: {report['summary']['validity_rate']:.1%}")
        print(f"Average consistency: {report['summary']['average_consistency']:.1%}")

        if report['problematic_clusters']:
            print(f"\nProblematic clusters: {len(report['problematic_clusters'])}")
            for p in report['problematic_clusters'][:5]:
                print(f"  {p['cluster_id']}: consistency={p['consistency']:.1%}, outliers={p['outliers']}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\nReport saved to {args.output}")

    else:
        print("Please specify --cluster or --cluster-dir")
