"""
Cluster Size Distribution Analyzer

This module analyzes the distribution of cluster sizes across discovered clusters
and identifies gaps in K-value coverage for the cross-entity dataset.

Phase A.2.4: Analyze cluster size distribution and identify gaps
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from sparql_profiler import SPARQLProfiler, PREDEFINED_CLUSTERS, run_cluster_profiling
from intersection_miner import IntersectionMiner, INTERSECTION_TEMPLATES


@dataclass
class ClusterSizeStats:
    """Statistics about cluster size distribution."""
    total_clusters: int = 0
    viable_clusters: int = 0
    size_distribution: dict = field(default_factory=dict)
    k_bucket_coverage: dict = field(default_factory=dict)
    gaps: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)


@dataclass
class KBucket:
    """Represents a K-value bucket for analysis."""
    name: str
    min_k: int
    max_k: int
    target_count: int
    actual_count: int = 0
    clusters: list = field(default_factory=list)

    @property
    def coverage_ratio(self) -> float:
        return self.actual_count / self.target_count if self.target_count > 0 else 0.0

    @property
    def is_sufficient(self) -> bool:
        return self.actual_count >= self.target_count


# Target K-value buckets based on dataset design
K_BUCKETS = [
    KBucket(name="tiny", min_k=3, max_k=5, target_count=40),
    KBucket(name="small", min_k=6, max_k=10, target_count=60),
    KBucket(name="medium", min_k=11, max_k=20, target_count=80),
    KBucket(name="large", min_k=21, max_k=50, target_count=60),
    KBucket(name="xlarge", min_k=51, max_k=100, target_count=40),
]


class ClusterAnalyzer:
    """
    Analyzes cluster size distribution and identifies coverage gaps.
    """

    def __init__(self):
        self.profiler = SPARQLProfiler()
        self.miner = IntersectionMiner()
        self.clusters = []
        self.k_buckets = [KBucket(
            name=b.name,
            min_k=b.min_k,
            max_k=b.max_k,
            target_count=b.target_count
        ) for b in K_BUCKETS]

    def add_cluster(self, name: str, count: int, cluster_type: str, metadata: dict = None):
        """Add a cluster to the analysis."""
        self.clusters.append({
            "name": name,
            "count": count,
            "cluster_type": cluster_type,
            "metadata": metadata or {}
        })

        # Assign to appropriate bucket
        for bucket in self.k_buckets:
            if bucket.min_k <= count <= bucket.max_k:
                bucket.actual_count += 1
                bucket.clusters.append(name)
                break

    def profile_predefined_clusters(self, verbose: bool = True):
        """Profile all predefined clusters and add to analysis."""
        if verbose:
            print("Profiling predefined single-property clusters...")
            print("=" * 60)

        results = run_cluster_profiling(count_only=True)

        for name, data in results.items():
            if data["result"].success:
                self.add_cluster(
                    name=name,
                    count=data["result"].count,
                    cluster_type=data["query"].cluster_type,
                    metadata={
                        "expected_k": data["expected_k"],
                        "primary_property": data["query"].primary_property
                    }
                )

        if verbose:
            print(f"\nProfiled {len(results)} predefined clusters")

    def profile_intersection_clusters(self, verbose: bool = True):
        """Profile intersection clusters and add to analysis."""
        if verbose:
            print("\nProfiling intersection clusters...")
            print("=" * 60)

        for template_name, config in INTERSECTION_TEMPLATES.items():
            if verbose:
                print(f"  Processing {template_name}...")

            try:
                if config["type"] == "award_occupation":
                    params = config["params"]
                    if params.get("award_qid"):
                        cluster = self.miner.mine_award_occupation_intersection(
                            award_qid=params["award_qid"],
                            occupation_qid=params["occupation_qid"],
                            country_qid=params.get("country_qid")
                        )
                        self.add_cluster(
                            name=template_name,
                            count=cluster.entity_count,
                            cluster_type="intersection",
                            metadata={"properties": cluster.properties}
                        )

                elif config["type"] == "education_achievement":
                    params = config["params"]
                    cluster = self.miner.mine_education_achievement_intersection(
                        university_qid=params["university_qid"],
                        achievement_property=params["achievement_property"],
                        achievement_value=params["achievement_value"]
                    )
                    self.add_cluster(
                        name=template_name,
                        count=cluster.entity_count,
                        cluster_type="intersection",
                        metadata={"properties": cluster.properties}
                    )

                elif config["type"] == "geographic_categorical":
                    params = config["params"]
                    cluster = self.miner.mine_geographic_categorical_intersection(
                        entity_type_qid=params["entity_type_qid"],
                        country_qid=params["country_qid"]
                    )
                    self.add_cluster(
                        name=template_name,
                        count=cluster.entity_count,
                        cluster_type="intersection",
                        metadata={"properties": cluster.properties}
                    )

            except Exception as e:
                if verbose:
                    print(f"    Error: {e}")

    def analyze_distribution(self) -> ClusterSizeStats:
        """Analyze the cluster size distribution and identify gaps."""
        stats = ClusterSizeStats()
        stats.total_clusters = len(self.clusters)

        # Count clusters by size buckets
        size_buckets = defaultdict(int)
        for cluster in self.clusters:
            count = cluster["count"]
            if count <= 5:
                size_buckets["1-5"] += 1
            elif count <= 10:
                size_buckets["6-10"] += 1
            elif count <= 20:
                size_buckets["11-20"] += 1
            elif count <= 50:
                size_buckets["21-50"] += 1
            elif count <= 100:
                size_buckets["51-100"] += 1
            else:
                size_buckets["100+"] += 1

        stats.size_distribution = dict(size_buckets)

        # Analyze K-bucket coverage
        for bucket in self.k_buckets:
            stats.k_bucket_coverage[bucket.name] = {
                "range": f"{bucket.min_k}-{bucket.max_k}",
                "target": bucket.target_count,
                "actual": bucket.actual_count,
                "coverage": f"{bucket.coverage_ratio:.1%}",
                "sufficient": bucket.is_sufficient,
                "clusters": bucket.clusters
            }

            # Identify gaps
            if not bucket.is_sufficient:
                gap = bucket.target_count - bucket.actual_count
                stats.gaps.append({
                    "bucket": bucket.name,
                    "range": f"K={bucket.min_k}-{bucket.max_k}",
                    "missing": gap,
                    "coverage": f"{bucket.coverage_ratio:.1%}"
                })

        # Count viable clusters (within any target bucket)
        stats.viable_clusters = sum(b.actual_count for b in self.k_buckets)

        # Generate recommendations
        stats.recommendations = self._generate_recommendations()

        return stats

    def _generate_recommendations(self) -> list:
        """Generate recommendations for filling coverage gaps."""
        recommendations = []

        for bucket in self.k_buckets:
            if bucket.is_sufficient:
                continue

            gap = bucket.target_count - bucket.actual_count
            rec = {
                "bucket": bucket.name,
                "gap": gap,
                "strategies": []
            }

            if bucket.name == "tiny":
                rec["strategies"] = [
                    "Add more family relationship clusters (spouses, children of historical figures)",
                    "Look for niche award categories with few winners",
                    "Find small creative ensembles (bands, writing partnerships)",
                    "Search for exclusive group memberships (G7, security council)"
                ]
            elif bucket.name == "small":
                rec["strategies"] = [
                    "Add decade-constrained award winners",
                    "Look for specific cast lists from notable films",
                    "Find sports championship rosters from specific years",
                    "Search for signatories of historical documents"
                ]
            elif bucket.name == "medium":
                rec["strategies"] = [
                    "Add geographic constraints to occupation queries",
                    "Look for event participants (conferences, summits)",
                    "Find franchise film series",
                    "Search for academic department faculty"
                ]
            elif bucket.name == "large":
                rec["strategies"] = [
                    "Expand temporal ranges on award queries",
                    "Look for larger organizational memberships",
                    "Find national team rosters across multiple events",
                    "Search for alumni of specific programs"
                ]
            elif bucket.name == "xlarge":
                rec["strategies"] = [
                    "Use broader categorical queries with country filter",
                    "Look for full award winner lists without time constraint",
                    "Find large institutional memberships",
                    "Search for professional organization rosters"
                ]

            recommendations.append(rec)

        return recommendations

    def get_clusters_by_type(self) -> dict:
        """Group clusters by their type."""
        by_type = defaultdict(list)
        for cluster in self.clusters:
            by_type[cluster["cluster_type"]].append(cluster)
        return dict(by_type)

    def get_clusters_in_range(self, min_k: int, max_k: int) -> list:
        """Get clusters within a specific K range."""
        return [c for c in self.clusters if min_k <= c["count"] <= max_k]

    def print_analysis_report(self, stats: ClusterSizeStats):
        """Print a formatted analysis report."""
        print("\n" + "=" * 70)
        print("CLUSTER SIZE DISTRIBUTION ANALYSIS")
        print("=" * 70)

        print(f"\nTotal clusters analyzed: {stats.total_clusters}")
        print(f"Viable clusters (within target K ranges): {stats.viable_clusters}")

        print("\n--- Size Distribution ---")
        for size_range, count in sorted(stats.size_distribution.items()):
            bar = "█" * min(count, 50)
            print(f"  {size_range:>8}: {count:>4} {bar}")

        print("\n--- K-Bucket Coverage ---")
        print(f"{'Bucket':<10} {'Range':<12} {'Target':<8} {'Actual':<8} {'Coverage':<10} {'Status':<10}")
        print("-" * 70)

        for bucket_name, info in stats.k_bucket_coverage.items():
            status = "OK" if info["sufficient"] else "GAP"
            print(f"{bucket_name:<10} {info['range']:<12} {info['target']:<8} {info['actual']:<8} {info['coverage']:<10} {status:<10}")

        if stats.gaps:
            print("\n--- Coverage Gaps ---")
            for gap in stats.gaps:
                print(f"  {gap['bucket']}: Missing {gap['missing']} clusters for K={gap['range']}")

        if stats.recommendations:
            print("\n--- Recommendations ---")
            for rec in stats.recommendations:
                print(f"\n  {rec['bucket'].upper()} bucket (need {rec['gap']} more clusters):")
                for strategy in rec["strategies"]:
                    print(f"    • {strategy}")

        print("\n" + "=" * 70)

    def export_analysis(self, filepath: str, stats: ClusterSizeStats):
        """Export analysis results to JSON."""
        output = {
            "summary": {
                "total_clusters": stats.total_clusters,
                "viable_clusters": stats.viable_clusters,
                "size_distribution": stats.size_distribution
            },
            "k_bucket_coverage": stats.k_bucket_coverage,
            "gaps": stats.gaps,
            "recommendations": stats.recommendations,
            "clusters": self.clusters
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nAnalysis exported to {filepath}")


def run_full_analysis(output_file: Optional[str] = None):
    """Run complete cluster analysis pipeline."""
    analyzer = ClusterAnalyzer()

    # Profile predefined clusters
    analyzer.profile_predefined_clusters(verbose=True)

    # Profile intersection clusters
    analyzer.profile_intersection_clusters(verbose=True)

    # Analyze distribution
    stats = analyzer.analyze_distribution()

    # Print report
    analyzer.print_analysis_report(stats)

    # Export if requested
    if output_file:
        analyzer.export_analysis(output_file, stats)

    return analyzer, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze cluster size distribution")
    parser.add_argument("--output", type=str, help="Output JSON file for analysis")
    parser.add_argument("--skip-profiling", action="store_true", help="Skip SPARQL profiling (use cached data)")

    args = parser.parse_args()

    if args.skip_profiling:
        # Create analyzer with sample data for testing
        analyzer = ClusterAnalyzer()

        # Add sample clusters manually
        sample_clusters = [
            ("sample_tiny_1", 4, "relational"),
            ("sample_tiny_2", 5, "relational"),
            ("sample_small_1", 8, "award_recognition"),
            ("sample_small_2", 10, "creative_ensemble"),
            ("sample_medium_1", 15, "organizational"),
            ("sample_medium_2", 18, "event_temporal"),
            ("sample_large_1", 35, "geographic_spatial"),
            ("sample_large_2", 45, "categorical"),
            ("sample_xlarge_1", 75, "award_recognition"),
        ]

        for name, count, ctype in sample_clusters:
            analyzer.add_cluster(name, count, ctype)

        stats = analyzer.analyze_distribution()
        analyzer.print_analysis_report(stats)

        if args.output:
            analyzer.export_analysis(args.output, stats)
    else:
        run_full_analysis(output_file=args.output)
