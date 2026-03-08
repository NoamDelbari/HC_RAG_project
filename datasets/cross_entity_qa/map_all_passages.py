"""
Map All Clusters to Wikipedia Passages

This script processes all cluster files and extracts Wikipedia passages
for each entity. It provides progress tracking and summary statistics.

Usage:
    python map_all_passages.py [--output data/passages] [--delay 0.5]
"""

import argparse
import glob
import json
import time
from pathlib import Path
from datetime import datetime

from wikipedia_mapper import (
    ClusterWikipediaMapper,
    save_cluster_passages,
    coverage_to_dict
)


def map_all_clusters(
    clusters_dir: str = "data/clusters",
    output_dir: str = "data/passages",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    delay: float = 0.5,
    max_entities_per_cluster: int = None
):
    """
    Map all clusters to Wikipedia passages.
    
    Args:
        clusters_dir: Directory containing cluster JSON files
        output_dir: Directory to save passage files
        chunk_size: Chunk size for passage extraction
        chunk_overlap: Overlap between chunks
        delay: Delay between API requests
        max_entities_per_cluster: Optional limit per cluster (for testing)
    
    Returns:
        Summary dictionary with statistics
    """
    # Find all cluster files
    cluster_files = list(glob.glob(f"{clusters_dir}/*.json"))
    
    if not cluster_files:
        print(f"No cluster files found in {clusters_dir}")
        return None
    
    print("=" * 70)
    print("WIKIPEDIA PASSAGE MAPPING")
    print("=" * 70)
    print(f"Found {len(cluster_files)} cluster files")
    print(f"Output directory: {output_dir}")
    print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    print(f"API delay: {delay}s")
    print("=" * 70)
    
    # Initialize mapper
    mapper = ClusterWikipediaMapper(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Process each cluster
    results = []
    total_entities = 0
    total_passages = 0
    start_time = time.time()
    
    for i, cluster_file in enumerate(cluster_files, 1):
        print(f"\n[{i}/{len(cluster_files)}] Processing: {Path(cluster_file).name}")
        print("-" * 50)
        
        try:
            # Load cluster
            with open(cluster_file, "r", encoding="utf-8") as f:
                cluster_data = json.load(f)
            
            cluster_id = cluster_data.get("cluster_id", "unknown")
            entity_count = len(cluster_data.get("entities", []))
            print(f"Cluster: {cluster_id}")
            print(f"Entities: {entity_count}")
            
            # Map to Wikipedia
            articles, coverage = mapper.map_cluster(
                cluster_data,
                max_entities=max_entities_per_cluster,
                delay=delay
            )
            
            # Validate coverage
            validation = mapper.validate_coverage(coverage)
            
            # Save results
            save_cluster_passages(cluster_id, articles, coverage, output_dir)
            
            # Track statistics
            total_entities += coverage.entities_with_wikipedia
            total_passages += coverage.total_passages
            
            results.append({
                "cluster_id": cluster_id,
                "cluster_file": cluster_file,
                "total_entities": coverage.total_entities,
                "entities_with_wikipedia": coverage.entities_with_wikipedia,
                "coverage_rate": coverage.coverage_rate,
                "total_passages": coverage.total_passages,
                "avg_passages": coverage.avg_passages_per_entity,
                "is_valid": validation["is_valid"],
                "quality_distribution": coverage.quality_distribution
            })
            
            print(f"\n  ✓ Coverage: {coverage.coverage_rate:.1%}")
            print(f"  ✓ Passages: {coverage.total_passages}")
            print(f"  ✓ Status: {'VALID' if validation['is_valid'] else 'NEEDS REVIEW'}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                "cluster_file": cluster_file,
                "error": str(e)
            })
    
    # Summary
    elapsed = time.time() - start_time
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    print("\n" + "=" * 70)
    print("MAPPING COMPLETE")
    print("=" * 70)
    print(f"Clusters processed: {len(successful)}/{len(cluster_files)}")
    print(f"Total entities mapped: {total_entities}")
    print(f"Total passages extracted: {total_passages}")
    print(f"Time elapsed: {elapsed:.1f}s")
    
    if failed:
        print(f"\nFailed clusters ({len(failed)}):")
        for r in failed:
            print(f"  - {r['cluster_file']}: {r['error']}")
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "clusters_processed": len(successful),
        "clusters_failed": len(failed),
        "total_entities": total_entities,
        "total_passages": total_passages,
        "elapsed_seconds": elapsed,
        "config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "delay": delay
        },
        "results": results
    }
    
    summary_file = Path(output_dir) / "mapping_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary saved to: {summary_file}")
    
    return summary


def print_stats(output_dir: str = "data/passages"):
    """Print statistics from existing passage files."""
    print("\n" + "=" * 70)
    print("PASSAGE STATISTICS")
    print("=" * 70)
    
    coverage_files = list(glob.glob(f"{output_dir}/*_coverage.json"))
    
    if not coverage_files:
        print("No coverage files found.")
        return
    
    total_entities = 0
    total_passages = 0
    
    for cov_file in coverage_files:
        with open(cov_file, "r", encoding="utf-8") as f:
            cov = json.load(f)
        
        cluster_id = cov.get("cluster_id", Path(cov_file).stem)
        entities = cov.get("entities_with_wikipedia", 0)
        passages = cov.get("total_passages", 0)
        rate = cov.get("coverage_rate", 0)
        
        total_entities += entities
        total_passages += passages
        
        print(f"  {cluster_id:<40} {entities:>3} entities, {passages:>4} passages ({rate:.0%})")
    
    print("-" * 70)
    print(f"  {'TOTAL':<40} {total_entities:>3} entities, {total_passages:>4} passages")
    

def main():
    parser = argparse.ArgumentParser(description="Map all clusters to Wikipedia passages")
    parser.add_argument("--clusters", type=str, default="data/clusters",
                        help="Directory containing cluster JSON files")
    parser.add_argument("--output", type=str, default="data/passages",
                        help="Output directory for passage files")
    parser.add_argument("--chunk-size", type=int, default=800,
                        help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=100,
                        help="Chunk overlap in characters")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between API requests (seconds)")
    parser.add_argument("--max-entities", type=int, default=None,
                        help="Maximum entities per cluster (for testing)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only print statistics from existing files")
    
    args = parser.parse_args()
    
    if args.stats_only:
        print_stats(args.output)
    else:
        map_all_clusters(
            clusters_dir=args.clusters,
            output_dir=args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            delay=args.delay,
            max_entities_per_cluster=args.max_entities
        )


if __name__ == "__main__":
    main()
