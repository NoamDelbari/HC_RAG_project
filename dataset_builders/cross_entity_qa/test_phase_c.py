"""
Test script for Phase C: Query Generation, Verification, and Dataset Assembly

Tests all Phase C modules without requiring API calls.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_generation_planner():
    """Test GenerationPlanner."""
    print("\n" + "=" * 60)
    print("TEST 1: GenerationPlanner")
    print("=" * 60)

    from generation_planner import GenerationPlanner

    planner = GenerationPlanner()
    all_passed = True

    # Test target matrix
    print(planner.get_matrix_display())

    # Check targets exist
    if len(planner.targets) == 12:  # 4 K-ranges x 3 difficulties
        print("  [PASS] 12 targets created (4 K-ranges x 3 difficulties)")
    else:
        print(f"  [FAIL] Expected 12 targets, got {len(planner.targets)}")
        all_passed = False

    # Test allocation
    sample_clusters = [
        {"cluster_id": "c1", "cluster_type": "creative_ensemble", "entities": [{"qid": f"Q{i}"} for i in range(5)]},
        {"cluster_id": "c2", "cluster_type": "award_recognition", "entities": [{"qid": f"Q{i}"} for i in range(8)]},
    ]

    allocations = planner.allocate_clusters(sample_clusters)
    if len(allocations) > 0:
        print(f"  [PASS] Created {len(allocations)} allocations")
    else:
        print("  [FAIL] No allocations created")
        all_passed = False

    return all_passed


def test_generation_executor():
    """Test GenerationExecutor initialization."""
    print("\n" + "=" * 60)
    print("TEST 2: GenerationExecutor")
    print("=" * 60)

    from generation_executor import GenerationExecutor, GenerationResult

    all_passed = True

    # Test result dataclass
    result = GenerationResult(
        query_id="test_001",
        cluster_id="c1",
        cluster_type="creative_ensemble",
        difficulty="easy",
        target_k=5
    )

    if result.status == "pending":
        print("  [PASS] GenerationResult created with default status")
    else:
        print(f"  [FAIL] Unexpected status: {result.status}")
        all_passed = False

    # Test executor init (without API)
    try:
        executor = GenerationExecutor(output_dir="test_output")
        print("  [PASS] GenerationExecutor initialized")
    except Exception as e:
        print(f"  [FAIL] Executor init failed: {e}")
        all_passed = False

    return all_passed


def test_generation_processor():
    """Test GenerationProcessor."""
    print("\n" + "=" * 60)
    print("TEST 3: GenerationProcessor")
    print("=" * 60)

    from generation_processor import GenerationProcessor

    processor = GenerationProcessor()
    all_passed = True

    # Sample results
    sample_results = [
        {"query_id": "q1", "query_text": "Who are the actors?", "difficulty": "easy",
         "cluster_type": "creative_ensemble", "is_valid": True},
        {"query_id": "q2", "query_text": "What nationalities do they have?", "difficulty": "medium",
         "cluster_type": "creative_ensemble", "is_valid": True},
        {"query_id": "q3", "query_text": "Invalid query", "difficulty": "hard",
         "cluster_type": "award_recognition", "is_valid": False, "validation_errors": ["No question mark"]},
    ]

    metrics = processor.compute_metrics(sample_results)

    if metrics.total_count == 3:
        print("  [PASS] Computed metrics for 3 queries")
    else:
        print(f"  [FAIL] Expected 3 queries, got {metrics.total_count}")
        all_passed = False

    if metrics.valid_count == 2:
        print("  [PASS] Correctly counted 2 valid queries")
    else:
        print(f"  [FAIL] Expected 2 valid, got {metrics.valid_count}")
        all_passed = False

    # Test report formatting
    report = processor.format_report(metrics)
    if "GENERATION QUALITY REPORT" in report:
        print("  [PASS] Report formatted correctly")
    else:
        print("  [FAIL] Report format issue")
        all_passed = False

    return all_passed


def test_ground_truth_verifier():
    """Test SPARQLVerifier."""
    print("\n" + "=" * 60)
    print("TEST 4: Ground Truth Verifier")
    print("=" * 60)

    from ground_truth_verifier import SPARQLVerifier, GroundTruth

    all_passed = True

    # Test GroundTruth dataclass
    gt = GroundTruth(
        query_id="q1",
        cluster_id="c1",
        expected_entities=["Q1", "Q2", "Q3"],
        verified_entities=["Q1", "Q2"],
        entity_count=2,
        target_k=3
    )

    if gt.query_id == "q1":
        print("  [PASS] GroundTruth dataclass works")
    else:
        print("  [FAIL] GroundTruth issue")
        all_passed = False

    # Test verifier init
    try:
        verifier = SPARQLVerifier()
        print("  [PASS] SPARQLVerifier initialized")
    except Exception as e:
        print(f"  [FAIL] Verifier init failed: {e}")
        all_passed = False

    return all_passed


def test_passage_mapper():
    """Test PassageMapper."""
    print("\n" + "=" * 60)
    print("TEST 5: PassageMapper")
    print("=" * 60)

    from passage_mapper import PassageMapper

    mapper = PassageMapper()
    all_passed = True

    # Add sample passages
    sample_passages = [
        {"passage_id": "p1", "entity_qid": "Q1", "entity_label": "Entity 1", "passage_text": "Text 1"},
        {"passage_id": "p2", "entity_qid": "Q1", "entity_label": "Entity 1", "passage_text": "Text 2"},
        {"passage_id": "p3", "entity_qid": "Q2", "entity_label": "Entity 2", "passage_text": "Text 3"},
    ]

    for p in sample_passages:
        mapper._index_passage(p)

    stats = mapper.get_corpus_stats()
    if stats["total_passages"] == 3:
        print("  [PASS] Indexed 3 passages")
    else:
        print(f"  [FAIL] Expected 3 passages, got {stats['total_passages']}")
        all_passed = False

    # Test mapping
    qp = mapper.map_query("q1", "c1", ["Q1", "Q2", "Q3"])

    if qp.passage_count == 3:
        print("  [PASS] Mapped 3 passages for query")
    else:
        print(f"  [FAIL] Expected 3 passages, got {qp.passage_count}")
        all_passed = False

    if len(qp.missing_entities) == 1:
        print("  [PASS] Correctly identified 1 missing entity")
    else:
        print(f"  [FAIL] Expected 1 missing, got {len(qp.missing_entities)}")
        all_passed = False

    return all_passed


def test_k_validator():
    """Test KValidator."""
    print("\n" + "=" * 60)
    print("TEST 6: KValidator")
    print("=" * 60)

    from k_validator import KValidator

    validator = KValidator()
    all_passed = True

    # Test exact match
    result = validator.validate_query("q1", "c1", "easy", 5, ["Q1"]*5, 5, 1.0)
    if result.accepted and result.discrepancy_category == "exact":
        print("  [PASS] Exact match accepted")
    else:
        print(f"  [FAIL] Exact match not handled correctly")
        all_passed = False

    # Test close match
    result = validator.validate_query("q2", "c1", "medium", 5, ["Q1"]*4, 4, 0.8)
    if result.accepted and result.discrepancy_category == "close":
        print("  [PASS] Close match accepted")
    else:
        print(f"  [FAIL] Close match not handled correctly")
        all_passed = False

    # Test rejection (too few)
    result = validator.validate_query("q3", "c2", "hard", 5, ["Q1"], 1, 0.2)
    if not result.accepted:
        print("  [PASS] Low K rejected")
    else:
        print(f"  [FAIL] Low K should be rejected")
        all_passed = False

    return all_passed


def test_quality_filter():
    """Test QualityFilterPipeline."""
    print("\n" + "=" * 60)
    print("TEST 7: QualityFilterPipeline")
    print("=" * 60)

    from quality_filter import QualityFilterPipeline

    pipeline = QualityFilterPipeline()
    all_passed = True

    sample_queries = [
        {"query_id": "q1", "query_text": "Who are the actors in Inception?",
         "cluster_id": "c1", "cluster_type": "creative_ensemble", "difficulty": "easy",
         "target_k": 5, "actual_k": 5, "entity_coverage": 1.0, "passage_count": 5},
        {"query_id": "q2", "query_text": "No question mark",
         "cluster_id": "c2", "cluster_type": "award_recognition", "difficulty": "hard",
         "target_k": 3, "actual_k": 1, "entity_coverage": 0.3, "passage_count": 1},
    ]

    report = pipeline.filter_queries(sample_queries)

    if report.total_input == 2:
        print("  [PASS] Processed 2 queries")
    else:
        print(f"  [FAIL] Expected 2 queries")
        all_passed = False

    if report.total_passed >= 1:
        print(f"  [PASS] {report.total_passed} queries passed filters")
    else:
        print(f"  [FAIL] Expected at least 1 to pass")
        all_passed = False

    # Check report format
    formatted = pipeline.format_report(report)
    if "QUALITY FILTERING REPORT" in formatted:
        print("  [PASS] Report formatted correctly")
    else:
        print("  [FAIL] Report format issue")
        all_passed = False

    return all_passed


def test_dataset_balancer():
    """Test DatasetBalancer."""
    print("\n" + "=" * 60)
    print("TEST 8: DatasetBalancer")
    print("=" * 60)

    from dataset_balancer import DatasetBalancer
    import random

    balancer = DatasetBalancer(total_target=20)
    all_passed = True

    # Generate sample queries
    sample_queries = []
    for i in range(50):
        sample_queries.append({
            "query_id": f"q{i}",
            "query_text": f"Query {i}?",
            "cluster_id": f"c{i % 10}",
            "cluster_type": random.choice(["categorical", "creative_ensemble", "award_recognition"]),
            "difficulty": random.choice(["easy", "medium", "hard"]),
            "actual_k": random.choice([3, 5, 8, 12, 18]),
            "total_score": random.random()
        })

    result = balancer.select_queries(sample_queries)

    if result.total_selected > 0:
        print(f"  [PASS] Selected {result.total_selected} queries")
    else:
        print("  [FAIL] No queries selected")
        all_passed = False

    # Check distribution
    if result.by_difficulty:
        print(f"  [PASS] Distribution by difficulty: {result.by_difficulty}")
    else:
        print("  [FAIL] No difficulty distribution")
        all_passed = False

    return all_passed


def test_dataset_assembler():
    """Test dataset assembly components."""
    print("\n" + "=" * 60)
    print("TEST 9: DatasetAssembler")
    print("=" * 60)

    from dataset_assembler import CorpusAssembler, DatasetSplitter, MetadataGenerator

    all_passed = True

    # Test CorpusAssembler
    assembler = CorpusAssembler(target_sparsity=0.2)

    sample_queries = [
        {"query_id": "q1", "cluster_id": "c1", "difficulty": "easy", "actual_k": 3,
         "cluster_type": "creative_ensemble", "target_k": 3}
    ]
    sample_passages = [
        {"passage_id": f"p{i}", "passage_text": f"Text {i}", "entity_qid": f"Q{i}"}
        for i in range(10)
    ]
    sample_mappings = {
        "q1": {"matches": [{"passage_id": "p1", "entity_qid": "Q1"}, {"passage_id": "p2", "entity_qid": "Q2"}]}
    }

    corpus, qrels, stats = assembler.assemble_corpus(sample_queries, sample_mappings, sample_passages)

    if stats.relevant_passages == 2:
        print(f"  [PASS] Assembled corpus with {stats.total_passages} passages")
    else:
        print(f"  [FAIL] Corpus assembly issue")
        all_passed = False

    # Test DatasetSplitter
    splitter = DatasetSplitter()
    splits = splitter.create_splits(sample_queries * 10)  # Need more for splits

    if "train" in splits and "dev" in splits and "test" in splits:
        print("  [PASS] Created train/dev/test splits")
    else:
        print("  [FAIL] Split creation issue")
        all_passed = False

    # Test MetadataGenerator
    meta_gen = MetadataGenerator()
    metadata = meta_gen.generate_metadata(sample_queries, corpus, qrels, splits, stats)

    if "statistics" in metadata and "schema" in metadata:
        print("  [PASS] Generated metadata with required fields")
    else:
        print("  [FAIL] Metadata generation issue")
        all_passed = False

    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PHASE C TEST SUITE")
    print("Query Generation, Verification, and Dataset Assembly")
    print("=" * 60)

    results = {}

    results["GenerationPlanner"] = test_generation_planner()
    results["GenerationExecutor"] = test_generation_executor()
    results["GenerationProcessor"] = test_generation_processor()
    results["GroundTruthVerifier"] = test_ground_truth_verifier()
    results["PassageMapper"] = test_passage_mapper()
    results["KValidator"] = test_k_validator()
    results["QualityFilter"] = test_quality_filter()
    results["DatasetBalancer"] = test_dataset_balancer()
    results["DatasetAssembler"] = test_dataset_assembler()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  All Phase C tests passed!")
        return 0
    else:
        print("\n  Some tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
