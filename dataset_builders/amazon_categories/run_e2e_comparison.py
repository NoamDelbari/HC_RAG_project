"""
E2E Comparison: HC vs Top-K on Amazon Categories Dataset

Runs both retrieval strategies with structured output (product title extraction),
then produces a side-by-side comparison of retrieval IR metrics, QA metrics
(EM, Title P/R/F1), and LLM judge scores.

Configurations (based on best IR-only results):
  - Baseline Top-K: k=50 (best F1=0.1275 among k={5,10,20,50,100})
  - HC Adaptive:    gamma=0.05, pool=1000 (best F1=0.1326, mean_k=31.5)

Usage:
  python run_e2e_comparison.py                     # full 131 queries
  python run_e2e_comparison.py --max-queries 10    # quick test
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from embeddings.embedding_model import EmbeddingModel
from embeddings.vector_database import VectorDatabase
from retrieval.baseline_retrieval import BaselineRetrieval
from hc.null_distribution import NullDistribution
from retrieval.hc_retrieval import HCRetrieval
from evaluation.e2e_evaluator import E2EEvaluator, E2EAggregateMetrics
from src.llm.openai_llm import OpenAILLM
from src.llm.llm_io import retrieval_output_to_llm_input, format_docs

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_amazon_dataset, load_corpus, generate_gold_answers


# ── Experiment configurations ────────────────────────────────────────
CONFIGS = {
    "baseline_k50": {
        "label": "Baseline Top-K (k=50)",
        "method": "baseline",
        "k": 50,
    },
    "hc_g005_p1000": {
        "label": "HC Adaptive (gamma=0.05, pool=1000)",
        "method": "hc",
        "gamma": 0.05,
        "pool_size": 1000,
    },
}

QA_MODEL = "gpt-5-mini"
JUDGE_MODEL = "gpt-5.2"
# ─────────────────────────────────────────────────────────────────────


def call_with_retry(fn, *args, max_retries=3, base_delay=2.0, **kwargs):
    """Call a function with exponential backoff retry on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "rate" in error_str or "429" in error_str or "quota" in error_str
            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"    Rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                raise


def run_single_config(
    config_key, config, queries, qrels, corpus, gold_answers,
    vector_db, embedding_model, global_null, qa_llm, judge_llm, evaluator, delay,
):
    """Run E2E evaluation for a single retrieval configuration."""
    label = config["label"]
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # Set up retriever
    if config["method"] == "baseline":
        retriever = BaselineRetrieval(vector_db=vector_db, k=config["k"])
        print(f"  k={config['k']}")
    else:
        retriever = HCRetrieval(
            vector_db=vector_db,
            global_null_distribution=global_null,
            gamma=config["gamma"],
            max_candidates=config["pool_size"],
            min_hc=0.0,
            allow_empty=False,
            embedding_model=embedding_model,
            use_zscore=True,
        )
        print(f"  gamma={config['gamma']}, pool_size={config['pool_size']}")

    e2e_results = []
    for i, query in enumerate(queries):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(queries)}] {query.text[:50]}...")

        # Retrieve
        if config["method"] == "baseline":
            query_embedding = embedding_model.embed_query(query.text)
            retrieval_output = retriever.retrieve(query.query_id, query_embedding)
        else:
            retrieval_output = retriever.retrieve_from_text(query.query_id, query.text)

        # Generate answer (structured)
        llm_input = retrieval_output_to_llm_input(
            retrieval_output, query.text, corpus, text_field="text"
        )
        generated_answer = call_with_retry(qa_llm.generate_answer, llm_input)

        if delay > 0:
            time.sleep(delay)

        # Judge
        gold_answer = gold_answers.get(query.query_id, "No gold answer available.")
        relevant_ids = qrels.get(query.query_id, set())

        ground_truth_doc_texts = []
        for doc_id in sorted(relevant_ids):
            doc = corpus.get(doc_id)
            if doc and "text" in doc:
                ground_truth_doc_texts.append(str(doc["text"]))
        ground_truth_docs_str = "\n\n".join(ground_truth_doc_texts) if ground_truth_doc_texts else "N/A"
        retrieved_docs_str = format_docs(llm_input, qa_llm.max_tokens)

        judge_scores = call_with_retry(
            judge_llm.judge_answer,
            query=query.text,
            ground_truth_docs=ground_truth_docs_str,
            ground_truth_answer=gold_answer,
            retrieved_docs=retrieved_docs_str,
            generated_answer=generated_answer,
        )

        if delay > 0:
            time.sleep(delay)

        # Evaluate
        e2e_result = evaluator.evaluate_single(
            query_id=query.query_id,
            retrieved_ids=retrieval_output.retrieved_ids,
            retrieved_scores=retrieval_output.retrieved_scores,
            relevant_ids=relevant_ids,
            generated_answer=generated_answer,
            gold_answer=gold_answer,
            judge_scores=judge_scores,
            metadata={
                "subcategory": query.subcategory,
                "k_bucket": query.k_bucket,
                "true_k": len(relevant_ids),
            },
        )
        e2e_results.append(e2e_result)

    aggregate = evaluator.evaluate_batch(e2e_results)
    print(f"\n  Done. {label}")
    return e2e_results, aggregate


def print_comparison(results_map):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 80)

    configs = list(results_map.keys())
    labels = [results_map[c]["config"]["label"] for c in configs]

    # Header
    col_w = 30
    print(f"\n{'Metric':<30}", end="")
    for label in labels:
        print(f" {label:>{col_w}}", end="")
    print()
    print("-" * (30 + (col_w + 1) * len(labels)))

    aggs = [results_map[c]["aggregate"] for c in configs]

    # Retrieval metrics
    print("\n  RETRIEVAL")
    metrics = [
        ("Recall@k", lambda a: a.retrieval_metrics.mean_recall_at_k_labeled),
        ("Precision@k", lambda a: a.retrieval_metrics.mean_precision_at_k_labeled),
        ("Retrieval F1", lambda a: 2 * a.retrieval_metrics.mean_recall_at_k_labeled * a.retrieval_metrics.mean_precision_at_k_labeled / (a.retrieval_metrics.mean_recall_at_k_labeled + a.retrieval_metrics.mean_precision_at_k_labeled) if (a.retrieval_metrics.mean_recall_at_k_labeled + a.retrieval_metrics.mean_precision_at_k_labeled) > 0 else 0),
        ("MRR", lambda a: a.retrieval_metrics.mean_reciprocal_rank_labeled),
        ("NDCG@k", lambda a: a.retrieval_metrics.mean_ndcg_at_k_labeled),
        ("MAP@k", lambda a: a.retrieval_metrics.mean_average_precision_labeled),
        ("Mean k retrieved", lambda a: a.retrieval_metrics.mean_k),
        ("Min k", lambda a: float(a.retrieval_metrics.min_k)),
        ("Max k", lambda a: float(a.retrieval_metrics.max_k)),
    ]
    for name, fn in metrics:
        print(f"  {name:<28}", end="")
        vals = [fn(a) for a in aggs]
        best = max(vals) if "k retrieved" not in name and "Min" not in name and "Max" not in name else None
        for v in vals:
            marker = " *" if best is not None and v == best and vals.count(v) == 1 else "  "
            print(f" {v:>{col_w - 2}.4f}{marker}", end="")
        print()

    # QA metrics
    print("\n  QA (STRUCTURED OUTPUT)")
    qa_metrics = [
        ("Exact Match", lambda a: a.mean_exact_match),
        ("Token Precision (SQuAD)", lambda a: a.mean_token_precision),
        ("Token Recall (SQuAD)", lambda a: a.mean_token_recall),
        ("Token F1 (SQuAD)", lambda a: a.mean_token_f1),
        ("Title Precision (exact)", lambda a: a.mean_title_precision),
        ("Title Recall (exact)", lambda a: a.mean_title_recall),
        ("Title F1 (exact)", lambda a: a.mean_title_f1),
        ("Title Precision (fuzzy)", lambda a: a.mean_fuzzy_title_precision),
        ("Title Recall (fuzzy)", lambda a: a.mean_fuzzy_title_recall),
        ("Title F1 (fuzzy)", lambda a: a.mean_fuzzy_title_f1),
    ]
    for name, fn in qa_metrics:
        print(f"  {name:<28}", end="")
        vals = [fn(a) for a in aggs]
        best = max(vals)
        for v in vals:
            marker = " *" if v == best and vals.count(v) == 1 else "  "
            print(f" {v:>{col_w - 2}.4f}{marker}", end="")
        print()

    # Judge scores
    print("\n  JUDGE SCORES (1-5)")
    judge_metrics = [
        ("Context Precision", lambda a: a.mean_context_precision),
        ("Context Recall", lambda a: a.mean_context_recall),
        ("Answer Faithfulness", lambda a: a.mean_answer_faithfulness),
        ("Answer Correctness", lambda a: a.mean_answer_correctness),
    ]
    for name, fn in judge_metrics:
        print(f"  {name:<28}", end="")
        vals = [fn(a) for a in aggs]
        best = max(vals)
        for v in vals:
            marker = " *" if v == best and vals.count(v) == 1 else "  "
            print(f" {v:>{col_w - 2}.2f}{marker}", end="")
        print()

    print(f"\n  (* = better)\n")

    # Per-bucket breakdown
    print("=" * 80)
    print("PER-BUCKET BREAKDOWN")
    print("=" * 80)
    for config_key in configs:
        label = results_map[config_key]["config"]["label"]
        results = results_map[config_key]["results"]
        print(f"\n  {label}:")

        buckets = {"small": [], "medium": [], "large": []}
        for r in results:
            bucket = r.metadata.get("k_bucket", "unknown")
            if bucket in buckets:
                buckets[bucket].append(r)

        for bucket, bucket_results in buckets.items():
            if not bucket_results:
                continue
            mean_k = np.mean([r.retrieval_result.k for r in bucket_results])
            mean_true_k = np.mean([r.metadata.get("true_k", 0) for r in bucket_results])
            recall_ir = np.mean([r.retrieval_result.recall_at_k for r in bucket_results])
            mean_tok_f1 = np.mean([r.token_f1 for r in bucket_results])
            mean_tf1 = np.mean([r.title_f1 for r in bucket_results])
            mean_ftf1 = np.mean([r.fuzzy_title_f1 for r in bucket_results])
            print(f"    {bucket:>8} (n={len(bucket_results):>2}, avg_true_k={mean_true_k:>5.1f}): "
                  f"IR_Recall={recall_ir:.3f}  mean_k={mean_k:>5.1f}  "
                  f"TokenF1={mean_tok_f1:.3f}  TitleF1={mean_tf1:.3f}  FuzzyF1={mean_ftf1:.3f}")


def serialize_results(config_key, config, e2e_results, aggregate):
    """Serialize results for JSON output."""
    per_query = []
    for r in e2e_results:
        rr = r.retrieval_result
        per_query.append({
            "query_id": r.query_id,
            "k": rr.k,
            "true_k": len(rr.relevant_ids),
            "recall": rr.recall_at_k,
            "precision": rr.precision_at_k,
            "mrr": rr.reciprocal_rank,
            "ndcg": rr.ndcg_at_k,
            "map": rr.average_precision,
            "exact_match": r.exact_match,
            "token_precision": r.token_precision,
            "token_recall": r.token_recall,
            "token_f1": r.token_f1,
            "title_precision": r.title_precision,
            "title_recall": r.title_recall,
            "title_f1": r.title_f1,
            "fuzzy_title_precision": r.fuzzy_title_precision,
            "fuzzy_title_recall": r.fuzzy_title_recall,
            "fuzzy_title_f1": r.fuzzy_title_f1,
            "generated_answer": r.generated_answer,
            "gold_answer": r.gold_answer,
            "judge_scores": {k: v["rating"] for k, v in r.judge_scores.items()},
            "k_bucket": r.metadata.get("k_bucket", ""),
            "subcategory": r.metadata.get("subcategory", ""),
        })

    rm = aggregate.retrieval_metrics
    rec = rm.mean_recall_at_k_labeled
    prec = rm.mean_precision_at_k_labeled
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0

    return {
        "config": {
            **config,
            "qa_model": QA_MODEL,
            "judge_model": JUDGE_MODEL,
            "structured_output": True,
            "n_queries": len(e2e_results),
        },
        "aggregate": {
            "retrieval": {
                "recall": rec,
                "precision": prec,
                "f1": f1,
                "mrr": rm.mean_reciprocal_rank_labeled,
                "ndcg": rm.mean_ndcg_at_k_labeled,
                "map": rm.mean_average_precision_labeled,
                "mean_k": rm.mean_k,
                "min_k": rm.min_k,
                "max_k": rm.max_k,
            },
            "qa": {
                "exact_match": aggregate.mean_exact_match,
                "token_precision": aggregate.mean_token_precision,
                "token_recall": aggregate.mean_token_recall,
                "token_f1": aggregate.mean_token_f1,
                "title_precision": aggregate.mean_title_precision,
                "title_recall": aggregate.mean_title_recall,
                "title_f1": aggregate.mean_title_f1,
                "fuzzy_title_precision": aggregate.mean_fuzzy_title_precision,
                "fuzzy_title_recall": aggregate.mean_fuzzy_title_recall,
                "fuzzy_title_f1": aggregate.mean_fuzzy_title_f1,
            },
            "judge": {
                "context_precision": aggregate.mean_context_precision,
                "context_recall": aggregate.mean_context_recall,
                "answer_faithfulness": aggregate.mean_answer_faithfulness,
                "answer_correctness": aggregate.mean_answer_correctness,
            },
        },
        "per_query": per_query,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="E2E Comparison: HC vs Top-K")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Max queries to evaluate (default: all 131)")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Delay between LLM calls in seconds (default: 0.3)")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = PROJECT_ROOT / "datasets" / "amazon_categories"
    results_dir = PROJECT_ROOT / "results" / "amazon_categories" / "e2e"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("E2E Comparison: HC Adaptive vs Baseline Top-K")
    print("=" * 80)
    print(f"  QA Model:    {QA_MODEL} (structured output)")
    print(f"  Judge Model: {JUDGE_MODEL}")
    print()
    for key, cfg in CONFIGS.items():
        print(f"  [{key}] {cfg['label']}")
        if cfg["method"] == "baseline":
            print(f"    k={cfg['k']}")
        else:
            print(f"    gamma={cfg['gamma']}, pool_size={cfg['pool_size']}")
    print()

    # Load shared resources
    print("Loading dataset...")
    queries, qrels = load_amazon_dataset(str(output_dir))

    print("Loading corpus...")
    corpus = load_corpus(str(output_dir))

    print("Generating gold answers...")
    gold_answers = generate_gold_answers(str(output_dir))

    if args.max_queries is not None:
        queries = queries[: args.max_queries]
        print(f"Limited to {len(queries)} queries")

    print("\nLoading vector database...")
    vector_db = VectorDatabase.load(str(output_dir / "amazon_categories_vector_db"))
    print(f"  Documents: {vector_db.get_num_documents()}")

    print("Loading embedding model...")
    embedding_model = EmbeddingModel(
        model_name=EmbeddingModel.BGE_MODEL, normalize_embeddings=True
    )

    print("Loading global null distribution...")
    null_path = str(output_dir / "null_distributions" / "amazon_global_null")
    global_null = NullDistribution.load(null_path)
    print(f"  {global_null}")

    print("\nInitializing LLMs...")
    qa_llm = OpenAILLM(model_name=QA_MODEL, qa_mode=True, structured_output=True)
    judge_llm = OpenAILLM(model_name=JUDGE_MODEL, qa_mode=False)
    print(f"  QA: {QA_MODEL} (structured)")
    print(f"  Judge: {JUDGE_MODEL}")

    evaluator = E2EEvaluator()

    # Run each config
    results_map = {}
    for config_key, config in CONFIGS.items():
        e2e_results, aggregate = run_single_config(
            config_key, config, queries, qrels, corpus, gold_answers,
            vector_db, embedding_model, global_null, qa_llm, judge_llm, evaluator, args.delay,
        )
        results_map[config_key] = {
            "config": config,
            "results": e2e_results,
            "aggregate": aggregate,
        }

    # Print comparison
    print_comparison(results_map)

    # Save results
    all_results = {}
    for config_key in results_map:
        all_results[config_key] = serialize_results(
            config_key,
            results_map[config_key]["config"],
            results_map[config_key]["results"],
            results_map[config_key]["aggregate"],
        )

    results_path = results_dir / "e2e_comparison_hc_vs_topk.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
