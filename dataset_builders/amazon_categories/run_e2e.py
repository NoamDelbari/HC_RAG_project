"""
Run End-to-End RAG Evaluation on Amazon Categories Dataset

Combines retrieval (baseline or HC) with LLM generation and judge evaluation.
Saves per-query and aggregate results including both IR metrics and judge scores.
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
from evaluation.e2e_evaluator import E2EEvaluator
from src.llm.openai_llm import OpenAILLM
from src.llm.llm_io import retrieval_output_to_llm_input, format_docs

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_amazon_dataset, load_corpus, generate_gold_answers


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


def parse_args():
    parser = argparse.ArgumentParser(description="E2E RAG Evaluation on Amazon Categories")
    parser.add_argument("--method", choices=["baseline", "hc"], default="baseline",
                        help="Retrieval method (default: baseline)")
    parser.add_argument("--k", type=int, default=10,
                        help="Top-k for baseline retrieval (default: 10)")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="HC gamma parameter (default: 0.1)")
    parser.add_argument("--pool-size", type=int, default=100,
                        help="HC candidate pool size (default: 100)")
    parser.add_argument("--qa-model", type=str, default="gpt-5-mini",
                        help="QA generation model (default: gpt-5-mini)")
    parser.add_argument("--judge-model", type=str, default="gpt-5.2",
                        help="Judge model (default: gpt-5.2)")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Max queries to evaluate (default: all)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between LLM calls in seconds (default: 0.5)")
    parser.add_argument("--structured", action="store_true",
                        help="Use structured output (JSON schema) for QA model to return product titles")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = PROJECT_ROOT / "datasets" / "amazon_categories"
    results_dir = PROJECT_ROOT / "results" / "amazon_categories" / "e2e"
    results_dir.mkdir(parents=True, exist_ok=True)

    db_path = str(output_dir / "amazon_categories_vector_db")

    print("=" * 70)
    print("End-to-End RAG Evaluation (Amazon Categories Dataset)")
    print("=" * 70)
    print(f"  Method: {args.method}")
    print(f"  QA model: {args.qa_model}")
    print(f"  Judge model: {args.judge_model}")
    print(f"  Structured output: {args.structured}")
    if args.method == "baseline":
        print(f"  k: {args.k}")
    else:
        print(f"  gamma: {args.gamma}, pool_size: {args.pool_size}")

    # Load dataset
    print("\nLoading dataset...")
    queries, qrels = load_amazon_dataset(str(output_dir))

    print("Loading corpus...")
    corpus = load_corpus(str(output_dir))

    print("Generating gold answers...")
    gold_answers = generate_gold_answers(str(output_dir))

    # Limit queries if requested
    if args.max_queries is not None:
        queries = queries[: args.max_queries]
        print(f"  Limited to {len(queries)} queries")

    # Load vector DB
    print("\nLoading vector database...")
    vector_db = VectorDatabase.load(db_path)
    print(f"  Documents: {vector_db.get_num_documents()}")

    # Load embedding model
    print("Loading embedding model...")
    embedding_model = EmbeddingModel(
        model_name=EmbeddingModel.BGE_MODEL, normalize_embeddings=True
    )

    # Set up retriever
    if args.method == "baseline":
        retriever = BaselineRetrieval(vector_db=vector_db, k=args.k)
    else:
        null_path = str(output_dir / "null_distributions" / "amazon_global_null")
        print("Loading global null distribution...")
        global_null = NullDistribution.load(null_path)
        print(f"  {global_null}")
        retriever = HCRetrieval(
            vector_db=vector_db,
            global_null_distribution=global_null,
            gamma=args.gamma,
            max_candidates=args.pool_size,
            min_hc=0.0,
            allow_empty=False,
            embedding_model=embedding_model,
            use_zscore=True,
        )

    # Set up LLMs
    print("\nInitializing LLMs...")
    qa_llm = OpenAILLM(model_name=args.qa_model, qa_mode=True, structured_output=args.structured)
    judge_llm = OpenAILLM(model_name=args.judge_model, qa_mode=False)
    print(f"  QA: {args.qa_model}")
    print(f"  Judge: {args.judge_model}")

    # Set up evaluator
    evaluator = E2EEvaluator()

    # Run E2E pipeline
    print(f"\nRunning E2E evaluation on {len(queries)} queries...")
    print("-" * 70)

    e2e_results = []
    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] Query: {query.text[:60]}...")

        # Step 1: Retrieve
        if args.method == "baseline":
            query_embedding = embedding_model.embed_query(query.text)
            retrieval_output = retriever.retrieve(query.query_id, query_embedding)
        else:
            retrieval_output = retriever.retrieve_from_text(query.query_id, query.text)

        print(f"  Retrieved {retrieval_output.k} docs")

        # Step 2: Build LLM input
        llm_input = retrieval_output_to_llm_input(
            retrieval_output, query.text, corpus, text_field="text"
        )

        # Step 3: Generate answer
        generated_answer = call_with_retry(qa_llm.generate_answer, llm_input)
        print(f"  Answer: {generated_answer[:80].encode('ascii', 'replace').decode()}...")

        if args.delay > 0:
            time.sleep(args.delay)

        # Step 4: Get gold answer
        gold_answer = gold_answers.get(query.query_id, "No gold answer available.")

        # Step 5: Judge
        relevant_ids = qrels.get(query.query_id, set())
        # Build doc text strings for the judge
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

        tok_p, tok_r, tok_f1 = E2EEvaluator.compute_token_f1(generated_answer, gold_answer)
        ft_prec, ft_rec, ft_f1 = E2EEvaluator.compute_fuzzy_title_f1(generated_answer, gold_answer)
        print(f"  Judge: CP={judge_scores['context_precision']['rating']} "
              f"CR={judge_scores['context_recall']['rating']} "
              f"AF={judge_scores['answer_faithfulness']['rating']} "
              f"AC={judge_scores['answer_correctness']['rating']} "
              f"| TokenF1={tok_f1:.2f} FuzzyTitleF1={ft_f1:.2f}")

        if args.delay > 0:
            time.sleep(args.delay)

        # Step 6: Collect E2E result
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

    # Aggregate
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    aggregate = evaluator.evaluate_batch(e2e_results)
    print(aggregate)

    # Save results
    results_data = {
        "config": {
            "method": args.method,
            "qa_model": args.qa_model,
            "judge_model": args.judge_model,
            "k": args.k if args.method == "baseline" else None,
            "gamma": args.gamma if args.method == "hc" else None,
            "pool_size": args.pool_size if args.method == "hc" else None,
            "n_queries": len(e2e_results),
        },
        "aggregate": {
            "retrieval": {
                "recall": aggregate.retrieval_metrics.mean_recall_at_k_labeled,
                "precision": aggregate.retrieval_metrics.mean_precision_at_k_labeled,
                "mrr": aggregate.retrieval_metrics.mean_reciprocal_rank_labeled,
                "ndcg": aggregate.retrieval_metrics.mean_ndcg_at_k_labeled,
                "map": aggregate.retrieval_metrics.mean_average_precision_labeled,
                "mean_k": aggregate.retrieval_metrics.mean_k,
            },
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
            "judge": {
                "context_precision": aggregate.mean_context_precision,
                "context_recall": aggregate.mean_context_recall,
                "answer_faithfulness": aggregate.mean_answer_faithfulness,
                "answer_correctness": aggregate.mean_answer_correctness,
            },
        },
        "per_query": [],
    }

    for r in e2e_results:
        rr = r.retrieval_result
        results_data["per_query"].append({
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
            "judge_scores": {
                k: v["rating"] for k, v in r.judge_scores.items()
            },
            "k_bucket": r.metadata.get("k_bucket", ""),
            "subcategory": r.metadata.get("subcategory", ""),
        })

    # Build filename
    mode_tag = "_structured" if args.structured else ""
    if args.method == "baseline":
        filename = f"e2e_{args.method}_k{args.k}_{args.qa_model.replace('/', '_')}{mode_tag}.json"
    else:
        filename = f"e2e_{args.method}_g{args.gamma}_p{args.pool_size}_{args.qa_model.replace('/', '_')}{mode_tag}.json"

    results_path = results_dir / filename
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
