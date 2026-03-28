"""Run end-to-end RAG evaluation: retrieval -> QA -> judge -> metrics.

Loads saved retrieval outputs from run_retrieval.py, then runs LLM evaluation.

Usage: python -m experiments.scripts.run_eval --config experiments/configs/amazon_compound.yaml
"""

import argparse
import json
import pickle
import sys
import time
import numpy as np
from pathlib import Path

from openai import OpenAI

from hc_rag.llm.llm_io import format_docs_v2, sanitize_text, load_prompt, render_prompt
from hc_rag.evaluation.e2e_evaluator import E2EEvaluator

from experiments.lib.config import load_config
from experiments.lib.registry import get_adapter
from experiments.lib.retry import call_with_retry
from experiments.lib.results_io import save_results_json, analyze_by_bucket

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def qa_call(client, qa_model, qa_prompt, qa_schema, query_text, docs_str, title_truncate):
    """Call QA model and parse structured response."""
    system_msg = qa_prompt["system"].strip()
    user_msg = render_prompt(qa_prompt["user"], query=query_text, documents=docs_str)

    response = client.chat.completions.create(
        model=qa_model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        response_format=qa_schema,
        temperature=0.0,
    )

    raw = json.loads(response.choices[0].message.content)
    titles = raw.get("titles", [])
    titles = [t[:title_truncate] for t in titles]
    return ", ".join(titles) if titles else "No matching products found."


def judge_call(client, judge_model, judge_prompt, judge_schema,
               query_text, gold_answer, docs_str, generated_answer):
    """Call judge model and parse structured response."""
    system_msg = judge_prompt["system"].strip()
    user_msg = render_prompt(
        judge_prompt["user"],
        query=query_text,
        ground_truth_answer=gold_answer,
        retrieved_docs=docs_str,
        generated_answer=generated_answer,
    )

    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        response_format=judge_schema,
        temperature=0.0,
    )

    raw = json.loads(response.choices[0].message.content)
    return {
        "context_precision": raw["context_precision"]["rating"],
        "context_recall": raw["context_recall"]["rating"],
        "answer_faithfulness": raw["answer_faithfulness"]["rating"],
        "answer_correctness": raw["answer_correctness"]["rating"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run E2E RAG evaluation")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--force", action="store_true", help="Rerun even if output exists")
    args = parser.parse_args()

    config = load_config(args.config)
    import experiments.datasets  # noqa: F401
    adapter = get_adapter(config.dataset.name)

    e2e_dir = Path(config.dataset.results_dir) / "e2e"
    e2e_dir.mkdir(parents=True, exist_ok=True)
    retrieval_dir = Path(config.dataset.results_dir) / "retrieval"

    queries, qrels, corpus = adapter.load_dataset(config.dataset.data_dir)
    query_map = {q.query_id: q for q in queries}

    prompts_dir = str(Path(__file__).parent.parent / "prompts")
    qa_prompt = load_prompt(config.eval.qa_prompt, prompts_dir)
    judge_prompt = load_prompt(config.eval.judge_prompt, prompts_dir)
    qa_schema = qa_prompt.get("schema")
    judge_schema = judge_prompt.get("schema")

    client = OpenAI()
    evaluator = E2EEvaluator()

    for method in config.methods:
        result_path = e2e_dir / f"{method.name}_e2e.json"
        if result_path.exists() and not args.force:
            print(f"Skipping {method.name} — results exist. Use --force to rerun.")
            continue

        outputs_path = retrieval_dir / f"{method.name}_outputs.pkl"
        if not outputs_path.exists():
            print(f"ERROR: No retrieval outputs for {method.name}. Run run_retrieval.py first.")
            continue

        with open(outputs_path, "rb") as f:
            retrieval_outputs = pickle.load(f)

        output_map = {ro.query_id: ro for ro in retrieval_outputs}

        print(f"\n{'=' * 60}")
        print(f"E2E Evaluation: {method.name}")
        print(f"{'=' * 60}")

        all_results = []
        for i, query in enumerate(queries):
            ro = output_map.get(query.query_id)
            if ro is None:
                continue

            doc_texts = []
            for doc_id in ro.retrieved_ids:
                doc = corpus.get(doc_id)
                if doc and "text" in doc:
                    text = str(doc["text"])
                    title = text.split(". ")[0][:config.eval.title_truncate]
                    doc_texts.append(sanitize_text(title))
                else:
                    doc_texts.append(f"[Document {doc_id} not found]")

            docs_str = format_docs_v2(doc_texts)

            generated_answer = call_with_retry(
                qa_call, client, config.eval.qa_model, qa_prompt, qa_schema,
                query.text, docs_str, config.eval.title_truncate,
            )
            time.sleep(config.eval.delay)

            gold_answers = adapter.get_gold_answers(query, corpus, qrels)
            gold_answer = ", ".join(gold_answers)

            judge_scores = call_with_retry(
                judge_call, client, config.eval.judge_model, judge_prompt, judge_schema,
                query.text, gold_answer, docs_str, generated_answer,
            )
            time.sleep(config.eval.delay)

            _, _, token_f1 = E2EEvaluator.compute_token_f1(generated_answer, gold_answer)
            relevant_ids = qrels.get(query.query_id, set())

            result = {
                "query_id": query.query_id,
                "k": ro.k,
                "generated_answer": generated_answer,
                "gold_answer": gold_answer,
                "judge_scores": judge_scores,
                "token_f1": token_f1,
                "recall": len(set(ro.retrieved_ids) & relevant_ids) / max(len(relevant_ids), 1),
                "metadata": adapter.get_result_metadata(query),
            }
            all_results.append(result)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(queries)} queries")

        aggregate = {
            "method": method.name,
            "n_queries": len(all_results),
            "mean_context_precision": float(np.mean([r["judge_scores"]["context_precision"] for r in all_results])),
            "mean_context_recall": float(np.mean([r["judge_scores"]["context_recall"] for r in all_results])),
            "mean_answer_faithfulness": float(np.mean([r["judge_scores"]["answer_faithfulness"] for r in all_results])),
            "mean_answer_correctness": float(np.mean([r["judge_scores"]["answer_correctness"] for r in all_results])),
            "mean_token_f1": float(np.mean([r["token_f1"] for r in all_results])),
        }

        bucket_analysis = analyze_by_bucket(all_results, adapter.get_bucket_field())

        save_results_json(
            {"aggregate": aggregate, "per_query": all_results, "by_bucket": bucket_analysis},
            str(result_path),
        )

        print(f"\n  Context Precision: {aggregate['mean_context_precision']:.2f}")
        print(f"  Context Recall: {aggregate['mean_context_recall']:.2f}")
        print(f"  Answer Correctness: {aggregate['mean_answer_correctness']:.2f}")
        print(f"  Token F1: {aggregate['mean_token_f1']:.4f}")
        print(f"  Saved to {result_path}")


if __name__ == "__main__":
    main()
