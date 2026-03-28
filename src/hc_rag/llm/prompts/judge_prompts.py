# TODO: Refine prompts for better accuracy

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems. 
Your role is to be an impartial judge, assessing the quality of both the retrieval and generation components based on the provided materials. 
You must follow the instructions precisely and provide your evaluation in the specified JSON format."""

JUDGE_USER_PROMPT = """Please evaluate the RAG system's performance on the following query.

**The User's Query:**
{query}

**Ground Truth Documents (Containing the correct information):**
{ground_truth_docs}

**Ground Truth Answer:**
{ground_truth_answer}

**Retrieved Documents (Provided to the RAG model):**
{retrieved_docs}

**Generated Answer (From the RAG model):**
{generated_answer}

Now, please provide your evaluation based on the instructions in the JSON format specified.
"""

JUDGE_INSTRUCTIONS_PROMPT = """
**Evaluation Instructions:**

You must evaluate the RAG system on four key metrics: Context Precision, Context Recall, Answer Faithfulness, and Answer Correctness.
Provide a rating on a scale of 1 to 5 (where 1 is worst and 5 is best) and a brief reasoning for each.

1.  **Context Precision (Rating: 1-5):**
    - Are the `Retrieved Documents` relevant to the `User's Query`?
    - 5: All retrieved documents are highly relevant and essential.
    - 1: None of the retrieved documents are relevant.

2.  **Context Recall (Rating: 1-5):**
    - Do the `Retrieved Documents` contain all the necessary information to answer the query, when compared to the `Ground Truth Documents`?
    - 5: The retrieved documents contain all the necessary information.
    - 1: The retrieved documents are missing crucial information.

3.  **Answer Faithfulness (Rating: 1-5):**
    - Is the `Generated Answer` grounded in the `Retrieved Documents`? The answer should not contain information that is not present in the retrieved documents.
    - 5: The answer is fully supported by the retrieved documents.
    - 1: The answer contains significant hallucinations or information not from the retrieved documents.

4.  **Answer Correctness (Rating: 1-5):**
    - Is the `Generated Answer` correct and accurate when compared to the `Ground Truth Answer`?
    - 5: The answer is completely correct and comprehensive.
    - 1: The answer is completely incorrect.

**Output Format:**
Provide your response as a single JSON object with the following structure:
```json
{
  "context_precision": {
    "rating": <integer>,
    "reasoning": "<string>"
  },
  "context_recall": {
    "rating": <integer>,
    "reasoning": "<string>"
  },
  "answer_faithfulness": {
    "rating": <integer>,
    "reasoning": "<string>"
  },
  "answer_correctness": {
    "rating": <integer>,
    "reasoning": "<string>"
  }
}
```
"""

# ── V2 Judge Prompts (with structured output) ────────────────

V2_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.
Evaluate retrieval and answer quality precisely. Be strict and consistent."""

V2_JUDGE_USER_PROMPT = """Evaluate the RAG system's performance.

**Query:** {query}

**Ground Truth Products (correct answers):**
{ground_truth_answer}

**Retrieved Products (given to the system):**
{retrieved_docs}

**System's Answer:**
{generated_answer}

Evaluate on these 4 metrics (1-5 scale):

1. **context_precision**: What fraction of retrieved products are relevant to the query?
   5=all relevant, 3=about half, 1=none relevant

2. **context_recall**: What fraction of ground truth products appear in the retrieved set?
   5=all found, 3=about half found, 1=none found

3. **answer_faithfulness**: Is the answer based only on the retrieved products (no hallucination)?
   5=fully grounded, 3=some unsupported claims, 1=mostly hallucinated

4. **answer_correctness**: Does the answer match the ground truth?
   5=perfect match, 3=partially correct, 1=completely wrong"""

# JSON schema for judge structured output
JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "context_precision": {
                    "type": "object",
                    "properties": {
                        "rating": {"type": "integer", "description": "1-5 rating"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["rating", "reasoning"],
                    "additionalProperties": False,
                },
                "context_recall": {
                    "type": "object",
                    "properties": {
                        "rating": {"type": "integer", "description": "1-5 rating"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["rating", "reasoning"],
                    "additionalProperties": False,
                },
                "answer_faithfulness": {
                    "type": "object",
                    "properties": {
                        "rating": {"type": "integer", "description": "1-5 rating"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["rating", "reasoning"],
                    "additionalProperties": False,
                },
                "answer_correctness": {
                    "type": "object",
                    "properties": {
                        "rating": {"type": "integer", "description": "1-5 rating"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["rating", "reasoning"],
                    "additionalProperties": False,
                },
            },
            "required": ["context_precision", "context_recall", "answer_faithfulness", "answer_correctness"],
            "additionalProperties": False,
        },
    },
}