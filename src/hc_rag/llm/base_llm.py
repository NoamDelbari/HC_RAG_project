import json
import re
from logging import Logger
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from hc_rag.llm.llm_io import LLMRAGInput, format_docs, clean_output


class BaseLLM(ABC):
    """
    Abstract LLM class defining the interface for all LLM backends.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 100000,
        qa_mode: bool = True,
        combine_system_user_prompt: bool = False,
        system_prompt: str = None,
        user_prompt: str = None,
        logger: Optional[Logger] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.qa_mode = qa_mode
        self.combine_system_user_prompt = combine_system_user_prompt
        if system_prompt is None or user_prompt is None:
            raise ValueError(
                "system_prompt and user_prompt are required. "
                "Use load_prompt() from hc_rag.llm.llm_io to load prompt templates."
            )
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.logger = logger
        self.client = self.init_client()

    @abstractmethod
    def init_client(self):
        """Initialize the LLM client."""
        pass

    @abstractmethod
    def call_model(self, prompt: str) -> str:
        """Call the LLM model with the given prompt and return the raw output."""
        pass

    @abstractmethod
    def format_output(self, raw_output: str) -> str:
        """Format the raw output from the LLM into a clean answer."""
        return raw_output

    def log(self, message: str):
        if self.logger:
            self.logger.info(message)

    def build_prompt(self, query: str, documents_str: str) -> str:
        """Build the prompt to be sent to the LLM."""
        user_part = self.user_prompt.format(query=query, documents=documents_str)
        if self.combine_system_user_prompt:
            full_prompt = self.system_prompt + "\n" + user_part
            return full_prompt
        else:
            return user_part

    def generate_answer(
        self,
        llm_rag_input: LLMRAGInput
    ) -> str:
        """Generate an answer from the LLM based on the given input."""
        if not self.qa_mode:
            raise ValueError("generate_answer is only available in QA mode.")
        try:
            query = llm_rag_input.query
            self.log(f"Generating answer for query: {query}")
            docs = format_docs(llm_rag_input, self.max_tokens)
            self.log(f"Formatted documents: {docs}")
            prompt = self.build_prompt(query, docs)
            self.log(f"Built prompt: {prompt}")
            raw_output = self.call_model(prompt)
            self.log(f"Raw model output: {raw_output}")
            output = self.format_output(raw_output)
            self.log(f"Formatted output: {output}")
            output = clean_output(output)
            self.log(f"Cleaned output: {output}")
            return output
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating answer: {e}")
            raise

    def batch_answers(self, llm_rag_inputs: List[LLMRAGInput]) -> List[str]:
        answers = []
        for llm_rag_input in llm_rag_inputs:
            answer = self.generate_answer(llm_rag_input)
            answers.append(answer)
        return answers

    def judge_answer(
        self,
        query: str,
        ground_truth_docs: str,
        ground_truth_answer: str,
        retrieved_docs: str,
        generated_answer: str,
    ) -> Dict[str, dict]:
        """
        Judge the quality of a RAG system's response.

        Returns dict with 4 metrics: context_precision, context_recall,
        answer_faithfulness, answer_correctness. Each has 'rating' (1-5) and 'reasoning'.
        """
        if self.qa_mode:
            raise ValueError("judge_answer is only available in Judge mode.")

        # Build the judge prompt using instance templates
        user_prompt = self.user_prompt.format(
            query=query,
            ground_truth_docs=ground_truth_docs,
            ground_truth_answer=ground_truth_answer,
            retrieved_docs=retrieved_docs,
            generated_answer=generated_answer,
        )

        # Build messages based on combine mode
        if self.combine_system_user_prompt:
            prompt = self.system_prompt + "\n" + user_prompt
        else:
            prompt = user_prompt

        raw_output = self.call_model(prompt)
        self.log(f"Judge raw output: {raw_output}")

        return self._parse_judge_response(raw_output)

    def _parse_judge_response(self, raw_output: str) -> Dict[str, dict]:
        """Parse judge LLM response into structured metrics dict."""
        # Try direct JSON parse first
        try:
            parsed = json.loads(raw_output)
            return self._validate_judge_scores(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: extract JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                return self._validate_judge_scores(parsed)
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: find first { ... } block
        brace_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if brace_match:
            try:
                parsed = json.loads(brace_match.group(0))
                return self._validate_judge_scores(parsed)
            except (json.JSONDecodeError, ValueError):
                pass

        # All parsing failed — return default scores
        self.log(f"WARNING: Could not parse judge response, returning defaults")
        return self._default_judge_scores()

    def _validate_judge_scores(self, parsed: dict) -> Dict[str, dict]:
        """Validate and normalize parsed judge scores."""
        expected_keys = ["context_precision", "context_recall", "answer_faithfulness", "answer_correctness"]
        result = {}
        for key in expected_keys:
            if key in parsed and isinstance(parsed[key], dict) and "rating" in parsed[key]:
                rating = parsed[key]["rating"]
                if isinstance(rating, (int, float)) and 1 <= rating <= 5:
                    result[key] = {
                        "rating": int(rating),
                        "reasoning": parsed[key].get("reasoning", ""),
                    }
                else:
                    result[key] = {"rating": 3, "reasoning": "Invalid rating value"}
            else:
                result[key] = {"rating": 3, "reasoning": "Missing from response"}
        return result

    def _default_judge_scores(self) -> Dict[str, dict]:
        """Return default scores when parsing fails."""
        return {
            "context_precision": {"rating": 3, "reasoning": "Parse error"},
            "context_recall": {"rating": 3, "reasoning": "Parse error"},
            "answer_faithfulness": {"rating": 3, "reasoning": "Parse error"},
            "answer_correctness": {"rating": 3, "reasoning": "Parse error"},
        }
