from logging import Logger
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from src.llm.llm_io import LLMRAGInput, format_docs, clean_output
from .prompts.qa_prompts import QA_SYSTEM_PROMPT, QA_USER_PROMPT
from .prompts.judge_prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT, JUDGE_INSTRUCTIONS_PROMPT


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
        self.system_prompt = system_prompt or (QA_SYSTEM_PROMPT if qa_mode else JUDGE_SYSTEM_PROMPT)
        self.user_prompt = user_prompt or (QA_USER_PROMPT if qa_mode else JUDGE_USER_PROMPT)
        self.logger = logger
        self.client = self.init_client()

    @abstractmethod
    def init_client(self):
        """Initialize the LLM client."""
        pass

    @abstractmethod
    async def call_model(self, prompt: str) -> str:
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

    async def generate_answer(
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
            raw_output = await self.call_model(prompt)
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

    # TODO: Implement batch processing if supported by the LLM backend, or parallelize calls
    async def batch_answers(self, llm_rag_inputs: List[LLMRAGInput]) -> List[str]:
        answers = []
        for llm_rag_input in llm_rag_inputs:
            answer = await self.generate_answer(llm_rag_input)
            answers.append(answer)
        return answers
    
    # TODO: Implement evaluation output class
    # TODO: Define metrics for accuracy and hallucination
    # TODO: Define LLM as a judge prompts
    async def judge_answer(self, llm_rag_input: LLMRAGInput, generated_answer: str, reference_answer: str) -> bool:
        """Judge if the generated answer matches the reference answer, calculating metrics of accuracy and hallucination"""
        # Simple string comparison; can be enhanced with more sophisticated methods
        if self.qa_mode:
            raise ValueError("judge_answer is only available in Judge mode.")
        return generated_answer.strip().lower() == reference_answer.strip().lower()