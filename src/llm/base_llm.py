from logging import Logger
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from src.llm.llm_io import LLMRAGInput, format_docs, clean_output
from .prompts.qa_base_prompt import qa_system_prompt, qa_user_prompt


class BaseLLM(ABC):
    """
    Abstract LLM class defining the interface for all LLM backends.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 100000,
        system_prompt: str = None,
        user_prompt: str = None,
        logger: Optional[Logger] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or qa_system_prompt
        self.user_prompt = user_prompt or qa_user_prompt
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
        """
        Build the prompt to be sent to the LLM.
        Basic implementation using system and user prompts.
        """
        system_part = self.system_prompt + "\n\n"
        user_part = self.user_prompt.format(query=query, documents=documents_str)
        full_prompt = system_part + user_part
        return full_prompt

    async def generate_answer(
        self,
        llm_rag_input: LLMRAGInput
    ) -> str:
        """
        Generate an answer from the LLM based on the given input.
        """
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

    async def batch_answers(self, llm_rag_inputs: List[LLMRAGInput]) -> List[str]:
        answers = []
        for llm_rag_input in llm_rag_inputs:
            answer = await self.generate_answer(llm_rag_input)
            answers.append(answer)
        return answers