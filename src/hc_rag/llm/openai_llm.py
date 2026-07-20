import json
import os
from dotenv import load_dotenv
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError
from .base_llm import BaseLLM
from .llm_io import LLMRAGInput, format_docs, clean_output, TITLE_TRUNCATE


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM implementation using the official OpenAI API.
    Supports structured output mode for extracting product titles as JSON.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        structured_output: bool = False,
        response_format: dict = None,
        **kwargs,
    ):
        self._skip_temperature = False
        self.structured_output = structured_output
        self.response_format = response_format
        super().__init__(model_name=model_name, **kwargs)

    def init_client(self):
        """Initialize the OpenAI client."""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please add it to your .env file or set it in your environment."
            )
        try:
            return OpenAI(api_key=api_key)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def call_model(self, user_prompt: str) -> str:
        """Call the OpenAI model and return the raw output."""
        try:
            kwargs = dict(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            if not self._skip_temperature:
                kwargs["temperature"] = self.temperature

            if self.structured_output and self.response_format:
                kwargs["response_format"] = self.response_format

            response = self.client.chat.completions.create(**kwargs)

            if not response.choices or response.choices[0].message.content is None:
                raise ValueError("API returned invalid or empty response")

            return response.choices[0].message.content

        except (AuthenticationError, RateLimitError, APIConnectionError, APIError) as e:
            # Auto-retry without temperature if the model doesn't support it
            if not self._skip_temperature and "temperature" in str(e).lower():
                self._skip_temperature = True
                return self.call_model(user_prompt)
            if self.logger:
                self.logger.error(f"OpenAI API error: {e}")
            raise

        except Exception as e:
            error_msg = f"Error calling model {self.model_name}: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def format_output(self, raw_output: str) -> str:
        """Format the raw output from the LLM into a clean answer.

        In structured QA mode, parses JSON and returns sorted, truncated,
        comma-separated titles matching the gold answer format.
        """
        if not self.structured_output or not self.qa_mode:
            return raw_output

        try:
            parsed = json.loads(raw_output)
            titles = parsed.get("titles", [])
            truncated = sorted(t[:TITLE_TRUNCATE].strip() for t in titles if t.strip())
            return ", ".join(truncated) if truncated else "No products found."
        except (json.JSONDecodeError, KeyError, TypeError):
            self.log(f"WARNING: Failed to parse structured output: {raw_output[:200]}")
            return raw_output
