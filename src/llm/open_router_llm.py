import os
from dotenv import load_dotenv
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError
from base_llm import BaseLLM

load_dotenv()

# TODO: Check tokens limit per model
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL="https://openrouter.ai/api/v1"

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY environment variable is not set. "
        "Please add it to your .env file or set it in your environment."
    )

# Models:
# deepseek/deepseek-chat-v3.1:free
# deepseek/deepseek-r1:free
# openai/gpt-oss-20b:free
# openai/gpt-5-nano

class OpenRouterLLM(BaseLLM):
    """
    OpenRouter LLM implementation.
    """

    def init_client(self):
        """Initialize the OpenRouter client."""
        try:
            return OpenAI(api_key=OPENROUTER_API_KEY, base_url=BASE_URL)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize OpenRouter client: {e}")
            raise
    
    async def call_model(self, user_prompt: str) -> str:
        """Call the OpenRouter model with system and user prompts and return the raw output."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            if not response.choices or response.choices[0].message.content is None:
                raise ValueError("API returned invalid or empty response")
            
            return response.choices[0].message.content
            
        except (AuthenticationError, RateLimitError, APIConnectionError, APIError) as e:
            if self.logger:
                self.logger.error(f"OpenRouter API error: {e}")
            raise
        
        except Exception as e:
            error_msg = f"Error calling model {self.model_name}: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    # TODO: Implement format_output method if needed
    def format_output(self, raw_output: str) -> str:
        """Format the raw output from the LLM into a clean answer."""
        return raw_output
