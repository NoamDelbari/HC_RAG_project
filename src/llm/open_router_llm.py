import os
from dotenv import load_dotenv
from openai import OpenAI
from base_llm import BaseLLM

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL="https://openrouter.ai/api/v1"

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
        return OpenAI(api_key=OPENROUTER_API_KEY, base_url=BASE_URL)
    
    async def call_model(self, prompt: str) -> str:
        """Call the OpenRouter model with the given prompt and return the raw output."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message['content']