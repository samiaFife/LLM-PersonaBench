import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.messages import HumanMessage
from src.models.base import BaseLLM

load_dotenv()


class OpenRouterAPIModel(BaseLLM):
    """
    Wrapper для OpenRouter API с поддержкой rate limiting.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        timeout: float | None = None,
        max_retries: int | None = None,
        max_completion_tokens: int = 4000,
        extra_body: dict | None = None,
        rate_limit_enabled: bool = True,
        requests_per_second: float = 1.0,
        max_bucket_size: int = 10,
    ):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment variables"
            )

        base_url = "https://openrouter.ai/api/v1"

        super().__init__(model_name)

        # Rate limiting
        rate_limiter = None
        if rate_limit_enabled:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=requests_per_second,
                check_every_n_seconds=0.1,
                max_bucket_size=max_bucket_size,
            )

        llm_kwargs: dict = {
            "model": model_name,
            "api_key": api_key,
            "base_url": base_url,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        }

        if timeout is not None:
            llm_kwargs["timeout"] = timeout
        if max_retries is not None:
            llm_kwargs["max_retries"] = max_retries
        if rate_limiter is not None:
            llm_kwargs["rate_limiter"] = rate_limiter
        if extra_body is not None:
            llm_kwargs["extra_body"] = extra_body

        self.llm = ChatOpenAI(**llm_kwargs)

    def generate(self, prompt) -> str:
        """Генерирует ответ на промпт (строка или ChatPromptTemplate)."""
        # Если это LangChain объект (ChatPromptTemplate и т.д.)
        if hasattr(prompt, "invoke"):
            chain = prompt | self.llm
            result = chain.invoke({})
        else:
            # Если это строка - оборачиваем в HumanMessage
            result = self.llm.invoke([HumanMessage(content=str(prompt))])

        return result.content if hasattr(result, "content") else str(result)

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Генерирует ответы на несколько промптов."""
        return [self.generate(p) for p in prompts]
