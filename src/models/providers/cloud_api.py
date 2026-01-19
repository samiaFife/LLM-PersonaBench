import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.models.base import BaseLLM

load_dotenv()

class CloudAPIModel(BaseLLM):
    """
    Wrapper для Cloud API 
    """
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        timeout: float | None = None,
        max_retries: int | None = None,
    ):
        api_key = os.environ["CLOUD_API_KEY"]
        base_url = "https://foundation-models.api.cloud.ru/v1"

        super().__init__(model_name)
        # timeout/request_timeout: если не задан — может ждать бесконечно (и "висеть" десятки минут)
        # max_retries: полезно при временных сетевых/429 ошибках
        llm_kwargs: dict = {
            "model": model_name,
            "api_key": api_key,
            "base_url": base_url,
            "temperature": temperature,
        }
        if timeout is not None:
            llm_kwargs["timeout"] = timeout
        if max_retries is not None:
            llm_kwargs["max_retries"] = max_retries

        self.llm = ChatOpenAI(**llm_kwargs)

    def generate(self, prompt: str) -> str:
        response = prompt | self.llm
        return response.invoke({})

    def generate_batch(self, prompts: list[str]) -> list[str]:
        return [self.generate(p) for p in prompts]
