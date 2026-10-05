from typing import Protocol, Any

from ollama import Client


class LLMClient(Protocol):
    """Interface implemented by any LLM provider."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any] | None = None,
    ) -> str:
        ...


class OllamaLLMClient:
    """LLM client that communicates with a local Ollama server."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.client = Client(host=host)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any] | None = None,
    ) -> str:

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            format = output_schema,
            options={
                "temperature": 0
            },
        )

        return response["message"]["content"]