"""OpenAI chat completion service with an injectable client."""

from typing import Any, TypeVar

import openai
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ChatService:
    """Sends system/user prompts to the chat model.

    Temperature defaults to 0: grounded answers should be deterministic,
    not creative.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for chat")
        self.model = model
        self.temperature = temperature
        self._client = client or openai.OpenAI(api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        """Return the assistant message for a system/user prompt pair."""
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    def complete_structured(
        self, system: str, user: str, response_model: type[T]
    ) -> T:
        """Return a Pydantic-validated structured output for the prompt.

        Uses the SDK's `parse` helper, which constrains the model to the
        schema instead of hoping free-form JSON parses.
        """
        response = self._client.chat.completions.parse(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=response_model,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("structured output parse returned nothing")
        return parsed
