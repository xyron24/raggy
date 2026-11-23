"""OpenAI chat completion service with an injectable client."""

from typing import Any

import openai


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
