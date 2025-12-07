"""
OpenAI API client for image captioning.
"""

import logging
from typing import Any

from openai import AsyncOpenAI, RateLimitError as OpenAIRateLimitError

from kohakucaption.clients.base import ClientConfig, MLLMClient, RateLimitError, StatsLogger

logger = logging.getLogger(__name__)


class OpenAIClient(MLLMClient):
    """
    Client for OpenAI's Chat Completions API with vision support.

    Uses the official OpenAI SDK for reliable API access.

    Supports:
    - GPT-4o, GPT-4o-mini, GPT-4-turbo, and other vision-capable models
    - Base64-encoded images and image URLs
    - Structured outputs via response_format

    Example:
        config = ClientConfig(
            api_key="sk-...",
            model="gpt-4o",
        )
        client = OpenAIClient(config)
        result = await client.caption(image, "Describe this image")
    """

    def __init__(
        self,
        config: ClientConfig,
        stats_logger: StatsLogger | None = None,
    ):
        super().__init__(config, stats_logger)
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.close()

    async def _send_request(
        self,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> str:
        """Send a request to the OpenAI API."""
        # Build request parameters
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        # Add response_format if specified (for structured outputs)
        if "response_format" in kwargs:
            params["response_format"] = kwargs["response_format"]

        # Add any additional parameters
        for key in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
            if key in kwargs:
                params[key] = kwargs[key]

        logger.debug(f"Sending request with model {self.config.model}")

        try:
            response = await self._client.chat.completions.create(**params)
        except OpenAIRateLimitError as e:
            # Convert to our RateLimitError for special handling
            raise RateLimitError(str(e)) from e

        # Extract content from response
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("Response content is None")
        return content

    async def caption_with_schema(
        self,
        image: "ImageInput",
        prompt: str,
        json_schema: dict[str, Any],
        system_prompt: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> "CaptionResult[str]":
        """
        Generate a caption with structured output using JSON schema.

        This uses OpenAI's structured outputs feature to guarantee
        the response matches the provided schema.

        Args:
            image: Image to caption
            prompt: User prompt
            json_schema: JSON schema for the response
            system_prompt: Optional system prompt
            strict: Use strict mode for schema validation
            **kwargs: Additional API parameters

        Returns:
            CaptionResult with structured response
        """
        from kohakucaption.types import CaptionResult, ImageInput

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "caption_response",
                "strict": strict,
                "schema": json_schema,
            },
        }

        return await self.caption(
            image=image,
            prompt=prompt,
            system_prompt=system_prompt,
            response_format=response_format,
            **kwargs,
        )

    async def __aenter__(self) -> "OpenAIClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
