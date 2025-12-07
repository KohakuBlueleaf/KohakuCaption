"""
OpenRouter API client for image captioning.
Uses OpenAI SDK with OpenRouter endpoint.
"""

import logging
from typing import Any

from openai import AsyncOpenAI, RateLimitError as OpenAIRateLimitError

from kohakucaption.clients.base import ClientConfig, MLLMClient, RateLimitError, StatsLogger

logger = logging.getLogger(__name__)

# OpenRouter API base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(MLLMClient):
    """
    Client for OpenRouter's API using OpenAI SDK.

    OpenRouter provides access to multiple model providers through a
    unified OpenAI-compatible API. Model names follow the format:
    "provider/model-name" (e.g., "openai/gpt-4o", "anthropic/claude-sonnet-4").

    The client automatically sets the base_url to OpenRouter's endpoint,
    so you only need to provide your OpenRouter API key in config.api_key.

    Example:
        config = ClientConfig(
            api_key="sk-or-...",  # Your OpenRouter key, NOT OpenAI key
            model="openai/gpt-4o",
        )
        client = OpenRouterClient(config)
        result = await client.caption(image, "Describe this image")
    """

    def __init__(
        self,
        config: ClientConfig,
        stats_logger: StatsLogger | None = None,
        site_url: str | None = None,
        site_name: str | None = None,
    ):
        """
        Initialize OpenRouter client.

        Args:
            config: Client configuration (api_key should be OpenRouter key)
            stats_logger: Optional stats logger
            site_url: Your site URL for OpenRouter rankings
            site_name: Your site name for OpenRouter rankings
        """
        super().__init__(config, stats_logger)
        self.site_url = site_url
        self.site_name = site_name

        # Build default headers for OpenRouter
        default_headers = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if site_name:
            default_headers["X-Title"] = site_name

        # Use OpenRouter base URL, override if user specified custom one
        base_url = config.base_url or OPENROUTER_BASE_URL

        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=base_url,
            timeout=config.timeout,
            default_headers=default_headers if default_headers else None,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.close()

    async def _send_request(
        self,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> str:
        """Send a request to the OpenRouter API."""
        # Build request parameters
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        # Add response_format if specified
        if "response_format" in kwargs:
            params["response_format"] = kwargs["response_format"]

        # Add any additional parameters
        for key in ["top_p", "frequency_penalty", "presence_penalty", "stop", "top_k"]:
            if key in kwargs:
                params[key] = kwargs[key]

        # OpenRouter-specific: provider routing
        extra_body = {}
        if "provider" in kwargs:
            extra_body["provider"] = kwargs["provider"]
        if "transforms" in kwargs:
            extra_body["transforms"] = kwargs["transforms"]
        if extra_body:
            params["extra_body"] = extra_body

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

    async def __aenter__(self) -> "OpenRouterClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
