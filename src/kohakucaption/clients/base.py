"""
Base MLLM client with retry logic and monitoring.
"""

import asyncio
import json
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kohakucaption.types import (
    AggregateStats,
    CaptionResult,
    ImageInput,
    RequestStats,
)

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when API returns 429 rate limit error."""

    pass


@dataclass
class ClientConfig:
    """Configuration for MLLM clients."""

    api_key: str
    base_url: str | None = None
    model: str = "gpt-4o"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_multiplier: float = 2.0
    retry_max_delay: float = 30.0
    detail: str = "auto"  # "low", "high", or "auto" for vision


@dataclass
class RetryState:
    """Tracks retry state for a request."""

    attempt: int = 0
    last_error: str | None = None
    delay: float = 1.0


class StatsLogger:
    """Logs request statistics to file and memory."""

    def __init__(
        self,
        log_file: str | Path | None = None,
        enable_file_logging: bool = True,
    ):
        self.stats = AggregateStats()
        self.request_history: list[RequestStats] = []
        self.log_file = Path(log_file) if log_file else None
        self.enable_file_logging = enable_file_logging and log_file is not None

        if self.enable_file_logging and self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def record(self, stats: RequestStats) -> None:
        """Record a request's statistics."""
        self.stats.record(stats)
        self.request_history.append(stats)

        if self.enable_file_logging and self.log_file:
            self._write_to_file(stats)

    def _write_to_file(self, stats: RequestStats) -> None:
        """Write stats to log file."""
        entry = {
            "request_id": stats.request_id,
            "model": stats.model,
            "success": stats.success,
            "latency_ms": stats.latency_ms,
            "retries": stats.retries,
            "error": stats.error,
            "timestamp": stats.timestamp.isoformat(),
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all statistics."""
        return {
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "success_rate": f"{self.stats.success_rate:.1%}",
            "failure_rate": f"{self.stats.failure_rate:.1%}",
            "average_latency_ms": f"{self.stats.average_latency_ms:.1f}",
            "average_retries": f"{self.stats.average_retries:.2f}",
            "total_retries": self.stats.total_retries,
            "error_counts": self.stats.errors,
        }


class MLLMClient(ABC):
    """
    Abstract base class for MLLM API clients.
    Provides retry logic, monitoring, and common utilities.
    """

    def __init__(
        self,
        config: ClientConfig,
        stats_logger: StatsLogger | None = None,
    ):
        self.config = config
        self.stats_logger = stats_logger or StatsLogger(enable_file_logging=False)

    @abstractmethod
    async def _send_request(
        self,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> str:
        """
        Send a request to the API. Must be implemented by subclasses.

        Args:
            messages: List of message dicts in OpenAI format
            **kwargs: Additional API-specific parameters

        Returns:
            Raw response content string

        Raises:
            Exception: On API errors
        """
        pass

    def _build_image_content(self, image: ImageInput) -> dict[str, Any]:
        """Build the image content part of a message."""
        if image.is_url():
            url = str(image.source)
        else:
            url = image.to_base64()

        return {
            "type": "image_url",
            "image_url": {
                "url": url,
                "detail": self.config.detail,
            },
        }

    def _build_messages(
        self,
        prompt: str,
        image: ImageInput,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the messages list for the API request."""
        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    self._build_image_content(image),
                ],
            }
        )

        return messages

    async def _execute_with_retry(
        self,
        messages: list[dict[str, Any]],
        max_retries: int | None = None,
        **kwargs,
    ) -> tuple[str, int]:
        """
        Execute request with retry logic.

        Rate limit errors (429) are retried infinitely with exponential backoff.
        Other errors respect max_retries limit.

        Returns:
            Tuple of (response_content, retries_used)
        """
        max_retries = (
            max_retries if max_retries is not None else self.config.max_retries
        )
        state = RetryState(delay=self.config.retry_delay)
        rate_limit_retries = 0

        while state.attempt <= max_retries:
            try:
                response = await self._send_request(messages, **kwargs)
                return response, state.attempt + rate_limit_retries
            except RateLimitError as e:
                # 429 error: infinite retry with exponential backoff
                rate_limit_retries += 1
                # Add jitter to prevent thundering herd
                jitter = random.uniform(0.5, 1.5)
                wait_time = min(state.delay * jitter, self.config.retry_max_delay)
                logger.warning(
                    f"Rate limited (429). Waiting {wait_time:.1f}s before retry #{rate_limit_retries}..."
                )
                await asyncio.sleep(wait_time)
                # Exponential backoff for rate limit
                state.delay = min(
                    state.delay * self.config.retry_multiplier,
                    self.config.retry_max_delay,
                )
                # Don't increment attempt count for rate limit - infinite retry
                continue
            except Exception as e:
                state.last_error = str(e)
                state.attempt += 1

                if state.attempt > max_retries:
                    raise

                # Exponential backoff
                logger.warning(
                    f"Request failed (attempt {state.attempt}/{max_retries + 1}): {e}. "
                    f"Retrying in {state.delay:.1f}s..."
                )
                await asyncio.sleep(state.delay)
                state.delay = min(
                    state.delay * self.config.retry_multiplier,
                    self.config.retry_max_delay,
                )

        raise RuntimeError(f"Max retries exceeded. Last error: {state.last_error}")

    async def caption(
        self,
        image: ImageInput,
        prompt: str,
        system_prompt: str | None = None,
        max_retries: int | None = None,
        **kwargs,
    ) -> CaptionResult[str]:
        """
        Generate a caption for an image.

        Args:
            image: Image to caption
            prompt: User prompt for captioning
            system_prompt: Optional system prompt
            max_retries: Override default max retries
            **kwargs: Additional API parameters

        Returns:
            CaptionResult with raw response string
        """
        request_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        try:
            messages = self._build_messages(prompt, image, system_prompt)
            response, retries = await self._execute_with_retry(
                messages,
                max_retries=max_retries,
                **kwargs,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Record stats
            stats = RequestStats(
                request_id=request_id,
                model=self.config.model,
                success=True,
                latency_ms=latency_ms,
                retries=retries,
            )
            self.stats_logger.record(stats)

            return CaptionResult(
                success=True,
                content=response,
                raw_response=response,
                retries_used=retries,
                latency_ms=latency_ms,
                model=self.config.model,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            error_msg = str(e)

            stats = RequestStats(
                request_id=request_id,
                model=self.config.model,
                success=False,
                latency_ms=latency_ms,
                retries=max_retries or self.config.max_retries,
                error=error_msg,
            )
            self.stats_logger.record(stats)

            logger.error(f"Caption request failed: {error_msg}")

            return CaptionResult(
                success=False,
                error=error_msg,
                retries_used=max_retries or self.config.max_retries,
                latency_ms=latency_ms,
                model=self.config.model,
            )

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics summary."""
        return self.stats_logger.get_summary()
