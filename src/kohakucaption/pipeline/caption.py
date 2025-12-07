"""
Main captioning pipeline that orchestrates the entire process.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from kohakucaption.clients.base import MLLMClient
from kohakucaption.context.providers import format_all_context
from kohakucaption.context.template import TemplateEngine, DEFAULT_CAPTION_TEMPLATE
from kohakucaption.types import CaptionRequest, CaptionResult, ImageInput
from kohakucaption.validation.validators import (
    OutputValidator,
    PydanticValidator,
    ValidationResult,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class PipelineConfig:
    """Configuration for the captioning pipeline."""

    max_retries: int = 3
    validation_retries: int = 2  # Retries specifically for validation failures
    timeout: float = 60.0
    system_prompt: str | None = None
    include_schema_in_prompt: bool = True  # Include validation schema in prompt


class CaptionPipeline:
    """
    High-level pipeline for generating structured image captions.

    Combines:
    - Context formatting and template rendering
    - MLLM API calls with retry logic
    - Output validation with auto-retry on failure
    - Statistics tracking

    Example:
        client = OpenAIClient(config)
        pipeline = CaptionPipeline(client)

        # Simple caption
        result = await pipeline.caption(image, prompt)

        # Structured caption with Pydantic model
        result = await pipeline.caption_structured(
            image, prompt, BasicCaption
        )
    """

    def __init__(
        self,
        client: MLLMClient,
        config: PipelineConfig | None = None,
        template_engine: TemplateEngine | None = None,
    ):
        self.client = client
        self.config = config or PipelineConfig()
        self.template_engine = template_engine or TemplateEngine()

    def _build_prompt(
        self,
        request: CaptionRequest,
        validator: OutputValidator | None = None,
    ) -> str:
        """Build the final prompt from request and template."""
        # Format context items
        context_str = format_all_context(request.context_items)

        # Merge context dict with formatted context
        template_context = {
            **request.context_dict,
            "context": context_str if context_str else "",
        }

        # Add schema description if we have a validator
        if validator and self.config.include_schema_in_prompt:
            template_context["schema"] = validator.get_schema_description()

        # Use custom template or default
        if request.prompt_template:
            prompt = self.template_engine.render(
                request.prompt_template,
                template_context,
            )
        else:
            prompt = DEFAULT_CAPTION_TEMPLATE.render(template_context)

        return prompt.strip()

    async def caption(
        self,
        image: ImageInput | str,
        prompt: str | None = None,
        context: dict[str, Any] | None = None,
        **kwargs,
    ) -> CaptionResult[str]:
        """
        Generate a caption for an image.

        Args:
            image: Image to caption (path, URL, or ImageInput)
            prompt: Custom prompt (uses default if None)
            context: Additional context dict for template
            **kwargs: Passed to client

        Returns:
            CaptionResult with raw response string
        """
        if isinstance(image, str):
            image = ImageInput(source=image)

        request = CaptionRequest(
            image=image,
            prompt_template=prompt,
            context_dict=context or {},
        )

        final_prompt = self._build_prompt(request)

        return await self.client.caption(
            image=image,
            prompt=final_prompt,
            system_prompt=self.config.system_prompt,
            max_retries=self.config.max_retries,
            **kwargs,
        )

    async def caption_structured(
        self,
        image: ImageInput | str,
        prompt: str | None,
        output_model: type[BaseModel],
        context: dict[str, Any] | None = None,
        strict: bool = False,
        **kwargs,
    ) -> CaptionResult[BaseModel]:
        """
        Generate a structured caption validated against a Pydantic model.

        Args:
            image: Image to caption
            prompt: Custom prompt template
            output_model: Pydantic model for output validation
            context: Additional context dict
            strict: Use strict Pydantic validation
            **kwargs: Passed to client

        Returns:
            CaptionResult with parsed Pydantic model instance
        """
        validator = PydanticValidator(output_model, strict=strict)
        return await self.caption_validated(
            image=image,
            prompt=prompt,
            validator=validator,
            context=context,
            **kwargs,
        )

    async def caption_validated(
        self,
        image: ImageInput | str,
        prompt: str | None,
        validator: OutputValidator[T],
        context: dict[str, Any] | None = None,
        **kwargs,
    ) -> CaptionResult[T]:
        """
        Generate a caption with custom output validation.

        Will retry on validation failures up to validation_retries times.

        Args:
            image: Image to caption
            prompt: Custom prompt template
            validator: Output validator instance
            context: Additional context dict
            **kwargs: Passed to client

        Returns:
            CaptionResult with validated/parsed content
        """
        if isinstance(image, str):
            image = ImageInput(source=image)

        request = CaptionRequest(
            image=image,
            prompt_template=prompt,
            context_dict=context or {},
        )

        final_prompt = self._build_prompt(request, validator)

        total_retries = 0
        validation_attempts = 0
        last_result: CaptionResult[str] | None = None
        last_validation: ValidationResult | None = None

        while validation_attempts <= self.config.validation_retries:
            # Get raw caption
            result = await self.client.caption(
                image=image,
                prompt=final_prompt,
                system_prompt=self.config.system_prompt,
                max_retries=self.config.max_retries,
                **kwargs,
            )
            total_retries += result.retries_used
            last_result = result

            if not result.success:
                # API call failed
                return CaptionResult(
                    success=False,
                    error=result.error,
                    raw_response=result.raw_response,
                    retries_used=total_retries,
                    latency_ms=result.latency_ms,
                    model=result.model,
                )

            # Validate the response
            validation = validator.validate(result.content or "")
            last_validation = validation

            if validation.valid:
                return CaptionResult(
                    success=True,
                    content=validation.value,
                    raw_response=result.raw_response,
                    retries_used=total_retries,
                    latency_ms=result.latency_ms,
                    model=result.model,
                )

            # Validation failed, retry
            validation_attempts += 1
            logger.warning(
                f"Validation failed (attempt {validation_attempts}/"
                f"{self.config.validation_retries + 1}): {validation.error}"
            )

        # All validation attempts exhausted
        return CaptionResult(
            success=False,
            error=f"Validation failed after {validation_attempts} attempts: "
                  f"{last_validation.error if last_validation else 'Unknown error'}",
            raw_response=last_result.raw_response if last_result else None,
            retries_used=total_retries,
            latency_ms=last_result.latency_ms if last_result else 0,
            model=last_result.model if last_result else None,
        )

    async def caption_batch(
        self,
        images: list[ImageInput | str],
        prompt: str | None = None,
        context: dict[str, Any] | None = None,
        concurrency: int = 5,
        **kwargs,
    ) -> list[CaptionResult[str]]:
        """
        Generate captions for multiple images concurrently.

        Args:
            images: List of images to caption
            prompt: Shared prompt template
            context: Shared context dict
            concurrency: Maximum concurrent requests
            **kwargs: Passed to client

        Returns:
            List of CaptionResults in same order as inputs
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def caption_with_semaphore(img):
            async with semaphore:
                return await self.caption(
                    image=img,
                    prompt=prompt,
                    context=context,
                    **kwargs,
                )

        tasks = [caption_with_semaphore(img) for img in images]
        return await asyncio.gather(*tasks)

    async def caption_batch_structured(
        self,
        images: list[ImageInput | str],
        prompt: str | None,
        output_model: type[BaseModel],
        context: dict[str, Any] | None = None,
        concurrency: int = 5,
        **kwargs,
    ) -> list[CaptionResult[BaseModel]]:
        """
        Generate structured captions for multiple images concurrently.

        Args:
            images: List of images to caption
            prompt: Shared prompt template
            output_model: Pydantic model for output validation
            context: Shared context dict
            concurrency: Maximum concurrent requests
            **kwargs: Passed to client

        Returns:
            List of CaptionResults in same order as inputs
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def caption_with_semaphore(img):
            async with semaphore:
                return await self.caption_structured(
                    image=img,
                    prompt=prompt,
                    output_model=output_model,
                    context=context,
                    **kwargs,
                )

        tasks = [caption_with_semaphore(img) for img in images]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics from the client."""
        return self.client.get_stats()
