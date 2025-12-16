"""
Base classes for local VLM inference.

Local inference provides additional functionality over remote API:
- Direct GPU control and multi-GPU support
- Batch inference with preprocessing pipeline
- Memory management and model loading/unloading
- Access to logits and token probabilities
"""

import base64
import io
import logging
import time
import urllib.request
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
from PIL import Image

from kohakucaption.types import AggregateStats, CaptionResult, ImageInput, RequestStats

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class LocalModelConfig:
    """Base configuration for local VLM inference."""

    model: str
    # GPU settings
    device: str | None = None  # Auto-detect if None
    tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism
    # Generation settings
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    # Memory settings
    gpu_memory_utilization: float = 0.95  # Use 95% VRAM for better KV cache
    # Model loading
    trust_remote_code: bool = True
    dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"
    # Quantization settings
    quantization: str | None = None  # "fp8" (W8A8), "fp8_w8a16", "awq", "gptq", etc.
    kv_cache_dtype: str = (
        "fp8"  # "auto", "fp8", "fp8_e4m3" - default fp8 for efficiency
    )


@dataclass
class GenerationOutput:
    """Output from local VLM generation with extended info."""

    text: str
    # Token-level info (available in local inference)
    tokens: list[int] | None = None
    token_logprobs: list[float] | None = None
    # Generation stats
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Timing
    time_to_first_token_ms: float | None = None
    generation_time_ms: float = 0.0


@dataclass
class BatchOutput:
    """Output from batch inference."""

    outputs: list[GenerationOutput]
    total_time_ms: float
    throughput_tokens_per_sec: float


class LocalVLMBase(ABC):
    """
    Abstract base class for local VLM inference.

    Provides unified interface for different backends (vLLM, LMDeploy, etc.)
    with additional local-only features.
    """

    def __init__(self, config: LocalModelConfig):
        self.config = config
        self._model = None
        self._loaded = False
        self._stats = AggregateStats()
        self._request_history: list[RequestStats] = []

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def device(self) -> str:
        """Get the device being used."""
        if self.config.device:
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free memory."""
        pass

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate text for a single image.

        Args:
            image: PIL Image
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Generation parameter overrides

        Returns:
            GenerationOutput with text and optional token info
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        images: list[Image.Image],
        prompts: list[str],
        system_prompt: str | None = None,
        **kwargs,
    ) -> BatchOutput:
        """
        Generate text for multiple images.

        Args:
            images: List of PIL Images
            prompts: List of prompts (one per image or single for all)
            system_prompt: Optional system prompt
            **kwargs: Generation parameter overrides

        Returns:
            BatchOutput with list of GenerationOutputs
        """
        pass

    def _prepare_image(
        self, image: ImageInput | Image.Image | str | Path
    ) -> Image.Image:
        """Convert various image types to PIL Image."""
        if isinstance(image, Image.Image):
            return self._ensure_rgb(image)

        if isinstance(image, ImageInput):
            if image.base64_data:
                img_bytes = base64.b64decode(image.base64_data)
                return self._ensure_rgb(Image.open(io.BytesIO(img_bytes)))
            elif image.is_url():
                with urllib.request.urlopen(str(image.source)) as response:
                    return self._ensure_rgb(Image.open(io.BytesIO(response.read())))
            else:
                return self._ensure_rgb(Image.open(image.source))

        # str or Path
        return self._ensure_rgb(Image.open(image))

    def _ensure_rgb(self, image: Image.Image) -> Image.Image:
        """Ensure image is in RGB mode."""
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            return background
        elif image.mode == "P":
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            if rgba.mode == "RGBA":
                background.paste(rgba, mask=rgba.split()[3])
            return background
        return image.convert("RGB")

    def caption(
        self,
        image: ImageInput | Image.Image | str | Path,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> CaptionResult[str]:
        """
        Generate a caption for an image.

        This provides API compatibility with remote clients.

        Args:
            image: Image to caption (various formats supported)
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Generation parameter overrides

        Returns:
            CaptionResult with generated caption
        """
        if not self._loaded:
            self.load()

        request_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        try:
            pil_image = self._prepare_image(image)
            output = self.generate(pil_image, prompt, system_prompt, **kwargs)

            latency_ms = (time.perf_counter() - start_time) * 1000

            stats = RequestStats(
                request_id=request_id,
                model=self.config.model,
                success=True,
                latency_ms=latency_ms,
                retries=0,
            )
            self._stats.record(stats)
            self._request_history.append(stats)

            return CaptionResult(
                success=True,
                content=output.text,
                raw_response=output.text,
                retries_used=0,
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
                retries=0,
                error=error_msg,
            )
            self._stats.record(stats)
            self._request_history.append(stats)

            logger.error(f"Caption failed: {error_msg}")

            return CaptionResult(
                success=False,
                error=error_msg,
                retries_used=0,
                latency_ms=latency_ms,
                model=self.config.model,
            )

    def caption_batch(
        self,
        images: list[ImageInput | Image.Image | str | Path],
        prompts: list[str],
        system_prompt: str | None = None,
        **kwargs,
    ) -> list[CaptionResult[str]]:
        """
        Generate captions for multiple images.

        Args:
            images: List of images
            prompts: List of prompts (one per image or single for all)
            system_prompt: Optional system prompt
            **kwargs: Generation parameter overrides

        Returns:
            List of CaptionResults
        """
        if not self._loaded:
            self.load()

        # Handle single prompt for all images
        if len(prompts) == 1:
            prompts = prompts * len(images)
        elif len(prompts) != len(images):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) must match images ({len(images)}) or be 1"
            )

        start_time = time.perf_counter()

        try:
            pil_images = [self._prepare_image(img) for img in images]
            batch_output = self.generate_batch(
                pil_images, prompts, system_prompt, **kwargs
            )

            results = []
            per_image_latency = batch_output.total_time_ms / len(images)

            for output in batch_output.outputs:
                results.append(
                    CaptionResult(
                        success=True,
                        content=output.text,
                        raw_response=output.text,
                        retries_used=0,
                        latency_ms=per_image_latency,
                        model=self.config.model,
                    )
                )

            return results

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            error_msg = str(e)
            logger.error(f"Batch caption failed: {error_msg}")

            return [
                CaptionResult(
                    success=False,
                    error=error_msg,
                    retries_used=0,
                    latency_ms=latency_ms / len(images),
                    model=self.config.model,
                )
                for _ in images
            ]

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics summary."""
        return {
            "total_requests": self._stats.total_requests,
            "successful_requests": self._stats.successful_requests,
            "failed_requests": self._stats.failed_requests,
            "success_rate": f"{self._stats.success_rate:.1%}",
            "average_latency_ms": f"{self._stats.average_latency_ms:.1f}",
            "model": self.config.model,
            "device": self.device,
            "tensor_parallel": self.config.tensor_parallel_size,
        }

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()

    def __del__(self):
        if self._loaded:
            self.unload()
