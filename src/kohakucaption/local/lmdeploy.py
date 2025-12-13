"""
LMDeploy backend for local VLM inference.

Features:
- TurboMind engine for optimized inference
- PyTorch engine for flexibility
- Multi-GPU tensor parallelism
- Persistent batch (continuous batching)
- Native chat template support
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

try:
    from lmdeploy import pipeline, GenerationConfig
    from lmdeploy import TurbomindEngineConfig, PytorchEngineConfig
    from lmdeploy import VisionConfig
    LMDEPLOY_AVAILABLE = True
except ImportError:
    LMDEPLOY_AVAILABLE = False
    pipeline = None
    GenerationConfig = None
    TurbomindEngineConfig = None
    PytorchEngineConfig = None
    VisionConfig = None

# Optional imports
try:
    from lmdeploy import ChatTemplateConfig
    CHAT_TEMPLATE_AVAILABLE = True
except ImportError:
    CHAT_TEMPLATE_AVAILABLE = False
    ChatTemplateConfig = None

from kohakucaption.local.base import (
    LocalModelConfig,
    LocalVLMBase,
    GenerationOutput,
    BatchOutput,
)

logger = logging.getLogger(__name__)


@dataclass
class LMDeployConfig(LocalModelConfig):
    """Configuration for LMDeploy inference."""

    model: str = "OpenGVLab/InternVL2-8B"
    # LMDeploy specific
    backend: str = "turbomind"  # "turbomind" or "pytorch"
    session_len: int = 8192  # Max context length
    cache_max_entry_count: float = 0.8  # KV cache memory ratio
    # Vision specific
    vision_batch_size: int = 8  # Batch size for vision encoder
    # Chat template (for custom templates)
    chat_template: str | None = None
    # Quantization
    quant_policy: int = 0  # 0=none, 4=w4a16, 8=w8a8


class LMDeployModel(LocalVLMBase):
    """
    LMDeploy backend for local VLM inference.

    Uses the model's native chat template - no manual prompt building.

    Example:
        config = LMDeployConfig(
            model="OpenGVLab/InternVL2-8B",
            tensor_parallel_size=2,
        )
        model = LMDeployModel(config)

        with model:
            result = model.generate(image, "Describe this image")
            print(result.text)

    Backends:
        - turbomind: Optimized C++/CUDA engine (faster, recommended)
        - pytorch: Pure PyTorch engine (more flexible)

    Multi-GPU:
        config = LMDeployConfig(
            model="OpenGVLab/InternVL2-26B",
            tensor_parallel_size=4,  # Use 4 GPUs
            backend="turbomind",
        )
    """

    def __init__(self, config: LMDeployConfig | None = None):
        if not LMDEPLOY_AVAILABLE:
            raise ImportError("LMDeploy is not installed. Install with: pip install lmdeploy")
        super().__init__(config or LMDeployConfig())
        self.config: LMDeployConfig = self.config
        self._pipe = None
        self._gen_config = None

    def load(self) -> None:
        """Load LMDeploy pipeline."""
        if self._loaded:
            return

        logger.info(
            f"Loading LMDeploy: {self.config.model} "
            f"(tp={self.config.tensor_parallel_size}, backend={self.config.backend})"
        )

        # Build engine config
        if self.config.backend == "turbomind":
            backend_config = TurbomindEngineConfig(
                tp=self.config.tensor_parallel_size,
                session_len=self.config.session_len,
                cache_max_entry_count=self.config.cache_max_entry_count,
                quant_policy=self.config.quant_policy,
            )
        else:
            backend_config = PytorchEngineConfig(
                tp=self.config.tensor_parallel_size,
                session_len=self.config.session_len,
                cache_max_entry_count=self.config.cache_max_entry_count,
            )

        # Vision config
        vision_config = VisionConfig(max_batch_size=self.config.vision_batch_size)

        # Build pipeline kwargs
        pipe_kwargs: dict[str, Any] = {
            "backend_config": backend_config,
            "vision_config": vision_config,
        }

        # Chat template config if specified
        if self.config.chat_template and CHAT_TEMPLATE_AVAILABLE:
            pipe_kwargs["chat_template_config"] = ChatTemplateConfig(
                model_name=self.config.chat_template
            )

        self._pipe = pipeline(self.config.model, **pipe_kwargs)

        # Default generation config
        self._gen_config = GenerationConfig(
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )

        self._loaded = True
        logger.info("LMDeploy model loaded successfully")

    def unload(self) -> None:
        """Unload LMDeploy pipeline."""
        if not self._loaded:
            return

        if self._pipe is not None:
            if hasattr(self._pipe, 'close'):
                self._pipe.close()
            del self._pipe
            self._pipe = None

        self._gen_config = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("LMDeploy model unloaded")

    def _get_gen_config(self, **kwargs) -> GenerationConfig:
        """Get generation config with overrides."""
        return GenerationConfig(
            max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
        )

    def _extract_text(self, response) -> str:
        """Extract text from LMDeploy response."""
        if hasattr(response, 'text'):
            return response.text
        elif isinstance(response, str):
            return response
        return str(response)

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """Generate text for a single image."""
        if not self._loaded:
            self.load()

        # Build full prompt - LMDeploy handles chat template internally
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        gen_config = self._get_gen_config(**kwargs)

        start_time = time.perf_counter()

        # LMDeploy uses tuple format: (prompt, image)
        response = self._pipe((full_prompt, image), gen_config=gen_config)

        generation_time = (time.perf_counter() - start_time) * 1000

        text = self._extract_text(response)

        return GenerationOutput(
            text=text,
            generation_time_ms=generation_time,
        )

    def generate_batch(
        self,
        images: list[Image.Image],
        prompts: list[str],
        system_prompt: str | None = None,
        **kwargs,
    ) -> BatchOutput:
        """Generate text for multiple images."""
        if not self._loaded:
            self.load()

        # Handle single prompt for all images
        if len(prompts) == 1:
            prompts = prompts * len(images)

        # Build inputs as list of tuples
        inputs = []
        for image, prompt in zip(images, prompts):
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            inputs.append((full_prompt, image))

        gen_config = self._get_gen_config(**kwargs)

        start_time = time.perf_counter()
        responses = self._pipe(inputs, gen_config=gen_config)
        total_time = (time.perf_counter() - start_time) * 1000

        # Process outputs
        results = []
        for response in responses:
            text = self._extract_text(response)
            results.append(GenerationOutput(
                text=text,
                generation_time_ms=total_time / len(images),
            ))

        return BatchOutput(
            outputs=results,
            total_time_ms=total_time,
            throughput_tokens_per_sec=0,  # LMDeploy doesn't expose token counts easily
        )

    def generate_with_logits(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate text with output logits.

        This is a local-only feature. Requires LMDeploy >= 0.5.0.

        Args:
            image: PIL Image
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Generation parameter overrides

        Returns:
            GenerationOutput with token_logprobs populated
        """
        if not self._loaded:
            self.load()

        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        gen_config = GenerationConfig(
            max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
            output_logits='generation',  # Return logits for generated tokens
        )

        start_time = time.perf_counter()
        response = self._pipe((full_prompt, image), gen_config=gen_config)
        generation_time = (time.perf_counter() - start_time) * 1000

        text = self._extract_text(response)

        # Extract logits if available
        token_logprobs = None
        if hasattr(response, 'logits') and response.logits is not None:
            logits = torch.tensor(response.logits)
            logprobs = torch.log_softmax(logits, dim=-1)
            # Get logprob of selected tokens
            if hasattr(response, 'token_ids'):
                token_ids = response.token_ids
                token_logprobs = [
                    logprobs[i, tid].item()
                    for i, tid in enumerate(token_ids)
                ]

        return GenerationOutput(
            text=text,
            token_logprobs=token_logprobs,
            generation_time_ms=generation_time,
        )

    def chat(
        self,
        image: Image.Image,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> GenerationOutput:
        """
        Multi-turn chat with an image.

        This is a local-only feature for multi-turn conversations.

        Args:
            image: PIL Image
            messages: List of {"role": "user"|"assistant", "content": str}
            **kwargs: Generation parameter overrides

        Returns:
            GenerationOutput with assistant's response
        """
        if not self._loaded:
            self.load()

        gen_config = self._get_gen_config(**kwargs)

        start_time = time.perf_counter()

        # Use chat interface
        response = self._pipe.chat((messages[-1]["content"], image), gen_config=gen_config)

        generation_time = (time.perf_counter() - start_time) * 1000

        text = self._extract_text(response)

        return GenerationOutput(
            text=text,
            generation_time_ms=generation_time,
        )
