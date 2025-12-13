"""
vLLM backend for local VLM inference.

Features:
- High-throughput inference with continuous batching
- Multi-GPU tensor parallelism
- PagedAttention for memory efficiency
- Prefix caching for repeated prompts
- Native chat template support via tokenizer
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    class LLM: pass
    class SamplingParams: pass

from kohakucaption.local.base import (
    LocalModelConfig,
    LocalVLMBase,
    GenerationOutput,
    BatchOutput,
)

logger = logging.getLogger(__name__)


@dataclass
class VLLMConfig(LocalModelConfig):
    """Configuration for vLLM inference."""

    model: str = "llava-hf/llava-1.5-7b-hf"
    # vLLM specific
    max_model_len: int | None = None  # Auto if None
    max_num_seqs: int = 256  # Max concurrent sequences
    enable_prefix_caching: bool = True
    # Multi-GPU
    pipeline_parallel_size: int = 1  # For multi-node
    distributed_executor_backend: str | None = None  # "mp" or "ray"
    # Vision specific
    limit_mm_per_prompt: dict[str, int] = field(default_factory=lambda: {"image": 1})
    # Sampling
    seed: int | None = None


class VLLMModel(LocalVLMBase):
    """
    vLLM backend for local VLM inference.

    Uses the model's native chat template via tokenizer - no manual prompt building.

    Example:
        config = VLLMConfig(
            model="llava-hf/llava-1.5-7b-hf",
            tensor_parallel_size=2,  # Use 2 GPUs
        )
        model = VLLMModel(config)

        with model:
            result = model.generate(image, "Describe this image")
            print(result.text)

    Multi-GPU:
        # Tensor parallelism (single node, multiple GPUs)
        config = VLLMConfig(
            model="llava-hf/llava-v1.6-34b-hf",
            tensor_parallel_size=4,
        )

        # Pipeline parallelism (multi-node, requires Ray)
        config = VLLMConfig(
            model="very-large-model",
            tensor_parallel_size=8,
            pipeline_parallel_size=2,
            distributed_executor_backend="ray",
        )
    """
    _model: LLM

    def __init__(self, config: VLLMConfig | None = None):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Install with: pip install vllm")
        super().__init__(config or VLLMConfig())
        self.config: VLLMConfig = self.config
        self._processor = None

    def load(self) -> None:
        """Load vLLM model."""
        if self._loaded:
            return

        logger.info(
            f"Loading vLLM: {self.config.model} "
            f"(tp={self.config.tensor_parallel_size}, pp={self.config.pipeline_parallel_size})"
        )

        # Build LLM kwargs
        llm_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "trust_remote_code": self.config.trust_remote_code,
            "max_num_seqs": self.config.max_num_seqs,
            "limit_mm_per_prompt": self.config.limit_mm_per_prompt,
            "enable_prefix_caching": self.config.enable_prefix_caching,
        }

        if self.config.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.config.max_model_len

        if self.config.dtype != "auto":
            llm_kwargs["dtype"] = self.config.dtype

        if self.config.pipeline_parallel_size > 1:
            llm_kwargs["pipeline_parallel_size"] = self.config.pipeline_parallel_size

        if self.config.distributed_executor_backend:
            llm_kwargs["distributed_executor_backend"] = self.config.distributed_executor_backend

        self._model = LLM(**llm_kwargs)
        self._loaded = True

        logger.info("vLLM model loaded successfully")

    def unload(self) -> None:
        """Unload vLLM model."""
        if not self._loaded:
            return

        if self._model is not None:
            del self._model
            self._model = None

        self._processor = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("vLLM model unloaded")

    def _build_conversation(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build conversation in OpenAI chat format - vLLM handles template."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # User message with image placeholder
        # vLLM's chat template will handle the actual formatting
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        })

        return messages

    def _get_sampling_params(self, **kwargs) -> SamplingParams:
        """Get sampling parameters with optional overrides."""
        top_k = kwargs.get("top_k", self.config.top_k)

        return SamplingParams(
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=top_k if top_k > 0 else None,
            seed=kwargs.get("seed", self.config.seed),
        )

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

        messages = self._build_conversation(prompt, system_prompt)
        sampling_params = self._get_sampling_params(**kwargs)

        start_time = time.perf_counter()

        # vLLM handles chat template internally
        outputs = self._model.chat(
            messages=messages,
            sampling_params=sampling_params,
            multi_modal_data={"image": image},
        )

        generation_time = (time.perf_counter() - start_time) * 1000

        output = outputs[0].outputs[0]
        return GenerationOutput(
            text=output.text,
            tokens=list(output.token_ids) if output.token_ids else None,
            completion_tokens=len(output.token_ids) if output.token_ids else 0,
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

        # Build inputs - each is a conversation with image
        conversations = []
        mm_data_list = []
        for image, prompt in zip(images, prompts):
            messages = self._build_conversation(prompt, system_prompt)
            conversations.append(messages)
            mm_data_list.append({"image": image})

        sampling_params = self._get_sampling_params(**kwargs)

        start_time = time.perf_counter()

        # Batch chat inference
        outputs = self._model.chat(
            messages=conversations,
            sampling_params=sampling_params,
            multi_modal_data=mm_data_list,
        )

        total_time = (time.perf_counter() - start_time) * 1000

        # Process outputs
        results = []
        total_tokens = 0
        for output in outputs:
            gen_output = output.outputs[0]
            tokens = list(gen_output.token_ids) if gen_output.token_ids else []
            total_tokens += len(tokens)

            results.append(GenerationOutput(
                text=gen_output.text,
                tokens=tokens if tokens else None,
                completion_tokens=len(tokens),
                generation_time_ms=total_time / len(images),
            ))

        throughput = (total_tokens / (total_time / 1000)) if total_time > 0 else 0

        return BatchOutput(
            outputs=results,
            total_time_ms=total_time,
            throughput_tokens_per_sec=throughput,
        )

    def generate_with_logprobs(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: str | None = None,
        top_logprobs: int = 5,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate text with token log probabilities.

        This is a local-only feature not available via remote APIs.

        Args:
            image: PIL Image
            prompt: User prompt
            system_prompt: Optional system prompt
            top_logprobs: Number of top logprobs to return per token
            **kwargs: Generation parameter overrides

        Returns:
            GenerationOutput with token_logprobs populated
        """
        if not self._loaded:
            self.load()

        messages = self._build_conversation(prompt, system_prompt)

        top_k = kwargs.get("top_k", self.config.top_k)
        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=top_k if top_k > 0 else None,
            seed=kwargs.get("seed", self.config.seed),
            logprobs=top_logprobs,
        )

        start_time = time.perf_counter()

        outputs = self._model.chat(
            messages=messages,
            sampling_params=sampling_params,
            multi_modal_data={"image": image},
        )

        generation_time = (time.perf_counter() - start_time) * 1000

        output = outputs[0].outputs[0]

        # Extract logprobs
        token_logprobs = None
        if output.logprobs:
            token_logprobs = [
                lp[token_id].logprob if token_id in lp else 0.0
                for lp, token_id in zip(output.logprobs, output.token_ids)
            ]

        return GenerationOutput(
            text=output.text,
            tokens=list(output.token_ids) if output.token_ids else None,
            token_logprobs=token_logprobs,
            completion_tokens=len(output.token_ids) if output.token_ids else 0,
            generation_time_ms=generation_time,
        )
