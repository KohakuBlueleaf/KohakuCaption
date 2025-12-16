"""
vLLM backend for local VLM inference.

Features:
- High-throughput inference with continuous batching
- Multi-GPU tensor parallelism
- PagedAttention for memory efficiency
- Prefix caching for repeated prompts
- Native support for various VLM architectures (Gemma3, LLaVA, Qwen2-VL, etc.)
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

    class LLM:
        pass

    class SamplingParams:
        pass


from kohakucaption.local.base import (
    LocalModelConfig,
    LocalVLMBase,
    GenerationOutput,
    BatchOutput,
)

logger = logging.getLogger(__name__)


# Model-specific prompt templates
# Each model has its own format for combining image and text
PROMPT_TEMPLATES = {
    # Gemma 3 series
    "gemma-3": {
        "template": "<bos><start_of_turn>user\n<start_of_image>{prompt}<end_of_turn>\n<start_of_turn>model\n",
        "with_system": "<bos><start_of_turn>user\n<start_of_image>{system}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
    },
    # Gemma 3n series (different image token)
    "gemma-3n": {
        "template": "<start_of_turn>user\n<image_soft_token>{prompt}<end_of_turn>\n<start_of_turn>model\n",
        "with_system": "<start_of_turn>user\n<image_soft_token>{system}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
    },
    # LLaVA 1.5
    "llava-1.5": {
        "template": "USER: <image>\n{prompt}\nASSISTANT:",
        "with_system": "USER: <image>\n{system}\n\n{prompt}\nASSISTANT:",
    },
    # LLaVA 1.6 / LLaVA-NeXT
    "llava-next": {
        "template": "[INST] <image>\n{prompt} [/INST]",
        "with_system": "[INST] <image>\n{system}\n\n{prompt} [/INST]",
    },
    # Qwen2-VL
    "qwen2-vl": {
        "template": "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "with_system": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n",
    },
    # InternVL
    "internvl": {
        "template": "<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "with_system": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    },
    # Default fallback (simple format)
    "default": {
        "template": "<image>\n{prompt}",
        "with_system": "<image>\n{system}\n\n{prompt}",
    },
}


def detect_model_family(model_name: str) -> str:
    """Detect model family from model name for prompt template selection."""
    model_lower = model_name.lower()

    if "gemma-3n" in model_lower or "gemma3n" in model_lower:
        return "gemma-3n"
    if "gemma-3" in model_lower or "gemma3" in model_lower:
        return "gemma-3"
    if "llava-v1.6" in model_lower or "llava-next" in model_lower:
        return "llava-next"
    if "llava-1.5" in model_lower or "llava-v1.5" in model_lower:
        return "llava-1.5"
    if "qwen2-vl" in model_lower or "qwen2.5-vl" in model_lower:
        return "qwen2-vl"
    if "internvl" in model_lower:
        return "internvl"

    return "default"


@dataclass
class VLLMConfig(LocalModelConfig):
    """Configuration for vLLM inference."""

    model: str = "unsloth/gemma-3-4b-it-FP8-Dynamic"
    # vLLM specific
    max_model_len: int | None = None  # Auto if None
    max_num_seqs: int = 256  # Max concurrent sequences
    enable_prefix_caching: bool = True
    # Multi-GPU
    pipeline_parallel_size: int = 1  # For multi-node
    distributed_executor_backend: str | None = None  # "mp" or "ray"
    # Vision specific
    limit_mm_per_prompt: dict[str, int] = field(default_factory=lambda: {"image": 1})
    mm_processor_kwargs: dict[str, Any] | None = None  # e.g., {"do_pan_and_scan": True}
    # Sampling
    seed: int | None = None
    # Model family override (auto-detected if None)
    model_family: str | None = None


class VLLMModel(LocalVLMBase):
    """
    vLLM backend for local VLM inference.

    Uses LLM.generate() with proper prompt templates for each model family.

    Example:
        config = VLLMConfig(
            model="unsloth/gemma-3-4b-it-FP8-Dynamic",
            tensor_parallel_size=2,  # Use 2 GPUs
        )
        model = VLLMModel(config)

        with model:
            result = model.generate(image, "Describe this image")
            print(result.text)

    Multi-GPU:
        # Tensor parallelism (single node, multiple GPUs)
        config = VLLMConfig(
            model="google/gemma-3-27b-it",
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
        self._model_family: str | None = None

    def load(self) -> None:
        """Load vLLM model."""
        if self._loaded:
            return

        logger.info(
            f"Loading vLLM: {self.config.model} "
            f"(tp={self.config.tensor_parallel_size}, pp={self.config.pipeline_parallel_size})"
        )

        # Detect model family for prompt template
        self._model_family = self.config.model_family or detect_model_family(
            self.config.model
        )
        logger.info(f"Detected model family: {self._model_family}")

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

        # Quantization: fp8 (W8A8), fp8_w8a16 (weight-only), awq, gptq, etc.
        if self.config.quantization:
            llm_kwargs["quantization"] = self.config.quantization

        # KV cache dtype: fp8 for memory efficiency (2x KV cache capacity)
        if self.config.kv_cache_dtype != "auto":
            llm_kwargs["kv_cache_dtype"] = self.config.kv_cache_dtype

        if self.config.pipeline_parallel_size > 1:
            llm_kwargs["pipeline_parallel_size"] = self.config.pipeline_parallel_size

        if self.config.distributed_executor_backend:
            llm_kwargs["distributed_executor_backend"] = (
                self.config.distributed_executor_backend
            )

        if self.config.mm_processor_kwargs:
            llm_kwargs["mm_processor_kwargs"] = self.config.mm_processor_kwargs

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

        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("vLLM model unloaded")

    def _build_prompt(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Build prompt string using model-specific template."""
        templates = PROMPT_TEMPLATES.get(
            self._model_family, PROMPT_TEMPLATES["default"]
        )

        if system_prompt:
            return templates["with_system"].format(system=system_prompt, prompt=prompt)
        else:
            return templates["template"].format(prompt=prompt)

    def _get_sampling_params(self, **kwargs) -> SamplingParams:
        """Get sampling parameters with optional overrides."""
        top_k = kwargs.get("top_k", self.config.top_k)

        return SamplingParams(
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=top_k if top_k > 0 else -1,
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

        full_prompt = self._build_prompt(prompt, system_prompt)
        sampling_params = self._get_sampling_params(**kwargs)

        start_time = time.perf_counter()

        # vLLM generate() with dict input containing multi_modal_data
        inputs = {
            "prompt": full_prompt,
            "multi_modal_data": {"image": image},
        }

        outputs = self._model.generate(inputs, sampling_params=sampling_params)

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

        # Build inputs - each is a dict with prompt and multi_modal_data
        inputs = []
        for image, prompt in zip(images, prompts):
            full_prompt = self._build_prompt(prompt, system_prompt)
            inputs.append(
                {
                    "prompt": full_prompt,
                    "multi_modal_data": {"image": image},
                }
            )

        sampling_params = self._get_sampling_params(**kwargs)

        start_time = time.perf_counter()

        # Batch inference
        outputs = self._model.generate(inputs, sampling_params=sampling_params)

        total_time = (time.perf_counter() - start_time) * 1000

        # Process outputs
        results = []
        total_tokens = 0
        for output in outputs:
            gen_output = output.outputs[0]
            tokens = list(gen_output.token_ids) if gen_output.token_ids else []
            total_tokens += len(tokens)

            results.append(
                GenerationOutput(
                    text=gen_output.text,
                    tokens=tokens if tokens else None,
                    completion_tokens=len(tokens),
                    generation_time_ms=total_time / len(images),
                )
            )

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

        full_prompt = self._build_prompt(prompt, system_prompt)

        top_k = kwargs.get("top_k", self.config.top_k)
        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=top_k if top_k > 0 else -1,
            seed=kwargs.get("seed", self.config.seed),
            logprobs=top_logprobs,
        )

        start_time = time.perf_counter()

        inputs = {
            "prompt": full_prompt,
            "multi_modal_data": {"image": image},
        }

        outputs = self._model.generate(inputs, sampling_params=sampling_params)

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
