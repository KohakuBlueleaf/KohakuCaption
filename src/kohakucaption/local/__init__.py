"""
Local VLM inference backends.

This module provides high-throughput local inference for Vision Language Models
using optimized backends like vLLM and LMDeploy.

Features over remote API:
- Direct GPU control and multi-GPU support (tensor/pipeline parallelism)
- Batch inference with preprocessing pipeline
- Memory management and model loading/unloading
- Access to logits and token probabilities
- No rate limits or API costs

Example:
    from kohakucaption.local import VLLMModel, VLLMConfig

    config = VLLMConfig(
        model="llava-hf/llava-1.5-7b-hf",
        tensor_parallel_size=2,  # Use 2 GPUs
    )

    with VLLMModel(config) as model:
        result = model.generate(image, "Describe this image")
        print(result.text)

Available backends:
- VLLMModel: High-throughput inference with vLLM
- LMDeployModel: Optimized inference with LMDeploy (TurboMind/PyTorch)
"""

from kohakucaption.local.base import (
    LocalModelConfig,
    LocalVLMBase,
    GenerationOutput,
    BatchOutput,
)
from kohakucaption.local.preprocess import (
    PreprocessPipeline,
    PreprocessedBatch,
    StreamingPreprocessor,
)

# Lazy imports to avoid hard dependencies
_vllm_classes = None
_lmdeploy_classes = None


def _get_vllm():
    global _vllm_classes
    if _vllm_classes is None:
        from kohakucaption.local.vllm import VLLMModel, VLLMConfig
        _vllm_classes = (VLLMModel, VLLMConfig)
    return _vllm_classes


def _get_lmdeploy():
    global _lmdeploy_classes
    if _lmdeploy_classes is None:
        from kohakucaption.local.lmdeploy import LMDeployModel, LMDeployConfig
        _lmdeploy_classes = (LMDeployModel, LMDeployConfig)
    return _lmdeploy_classes


# Public API via property-like access
class _LazyLoader:
    @property
    def VLLMModel(self):
        return _get_vllm()[0]

    @property
    def VLLMConfig(self):
        return _get_vllm()[1]

    @property
    def LMDeployModel(self):
        return _get_lmdeploy()[0]

    @property
    def LMDeployConfig(self):
        return _get_lmdeploy()[1]


_lazy = _LazyLoader()


def __getattr__(name: str):
    """Lazy load backend classes on first access."""
    if name in ("VLLMModel", "VLLMConfig"):
        return getattr(_lazy, name)
    if name in ("LMDeployModel", "LMDeployConfig"):
        return getattr(_lazy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base classes
    "LocalModelConfig",
    "LocalVLMBase",
    "GenerationOutput",
    "BatchOutput",
    # Preprocessing
    "PreprocessPipeline",
    "PreprocessedBatch",
    "StreamingPreprocessor",
    # Backend classes (lazy loaded)
    "VLLMModel",
    "VLLMConfig",
    "LMDeployModel",
    "LMDeployConfig",
]
