"""
KohakuCaption - Image Caption Pipeline for T2I Projects

A comprehensive system for generating structured image captions using
MLLM APIs (OpenAI, OpenRouter) with support for:
- Structured output validation (Pydantic, JSON Schema, custom validators)
- Template-based context formatting
- Async processing with retry/monitoring
- Multiple context types (metadata, bbox, depth maps, etc.)
"""

from kohakucaption.clients import MLLMClient, OpenAIClient, OpenRouterClient
from kohakucaption.context import ContextProvider, TemplateEngine
from kohakucaption.formats import DefaultFormat, JsonFormat, OutputFormat
from kohakucaption.pipeline import CaptionPipeline
from kohakucaption.tagger import (
    AnimeTimmTagger,
    AnimeTimmTagResult,
    PixAITagger,
    TagResult,
)
from kohakucaption.tokenizer import TokenCounter, count_caption_tokens, count_tokens
from kohakucaption.types import CaptionRequest, CaptionResult, ContextType, ImageInput
from kohakucaption.validation import (
    JsonSchemaValidator,
    OutputValidator,
    PydanticValidator,
    RegexValidator,
)

__version__ = "0.0.1"
__all__ = [
    # Types
    "CaptionResult",
    "CaptionRequest",
    "ImageInput",
    "ContextType",
    # Clients
    "MLLMClient",
    "OpenAIClient",
    "OpenRouterClient",
    # Pipeline
    "CaptionPipeline",
    # Context
    "ContextProvider",
    "TemplateEngine",
    # Formats
    "OutputFormat",
    "DefaultFormat",
    "JsonFormat",
    # Validation
    "OutputValidator",
    "PydanticValidator",
    "JsonSchemaValidator",
    "RegexValidator",
    # Tokenizer
    "TokenCounter",
    "count_tokens",
    "count_caption_tokens",
    # Tagger
    "PixAITagger",
    "TagResult",
    "AnimeTimmTagger",
    "AnimeTimmTagResult",
]
