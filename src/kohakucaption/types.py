"""
Core types and data structures for KohakuCaption.
"""

import base64
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field


class ContextType(str, Enum):
    """Predefined context types for image captioning."""

    METADATA = "metadata"  # Image metadata (EXIF, dimensions, etc.)
    OBJECT_BBOX = "object_bbox"  # Object bounding boxes
    DEPTH_MAP = "depth_map"  # Depth map information
    SEGMENTATION = "segmentation"  # Segmentation masks
    OCR = "ocr"  # OCR text results
    TAGS = "tags"  # Existing tags/labels
    CUSTOM = "custom"  # Custom user-defined context


@dataclass
class ImageInput:
    """
    Represents an image input for captioning.
    Supports both file paths and base64-encoded data.
    """

    source: str | Path  # File path or URL
    base64_data: str | None = None  # Pre-encoded base64 data
    mime_type: str = "image/png"  # MIME type for encoding

    def to_base64(self) -> str:
        """Convert image to base64-encoded data URL."""
        if self.base64_data:
            return f"data:{self.mime_type};base64,{self.base64_data}"

        path = Path(self.source)
        if path.exists():
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")

            # Auto-detect mime type from extension
            mime_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mime = mime_map.get(path.suffix.lower(), self.mime_type)
            return f"data:{mime};base64,{encoded}"

        # Assume it's a URL
        return str(self.source)

    def is_url(self) -> bool:
        """Check if source is a URL."""
        source_str = str(self.source)
        return source_str.startswith(("http://", "https://"))


@dataclass
class ContextItem:
    """A single context item with type and data."""

    context_type: ContextType
    data: dict[str, Any]
    label: str | None = None  # Optional human-readable label


@dataclass
class CaptionRequest:
    """Request for generating a caption."""

    image: ImageInput
    prompt_template: str | None = None  # Optional custom prompt template
    context_items: list[ContextItem] = field(default_factory=list)
    context_dict: dict[str, Any] = field(default_factory=dict)  # For template vars
    max_retries: int = 3
    timeout: float = 60.0

    def add_context(
        self,
        context_type: ContextType,
        data: dict[str, Any],
        label: str | None = None,
    ) -> "CaptionRequest":
        """Add context to the request. Returns self for chaining."""
        self.context_items.append(ContextItem(context_type, data, label))
        return self


T = TypeVar("T")


@dataclass
class CaptionResult(Generic[T]):
    """Result of a caption generation request."""

    success: bool
    content: T | None = None  # Parsed/validated content
    raw_response: str | None = None  # Raw model response
    error: str | None = None
    retries_used: int = 0
    latency_ms: float = 0.0
    model: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def failed(self) -> bool:
        return not self.success


# Common caption output schemas
class BasicCaption(BaseModel):
    """Basic caption output schema with scoring."""

    aesthetic_score: float = Field(description="Aesthetic quality score 0.0-1.0")
    nsfw_score: float = Field(description="NSFW content score 0.0-1.0 (0=safe, 1=explicit)")
    quality_score: float = Field(description="Technical quality score 0.0-1.0")
    title: str = Field(description="A short, descriptive title for the image")
    brief: str = Field(description="A brief one-sentence description")
    description: str = Field(
        description="A detailed description of the image. "
        "If nsfw_score is high, describe explicit content in detail."
    )


class DetailedCaption(BaseModel):
    """Extended caption output schema with more fields."""

    aesthetic_score: float = Field(description="Aesthetic quality score 0.0-1.0")
    nsfw_score: float = Field(description="NSFW content score 0.0-1.0 (0=safe, 1=explicit)")
    quality_score: float = Field(description="Technical quality score 0.0-1.0")
    title: str = Field(description="A short, descriptive title for the image")
    brief: str = Field(description="A brief one-sentence description")
    description: str = Field(
        description="A detailed description of the image. "
        "If nsfw_score is high, describe explicit content in detail."
    )
    style: str | None = Field(
        default=None, description="Art style (e.g., anime, realistic, sketch)"
    )
    mood: str | None = Field(
        default=None, description="Overall mood or atmosphere of the image"
    )
    colors: list[str] = Field(
        default_factory=list, description="Dominant colors in the image"
    )
    subjects: list[str] = Field(
        default_factory=list, description="Main subjects/characters in the image"
    )


# Statistics tracking
@dataclass
class RequestStats:
    """Statistics for a single request."""

    request_id: str
    model: str
    success: bool
    latency_ms: float
    retries: int
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AggregateStats:
    """Aggregate statistics across multiple requests."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_retries: int = 0
    total_latency_ms: float = 0.0
    errors: dict[str, int] = field(default_factory=dict)  # Error type -> count

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def failure_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def average_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests

    @property
    def average_retries(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_retries / self.total_requests

    def record(self, stats: RequestStats) -> None:
        """Record a request's statistics."""
        self.total_requests += 1
        self.total_retries += stats.retries
        self.total_latency_ms += stats.latency_ms

        if stats.success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if stats.error:
                error_type = type(stats.error).__name__ if isinstance(stats.error, Exception) else str(stats.error)[:50]
                self.errors[error_type] = self.errors.get(error_type, 0) + 1
