"""
Context providers for different types of image context.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from kohakucaption.types import ContextItem, ContextType


class ContextProvider(ABC):
    """Base class for context providers."""

    context_type: ContextType

    @abstractmethod
    def format(self, item: ContextItem) -> str:
        """Format the context item as a string for the prompt."""
        pass

    @abstractmethod
    def validate(self, data: dict[str, Any]) -> bool:
        """Validate that the data has the required fields."""
        pass


class MetadataProvider(ContextProvider):
    """Provider for image metadata context."""

    context_type = ContextType.METADATA

    def validate(self, data: dict[str, Any]) -> bool:
        """Metadata is flexible, any dict is valid."""
        return isinstance(data, dict)

    def format(self, item: ContextItem) -> str:
        """Format metadata as a readable string."""
        lines = ["Image Metadata:"]
        for key, value in item.data.items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for k, v in value.items():
                    lines.append(f"    {k}: {v}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


@dataclass
class BoundingBox:
    """Represents a bounding box."""

    x: float
    y: float
    width: float
    height: float
    label: str | None = None
    confidence: float | None = None


class BBoxProvider(ContextProvider):
    """Provider for object bounding box context."""

    context_type = ContextType.OBJECT_BBOX

    def validate(self, data: dict[str, Any]) -> bool:
        """Validate bbox data structure."""
        if "boxes" not in data:
            return False
        boxes = data["boxes"]
        if not isinstance(boxes, list):
            return False
        for box in boxes:
            if not all(k in box for k in ("x", "y", "width", "height")):
                return False
        return True

    def format(self, item: ContextItem) -> str:
        """Format bounding boxes as readable text."""
        lines = ["Detected Objects:"]
        for i, box in enumerate(item.data.get("boxes", []), 1):
            label = box.get("label", f"Object {i}")
            conf = box.get("confidence")
            conf_str = f" ({conf:.1%})" if conf is not None else ""
            lines.append(
                f"  - {label}{conf_str}: "
                f"x={box['x']:.0f}, y={box['y']:.0f}, "
                f"w={box['width']:.0f}, h={box['height']:.0f}"
            )
        return "\n".join(lines)


class DepthMapProvider(ContextProvider):
    """Provider for depth map context."""

    context_type = ContextType.DEPTH_MAP

    def validate(self, data: dict[str, Any]) -> bool:
        """Validate depth map data."""
        # At minimum, we expect some depth information
        return "description" in data or "regions" in data or "stats" in data

    def format(self, item: ContextItem) -> str:
        """Format depth map information."""
        lines = ["Depth Information:"]

        if "description" in item.data:
            lines.append(f"  Overview: {item.data['description']}")

        if "stats" in item.data:
            stats = item.data["stats"]
            lines.append(f"  Depth range: {stats.get('min', 'N/A')} - {stats.get('max', 'N/A')}")
            if "mean" in stats:
                lines.append(f"  Mean depth: {stats['mean']}")

        if "regions" in item.data:
            lines.append("  Regions:")
            for region in item.data["regions"]:
                name = region.get("name", "Unknown")
                depth = region.get("depth", "N/A")
                lines.append(f"    - {name}: depth={depth}")

        return "\n".join(lines)


class SegmentationProvider(ContextProvider):
    """Provider for segmentation mask context."""

    context_type = ContextType.SEGMENTATION

    def validate(self, data: dict[str, Any]) -> bool:
        """Validate segmentation data."""
        return "segments" in data or "classes" in data

    def format(self, item: ContextItem) -> str:
        """Format segmentation information."""
        lines = ["Segmentation:"]

        if "segments" in item.data:
            for seg in item.data["segments"]:
                label = seg.get("label", "Unknown")
                area = seg.get("area_percent")
                area_str = f" ({area:.1f}%)" if area is not None else ""
                lines.append(f"  - {label}{area_str}")

        if "classes" in item.data:
            lines.append("  Detected classes: " + ", ".join(item.data["classes"]))

        return "\n".join(lines)


class OCRProvider(ContextProvider):
    """Provider for OCR text context."""

    context_type = ContextType.OCR

    def validate(self, data: dict[str, Any]) -> bool:
        """Validate OCR data."""
        return "text" in data or "blocks" in data

    def format(self, item: ContextItem) -> str:
        """Format OCR results."""
        lines = ["Text detected in image:"]

        if "text" in item.data:
            # Simple text output
            lines.append(f'  "{item.data["text"]}"')

        if "blocks" in item.data:
            # Structured text blocks
            for block in item.data["blocks"]:
                text = block.get("text", "")
                conf = block.get("confidence")
                conf_str = f" (confidence: {conf:.1%})" if conf is not None else ""
                lines.append(f'  - "{text}"{conf_str}')

        return "\n".join(lines)


class TagsProvider(ContextProvider):
    """Provider for existing tags/labels context."""

    context_type = ContextType.TAGS

    def validate(self, data: dict[str, Any]) -> bool:
        """Validate tags data."""
        return "tags" in data

    def format(self, item: ContextItem) -> str:
        """Format tags."""
        tags = item.data.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]

        # Group by category if available
        if "categories" in item.data:
            lines = ["Existing tags:"]
            for cat, cat_tags in item.data["categories"].items():
                lines.append(f"  {cat}: {', '.join(cat_tags)}")
            return "\n".join(lines)

        return f"Existing tags: {', '.join(tags)}"


class CustomProvider(ContextProvider):
    """Provider for custom context with flexible formatting."""

    context_type = ContextType.CUSTOM

    def __init__(self, format_template: str | None = None):
        """
        Initialize with optional format template.

        Args:
            format_template: Template string with {key} placeholders
        """
        self.format_template = format_template

    def validate(self, data: dict[str, Any]) -> bool:
        """Any dict is valid for custom context."""
        return isinstance(data, dict)

    def format(self, item: ContextItem) -> str:
        """Format custom context."""
        if self.format_template:
            # Use template
            result = self.format_template
            for key, value in item.data.items():
                placeholder = "{" + key + "}"
                if placeholder in result:
                    if isinstance(value, (list, dict)):
                        value = json.dumps(value, ensure_ascii=False)
                    result = result.replace(placeholder, str(value))
            return result

        # Default: format as labeled JSON
        label = item.label or "Additional Context"
        return f"{label}:\n{json.dumps(item.data, indent=2, ensure_ascii=False)}"


# Registry of all providers
PROVIDERS: dict[ContextType, type[ContextProvider]] = {
    ContextType.METADATA: MetadataProvider,
    ContextType.OBJECT_BBOX: BBoxProvider,
    ContextType.DEPTH_MAP: DepthMapProvider,
    ContextType.SEGMENTATION: SegmentationProvider,
    ContextType.OCR: OCRProvider,
    ContextType.TAGS: TagsProvider,
    ContextType.CUSTOM: CustomProvider,
}


def get_provider(context_type: ContextType) -> ContextProvider:
    """Get a provider instance for the given context type."""
    provider_class = PROVIDERS.get(context_type)
    if provider_class is None:
        raise ValueError(f"No provider for context type: {context_type}")
    return provider_class()


def format_all_context(items: list[ContextItem]) -> str:
    """Format all context items into a single string."""
    if not items:
        return ""

    parts = []
    for item in items:
        provider = get_provider(item.context_type)
        if provider.validate(item.data):
            parts.append(provider.format(item))
        else:
            # Still include but mark as unvalidated
            parts.append(f"[Unvalidated {item.context_type.value}]: {item.data}")

    return "\n\n".join(parts)
