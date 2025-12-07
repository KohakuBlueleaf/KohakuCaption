"""
Template engine for context formatting.
Supports simple {variable} substitution with optional formatting.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TemplateVariable:
    """Represents a variable found in a template."""

    name: str
    full_match: str  # The full matched string including braces
    format_spec: str | None = None  # Optional format specification
    default: str | None = None  # Optional default value


class TemplateEngine:
    """
    Simple template engine supporting {variable} substitution.

    Features:
    - Basic substitution: {var_name}
    - Default values: {var_name:default_value}
    - Format specs: {var_name!format}
    - Nested access: {obj.attr} or {dict.key}
    - Custom formatters: register_formatter("uppercase", str.upper)

    Example:
        engine = TemplateEngine()
        result = engine.render(
            "Image: {title} - Tags: {tags}",
            {"title": "Sunset", "tags": "nature, sky"}
        )
    """

    # Pattern for matching {var_name}, {var_name:default}, {var_name!format}
    VARIABLE_PATTERN = re.compile(
        r"\{(?P<name>[a-zA-Z_][a-zA-Z0-9_\.]*)"
        r"(?::(?P<default>[^}!]*))?"
        r"(?:!(?P<format>[a-zA-Z_][a-zA-Z0-9_]*))?"
        r"\}"
    )

    def __init__(self) -> None:
        self._formatters: dict[str, Callable[[Any], str]] = {}
        self._register_default_formatters()

    def _register_default_formatters(self) -> None:
        """Register built-in formatters."""
        self._formatters["upper"] = lambda x: str(x).upper()
        self._formatters["lower"] = lambda x: str(x).lower()
        self._formatters["title"] = lambda x: str(x).title()
        self._formatters["strip"] = lambda x: str(x).strip()
        self._formatters["json"] = self._format_json
        self._formatters["list"] = self._format_list
        self._formatters["csv"] = lambda x: ", ".join(str(i) for i in x) if isinstance(x, (list, tuple)) else str(x)

    @staticmethod
    def _format_json(value: Any) -> str:
        """Format value as JSON string."""
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _format_list(value: Any) -> str:
        """Format list as bullet points."""
        if isinstance(value, (list, tuple)):
            return "\n".join(f"- {item}" for item in value)
        return str(value)

    def register_formatter(self, name: str, func: Callable[[Any], str]) -> None:
        """Register a custom formatter function."""
        self._formatters[name] = func

    def parse_variables(self, template: str) -> list[TemplateVariable]:
        """Parse and extract all variables from a template."""
        variables = []
        for match in self.VARIABLE_PATTERN.finditer(template):
            variables.append(
                TemplateVariable(
                    name=match.group("name"),
                    full_match=match.group(0),
                    format_spec=match.group("format"),
                    default=match.group("default"),
                )
            )
        return variables

    def _get_nested_value(self, data: dict[str, Any], key: str) -> Any:
        """Get a nested value using dot notation (e.g., 'obj.attr.value')."""
        parts = key.split(".")
        value = data
        for part in parts:
            if isinstance(value, dict):
                if part not in value:
                    return None
                value = value[part]
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None
        return value

    def render(
        self,
        template: str,
        context: dict[str, Any],
        strict: bool = False,
    ) -> str:
        """
        Render a template with the given context.

        Args:
            template: Template string with {variable} placeholders
            context: Dictionary of variable values
            strict: If True, raise KeyError for missing variables

        Returns:
            Rendered string with variables substituted

        Raises:
            KeyError: If strict=True and a variable is missing
            ValueError: If a formatter is not found
        """

        def replace_match(match: re.Match) -> str:
            name = match.group("name")
            default = match.group("default")
            format_spec = match.group("format")

            # Get the value
            value = self._get_nested_value(context, name)

            if value is None:
                if default is not None:
                    value = default
                elif strict:
                    raise KeyError(f"Missing template variable: {name}")
                else:
                    return match.group(0)  # Return original if not strict

            # Apply formatter if specified
            if format_spec:
                if format_spec not in self._formatters:
                    raise ValueError(f"Unknown formatter: {format_spec}")
                value = self._formatters[format_spec](value)
            else:
                value = str(value)

            return value

        return self.VARIABLE_PATTERN.sub(replace_match, template)

    def validate_template(self, template: str, available_vars: set[str]) -> list[str]:
        """
        Validate a template against available variables.

        Args:
            template: Template string to validate
            available_vars: Set of available variable names

        Returns:
            List of missing variable names
        """
        variables = self.parse_variables(template)
        missing = []
        for var in variables:
            # Check top-level name for nested vars
            top_level = var.name.split(".")[0]
            if top_level not in available_vars and var.default is None:
                missing.append(var.name)
        return missing


@dataclass
class PromptTemplate:
    """
    Pre-defined prompt template with metadata.
    """

    name: str
    template: str
    description: str = ""
    required_vars: list[str] = field(default_factory=list)
    optional_vars: list[str] = field(default_factory=list)

    def render(self, context: dict[str, Any], engine: TemplateEngine | None = None) -> str:
        """Render this template with the given context."""
        if engine is None:
            engine = TemplateEngine()
        return engine.render(self.template, context)

    def validate(self, context: dict[str, Any]) -> list[str]:
        """Check if all required variables are present. Returns missing vars."""
        missing = []
        for var in self.required_vars:
            if var not in context:
                missing.append(var)
        return missing


# Pre-defined prompt templates
DEFAULT_CAPTION_TEMPLATE = PromptTemplate(
    name="default_caption",
    template="""Analyze this image and provide a structured caption.

{context}

Respond with a JSON object (in this exact order):
- aesthetic_score: Float 0.0-1.0, aesthetic quality
- nsfw_score: Float 0.0-1.0, NSFW level (0=safe, 1=explicit)
- quality_score: Float 0.0-1.0, technical quality
- title: Short descriptive title
- brief: One-sentence summary
- description: One comprehensive paragraph covering ALL visual elements - subjects, their attributes, actions, background, environment, lighting, colors, composition, and artistic style. Never omit background or secondary elements. If nsfw_score is high (>0.3), describe the explicit/sexual content in detail.

Write in direct declarative sentences describing the content. Never use phrases like "this image shows" or "the image depicts".

Respond ONLY with JSON, no markdown or extra text.""",
    description="Default template for basic image captioning",
    required_vars=[],
    optional_vars=["context"],
)

DETAILED_CAPTION_TEMPLATE = PromptTemplate(
    name="detailed_caption",
    template="""Analyze this image in detail.

{context}

Respond with a JSON object (in this exact order):
- aesthetic_score: Float 0.0-1.0, aesthetic quality
- nsfw_score: Float 0.0-1.0, NSFW level (0=safe, 1=explicit)
- quality_score: Float 0.0-1.0, technical quality
- title: Short descriptive title
- brief: One-sentence summary
- description: 2-3 comprehensive paragraphs covering ALL visual elements - subjects, their attributes, actions, background, environment, lighting, colors, composition, and artistic style. Never omit background or secondary elements. If nsfw_score is high (>0.3), describe the explicit/sexual content in detail.
- style: Art style (anime, realistic, sketch, etc.)
- mood: Overall mood/atmosphere
- colors: Array of dominant colors
- subjects: Array of main subjects/characters

Write in direct declarative sentences describing the content. Never use phrases like "this image shows" or "the image depicts".

Respond ONLY with JSON, no markdown or extra text.""",
    description="Detailed template with extended caption fields",
    required_vars=[],
    optional_vars=["context"],
)

CUSTOM_PROMPT_TEMPLATE = PromptTemplate(
    name="custom",
    template="{prompt}",
    description="Use a fully custom prompt",
    required_vars=["prompt"],
    optional_vars=[],
)
