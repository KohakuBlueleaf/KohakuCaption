"""
Default output format: # key\\nvalue format.
"""

from typing import Any

from kohakucaption.formats.base import FormatField, OutputFormat, ParseResult
from kohakucaption.formats.state_machine import (
    HashSectionMatcher,
    StateMachineParser,
    apply_type_conversions,
)


# Default fields for caption
DEFAULT_CAPTION_FIELDS = [
    FormatField("aesthetic_score", "Aesthetic quality score 0.0-1.0", "float"),
    FormatField(
        "nsfw_score", "NSFW content score 0.0-1.0 (0=safe, 1=explicit)", "float"
    ),
    FormatField("quality_score", "Technical quality score 0.0-1.0", "float"),
    FormatField("title", "Short descriptive title (5-10 words)", "str"),
    FormatField("brief", "One-sentence summary", "str"),
    FormatField(
        "description",
        "One comprehensive paragraph covering ALL visual elements. "
        "If nsfw_score > 0.3, describe explicit content in detail.",
        "str",
    ),
]


class DefaultFormat(OutputFormat):
    """
    Default format using # key / value pairs.

    Format:
        # field_name
        field_value

        # field_name2
        field_value2
    """

    name = "default"

    def __init__(self, fields: list[FormatField] | None = None):
        self.fields = fields or DEFAULT_CAPTION_FIELDS
        self._init_parser()

    def _init_parser(self):
        """Initialize the state machine parser."""
        valid_fields = {f.name for f in self.fields}
        self.matcher = HashSectionMatcher(valid_fields, case_sensitive=False)
        self.parser = StateMachineParser(self.matcher)
        self.type_map = {f.name: f.field_type for f in self.fields}

    def get_format_instruction(self) -> str:
        """Get format instruction for the prompt."""
        # Build field names list for emphasis
        field_names = [f.name for f in self.fields]

        lines = [
            "IMPORTANT: You MUST respond using EXACTLY this format with these EXACT field names:",
            f"Required fields in order: {', '.join(field_names)}",
            "",
            "Format template:",
        ]

        for f in self.fields:
            lines.append(f"# {f.name}")
            lines.append(f"<{f.description}>")
            lines.append("")

        lines.append("STRICT RULES:")
        lines.append(
            f"- You MUST use EXACTLY these field names: {', '.join(field_names)}"
        )
        lines.append("- The FIRST line MUST be: # aesthetic_score")
        lines.append(
            "- Each field starts with # followed by the EXACT field name above"
        )
        lines.append("- Value goes on the next line(s) until the next # field or end")
        lines.append("- Do NOT use any other field names or formats")
        lines.append("- No markdown, no JSON, no extra formatting")
        lines.append(
            "- Write in direct declarative sentences, never use 'this image shows' or similar"
        )
        lines.append(
            "- Cover ALL visual elements: subjects, actions, background, lighting, colors, style"
        )
        lines.append("- If nsfw_score > 0.3, describe explicit content in detail")

        return "\n".join(lines)

    def parse(self, output: str) -> ParseResult:
        """Parse the # key / value format using state machine."""
        try:
            output = output.strip()

            # Parse using state machine
            data = self.parser.parse_to_dict(output)

            if not data:
                return ParseResult(
                    success=False,
                    error="No valid # key / value pairs found",
                    raw=output,
                )

            # Apply type conversions
            data = apply_type_conversions(data, self.type_map)

            # Check required fields
            missing = [f.name for f in self.fields if f.required and f.name not in data]

            if missing:
                return ParseResult(
                    success=False,
                    error=f"Missing required fields: {', '.join(missing)}",
                    data=data,
                    raw=output,
                )

            return ParseResult(
                success=True,
                data=data,
                raw=output,
            )

        except Exception as e:
            return ParseResult(
                success=False,
                error=f"Parse error: {e}",
                raw=output,
            )

    def format_output(self, data: dict[str, Any]) -> str:
        """Format data back to # key / value format."""
        lines = []
        for key, value in data.items():
            lines.append(f"# {key}")
            if isinstance(value, list):
                lines.append(", ".join(str(v).strip() for v in value))
            elif isinstance(value, str):
                lines.append(value.strip())
            else:
                lines.append(str(value))
            lines.append("")
        return "\n".join(lines).rstrip()
