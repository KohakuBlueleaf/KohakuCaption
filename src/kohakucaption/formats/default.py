"""
Default output format: # key\\nvalue format.
"""

import re
from typing import Any

from kohakucaption.formats.base import FormatField, OutputFormat, ParseResult


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

    def get_format_instruction(self) -> str:
        """Get format instruction for the prompt."""
        lines = ["Respond in this exact format (# key followed by value on next line):"]
        lines.append("")

        for f in self.fields:
            lines.append(f"# {f.name}")
            if f.field_type == "float":
                lines.append(f"<{f.description}>")
            else:
                lines.append(f"<{f.description}>")
            lines.append("")

        lines.append("Rules:")
        lines.append("- Each field starts with # followed by the field name")
        lines.append("- Value goes on the next line(s) until the next # or end")
        lines.append("- No extra formatting, no markdown, no JSON")
        lines.append(
            "- Write in direct declarative sentences, never use 'this image shows' or similar"
        )
        lines.append(
            "- Cover ALL visual elements: subjects, actions, background, lighting, colors, style"
        )
        lines.append("- If nsfw_score > 0.3, describe explicit content in detail")

        return "\n".join(lines)

    def parse(self, output: str) -> ParseResult:
        """Parse the # key / value format."""
        try:
            output = output.strip()
            data = {}

            # Pattern to match # key followed by value
            # Captures: key name and everything until next # or end
            pattern = r"#\s*(\w+)\s*\n([\s\S]*?)(?=\n#\s*\w+|$)"
            matches = re.findall(pattern, output)

            if not matches:
                return ParseResult(
                    success=False,
                    error="No valid # key / value pairs found",
                    raw=output,
                )

            for key, value in matches:
                key = key.strip().lower()
                value = value.strip()

                # Find field definition
                field_def = next((f for f in self.fields if f.name == key), None)

                if field_def:
                    # Convert type
                    if field_def.field_type == "float":
                        try:
                            data[key] = float(value)
                        except ValueError:
                            # Try to extract number from text
                            num_match = re.search(r"[\d.]+", value)
                            if num_match:
                                data[key] = float(num_match.group())
                            else:
                                data[key] = 0.0
                    elif field_def.field_type == "int":
                        try:
                            data[key] = int(value)
                        except ValueError:
                            num_match = re.search(r"\d+", value)
                            if num_match:
                                data[key] = int(num_match.group())
                            else:
                                data[key] = 0
                    elif field_def.field_type == "list":
                        # Split by comma or newline
                        items = re.split(r"[,\n]", value)
                        data[key] = [item.strip() for item in items if item.strip()]
                    else:
                        data[key] = value
                else:
                    # Unknown field, store as string
                    data[key] = value

            # Check required fields
            missing = []
            for f in self.fields:
                if f.required and f.name not in data:
                    missing.append(f.name)

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
