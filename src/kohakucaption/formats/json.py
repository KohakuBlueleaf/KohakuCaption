"""
JSON output format.
"""

import json
import re
from typing import Any

from kohakucaption.formats.base import FormatField, OutputFormat, ParseResult
from kohakucaption.formats.default import DEFAULT_CAPTION_FIELDS


class JsonFormat(OutputFormat):
    """
    JSON output format.
    """

    name = "json"

    def __init__(self, fields: list[FormatField] | None = None):
        self.fields = fields or DEFAULT_CAPTION_FIELDS

    def get_format_instruction(self) -> str:
        """Get format instruction for the prompt."""
        lines = ["Respond with a JSON object containing:"]

        for f in self.fields:
            type_hint = f.field_type
            if type_hint == "str":
                type_hint = "string"
            elif type_hint == "list":
                type_hint = "array"
            lines.append(f"- {f.name}: ({type_hint}) {f.description}")

        lines.append("")
        lines.append("Rules:")
        lines.append("- Respond ONLY with valid JSON, no markdown, no extra text")
        lines.append(
            "- Write in direct declarative sentences, never use 'this image shows' or similar"
        )

        return "\n".join(lines)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text, handling markdown code blocks."""
        text = text.strip()

        # Try markdown code block
        json_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        match = re.search(json_block_pattern, text)
        if match:
            return match.group(1).strip()

        # Find JSON object
        start = text.find("{")
        if start != -1:
            # Find matching closing brace
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]

        return text

    def parse(self, output: str) -> ParseResult:
        """Parse JSON output."""
        try:
            text = self._extract_json(output)

            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                return ParseResult(
                    success=False,
                    error=f"Invalid JSON: {e}",
                    raw=output,
                )

            # Type conversion
            for f in self.fields:
                if f.name in data:
                    value = data[f.name]
                    if f.field_type == "float" and not isinstance(value, float):
                        try:
                            data[f.name] = float(value)
                        except (ValueError, TypeError):
                            data[f.name] = 0.0
                    elif f.field_type == "int" and not isinstance(value, int):
                        try:
                            data[f.name] = int(value)
                        except (ValueError, TypeError):
                            data[f.name] = 0
                    elif f.field_type == "list" and not isinstance(value, list):
                        data[f.name] = [value] if value else []

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
        """Format data as JSON."""
        return json.dumps(data, indent=2, ensure_ascii=False)
