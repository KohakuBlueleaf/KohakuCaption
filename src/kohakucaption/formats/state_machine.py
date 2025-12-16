"""
State machine parser for structured text formats.

This module provides a reusable state machine for parsing key-value formats
where sections are delimited by markers (e.g., "# field_name").

Users can customize:
- Section marker pattern (default: lines starting with #)
- Valid field names (only these are recognized as section headers)
- Type conversion rules
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Section:
    """A parsed section with name and content."""

    name: str
    content: str
    start_line: int
    end_line: int


class SectionMatcher(ABC):
    """Base class for matching section headers."""

    @abstractmethod
    def match(self, line: str) -> str | None:
        """
        Check if line is a section header.

        Returns:
            Field name if this line starts a section, None otherwise.
        """
        pass


class HashSectionMatcher(SectionMatcher):
    """
    Matches lines starting with # followed by a field name.

    Example:
        # aesthetic_score
        # description
    """

    def __init__(self, valid_fields: set[str], case_sensitive: bool = False):
        """
        Args:
            valid_fields: Set of valid field names to recognize.
            case_sensitive: Whether field matching is case-sensitive.
        """
        self.case_sensitive = case_sensitive
        if case_sensitive:
            self.valid_fields = valid_fields
        else:
            self.valid_fields = {f.lower() for f in valid_fields}

    def match(self, line: str) -> str | None:
        stripped = line.strip()
        if not stripped.startswith("#"):
            return None

        # Extract word after #
        rest = stripped[1:].strip()
        parts = rest.split(None, 1)
        if not parts:
            return None

        potential_field = parts[0]
        check_field = (
            potential_field if self.case_sensitive else potential_field.lower()
        )

        if check_field in self.valid_fields:
            return check_field
        return None


class BracketSectionMatcher(SectionMatcher):
    """
    Matches lines like [field_name].

    Example:
        [aesthetic_score]
        [description]
    """

    def __init__(self, valid_fields: set[str], case_sensitive: bool = False):
        self.case_sensitive = case_sensitive
        if case_sensitive:
            self.valid_fields = valid_fields
        else:
            self.valid_fields = {f.lower() for f in valid_fields}

    def match(self, line: str) -> str | None:
        stripped = line.strip()
        if not (stripped.startswith("[") and stripped.endswith("]")):
            return None

        potential_field = stripped[1:-1].strip()
        check_field = (
            potential_field if self.case_sensitive else potential_field.lower()
        )

        if check_field in self.valid_fields:
            return check_field
        return None


class XMLTagMatcher(SectionMatcher):
    """
    Matches XML-style opening tags like <field_name>.

    Example:
        <aesthetic_score>
        <description>
    """

    def __init__(self, valid_fields: set[str], case_sensitive: bool = False):
        self.case_sensitive = case_sensitive
        if case_sensitive:
            self.valid_fields = valid_fields
        else:
            self.valid_fields = {f.lower() for f in valid_fields}
        self.pattern = re.compile(r"<(\w+)>")

    def match(self, line: str) -> str | None:
        stripped = line.strip()
        m = self.pattern.match(stripped)
        if not m:
            return None

        potential_field = m.group(1)
        check_field = (
            potential_field if self.case_sensitive else potential_field.lower()
        )

        if check_field in self.valid_fields:
            return check_field
        return None


class StateMachineParser:
    """
    State machine for parsing structured text into sections.

    The parser iterates through lines and uses a SectionMatcher to detect
    section boundaries. Only lines matching known field names are treated
    as new sections; all other lines are content.

    Example:
        matcher = HashSectionMatcher({"title", "description"})
        parser = StateMachineParser(matcher)
        sections = parser.parse(text)
    """

    def __init__(
        self,
        matcher: SectionMatcher,
        strip_content: bool = True,
        preserve_blank_lines: bool = True,
    ):
        """
        Args:
            matcher: SectionMatcher to detect section headers.
            strip_content: Whether to strip leading/trailing whitespace from content.
            preserve_blank_lines: Whether to keep blank lines within sections.
        """
        self.matcher = matcher
        self.strip_content = strip_content
        self.preserve_blank_lines = preserve_blank_lines

    def parse(self, text: str) -> list[Section]:
        """
        Parse text into a list of sections.

        Args:
            text: The text to parse.

        Returns:
            List of Section objects in order of appearance.
        """
        lines = text.split("\n")
        sections = []

        current_field: str | None = None
        current_lines: list[str] = []
        current_start: int = 0

        def save_current(end_line: int):
            nonlocal current_field, current_lines
            if current_field is not None and current_lines:
                content = "\n".join(current_lines)
                if self.strip_content:
                    content = content.strip()
                if content:
                    sections.append(
                        Section(
                            name=current_field,
                            content=content,
                            start_line=current_start,
                            end_line=end_line,
                        )
                    )
            current_lines = []

        for line_num, line in enumerate(lines):
            field_name = self.matcher.match(line)

            if field_name is not None:
                # This line is a section header
                save_current(line_num - 1)
                current_field = field_name
                current_start = line_num
            elif current_field is not None:
                # Inside a section, collect content
                if self.preserve_blank_lines or line.strip():
                    current_lines.append(line)

        # Save the last section
        save_current(len(lines) - 1)

        return sections

    def parse_to_dict(self, text: str) -> dict[str, str]:
        """
        Parse text and return as a dictionary.

        If a field appears multiple times, later occurrences overwrite earlier ones.

        Args:
            text: The text to parse.

        Returns:
            Dictionary mapping field names to their content.
        """
        sections = self.parse(text)
        return {s.name: s.content for s in sections}


# Type converters for common types
def convert_float(value: str) -> float:
    """Convert string to float, extracting number if needed."""
    try:
        return float(value)
    except ValueError:
        # Try to extract number from text
        match = re.search(r"[\d.]+", value)
        if match:
            return float(match.group())
        return 0.0


def convert_int(value: str) -> int:
    """Convert string to int, extracting number if needed."""
    try:
        return int(value)
    except ValueError:
        match = re.search(r"\d+", value)
        if match:
            return int(match.group())
        return 0


def convert_list(value: str, separators: str = r"[,\n]") -> list[str]:
    """Convert string to list by splitting on separators."""
    items = re.split(separators, value)
    return [item.strip() for item in items if item.strip()]


# Converter registry
TYPE_CONVERTERS: dict[str, Callable[[str], Any]] = {
    "float": convert_float,
    "int": convert_int,
    "list": convert_list,
    "str": lambda x: x,
}


def apply_type_conversions(
    data: dict[str, str],
    type_map: dict[str, str],
) -> dict[str, Any]:
    """
    Apply type conversions to parsed data.

    Args:
        data: Dictionary of field name -> string value.
        type_map: Dictionary of field name -> type name (float, int, list, str).

    Returns:
        Dictionary with converted values.
    """
    result = {}
    for name, value in data.items():
        field_type = type_map.get(name, "str")
        converter = TYPE_CONVERTERS.get(field_type, lambda x: x)
        result[name] = converter(value)
    return result
