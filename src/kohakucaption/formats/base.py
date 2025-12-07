"""
Base output format definition.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FormatField:
    """Definition of a field in the output format."""

    name: str
    description: str
    field_type: str = "str"  # str, float, int, list
    required: bool = True


@dataclass
class ParseResult:
    """Result of parsing output."""

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    raw: str | None = None


class OutputFormat(ABC):
    """
    Base class for output formats.
    Pairs format instructions (for prompt) with parser/validator.
    """

    name: str
    fields: list[FormatField]

    @abstractmethod
    def get_format_instruction(self) -> str:
        """Get the format instruction to include in the prompt."""
        pass

    @abstractmethod
    def parse(self, output: str) -> ParseResult:
        """Parse and validate the model output."""
        pass

    def format_output(self, data: dict[str, Any]) -> str:
        """Format parsed data back to string representation."""
        pass

    def get_field_names(self) -> list[str]:
        """Get list of field names."""
        return [f.name for f in self.fields]
