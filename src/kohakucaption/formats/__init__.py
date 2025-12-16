"""
Output format definitions with paired prompt instructions and validators.
"""

from kohakucaption.formats.base import OutputFormat, FormatField, ParseResult
from kohakucaption.formats.default import DefaultFormat
from kohakucaption.formats.json import JsonFormat
from kohakucaption.formats.state_machine import (
    SectionMatcher,
    HashSectionMatcher,
    BracketSectionMatcher,
    XMLTagMatcher,
    StateMachineParser,
    Section,
    apply_type_conversions,
    TYPE_CONVERTERS,
)

__all__ = [
    # Base
    "OutputFormat",
    "FormatField",
    "ParseResult",
    # Formats
    "DefaultFormat",
    "JsonFormat",
    # State machine
    "SectionMatcher",
    "HashSectionMatcher",
    "BracketSectionMatcher",
    "XMLTagMatcher",
    "StateMachineParser",
    "Section",
    "apply_type_conversions",
    "TYPE_CONVERTERS",
]
