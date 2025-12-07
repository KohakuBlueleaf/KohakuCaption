"""
Output format definitions with paired prompt instructions and validators.
"""

from kohakucaption.formats.base import OutputFormat, FormatField
from kohakucaption.formats.default import DefaultFormat
from kohakucaption.formats.json import JsonFormat

__all__ = [
    "OutputFormat",
    "FormatField",
    "DefaultFormat",
    "JsonFormat",
]
