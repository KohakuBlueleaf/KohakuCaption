"""
Context system for KohakuCaption.
Provides template-based context formatting and context providers.
"""

from kohakucaption.context.template import TemplateEngine
from kohakucaption.context.providers import (
    ContextProvider,
    MetadataProvider,
    BBoxProvider,
    DepthMapProvider,
    TagsProvider,
)

__all__ = [
    "TemplateEngine",
    "ContextProvider",
    "MetadataProvider",
    "BBoxProvider",
    "DepthMapProvider",
    "TagsProvider",
]
