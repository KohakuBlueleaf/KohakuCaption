"""
Image tagger module for generating tags using local models.
"""

from kohakucaption.tagger.animetimm import AnimeTimmTagger, AnimeTimmTagResult
from kohakucaption.tagger.pixai import PixAITagger, TagResult

__all__ = [
    "PixAITagger",
    "TagResult",
    "AnimeTimmTagger",
    "AnimeTimmTagResult",
]
