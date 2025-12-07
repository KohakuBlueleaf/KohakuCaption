"""
Tokenizer utilities for counting tokens in caption fields.
"""

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str = DEFAULT_TOKENIZER):
    """
    Get a cached tokenizer instance.

    Args:
        model_name: HuggingFace model name or path

    Returns:
        Tokenizer instance
    """
    from transformers import AutoTokenizer

    logger.debug(f"Loading tokenizer: {model_name}")
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def count_tokens(text: str, tokenizer_name: str = DEFAULT_TOKENIZER) -> int:
    """
    Count tokens in a text string.

    Args:
        text: Text to tokenize
        tokenizer_name: HuggingFace model name for tokenizer

    Returns:
        Number of tokens
    """
    if not text:
        return 0
    tokenizer = get_tokenizer(tokenizer_name)
    return len(tokenizer.encode(text, add_special_tokens=False))


def count_tokens_batch(texts: list[str], tokenizer_name: str = DEFAULT_TOKENIZER) -> list[int]:
    """
    Count tokens for multiple texts efficiently.

    Args:
        texts: List of texts to tokenize
        tokenizer_name: HuggingFace model name for tokenizer

    Returns:
        List of token counts
    """
    if not texts:
        return []
    tokenizer = get_tokenizer(tokenizer_name)
    return [
        len(tokenizer.encode(text, add_special_tokens=False)) if text else 0
        for text in texts
    ]


def count_caption_tokens(
    caption: dict[str, Any],
    tokenizer_name: str = DEFAULT_TOKENIZER,
    fields: list[str] | None = None,
) -> dict[str, int]:
    """
    Count tokens for each field in a caption dict.

    Args:
        caption: Caption dictionary with string fields
        tokenizer_name: HuggingFace model name for tokenizer
        fields: Specific fields to count (None = all string fields)

    Returns:
        Dict mapping field names to token counts
    """
    tokenizer = get_tokenizer(tokenizer_name)
    result = {}

    for key, value in caption.items():
        if fields is not None and key not in fields:
            continue

        if isinstance(value, str):
            result[key] = len(tokenizer.encode(value, add_special_tokens=False))
        elif isinstance(value, list):
            # For list fields like tags, count total tokens
            total = 0
            for item in value:
                if isinstance(item, str):
                    total += len(tokenizer.encode(item, add_special_tokens=False))
            result[key] = total

    return result


class TokenCounter:
    """
    Reusable token counter with configurable tokenizer.

    Example:
        counter = TokenCounter("Qwen/Qwen3-0.6B")
        count = counter.count("Hello world")
        field_counts = counter.count_fields(caption_dict)
    """

    def __init__(self, tokenizer_name: str = DEFAULT_TOKENIZER):
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy-load tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer(self.tokenizer_name)
        return self._tokenizer

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_batch(self, texts: list[str]) -> list[int]:
        """Count tokens for multiple texts."""
        return [self.count(t) for t in texts]

    def count_fields(
        self,
        data: dict[str, Any],
        fields: list[str] | None = None,
    ) -> dict[str, int]:
        """
        Count tokens for each field in a dict.

        Args:
            data: Dictionary with string/list fields
            fields: Specific fields to count (None = all)

        Returns:
            Dict mapping field names to token counts
        """
        result = {}

        for key, value in data.items():
            if fields is not None and key not in fields:
                continue

            if isinstance(value, str):
                result[key] = self.count(value)
            elif isinstance(value, list):
                total = sum(self.count(item) for item in value if isinstance(item, str))
                result[key] = total

        return result

    def count_total(self, data: dict[str, Any], fields: list[str] | None = None) -> int:
        """Get total token count across all fields."""
        return sum(self.count_fields(data, fields).values())
