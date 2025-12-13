"""
Output validators for structured response verification.
Supports Pydantic models, JSON Schema, regex patterns, and custom validators.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T")


@dataclass
class ValidationResult(Generic[T]):
    """Result of a validation attempt."""

    valid: bool
    value: T | None = None  # Parsed/converted value if successful
    error: str | None = None
    raw_input: str | None = None


class OutputValidator(ABC, Generic[T]):
    """Base class for output validators."""

    @abstractmethod
    def validate(self, output: str) -> ValidationResult[T]:
        """
        Validate and optionally parse the output string.

        Args:
            output: Raw string output from the model

        Returns:
            ValidationResult with parsed value or error
        """
        pass

    @abstractmethod
    def get_schema_description(self) -> str:
        """Get a description of the expected output format for prompts."""
        pass


class PydanticValidator(OutputValidator[T]):
    """Validator using Pydantic models for type-safe parsing."""

    def __init__(
        self,
        model: type[BaseModel],
        strict: bool = False,
        extract_json: bool = True,
    ):
        """
        Initialize with a Pydantic model.

        Args:
            model: Pydantic BaseModel class
            strict: If True, use strict validation mode
            extract_json: If True, try to extract JSON from markdown code blocks
        """
        self.model = model
        self.strict = strict
        self.extract_json = extract_json

    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON from text, handling markdown code blocks."""
        text = text.strip()

        # Try to extract from markdown code block
        json_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        match = re.search(json_block_pattern, text)
        if match:
            return match.group(1).strip()

        # Check if it starts with { or [
        if text.startswith(("{", "[")):
            # Find matching closing bracket
            return text

        return text

    def validate(self, output: str) -> ValidationResult[T]:
        """Validate output against the Pydantic model."""
        try:
            text = output
            if self.extract_json:
                text = self._extract_json_from_text(output)

            # Parse JSON first
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                return ValidationResult(
                    valid=False,
                    error=f"Invalid JSON: {e}",
                    raw_input=output,
                )

            # Validate with Pydantic
            if self.strict:
                instance = self.model.model_validate(data, strict=True)
            else:
                instance = self.model.model_validate(data)

            return ValidationResult(
                valid=True,
                value=instance,  # type: ignore
                raw_input=output,
            )

        except ValidationError as e:
            return ValidationResult(
                valid=False,
                error=f"Validation error: {e}",
                raw_input=output,
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                error=f"Unexpected error: {e}",
                raw_input=output,
            )

    def get_schema_description(self) -> str:
        """Get JSON schema description from the Pydantic model."""
        schema = self.model.model_json_schema()
        return json.dumps(schema, indent=2)


class JsonSchemaValidator(OutputValidator[dict[str, Any]]):
    """Validator using JSON Schema for flexible validation."""

    def __init__(
        self,
        schema: dict[str, Any],
        extract_json: bool = True,
    ):
        """
        Initialize with a JSON Schema.

        Args:
            schema: JSON Schema dict
            extract_json: If True, try to extract JSON from markdown code blocks
        """
        self.schema = schema
        self.extract_json = extract_json
        self._validator = None

    def _get_validator(self):
        """Lazy-load jsonschema validator."""
        if self._validator is None:
            try:
                import jsonschema

                self._validator = jsonschema.Draft7Validator(self.schema)
            except ImportError:
                raise ImportError(
                    "jsonschema is required for JsonSchemaValidator. "
                    "Install with: pip install jsonschema"
                )
        return self._validator

    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON from text, handling markdown code blocks."""
        text = text.strip()

        json_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        match = re.search(json_block_pattern, text)
        if match:
            return match.group(1).strip()

        if text.startswith(("{", "[")):
            return text

        return text

    def validate(self, output: str) -> ValidationResult[dict[str, Any]]:
        """Validate output against the JSON Schema."""
        try:
            text = output
            if self.extract_json:
                text = self._extract_json_from_text(output)

            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                return ValidationResult(
                    valid=False,
                    error=f"Invalid JSON: {e}",
                    raw_input=output,
                )

            validator = self._get_validator()
            errors = list(validator.iter_errors(data))

            if errors:
                error_messages = [f"- {e.message}" for e in errors[:5]]
                return ValidationResult(
                    valid=False,
                    error=f"Schema validation errors:\n" + "\n".join(error_messages),
                    raw_input=output,
                )

            return ValidationResult(
                valid=True,
                value=data,
                raw_input=output,
            )

        except Exception as e:
            return ValidationResult(
                valid=False,
                error=f"Unexpected error: {e}",
                raw_input=output,
            )

    def get_schema_description(self) -> str:
        """Get the JSON Schema as a string."""
        return json.dumps(self.schema, indent=2)


class RegexValidator(OutputValidator[str]):
    """Validator using regex pattern matching."""

    def __init__(
        self,
        pattern: str,
        flags: int = 0,
        extract_group: int | str | None = None,
    ):
        """
        Initialize with a regex pattern.

        Args:
            pattern: Regex pattern to match
            flags: Regex flags (e.g., re.IGNORECASE)
            extract_group: Group number or name to extract as value
        """
        self.pattern = re.compile(pattern, flags)
        self.extract_group = extract_group

    def validate(self, output: str) -> ValidationResult[str]:
        """Validate output matches the regex pattern."""
        match = self.pattern.search(output)

        if match:
            if self.extract_group is not None:
                try:
                    value = match.group(self.extract_group)
                except (IndexError, KeyError):
                    value = match.group(0)
            else:
                value = match.group(0)

            return ValidationResult(
                valid=True,
                value=value,
                raw_input=output,
            )

        return ValidationResult(
            valid=False,
            error=f"Output does not match pattern: {self.pattern.pattern}",
            raw_input=output,
        )

    def get_schema_description(self) -> str:
        """Get regex pattern description."""
        return f"Output must match regex pattern: {self.pattern.pattern}"


class CallableValidator(OutputValidator[T]):
    """Validator using a custom callable function."""

    def __init__(
        self,
        validator_func: Callable[[str], T | None],
        error_message: str = "Validation failed",
        description: str = "Custom validation",
    ):
        """
        Initialize with a validator function.

        Args:
            validator_func: Function that takes output string and returns
                           parsed value or None if invalid
            error_message: Error message when validation fails
            description: Description for prompts
        """
        self.validator_func = validator_func
        self.error_message = error_message
        self.description = description

    def validate(self, output: str) -> ValidationResult[T]:
        """Validate output using the custom function."""
        try:
            result = self.validator_func(output)
            if result is not None:
                return ValidationResult(
                    valid=True,
                    value=result,
                    raw_input=output,
                )
            return ValidationResult(
                valid=False,
                error=self.error_message,
                raw_input=output,
            )
        except Exception as e:
            return ValidationResult(
                valid=False,
                error=f"{self.error_message}: {e}",
                raw_input=output,
            )

    def get_schema_description(self) -> str:
        """Get the description."""
        return self.description


class CompositeValidator(OutputValidator[T]):
    """Validator that tries multiple validators in order."""

    def __init__(
        self,
        validators: list[OutputValidator],
        mode: str = "first_success",
    ):
        """
        Initialize with multiple validators.

        Args:
            validators: List of validators to try
            mode: "first_success" (return first valid result) or
                  "all" (all must pass, return last result)
        """
        self.validators = validators
        self.mode = mode

    def validate(self, output: str) -> ValidationResult[T]:
        """Validate using composite strategy."""
        errors = []

        for validator in self.validators:
            result = validator.validate(output)

            if self.mode == "first_success":
                if result.valid:
                    return result  # type: ignore
                errors.append(result.error)
            elif self.mode == "all":
                if not result.valid:
                    return result  # type: ignore

        if self.mode == "first_success":
            return ValidationResult(
                valid=False,
                error="All validators failed:\n"
                + "\n".join(f"- {e}" for e in errors if e),
                raw_input=output,
            )

        # All passed in "all" mode
        return result  # type: ignore

    def get_schema_description(self) -> str:
        """Get combined schema descriptions."""
        descriptions = [v.get_schema_description() for v in self.validators]
        return "\n\n".join(descriptions)


def create_json_extractor() -> CallableValidator[dict[str, Any]]:
    """Create a simple JSON extractor validator."""

    def extract_json(text: str) -> dict[str, Any] | None:
        text = text.strip()

        # Try markdown code block
        json_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        match = re.search(json_block_pattern, text)
        if match:
            text = match.group(1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    return CallableValidator(
        validator_func=extract_json,
        error_message="Failed to extract valid JSON",
        description="Output must be valid JSON",
    )


def create_type_coercing_validator(
    model: type[BaseModel],
) -> CallableValidator[BaseModel]:
    """
    Create a validator that attempts type coercion for Pydantic models.
    More lenient than PydanticValidator with strict=True.
    """

    def coerce_and_validate(text: str) -> BaseModel | None:
        text = text.strip()

        # Extract JSON
        json_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        match = re.search(json_block_pattern, text)
        if match:
            text = match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        # Try to coerce types
        try:
            # Pydantic handles most coercion automatically
            return model.model_validate(data)
        except ValidationError:
            return None

    return CallableValidator(
        validator_func=coerce_and_validate,
        error_message=f"Failed to validate as {model.__name__}",
        description=f"Output must be valid {model.__name__} JSON",
    )
