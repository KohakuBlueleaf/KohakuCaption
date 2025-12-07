#!/usr/bin/env python3
"""
Caption a single image using MLLM APIs (OpenAI or OpenRouter).

Generates structured captions with scores and descriptions, validates output,
and optionally counts tokens using a HuggingFace tokenizer.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import click

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kohakucaption.clients import ClientConfig, OpenAIClient, OpenRouterClient
from kohakucaption.formats import DefaultFormat
from kohakucaption.tokenizer import TokenCounter
from kohakucaption.types import ImageInput


CAPTION_PROMPT_TEMPLATE = """Provide a structured caption for this image.

{format_instruction}

Additional requirements:
- Cover ALL visual elements: subjects, attributes, actions, background, environment, lighting, colors, composition, artistic style
- Never omit background or secondary elements
- If nsfw_score > 0.3, describe explicit/sexual content in detail"""


def get_api_key(provider: str, api_key: str | None) -> str:
    """Get API key from argument or environment variable."""
    if api_key:
        return api_key
    env_var = "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
    key = os.environ.get(env_var)
    if not key:
        raise click.ClickException(
            f"API key not provided. Set {env_var} environment variable or use --api-key"
        )
    return key


def get_default_model(provider: str) -> str:
    """Get default model for the specified provider."""
    return "openai/gpt-4o" if provider == "openrouter" else "gpt-4o"


@click.command()
@click.argument(
    "image",
    type=str,
)
@click.option(
    "--provider", "-p",
    type=click.Choice(["openai", "openrouter"]),
    default="openai",
    help="API provider to use for captioning.",
)
@click.option(
    "--model", "-m",
    type=str,
    default=None,
    help="Model to use. Defaults to gpt-4o (OpenAI) or openai/gpt-4o (OpenRouter).",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    envvar=["OPENAI_API_KEY", "OPENROUTER_API_KEY"],
    help="API key. Can also be set via OPENAI_API_KEY or OPENROUTER_API_KEY env vars.",
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output file path. If not specified, prints to stdout.",
)
@click.option(
    "--detail",
    type=click.Choice(["low", "high", "auto"]),
    default="auto",
    show_default=True,
    help="Image detail level for vision API. 'low' uses fewer tokens, 'high' for detailed images.",
)
@click.option(
    "--max-retries",
    type=int,
    default=3,
    show_default=True,
    help="Maximum retries on API failure.",
)
@click.option(
    "--timeout",
    type=float,
    default=60.0,
    show_default=True,
    help="Request timeout in seconds.",
)
@click.option(
    "--tokenizer",
    type=str,
    default="Qwen/Qwen3-0.6B",
    show_default=True,
    help="HuggingFace tokenizer for counting output tokens. Set empty to disable.",
)
@click.option(
    "--validation-retries",
    type=int,
    default=3,
    show_default=True,
    help="Maximum retries on validation/parsing failure.",
)
def main(
    image: str,
    provider: str,
    model: str | None,
    api_key: str | None,
    output: Path | None,
    detail: str,
    max_retries: int,
    timeout: float,
    tokenizer: str,
    validation_retries: int,
):
    """
    Caption a single image using MLLM APIs.

    IMAGE can be a local file path or a URL (http:// or https://).

    \b
    Examples:
        # Basic usage with OpenAI
        python caption_single_image.py image.png

        # Using OpenRouter with a specific model
        python caption_single_image.py image.png -p openrouter -m anthropic/claude-sonnet-4

        # Save output to file
        python caption_single_image.py image.png -o caption.txt

        # Use high detail for complex images
        python caption_single_image.py detailed_art.png --detail high

        # Disable token counting for faster output
        python caption_single_image.py image.png --tokenizer ""

    \b
    Output includes:
        - RAW RESPONSE: The raw model output
        - PARSED DATA: Structured JSON with scores and descriptions
        - DEFAULT FORMAT: Human-readable # key/value format
        - TOKEN COUNTS: Token counts per field (if tokenizer enabled)
        - STATS: Request statistics (retries, latency)
    """
    asyncio.run(_async_main(
        image=image,
        provider=provider,
        model=model,
        api_key=api_key,
        output=output,
        detail=detail,
        max_retries=max_retries,
        timeout=timeout,
        tokenizer=tokenizer,
        validation_retries=validation_retries,
    ))


async def _async_main(
    image: str,
    provider: str,
    model: str | None,
    api_key: str | None,
    output: Path | None,
    detail: str,
    max_retries: int,
    timeout: float,
    tokenizer: str,
    validation_retries: int,
):
    """Async main function."""
    # Get API key and model
    resolved_api_key = get_api_key(provider, api_key)
    resolved_model = model or get_default_model(provider)

    # Validate image path (if not URL)
    if not image.startswith(("http://", "https://")):
        path = Path(image)
        if not path.exists():
            raise click.ClickException(f"Image file not found: {image}")
        if not path.is_file():
            raise click.ClickException(f"Not a file: {image}")

    # Create client config
    config = ClientConfig(
        api_key=resolved_api_key,
        model=resolved_model,
        timeout=timeout,
        max_retries=max_retries,
        detail=detail,
    )

    # Create client based on provider
    if provider == "openrouter":
        client = OpenRouterClient(config)
    else:
        client = OpenAIClient(config)

    # Create format (paired instruction + parser)
    output_format = DefaultFormat()

    # Build prompt with format instruction
    prompt = CAPTION_PROMPT_TEMPLATE.format(
        format_instruction=output_format.get_format_instruction()
    )

    # Create image input
    image_input = ImageInput(source=image)

    click.echo(f"Captioning image: {image}", err=True)
    click.echo(f"Using model: {resolved_model}", err=True)

    try:
        # Try with retries for validation failures
        last_error = None
        last_raw = None

        for attempt in range(validation_retries):
            # Get raw caption
            result = await client.caption(
                image=image_input,
                prompt=prompt,
            )

            if not result.success:
                raise click.ClickException(f"API error: {result.error}")

            last_raw = result.raw_response

            # Parse with format's validator
            parse_result = output_format.parse(result.raw_response)

            if parse_result.success:
                caption_data = parse_result.data

                # Count tokens if enabled
                token_counts = None
                if tokenizer:
                    counter = TokenCounter(tokenizer)
                    token_counts = counter.count_fields(caption_data)

                # Format output
                output_str = output_format.format_output(caption_data)

                # Print raw response
                click.echo("=" * 60)
                click.echo("RAW RESPONSE:")
                click.echo("=" * 60)
                click.echo(result.raw_response)

                # Print parsed data as JSON
                click.echo("\n" + "=" * 60)
                click.echo("PARSED DATA:")
                click.echo("=" * 60)
                click.echo(json.dumps(caption_data, indent=2, ensure_ascii=False))

                # Print formatted output
                click.echo("\n" + "=" * 60)
                click.echo("DEFAULT FORMAT:")
                click.echo("=" * 60)
                click.echo(output_str)

                # Print token counts
                if token_counts:
                    click.echo("\n" + "=" * 60)
                    click.echo(f"TOKEN COUNTS ({tokenizer}):")
                    click.echo("=" * 60)
                    for field, count in token_counts.items():
                        click.echo(f"  {field}: {count}")
                    click.echo(f"  total: {sum(token_counts.values())}")

                # Print stats
                click.echo("\n" + "=" * 60)
                click.echo("STATS:")
                click.echo("=" * 60)
                click.echo(f"  retries: {result.retries_used}")
                click.echo(f"  validation_attempts: {attempt + 1}")
                click.echo(f"  latency: {result.latency_ms:.0f}ms")

                # Save to file if specified
                if output:
                    output.write_text(output_str, encoding="utf-8")
                    click.echo(f"\nCaption saved to: {output}")

                break
            else:
                last_error = parse_result.error
                click.echo(
                    f"Validation failed (attempt {attempt + 1}/{validation_retries}): {last_error}",
                    err=True
                )
        else:
            # All attempts failed
            raise click.ClickException(
                f"Validation failed after {validation_retries} attempts: {last_error}\n"
                f"Raw response: {last_raw[:500] if last_raw else 'None'}"
            )

    finally:
        await client.close()


if __name__ == "__main__":
    main()
