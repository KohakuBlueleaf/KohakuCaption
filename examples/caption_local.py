#!/usr/bin/env python3
"""
Caption a single image using local VLM inference (vLLM).

Uses the DefaultFormat system to generate structured captions.
Optionally loads pre-computed tags from xxx.tag.json as context.
"""

import json
import sys
from pathlib import Path
from typing import Any

import click

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PIL import Image


def load_tags(tag_path: Path) -> dict[str, Any] | None:
    """Load tags from .tag.json file."""
    if not tag_path.exists():
        return None
    try:
        with open(tag_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "error" in data:
            return None
        return data
    except Exception:
        return None


def build_prompt_with_tags(base_prompt: str, tags: dict[str, Any] | None) -> str:
    """Build prompt with optional tags context."""
    if not tags:
        return base_prompt

    ctx_lines = []
    # Handle different tag formats
    if "general_tags" in tags:
        if tags.get("general_tags"):
            tag_list = [t.replace("_", " ") for t in tags["general_tags"]]
            ctx_lines.append(f"Features: {', '.join(tag_list)}")
        if tags.get("character_tags"):
            tag_list = [t.replace("_", " ") for t in tags["character_tags"]]
            ctx_lines.append(f"Characters: {', '.join(tag_list)}")
        if tags.get("rating_tags"):
            tag_list = [t.replace("_", " ") for t in tags["rating_tags"]]
            ctx_lines.append(f"Rating: {', '.join(tag_list)}")
    elif tags.get("general"):
        tag_list = [t.replace("_", " ") for t in tags["general"]]
        ctx_lines.append(f"Features: {', '.join(tag_list)}")
        if tags.get("character"):
            tag_list = [t.replace("_", " ") for t in tags["character"]]
            ctx_lines.append(f"Characters: {', '.join(tag_list)}")
        if tags.get("rating"):
            tag_list = [t.replace("_", " ") for t in tags["rating"]]
            ctx_lines.append(f"Rating: {', '.join(tag_list)}")

    if ctx_lines:
        return f"{base_prompt}\n\nExisting tags:\n" + "\n".join(ctx_lines)
    return base_prompt


@click.command()
@click.argument("image", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--model",
    "-m",
    type=str,
    default=None,
    help="Model name/path. Default: unsloth/gemma-3-4b-it-FP8-Dynamic",
)
@click.option(
    "--tensor-parallel",
    "-tp",
    type=int,
    default=1,
    show_default=True,
    help="Number of GPUs for tensor parallelism",
)
@click.option(
    "--system-prompt", "-s", type=str, default=None, help="Optional system prompt"
)
@click.option(
    "--load-tags/--no-load-tags",
    default=False,
    help="Load tags from .tag.json file as context",
)
@click.option(
    "--tag-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to .tag.json file (default: {image_stem}.tag.json)",
)
@click.option("--max-tokens", type=int, default=2048, show_default=True)
@click.option(
    "--max-model-len",
    type=int,
    default=8192,
    show_default=True,
    help="Maximum model context length",
)
@click.option("--temperature", type=float, default=0.7, show_default=True)
@click.option(
    "--gpu-memory-utilization",
    type=float,
    default=0.95,
    show_default=True,
    help="GPU memory utilization (0.0-1.0). Higher = more KV cache space",
)
@click.option(
    "--kv-cache-dtype",
    type=click.Choice(["auto", "fp8", "fp8_e4m3"]),
    default="fp8",
    show_default=True,
    help="KV cache dtype. fp8 gives 2x cache capacity with minimal quality loss",
)
@click.option(
    "--quantization",
    type=click.Choice(["none", "fp8", "fp8_w8a16"]),
    default="none",
    show_default=True,
    help="Weight quantization: none (default), fp8 (W8A8 true FP8), fp8_w8a16 (weight-only)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path (default: print to stdout)",
)
@click.option(
    "--raw/--parsed", default=False, help="Output raw text instead of parsed JSON"
)
def main(
    image: Path,
    model: str | None,
    tensor_parallel: int,
    system_prompt: str | None,
    load_tags: bool,
    tag_file: Path | None,
    max_tokens: int,
    max_model_len: int,
    temperature: float,
    gpu_memory_utilization: float,
    kv_cache_dtype: str,
    quantization: str,
    output: Path | None,
    raw: bool,
):
    """
    Caption a single image using local VLM inference.

    Uses the DefaultFormat system for structured output with:
    - aesthetic_score, nsfw_score, quality_score
    - title, brief, description

    \b
    Examples:
        # Basic usage
        python caption_local.py image.png

        # Multi-GPU inference with larger model
        python caption_local.py image.png -tp 2 -m google/gemma-3-12b-it

        # With pre-computed tags as context
        python caption_local.py image.png --load-tags

        # Raw output (unparsed)
        python caption_local.py image.png --raw
    """
    from kohakucaption.local import VLLMModel, VLLMConfig
    from kohakucaption.formats import DefaultFormat

    # Default model
    if model is None:
        model = "unsloth/gemma-3-4b-it-FP8-Dynamic"

    click.echo(f"Model: {model}")
    click.echo(f"Tensor Parallel: {tensor_parallel}")
    click.echo(f"Image: {image}")

    # Load tags if requested
    tags = None
    if load_tags:
        tag_path = tag_file or (image.parent / f"{image.stem}.tag.json")
        tags = load_tags(tag_path)
        if tags:
            click.echo(f"Tags: loaded from {tag_path}")
        else:
            click.echo(f"Tags: not found at {tag_path}")

    click.echo("-" * 60)

    # Create format and prompt
    output_format = DefaultFormat()
    base_prompt = f"""Provide a structured caption for this image.

{output_format.get_format_instruction()}"""

    # Build final prompt with tags
    final_prompt = build_prompt_with_tags(base_prompt, tags)

    # Load image
    pil_image = Image.open(image).convert("RGB")

    # Create model
    quant = None if quantization == "none" else quantization
    config = VLLMConfig(
        model=model,
        tensor_parallel_size=tensor_parallel,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        temperature=temperature,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_dtype=kv_cache_dtype,
        quantization=quant,
    )
    vlm = VLLMModel(config)

    try:
        # Generate caption
        click.echo("Generating caption...")
        result = vlm.generate(pil_image, final_prompt, system_prompt)

        click.echo()
        click.echo("=" * 60)

        if raw:
            click.echo("RAW OUTPUT:")
            click.echo("=" * 60)
            click.echo(result.text)
        else:
            # Parse output
            parse_result = output_format.parse(result.text)

            if parse_result.success:
                click.echo("PARSED CAPTION:")
                click.echo("=" * 60)
                click.echo(json.dumps(parse_result.data, indent=2, ensure_ascii=False))
            else:
                click.echo("PARSE FAILED:")
                click.echo("=" * 60)
                click.echo(f"Error: {parse_result.error}")
                click.echo()
                click.echo("Raw output:")
                click.echo(result.text)

        click.echo()
        click.echo(f"Generation time: {result.generation_time_ms:.0f}ms")

        if result.completion_tokens:
            click.echo(f"Tokens: {result.completion_tokens}")

        # Save if output specified
        if output:
            if raw:
                output.write_text(result.text, encoding="utf-8")
            else:
                parse_result = output_format.parse(result.text)
                if parse_result.success:
                    output.write_text(
                        json.dumps(parse_result.data, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                else:
                    output.write_text(result.text, encoding="utf-8")
            click.echo(f"\nSaved to: {output}")

    finally:
        vlm.unload()


if __name__ == "__main__":
    main()
