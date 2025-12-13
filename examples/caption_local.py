#!/usr/bin/env python3
"""
Caption a single image using local VLM inference (vLLM or LMDeploy).

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
    if tags.get("general"):
        tag_list = [t.replace("_", " ") for t in tags["general"]]
        ctx_lines.append(f"Features: {', '.join(tag_list)}")
    if tags.get("character"):
        tag_list = [t.replace("_", " ") for t in tags["character"]]
        ctx_lines.append(f"Characters: {', '.join(tag_list)}")
    if tags.get("rating"):
        tag_list = [t.replace("_", " ") for t in tags["rating"]]
        ctx_lines.append(f"Rating: {', '.join(tag_list)}")

    if ctx_lines:
        return f"{base_prompt}\n\nImage tags:\n" + "\n".join(ctx_lines)
    return base_prompt


@click.command()
@click.argument("image", type=click.Path(exists=True, path_type=Path))
@click.option("--backend", "-b", type=click.Choice(["vllm", "lmdeploy"]), default="vllm",
              help="Inference backend to use")
@click.option("--model", "-m", type=str, default=None,
              help="Model name/path")
@click.option("--tensor-parallel", "-tp", type=int, default=1, show_default=True,
              help="Number of GPUs for tensor parallelism")
@click.option("--prompt", "-p", type=str, default="Describe this image in detail.",
              help="Prompt for captioning")
@click.option("--system-prompt", "-s", type=str, default=None,
              help="Optional system prompt")
@click.option("--load-tags/--no-load-tags", default=False,
              help="Load tags from .tag.json file as context")
@click.option("--tag-file", type=click.Path(path_type=Path), default=None,
              help="Path to .tag.json file (default: {image_stem}.tag.json)")
@click.option("--max-tokens", type=int, default=512, show_default=True)
@click.option("--temperature", type=float, default=0.7, show_default=True)
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Output file path (default: print to stdout)")
def main(
    image: Path,
    backend: str,
    model: str | None,
    tensor_parallel: int,
    prompt: str,
    system_prompt: str | None,
    load_tags: bool,
    tag_file: Path | None,
    max_tokens: int,
    temperature: float,
    output: Path | None,
):
    """
    Caption a single image using local VLM inference.

    \b
    Examples:
        # Basic usage with vLLM
        python caption_local.py image.png

        # Use LMDeploy with InternVL2
        python caption_local.py image.png -b lmdeploy

        # Multi-GPU inference
        python caption_local.py image.png -tp 2 -m llava-hf/llava-v1.6-34b-hf

        # With pre-computed tags as context
        python caption_local.py image.png --load-tags

        # Custom prompt
        python caption_local.py image.png -p "What objects are in this image?"
    """
    # Default models
    if model is None:
        model = "llava-hf/llava-1.5-7b-hf" if backend == "vllm" else "OpenGVLab/InternVL2-8B"

    click.echo(f"Backend: {backend}")
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

    # Build final prompt
    final_prompt = build_prompt_with_tags(prompt, tags)

    # Load image
    pil_image = Image.open(image).convert("RGB")

    # Create model based on backend
    if backend == "vllm":
        from kohakucaption.local import VLLMModel, VLLMConfig

        config = VLLMConfig(
            model=model,
            tensor_parallel_size=tensor_parallel,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        vlm = VLLMModel(config)
    else:
        from kohakucaption.local import LMDeployModel, LMDeployConfig

        config = LMDeployConfig(
            model=model,
            tensor_parallel_size=tensor_parallel,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        vlm = LMDeployModel(config)

    try:
        # Generate caption
        click.echo("Generating caption...")
        result = vlm.generate(pil_image, final_prompt, system_prompt)

        click.echo()
        click.echo("=" * 60)
        click.echo("CAPTION:")
        click.echo("=" * 60)
        click.echo(result.text)
        click.echo()
        click.echo(f"Generation time: {result.generation_time_ms:.0f}ms")

        if result.completion_tokens:
            click.echo(f"Tokens: {result.completion_tokens}")

        # Save if output specified
        if output:
            output.write_text(result.text, encoding="utf-8")
            click.echo(f"\nSaved to: {output}")

    finally:
        vlm.unload()


if __name__ == "__main__":
    main()
