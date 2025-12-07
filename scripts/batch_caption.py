#!/usr/bin/env python3
"""
Batch caption images with LLM API + optional PixAI Tagger.

Two tagging modes:
1. --mode parallel: VLM captioning and tagger run simultaneously, results merged
2. --mode context: Tagger runs first, tags sent as context to LLM for captioning

Both modes output a single JSON file per image containing caption and tags.

Architecture:
- Tagger inference runs in thread pool via asyncio.to_thread
- API calls run as async tasks with semaphore rate limiting
- Results written immediately as they complete using aiofiles
- Full parallelization: model forward pass + API requests run concurrently
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import click

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import aiofiles
from PIL import Image

from kohakucaption.clients import ClientConfig, OpenAIClient, OpenRouterClient
from kohakucaption.formats import DefaultFormat
from kohakucaption.tokenizer import TokenCounter
from kohakucaption.types import ImageInput


# Global tagger reference for lazy loading
_tagger = None
_tagger_lock = asyncio.Lock()


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


def find_images(input_dir: Path, extensions: list[str]) -> list[Path]:
    """Find all image files in directory with given extensions."""
    images = []
    for ext in extensions:
        images.extend(input_dir.glob(f"*{ext}"))
        images.extend(input_dir.glob(f"*{ext.upper()}"))
    return sorted(set(images))


async def load_tagger(tagger_dir: str):
    """Load PixAI tagger lazily (thread-safe)."""
    global _tagger
    async with _tagger_lock:
        if _tagger is None:
            from kohakucaption.tagger import PixAITagger
            _tagger = PixAITagger(tagger_dir, use_fp16=True)
            # Load model in thread to avoid blocking
            await asyncio.to_thread(lambda: _tagger._load_model())
    return _tagger


async def tag_single_image(image_path: Path, tagger_dir: str) -> dict[str, Any]:
    """Tag a single image using thread pool."""
    tagger = await load_tagger(tagger_dir)

    def _tag():
        img = Image.open(image_path)
        return tagger.tag(img)

    result = await asyncio.to_thread(_tag)
    return {
        "feature_tags": result.feature_tags,
        "character_tags": result.character_tags,
        "ip_tags": result.ip_tags,
    }


async def tag_batch_images(
    image_paths: list[Path],
    tagger_dir: str,
    batch_size: int = 8,
) -> dict[Path, dict[str, Any]]:
    """Tag multiple images in batches using thread pool."""
    tagger = await load_tagger(tagger_dir)

    def _tag_batch():
        images = [Image.open(p) for p in image_paths]
        results = tagger.tag_batch(images, batch_size=batch_size)
        return {
            path: {
                "feature_tags": r.feature_tags,
                "character_tags": r.character_tags,
                "ip_tags": r.ip_tags,
            }
            for path, r in zip(image_paths, results)
        }

    return await asyncio.to_thread(_tag_batch)


async def caption_single_image(
    image_path: Path,
    client,
    output_format: DefaultFormat,
    prompt: str,
    semaphore: asyncio.Semaphore,
    token_counter: TokenCounter | None,
    tags_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Caption a single image with optional tags as context."""
    async with semaphore:
        # Build prompt with tags context if provided
        final_prompt = prompt
        if tags_context:
            context_lines = []
            if tags_context.get("feature_tags"):
                tags = [t.replace("_", " ") for t in tags_context["feature_tags"]]
                context_lines.append(f"Features: {', '.join(tags)}")
            if tags_context.get("character_tags"):
                tags = [t.replace("_", " ") for t in tags_context["character_tags"]]
                context_lines.append(f"Characters: {', '.join(tags)}")
            if tags_context.get("ip_tags"):
                tags = [t.replace("_", " ") for t in tags_context["ip_tags"]]
                context_lines.append(f"IPs/Series: {', '.join(tags)}")
            if context_lines:
                context = "\n".join(context_lines)
                final_prompt = f"{prompt}\n\nExisting tags for context:\n{context}"

        image = ImageInput(source=str(image_path))
        result = await client.caption(image=image, prompt=final_prompt)

        if result.success:
            parse_result = output_format.parse(result.raw_response)
            if parse_result.success:
                token_counts = None
                if token_counter:
                    token_counts = token_counter.count_fields(parse_result.data)

                return {
                    "success": True,
                    "caption": parse_result.data,
                    "token_counts": token_counts,
                    "latency_ms": result.latency_ms,
                    "retries": result.retries_used,
                }
            else:
                return {
                    "success": False,
                    "error": f"Parse error: {parse_result.error}",
                    "raw_response": result.raw_response,
                }
        else:
            return {
                "success": False,
                "error": result.error,
            }


async def write_result(output_path: Path, data: dict[str, Any]):
    """Write result to JSON file using aiofiles."""
    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(data, indent=2, ensure_ascii=False))


async def process_parallel_mode(
    image_paths: list[Path],
    output_dir: Path,
    client,
    output_format: DefaultFormat,
    prompt: str,
    semaphore: asyncio.Semaphore,
    token_counter: TokenCounter | None,
    tagger_dir: str,
    tagger_batch_size: int,
    on_progress,
) -> dict[str, Any]:
    """
    Parallel mode: VLM captioning and tagger run simultaneously.

    Both processes run concurrently:
    - Tagger batches run in thread pool
    - Caption API calls run as async tasks

    Results are merged and written as each image completes.
    """
    stats = {"caption_success": 0, "caption_failed": 0, "total_tokens": {}}

    # Create futures for both caption and tag results
    caption_futures: dict[Path, asyncio.Task] = {}
    results_data: dict[Path, dict[str, Any]] = {p: {} for p in image_paths}

    # Start all caption tasks immediately
    for image_path in image_paths:
        task = asyncio.create_task(
            caption_single_image(
                image_path=image_path,
                client=client,
                output_format=output_format,
                prompt=prompt,
                semaphore=semaphore,
                token_counter=token_counter,
                tags_context=None,
            )
        )
        caption_futures[image_path] = task

    # Start tagging in parallel (runs in thread pool)
    tag_task = asyncio.create_task(
        tag_batch_images(image_paths, tagger_dir, tagger_batch_size)
    )

    # Wait for tagging to complete
    tags_map = await tag_task

    # Process caption results as they complete and merge with tags
    for image_path, caption_task in caption_futures.items():
        caption_result = await caption_task
        tags = tags_map.get(image_path, {})

        # Merge results
        output_data = {
            "image": image_path.name,
            "tags": tags,
        }

        if caption_result["success"]:
            output_data["caption"] = caption_result["caption"]
            stats["caption_success"] += 1
            if caption_result.get("token_counts"):
                for k, v in caption_result["token_counts"].items():
                    stats["total_tokens"][k] = stats["total_tokens"].get(k, 0) + v
            on_progress(image_path, True, None)
        else:
            output_data["caption_error"] = caption_result.get("error")
            stats["caption_failed"] += 1
            on_progress(image_path, False, caption_result.get("error"))

        # Write merged result
        output_path = output_dir / f"{image_path.stem}.json"
        await write_result(output_path, output_data)

    return stats


async def process_context_mode(
    image_paths: list[Path],
    output_dir: Path,
    client,
    output_format: DefaultFormat,
    prompt: str,
    semaphore: asyncio.Semaphore,
    token_counter: TokenCounter | None,
    tagger_dir: str,
    tagger_batch_size: int,
    on_progress,
) -> dict[str, Any]:
    """
    Context mode: Tag first, then send tags as context to LLM.

    Process:
    1. Run tagger on all images (batched, in thread pool)
    2. Start caption tasks with tags as context
    3. Write results as each caption completes

    Still parallelized: all caption API calls run concurrently after tagging.
    """
    stats = {"caption_success": 0, "caption_failed": 0, "total_tokens": {}}

    # First, tag all images in batch (runs in thread)
    tags_map = await tag_batch_images(image_paths, tagger_dir, tagger_batch_size)

    # Create caption tasks with tags as context
    caption_tasks: list[tuple[Path, asyncio.Task]] = []
    for image_path in image_paths:
        tags = tags_map.get(image_path, {})
        task = asyncio.create_task(
            caption_single_image(
                image_path=image_path,
                client=client,
                output_format=output_format,
                prompt=prompt,
                semaphore=semaphore,
                token_counter=token_counter,
                tags_context=tags,
            )
        )
        caption_tasks.append((image_path, task))

    # Process results as they complete
    for image_path, task in caption_tasks:
        caption_result = await task
        tags = tags_map.get(image_path, {})

        # Build output data
        output_data = {
            "image": image_path.name,
            "tags": tags,
        }

        if caption_result["success"]:
            output_data["caption"] = caption_result["caption"]
            stats["caption_success"] += 1
            if caption_result.get("token_counts"):
                for k, v in caption_result["token_counts"].items():
                    stats["total_tokens"][k] = stats["total_tokens"].get(k, 0) + v
            on_progress(image_path, True, None)
        else:
            output_data["caption_error"] = caption_result.get("error")
            stats["caption_failed"] += 1
            on_progress(image_path, False, caption_result.get("error"))

        # Write result
        output_path = output_dir / f"{image_path.stem}.json"
        await write_result(output_path, output_data)

    return stats


async def process_caption_only(
    image_paths: list[Path],
    output_dir: Path,
    client,
    output_format: DefaultFormat,
    prompt: str,
    semaphore: asyncio.Semaphore,
    token_counter: TokenCounter | None,
    on_progress,
) -> dict[str, Any]:
    """Caption only mode (no tagger)."""
    stats = {"caption_success": 0, "caption_failed": 0, "total_tokens": {}}

    tasks: list[tuple[Path, asyncio.Task]] = []
    for image_path in image_paths:
        task = asyncio.create_task(
            caption_single_image(
                image_path=image_path,
                client=client,
                output_format=output_format,
                prompt=prompt,
                semaphore=semaphore,
                token_counter=token_counter,
                tags_context=None,
            )
        )
        tasks.append((image_path, task))

    for image_path, task in tasks:
        caption_result = await task

        output_data = {"image": image_path.name}

        if caption_result["success"]:
            output_data["caption"] = caption_result["caption"]
            stats["caption_success"] += 1
            if caption_result.get("token_counts"):
                for k, v in caption_result["token_counts"].items():
                    stats["total_tokens"][k] = stats["total_tokens"].get(k, 0) + v
            on_progress(image_path, True, None)
        else:
            output_data["caption_error"] = caption_result.get("error")
            stats["caption_failed"] += 1
            on_progress(image_path, False, caption_result.get("error"))

        output_path = output_dir / f"{image_path.stem}.json"
        await write_result(output_path, output_data)

    return stats


@click.command()
@click.argument(
    "input_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output directory for results. Defaults to input directory.",
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
    "--max-concurrent", "-c",
    type=int,
    default=5,
    show_default=True,
    help="Maximum concurrent API requests.",
)
@click.option(
    "--with-tags/--no-tags",
    default=False,
    help="Enable PixAI tagger for image tagging.",
)
@click.option(
    "--mode",
    type=click.Choice(["parallel", "context"]),
    default="parallel",
    help="""Tagging mode when --with-tags is enabled:

    \b
    parallel: Caption and tag run simultaneously, results merged.
              Best for throughput - model inference and API calls overlap.

    \b
    context:  Tag first, then send tags as context to LLM.
              May improve caption quality by giving LLM tag hints.""",
)
@click.option(
    "--tagger-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("pixai-tagger"),
    show_default=True,
    help="Directory containing PixAI tagger model files.",
)
@click.option(
    "--tagger-batch-size",
    type=int,
    default=8,
    show_default=True,
    help="Batch size for tagger inference. Larger = faster but more VRAM.",
)
@click.option(
    "--detail",
    type=click.Choice(["low", "high", "auto"]),
    default="auto",
    show_default=True,
    help="Image detail level for vision API. 'low' uses fewer tokens.",
)
@click.option(
    "--timeout",
    type=float,
    default=120.0,
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
    "--extensions",
    type=str,
    default=".jpg,.jpeg,.png,.webp,.gif",
    show_default=True,
    help="Comma-separated list of image file extensions to process.",
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=False,
    help="Skip images that already have output JSON files.",
)
def main(
    input_dir: Path,
    output_dir: Path | None,
    provider: str,
    model: str | None,
    api_key: str | None,
    max_concurrent: int,
    with_tags: bool,
    mode: str,
    tagger_dir: Path,
    tagger_batch_size: int,
    detail: str,
    timeout: float,
    tokenizer: str,
    extensions: str,
    skip_existing: bool,
):
    """
    Batch caption images using LLM API with optional PixAI tagger.

    INPUT_DIR is the directory containing images to process.

    \b
    Examples:
        # Caption only with OpenAI
        python batch_caption.py ./images

        # Caption + tags in parallel (fastest)
        python batch_caption.py ./images --with-tags --mode parallel

        # Caption with tags as context (potentially better quality)
        python batch_caption.py ./images --with-tags --mode context

        # Using OpenRouter with custom model
        python batch_caption.py ./images -p openrouter -m anthropic/claude-sonnet-4

        # High throughput settings
        python batch_caption.py ./images --max-concurrent 10 --with-tags

    \b
    Output format (JSON per image):
        {
          "image": "filename.png",
          "tags": {
            "feature_tags": [...],
            "character_tags": [...],
            "ip_tags": [...]
          },
          "caption": {
            "aesthetic_score": 0.85,
            "nsfw_score": 0.0,
            "quality_score": 0.9,
            "title": "...",
            "brief": "...",
            "description": "..."
          }
        }
    """
    asyncio.run(_async_main(
        input_dir=input_dir,
        output_dir=output_dir,
        provider=provider,
        model=model,
        api_key=api_key,
        max_concurrent=max_concurrent,
        with_tags=with_tags,
        mode=mode,
        tagger_dir=tagger_dir,
        tagger_batch_size=tagger_batch_size,
        detail=detail,
        timeout=timeout,
        tokenizer=tokenizer,
        extensions=extensions,
        skip_existing=skip_existing,
    ))


async def _async_main(
    input_dir: Path,
    output_dir: Path | None,
    provider: str,
    model: str | None,
    api_key: str | None,
    max_concurrent: int,
    with_tags: bool,
    mode: str,
    tagger_dir: Path,
    tagger_batch_size: int,
    detail: str,
    timeout: float,
    tokenizer: str,
    extensions: str,
    skip_existing: bool,
):
    """Async main function."""
    # Setup paths
    output_dir = output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    ext_list = [e.strip() for e in extensions.split(",")]
    image_paths = find_images(input_dir, ext_list)

    if not image_paths:
        click.echo(f"No images found in {input_dir}")
        return

    # Filter existing
    if skip_existing:
        image_paths = [
            p for p in image_paths
            if not (output_dir / f"{p.stem}.json").exists()
        ]
        if not image_paths:
            click.echo("All images already have output files.")
            return

    click.echo(f"Found {len(image_paths)} images to process")

    # Setup client
    resolved_api_key = get_api_key(provider, api_key)
    resolved_model = model or get_default_model(provider)

    config = ClientConfig(
        api_key=resolved_api_key,
        model=resolved_model,
        timeout=timeout,
        detail=detail,
    )

    client = OpenRouterClient(config) if provider == "openrouter" else OpenAIClient(config)

    # Setup format and prompt
    output_format = DefaultFormat()
    prompt = f"""Provide a structured caption for this image.

{output_format.get_format_instruction()}

Additional requirements:
- Cover ALL visual elements: subjects, attributes, actions, background, environment, lighting, colors, composition, artistic style
- Never omit background or secondary elements
- If nsfw_score > 0.3, describe explicit/sexual content in detail"""

    semaphore = asyncio.Semaphore(max_concurrent)
    token_counter = TokenCounter(tokenizer) if tokenizer else None

    # Progress callback
    completed = 0

    def on_progress(path: Path, success: bool, error: str | None):
        nonlocal completed
        completed += 1
        status = click.style("✓", fg="green") if success else click.style("✗", fg="red")
        msg = f"[{completed}/{len(image_paths)}] {status} {path.name}"
        if error:
            msg += f" - {error[:50]}"
        click.echo(msg)

    # Print configuration
    click.echo(f"Provider: {provider}")
    click.echo(f"Model: {resolved_model}")
    click.echo(f"Max concurrent: {max_concurrent}")
    click.echo(f"With tags: {with_tags}")
    if with_tags:
        click.echo(f"Mode: {mode}")
        click.echo(f"Tagger dir: {tagger_dir}")
    click.echo()

    try:
        if with_tags:
            # Validate tagger directory
            if not (tagger_dir / "model_v0.9.pth").exists():
                raise click.ClickException(
                    f"Tagger model not found in {tagger_dir}. "
                    "Run: python scripts/download_pixai_tagger.py"
                )

            if mode == "parallel":
                stats = await process_parallel_mode(
                    image_paths=image_paths,
                    output_dir=output_dir,
                    client=client,
                    output_format=output_format,
                    prompt=prompt,
                    semaphore=semaphore,
                    token_counter=token_counter,
                    tagger_dir=str(tagger_dir),
                    tagger_batch_size=tagger_batch_size,
                    on_progress=on_progress,
                )
            else:
                stats = await process_context_mode(
                    image_paths=image_paths,
                    output_dir=output_dir,
                    client=client,
                    output_format=output_format,
                    prompt=prompt,
                    semaphore=semaphore,
                    token_counter=token_counter,
                    tagger_dir=str(tagger_dir),
                    tagger_batch_size=tagger_batch_size,
                    on_progress=on_progress,
                )
        else:
            stats = await process_caption_only(
                image_paths=image_paths,
                output_dir=output_dir,
                client=client,
                output_format=output_format,
                prompt=prompt,
                semaphore=semaphore,
                token_counter=token_counter,
                on_progress=on_progress,
            )

    finally:
        await client.close()

    # Print summary
    click.echo()
    click.echo("=" * 60)
    click.echo("SUMMARY")
    click.echo("=" * 60)
    click.echo(f"  Total images: {len(image_paths)}")
    click.echo(f"  Succeeded: {stats['caption_success']}")
    click.echo(f"  Failed: {stats['caption_failed']}")
    if stats["total_tokens"]:
        click.echo(f"\n  Token counts ({tokenizer}):")
        for k, v in stats["total_tokens"].items():
            click.echo(f"    {k}: {v}")
        click.echo(f"    total: {sum(stats['total_tokens'].values())}")


if __name__ == "__main__":
    main()
