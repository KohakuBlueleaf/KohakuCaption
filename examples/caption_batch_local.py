#!/usr/bin/env python3
"""
Batch caption images using local VLM inference (vLLM).

Output: xxx.cap.json for each xxx.jpg/png/etc.

Uses the DefaultFormat system for structured captions.
Optionally loads pre-computed tags from xxx.tag.json as context.

Workflow:
    1. (Optional) python tag_batch.py ./images  # Creates xxx.tag.json
    2. python caption_batch_local.py ./images --load-tags  # Creates xxx.cap.json
"""

import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

import click

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tqdm import tqdm

# Global interrupt flag
_interrupted = False


def _signal_handler(signum, frame):
    global _interrupted
    _interrupted = True
    raise KeyboardInterrupt


def find_images(input_dir: Path, extensions: list[str]) -> list[Path]:
    images = []
    for ext in extensions:
        images.extend(input_dir.glob(f"*{ext}"))
        images.extend(input_dir.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def load_tags(tag_path: Path) -> dict[str, Any] | None:
    """Load tags from .tag.json file."""
    if not tag_path.exists():
        return None
    try:
        with open(tag_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Check for error
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
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory (default: same as input)",
)
@click.option(
    "--tag-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory containing .tag.json files (default: same as input)",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default=None,
    help="Model name/path. Default: google/gemma-3-4b-it",
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
    "--batch-size",
    type=int,
    default=8,
    show_default=True,
    help="Batch size for VLM inference",
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
    "--load-tags/--no-load-tags",
    default=False,
    help="Load tags from .tag.json files as context",
)
@click.option(
    "--num-workers",
    "-w",
    type=int,
    default=4,
    show_default=True,
    help="Number of DataLoader workers",
)
@click.option(
    "--prefetch",
    type=int,
    default=3,
    show_default=True,
    help="Number of batches to prefetch",
)
@click.option(
    "--extensions", type=str, default=".jpg,.jpeg,.png,.webp", show_default=True
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=False,
    help="Skip images that already have .cap.json files",
)
def main(
    input_dir: Path,
    output_dir: Path | None,
    tag_dir: Path | None,
    model: str | None,
    tensor_parallel: int,
    batch_size: int,
    max_tokens: int,
    max_model_len: int,
    temperature: float,
    load_tags: bool,
    num_workers: int,
    prefetch: int,
    extensions: str,
    skip_existing: bool,
):
    """
    Batch caption images using local VLM inference.

    Uses the DefaultFormat system for structured output with:
    - aesthetic_score, nsfw_score, quality_score
    - title, brief, description

    \b
    Output format: xxx.cap.json for each image
    {
        "caption": {...},  // parsed caption data
        "raw": "...",      // raw model output (if parse failed)
        "error": "..."     // error message (if failed)
    }

    \b
    Examples:
        # Basic captioning
        python caption_batch_local.py ./images

        # With pre-computed tags as context
        python caption_batch_local.py ./images --load-tags

        # Multi-GPU with larger model
        python caption_batch_local.py ./images -m google/gemma-3-12b-it -tp 2

    \b
    Workflow:
        1. python tag_batch.py ./images           # Create xxx.tag.json
        2. python caption_batch_local.py ./images --load-tags  # Create xxx.cap.json
    """
    from kohakucaption.local import VLLMModel, VLLMConfig, PreprocessPipeline
    from kohakucaption.formats import DefaultFormat

    signal.signal(signal.SIGINT, _signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, _signal_handler)

    output_dir = output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tag_dir = tag_dir or input_dir

    ext_list = [e.strip() for e in extensions.split(",")]
    image_paths = find_images(input_dir, ext_list)

    if not image_paths:
        click.echo(f"No images found in {input_dir}")
        return

    if skip_existing:
        orig = len(image_paths)
        image_paths = [
            p for p in image_paths if not (output_dir / f"{p.stem}.cap.json").exists()
        ]
        if orig - len(image_paths) > 0:
            click.echo(f"Skipped {orig - len(image_paths)} existing .cap.json files")
        if not image_paths:
            click.echo("All done.")
            return

    # Default model
    if model is None:
        model = "google/gemma-3-4b-it"

    click.echo("=" * 60)
    click.echo(f"Model: {model} | TP: {tensor_parallel}")
    click.echo(f"Images: {len(image_paths)} | Batch: {batch_size}")
    click.echo(f"Load tags: {load_tags} | Tag dir: {tag_dir if load_tags else 'N/A'}")
    click.echo(f"Output: {{name}}.cap.json")
    click.echo("=" * 60)

    # Create VLM model
    config = VLLMConfig(
        model=model,
        tensor_parallel_size=tensor_parallel,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        temperature=temperature,
    )
    vlm = VLLMModel(config)

    # Load VLM
    vlm.load()

    # Create preprocessing pipeline (no tagger transform - just load images)
    pipeline = PreprocessPipeline(
        image_paths=image_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_batches=prefetch,
        tagger_transform=None,
    )

    # Output format
    output_format = DefaultFormat()
    base_prompt = f"""Provide a structured caption for this image.

{output_format.get_format_instruction()}"""

    stats = {"success": 0, "failed": 0}
    start_time = time.time()

    try:
        with (
            pipeline,
            tqdm(total=len(image_paths), desc="Captioning", unit="img") as pbar,
        ):
            for batch in pipeline:
                paths = batch.paths
                images = batch.images
                errors = batch.errors

                # Load tags for each image if enabled
                tags_list: list[dict | None] = [None] * len(paths)
                if load_tags:
                    for i, path in enumerate(paths):
                        tag_path = tag_dir / f"{path.stem}.tag.json"
                        tags_list[i] = load_tags(tag_path)

                # Prepare valid images for VLM
                valid_items = []
                for i, (path, img, tags, error) in enumerate(
                    zip(paths, images, tags_list, errors)
                ):
                    if error or img is None:
                        # Write error
                        data = {"error": error or "Failed to load image"}
                        with open(
                            output_dir / f"{path.stem}.cap.json", "w", encoding="utf-8"
                        ) as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        stats["failed"] += 1
                        pbar.update(1)
                    else:
                        prompt = build_prompt_with_tags(base_prompt, tags)
                        valid_items.append(
                            {
                                "path": path,
                                "image": img,
                                "prompt": prompt,
                            }
                        )

                if not valid_items:
                    continue

                # Batch VLM inference
                batch_images = [item["image"] for item in valid_items]
                batch_prompts = [item["prompt"] for item in valid_items]

                batch_output = vlm.generate_batch(batch_images, batch_prompts)

                # Process results
                for item, gen_output in zip(valid_items, batch_output.outputs):
                    out_path = output_dir / f"{item['path'].stem}.cap.json"

                    # Parse output
                    parse_result = output_format.parse(gen_output.text)
                    if parse_result.success:
                        data = {"caption": parse_result.data}
                        stats["success"] += 1
                    else:
                        data = {
                            "raw": gen_output.text,
                            "parse_error": parse_result.error,
                        }
                        stats["failed"] += 1

                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)

                    pbar.update(1)

    except KeyboardInterrupt:
        click.echo("\n[!] Interrupted!")

    finally:
        vlm.unload()

    elapsed = time.time() - start_time
    click.echo()
    click.echo("=" * 60)
    click.echo(f"Done: {stats['success']} OK, {stats['failed']} failed")
    click.echo(
        f"Time: {elapsed:.1f}s | Throughput: {(stats['success'] + stats['failed'])/elapsed:.2f} img/s"
    )
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
