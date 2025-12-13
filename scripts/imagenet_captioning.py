#!/usr/bin/env python3
"""
ImageNet-1k Local Captioning Script.

Reads images from a tagged folder (output of imagenet_tagging.py) and generates captions.
Optionally includes tags as context for the captioning model.

Input folder structure:
    <id>.jpg        - The image
    <id>.tag.json   - Tags and metadata (from tagging script)

Output:
    <id>.caption.json - Generated caption

Supports multi-GPU inference with:
  - Data Parallelism: Multiple model replicas on different GPUs (default)
  - Tensor Parallelism: Single model sharded across GPUs (for large models)

Usage:
    # Data parallel on 4 GPUs (recommended for smaller models)
    python scripts/imagenet_captioning.py -i ./imagenet_data --gpus 0,1,2,3

    # With tags as context
    python scripts/imagenet_captioning.py -i ./imagenet_data --gpus 0,1,2,3 --with-tags

    # Tensor parallel on 4 GPUs (for large models like 34B)
    python scripts/imagenet_captioning.py -i ./imagenet_data --tensor-parallel 4

    # Single GPU
    python scripts/imagenet_captioning.py -i ./imagenet_data --gpus 0
"""

import json
import logging
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import torch

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PIL import Image
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silence noisy loggers
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("lmdeploy").setLevel(logging.WARNING)

# Global interrupt flag
_interrupted = False


def _signal_handler(signum, frame):
    """Handle interrupt signal."""
    global _interrupted
    _interrupted = True


# Default models for different backends
DEFAULT_MODELS = {
    "vllm": "llava-hf/llava-1.5-7b-hf",
    "lmdeploy": "OpenGVLab/InternVL2-8B",
}

# Caption prompts
DEFAULT_PROMPT = """Describe this image in detail. Include:
1. Main subjects and their characteristics
2. Actions or activities
3. Setting and environment
4. Colors, lighting, and mood
5. Any notable details

Provide a comprehensive description in 2-3 sentences."""

PROMPT_WITH_TAGS = """Describe this image in detail. The following tags have been detected:

{tags}

Using these tags as guidance, provide a comprehensive description that covers:
1. Main subjects and their characteristics
2. Actions or activities
3. Setting and environment
4. Colors, lighting, and mood

Provide a detailed description in 2-3 sentences."""


@dataclass
class CaptionTask:
    """Task for captioning worker."""
    image_id: str
    image_path: Path
    tag_path: Path
    tags: dict[str, Any] | None


@dataclass
class CaptionResult:
    """Result from captioning worker."""
    image_id: str
    caption: str | None
    error: str | None = None


def pil_to_rgb(image: Image.Image) -> Image.Image:
    """Convert image to RGB, handling RGBA and palette modes."""
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    elif image.mode == "P":
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        if rgba.mode == "RGBA":
            background.paste(rgba, mask=rgba.split()[3])
        return background
    else:
        return image.convert("RGB")


def format_tags_for_prompt(tags: dict[str, Any]) -> str:
    """Format tags dict into a string for the prompt."""
    parts = []

    general = tags.get("general_tags", [])
    if general:
        # Replace underscores with spaces
        formatted = [t.replace("_", " ") for t in general[:30]]  # Limit to top 30
        parts.append(f"Features: {', '.join(formatted)}")

    characters = tags.get("character_tags", [])
    if characters:
        formatted = [t.replace("_", " ") for t in characters]
        parts.append(f"Characters: {', '.join(formatted)}")

    rating = tags.get("rating_tags", [])
    if rating:
        parts.append(f"Rating: {rating[0].replace('_', ' ')}")

    return "\n".join(parts) if parts else "No tags detected"


class SpeedMonitor:
    """Monitor processing speed with moving average."""

    def __init__(self, window_size: int = 50):
        from collections import deque
        self.timestamps: deque[float] = deque(maxlen=window_size)
        self.counts: deque[int] = deque(maxlen=window_size)
        self.total_count = 0

    def tick(self, count: int = 1):
        self.timestamps.append(time.time())
        self.counts.append(count)
        self.total_count += count

    def get_speed(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        total = sum(self.counts) - self.counts[0]
        return total / elapsed if elapsed > 0 else 0.0

    def get_eta(self, remaining: int) -> str:
        speed = self.get_speed()
        if speed <= 0:
            return "?"
        eta_sec = remaining / speed
        if eta_sec < 60:
            return f"{eta_sec:.0f}s"
        elif eta_sec < 3600:
            return f"{eta_sec / 60:.1f}m"
        return f"{eta_sec / 3600:.1f}h"


def find_images(input_dir: Path) -> list[tuple[str, Path, Path]]:
    """
    Find all images with corresponding tag files.

    Returns list of (image_id, image_path, tag_path) tuples.
    """
    results = []

    # Find all tag.json files
    for tag_path in sorted(input_dir.glob("*.tag.json")):
        image_id = tag_path.stem.replace(".tag", "")

        # Find corresponding image
        image_path = None
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            candidate = input_dir / f"{image_id}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path:
            results.append((image_id, image_path, tag_path))

    return results


def worker_process_vllm(
    gpu_id: int,
    model_name: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    base_prompt: str,
    with_tags: bool,
    max_tokens: int,
    temperature: float,
):
    """
    Worker process for vLLM inference on a single GPU (data parallel mode).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        from kohakucaption.local.vllm import VLLMModel, VLLMConfig

        config = VLLMConfig(
            model=model_name,
            tensor_parallel_size=1,
            max_tokens=max_tokens,
            temperature=temperature,
            gpu_memory_utilization=0.85,
        )

        logger.info(f"[GPU {gpu_id}] Loading vLLM model: {model_name}")
        model = VLLMModel(config)
        model.load()
        logger.info(f"[GPU {gpu_id}] Model loaded successfully")

        while True:
            try:
                task = task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task is None:  # Shutdown signal
                break

            try:
                # Load image
                image = Image.open(task.image_path)
                image_rgb = pil_to_rgb(image)

                # Build prompt
                if with_tags and task.tags:
                    tags_str = format_tags_for_prompt(task.tags)
                    prompt = PROMPT_WITH_TAGS.format(tags=tags_str)
                else:
                    prompt = base_prompt

                output = model.generate(image_rgb, prompt)

                result_queue.put(CaptionResult(
                    image_id=task.image_id,
                    caption=output.text,
                ))
            except Exception as e:
                result_queue.put(CaptionResult(
                    image_id=task.image_id,
                    caption=None,
                    error=str(e),
                ))

        model.unload()

    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Worker failed: {e}")
        import traceback
        traceback.print_exc()


def worker_process_lmdeploy(
    gpu_id: int,
    model_name: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    base_prompt: str,
    with_tags: bool,
    max_tokens: int,
    temperature: float,
):
    """
    Worker process for LMDeploy inference on a single GPU (data parallel mode).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        from kohakucaption.local.lmdeploy import LMDeployModel, LMDeployConfig

        config = LMDeployConfig(
            model=model_name,
            tensor_parallel_size=1,
            max_tokens=max_tokens,
            temperature=temperature,
            backend="turbomind",
        )

        logger.info(f"[GPU {gpu_id}] Loading LMDeploy model: {model_name}")
        model = LMDeployModel(config)
        model.load()
        logger.info(f"[GPU {gpu_id}] Model loaded successfully")

        while True:
            try:
                task = task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task is None:
                break

            try:
                image = Image.open(task.image_path)
                image_rgb = pil_to_rgb(image)

                if with_tags and task.tags:
                    tags_str = format_tags_for_prompt(task.tags)
                    prompt = PROMPT_WITH_TAGS.format(tags=tags_str)
                else:
                    prompt = base_prompt

                output = model.generate(image_rgb, prompt)

                result_queue.put(CaptionResult(
                    image_id=task.image_id,
                    caption=output.text,
                ))
            except Exception as e:
                result_queue.put(CaptionResult(
                    image_id=task.image_id,
                    caption=None,
                    error=str(e),
                ))

        model.unload()

    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Worker failed: {e}")
        import traceback
        traceback.print_exc()


def process_data_parallel(
    image_items: list[tuple[str, Path, Path]],
    input_dir: Path,
    gpu_ids: list[int],
    model_name: str,
    backend: str,
    base_prompt: str,
    with_tags: bool,
    max_tokens: int,
    temperature: float,
    prefetch: int,
) -> dict[str, int]:
    """
    Process with data parallelism (multiple model copies).
    """
    global _interrupted

    stats = {"success": 0, "failed": 0}
    speed = SpeedMonitor()
    total = len(image_items)

    # Create queues
    task_queue = mp.Queue(maxsize=prefetch * len(gpu_ids))
    result_queue = mp.Queue()

    # Select worker function
    worker_fn = worker_process_vllm if backend == "vllm" else worker_process_lmdeploy

    # Start workers
    workers = []
    for gpu_id in gpu_ids:
        p = mp.Process(
            target=worker_fn,
            args=(gpu_id, model_name, task_queue, result_queue, base_prompt,
                  with_tags, max_tokens, temperature),
        )
        p.start()
        workers.append(p)

    logger.info(f"Started {len(workers)} workers on GPUs: {gpu_ids}")

    pbar = tqdm(total=total, desc="Captioning", unit="img")
    submitted = 0
    completed = 0

    def update_pbar():
        remaining = total - completed
        pbar.set_postfix_str(
            f"{speed.get_speed():.2f}/s ETA:{speed.get_eta(remaining)} "
            f"| Q:{submitted - completed} | OK:{stats['success']} ERR:{stats['failed']}"
        )

    try:
        item_iter = iter(image_items)
        done_submitting = False

        while completed < total and not _interrupted:
            # Submit tasks
            while submitted < total and not done_submitting:
                try:
                    image_id, image_path, tag_path = next(item_iter)

                    # Load tags if needed
                    tags = None
                    if with_tags and tag_path.exists():
                        try:
                            with open(tag_path, "r", encoding="utf-8") as f:
                                tag_data = json.load(f)
                                tags = tag_data.get("tags")
                        except Exception:
                            pass

                    task = CaptionTask(
                        image_id=image_id,
                        image_path=image_path,
                        tag_path=tag_path,
                        tags=tags,
                    )
                    task_queue.put_nowait(task)
                    submitted += 1
                except StopIteration:
                    done_submitting = True
                    break
                except queue.Full:
                    break

            # Collect results
            try:
                result = result_queue.get(timeout=0.1)

                # Save caption
                caption_path = input_dir / f"{result.image_id}.caption.json"
                data = {"image_id": result.image_id}

                if result.error:
                    data["error"] = result.error
                    stats["failed"] += 1
                else:
                    data["caption"] = result.caption
                    stats["success"] += 1

                with open(caption_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                completed += 1
                speed.tick()
                pbar.update(1)
                update_pbar()

            except queue.Empty:
                pass

    except KeyboardInterrupt:
        logger.info("\nInterrupted!")
        _interrupted = True

    finally:
        # Send shutdown signals
        for _ in workers:
            try:
                task_queue.put(None, timeout=1.0)
            except queue.Full:
                pass

        # Wait for workers
        for p in workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()

        pbar.close()

    return stats


def process_tensor_parallel(
    image_items: list[tuple[str, Path, Path]],
    input_dir: Path,
    tensor_parallel_size: int,
    model_name: str,
    backend: str,
    base_prompt: str,
    with_tags: bool,
    max_tokens: int,
    temperature: float,
    batch_size: int,
) -> dict[str, int]:
    """
    Process with tensor parallelism (single model across GPUs).
    """
    global _interrupted

    stats = {"success": 0, "failed": 0}
    speed = SpeedMonitor()
    total = len(image_items)

    # Load model
    if backend == "vllm":
        from kohakucaption.local.vllm import VLLMModel, VLLMConfig

        config = VLLMConfig(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_tokens=max_tokens,
            temperature=temperature,
            gpu_memory_utilization=0.85,
        )
        model = VLLMModel(config)
    else:
        from kohakucaption.local.lmdeploy import LMDeployModel, LMDeployConfig

        config = LMDeployConfig(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_tokens=max_tokens,
            temperature=temperature,
            backend="turbomind",
        )
        model = LMDeployModel(config)

    logger.info(f"Loading model with tensor_parallel_size={tensor_parallel_size}")
    model.load()
    logger.info("Model loaded successfully")

    pbar = tqdm(total=total, desc="Captioning", unit="img")

    def update_pbar():
        remaining = total - (stats["success"] + stats["failed"])
        pbar.set_postfix_str(
            f"{speed.get_speed():.2f}/s ETA:{speed.get_eta(remaining)} "
            f"| OK:{stats['success']} ERR:{stats['failed']}"
        )

    try:
        for batch_start in range(0, total, batch_size):
            if _interrupted:
                break

            batch_end = min(batch_start + batch_size, total)
            batch_items = image_items[batch_start:batch_end]

            # Load batch
            batch_images = []
            batch_prompts = []
            batch_ids = []

            for image_id, image_path, tag_path in batch_items:
                try:
                    image = Image.open(image_path)
                    image_rgb = pil_to_rgb(image)
                    batch_images.append(image_rgb)
                    batch_ids.append(image_id)

                    # Build prompt
                    if with_tags and tag_path.exists():
                        try:
                            with open(tag_path, "r", encoding="utf-8") as f:
                                tag_data = json.load(f)
                                tags = tag_data.get("tags")
                                if tags:
                                    tags_str = format_tags_for_prompt(tags)
                                    batch_prompts.append(PROMPT_WITH_TAGS.format(tags=tags_str))
                                else:
                                    batch_prompts.append(base_prompt)
                        except Exception:
                            batch_prompts.append(base_prompt)
                    else:
                        batch_prompts.append(base_prompt)

                except Exception as e:
                    # Write error
                    caption_path = input_dir / f"{image_id}.caption.json"
                    with open(caption_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "image_id": image_id,
                            "error": str(e),
                        }, f, indent=2, ensure_ascii=False)
                    stats["failed"] += 1
                    pbar.update(1)

            if not batch_images:
                continue

            # Batch inference
            try:
                batch_output = model.generate_batch(
                    images=batch_images,
                    prompts=batch_prompts,
                )

                for image_id, output in zip(batch_ids, batch_output.outputs):
                    caption_path = input_dir / f"{image_id}.caption.json"
                    data = {
                        "image_id": image_id,
                        "caption": output.text,
                    }
                    with open(caption_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    stats["success"] += 1

            except Exception as e:
                for image_id in batch_ids:
                    caption_path = input_dir / f"{image_id}.caption.json"
                    with open(caption_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "image_id": image_id,
                            "error": str(e),
                        }, f, indent=2, ensure_ascii=False)
                    stats["failed"] += 1

            speed.tick(len(batch_items))
            pbar.update(len(batch_items))
            update_pbar()

    except KeyboardInterrupt:
        logger.info("\nInterrupted!")
        _interrupted = True

    finally:
        pbar.close()
        model.unload()

    return stats


@click.command()
@click.option(
    "--input-dir", "-i",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Input directory containing images and tag.json files"
)
@click.option(
    "--backend",
    type=click.Choice(["vllm", "lmdeploy"]),
    default="vllm",
    show_default=True,
    help="Inference backend"
)
@click.option(
    "--model", "-m",
    type=str,
    default=None,
    help="Model name/path (default: backend-specific)"
)
@click.option(
    "--gpus",
    type=str,
    default=None,
    help="Comma-separated GPU IDs for data parallelism (e.g., '0,1,2,3')"
)
@click.option(
    "--tensor-parallel", "-tp",
    type=int,
    default=None,
    help="Tensor parallel size. Mutually exclusive with --gpus."
)
@click.option(
    "--with-tags/--no-tags",
    default=False,
    show_default=True,
    help="Include tags as context in the caption prompt"
)
@click.option(
    "--batch-size", "-b",
    type=int,
    default=8,
    show_default=True,
    help="Batch size for tensor parallel mode"
)
@click.option(
    "--max-tokens",
    type=int,
    default=512,
    show_default=True,
    help="Maximum tokens to generate"
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    show_default=True,
    help="Sampling temperature"
)
@click.option(
    "--max-samples", "-n",
    type=int,
    default=None,
    help="Maximum number of samples to process"
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    show_default=True,
    help="Skip already processed samples"
)
@click.option(
    "--prefetch",
    type=int,
    default=4,
    show_default=True,
    help="Number of images to prefetch per GPU (data parallel mode)"
)
@click.option(
    "--prompt",
    type=str,
    default=None,
    help="Custom caption prompt (overrides default)"
)
def main(
    input_dir: Path,
    backend: str,
    model: str | None,
    gpus: str | None,
    tensor_parallel: int | None,
    with_tags: bool,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    max_samples: int | None,
    skip_existing: bool,
    prefetch: int,
    prompt: str | None,
):
    """
    Caption images from a tagged folder.

    Reads images and tag.json files from input directory (output of imagenet_tagging.py)
    and generates caption.json files.

    \b
    Input structure:
        <id>.jpg        - The image
        <id>.tag.json   - Tags (from tagging script)

    Output:
        <id>.caption.json - Generated caption

    \b
    Parallelism modes:
        Data Parallel (--gpus): Multiple model copies, one per GPU.
        Tensor Parallel (--tensor-parallel): Single model sharded across GPUs.

    \b
    Examples:
        # Data parallel on 4 GPUs
        python scripts/imagenet_captioning.py -i ./imagenet_data --gpus 0,1,2,3

        # With tags as context
        python scripts/imagenet_captioning.py -i ./imagenet_data --gpus 0,1,2,3 --with-tags

        # Tensor parallel with LMDeploy
        python scripts/imagenet_captioning.py -i ./imagenet_data -tp 4 --backend lmdeploy
    """
    signal.signal(signal.SIGINT, _signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, _signal_handler)

    # Validate options
    if gpus and tensor_parallel:
        raise click.ClickException("Cannot use both --gpus and --tensor-parallel.")

    if not gpus and not tensor_parallel:
        gpus = "0"
        logger.info("No GPU option specified, defaulting to GPU 0")

    # Parse GPU IDs
    gpu_ids = None
    if gpus:
        gpu_ids = [int(x.strip()) for x in gpus.split(",")]

    # Use default model if not specified
    if model is None:
        model = DEFAULT_MODELS[backend]

    # Use custom or default prompt
    base_prompt = prompt if prompt else DEFAULT_PROMPT

    start_time = time.time()

    # Find images
    logger.info(f"Scanning {input_dir} for images...")
    image_items = find_images(input_dir)
    logger.info(f"Found {len(image_items)} images with tags")

    if not image_items:
        click.echo("No images found! Make sure to run imagenet_tagging.py first.")
        return

    # Limit samples
    if max_samples is not None:
        image_items = image_items[:max_samples]

    # Filter existing
    if skip_existing:
        original = len(image_items)
        image_items = [
            item for item in image_items
            if not (input_dir / f"{item[0]}.caption.json").exists()
        ]
        skipped = original - len(image_items)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already processed samples")

    if not image_items:
        click.echo("All samples already processed!")
        return

    # Print config
    click.echo("=" * 60)
    click.echo("ImageNet-1k Captioning")
    click.echo("=" * 60)
    click.echo(f"Input: {input_dir} | Samples: {len(image_items)}")
    click.echo(f"Backend: {backend} | Model: {model}")
    if gpu_ids:
        click.echo(f"Mode: Data Parallel | GPUs: {gpu_ids}")
    else:
        click.echo(f"Mode: Tensor Parallel | TP Size: {tensor_parallel}")
    click.echo(f"With tags: {with_tags} | Max tokens: {max_tokens}")
    click.echo("=" * 60)

    # Process
    if gpu_ids:
        stats = process_data_parallel(
            image_items=image_items,
            input_dir=input_dir,
            gpu_ids=gpu_ids,
            model_name=model,
            backend=backend,
            base_prompt=base_prompt,
            with_tags=with_tags,
            max_tokens=max_tokens,
            temperature=temperature,
            prefetch=prefetch,
        )
    else:
        stats = process_tensor_parallel(
            image_items=image_items,
            input_dir=input_dir,
            tensor_parallel_size=tensor_parallel,
            model_name=model,
            backend=backend,
            base_prompt=base_prompt,
            with_tags=with_tags,
            max_tokens=max_tokens,
            temperature=temperature,
            batch_size=batch_size,
        )

    elapsed = time.time() - start_time
    total_processed = stats["success"] + stats["failed"]

    click.echo()
    click.echo("=" * 60)
    click.echo(f"Completed in {elapsed:.1f}s")
    click.echo(f"Success: {stats['success']} | Failed: {stats['failed']}")
    if total_processed > 0:
        click.echo(f"Speed: {total_processed / elapsed:.2f} img/s")
    click.echo("=" * 60)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
