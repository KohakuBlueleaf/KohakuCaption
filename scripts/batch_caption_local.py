#!/usr/bin/env python3
"""
Batch Local Captioning Script using vLLM.

Generates captions for images in a folder (supports nested subfolders) using vLLM
with the DefaultFormat system for structured output.

Input folder structure (flat or nested):
    folder/
    ├── image1.jpg
    ├── image1.tag.json    (optional - for --with-tags)
    ├── subfolder/
    │   ├── image2.png
    │   └── image2.tag.json
    └── ...

Output:
    <image_stem>.caption.json - Generated caption with structured fields

Features:
    - Supports nested folder structures (recursive image discovery)
    - Multi-GPU data parallel via --gpus flag
    - Tensor parallel for large models via --tensor-parallel
    - FP8 KV cache and pre-quantized FP8 models for efficiency
    - Memory leak mitigation via --restart-after
    - Optional tag context from .tag.json files

Usage:
    # Single GPU (flat folder)
    python scripts/batch_caption_local.py -i ./images --gpus 0

    # Multi-GPU with nested folders
    python scripts/batch_caption_local.py -i ./dataset --gpus 0,1,2,3

    # With tags as context
    python scripts/batch_caption_local.py -i ./tagged_images --gpus 0,1 --with-tags

    # Tensor parallel for large models
    python scripts/batch_caption_local.py -i ./images --tensor-parallel 4 -m google/gemma-3-27b-it

    # With memory leak mitigation (restart workers every 50k samples)
    python scripts/batch_caption_local.py -i ./large_dataset --gpus 0,1,2,3 --restart-after 50000
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import click

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

# Global interrupt flag
_interrupted = False


def _signal_handler(signum, frame):
    """Handle interrupt signal."""
    global _interrupted
    _interrupted = True


# Default model
DEFAULT_MODEL = "unsloth/gemma-3-4b-it-FP8-Dynamic"


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


class SpeedMonitor:
    """Monitor processing speed with moving average."""

    def __init__(self, window_size: int = 50):
        self.timestamps: deque[float] = deque(maxlen=window_size)
        self.counts: deque[int] = deque(maxlen=window_size)
        self.start_time = time.time()
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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}


def get_caption_path(input_dir: Path, image_id: str) -> Path:
    """Get caption file path for an image ID, creating parent dirs if needed."""
    caption_path = input_dir / f"{image_id}.caption.json"
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    return caption_path


def find_images(
    input_dir: Path, recursive: bool = True
) -> list[tuple[str, Path, Path | None]]:
    """
    Find all images in a folder (optionally recursive).

    Returns list of (image_id, image_path, tag_path) tuples.
    - image_id: unique identifier (relative path without extension for nested, stem for flat)
    - image_path: full path to image
    - tag_path: path to .tag.json file (may not exist)

    For nested folders, image_id preserves directory structure with '/' separators.
    """
    results = []

    # Find all image files
    if recursive:
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(input_dir.rglob(f"*{ext}"))
            image_files.extend(input_dir.rglob(f"*{ext.upper()}"))
    else:
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    # Remove duplicates and sort
    image_files = sorted(set(image_files))

    for image_path in image_files:
        # Create unique image_id from relative path
        rel_path = image_path.relative_to(input_dir)
        # Use relative path without extension as ID (preserves folder structure)
        image_id = str(rel_path.with_suffix(""))

        # Look for corresponding tag file in same directory as image
        tag_path = image_path.with_suffix(".tag.json")

        results.append((image_id, image_path, tag_path))

    return results


def build_prompt_with_tags(base_prompt: str, tags: dict[str, Any] | None) -> str:
    """Build prompt with optional tags context."""
    if not tags:
        return base_prompt

    ctx_lines = []

    # Handle different tag formats
    if "general_tags" in tags:
        # Format from batch_caption.py tagger
        if tags.get("general_tags"):
            tag_list = [t.replace("_", " ") for t in tags["general_tags"]]
            ctx_lines.append(f"Features: {', '.join(tag_list)}")
        if tags.get("character_tags"):
            tag_list = [t.replace("_", " ") for t in tags["character_tags"]]
            ctx_lines.append(f"Characters: {', '.join(tag_list)}")
        if tags.get("rating_tags"):
            tag_list = [t.replace("_", " ") for t in tags["rating_tags"]]
            ctx_lines.append(f"Rating: {', '.join(tag_list)}")
    elif "tags" in tags:
        # Format from imagenet_tagging.py
        tag_data = tags["tags"]
        if isinstance(tag_data, dict):
            if tag_data.get("general"):
                tag_list = [t.replace("_", " ") for t in tag_data["general"]]
                ctx_lines.append(f"Features: {', '.join(tag_list)}")
            if tag_data.get("character"):
                tag_list = [t.replace("_", " ") for t in tag_data["character"]]
                ctx_lines.append(f"Characters: {', '.join(tag_list)}")
            if tag_data.get("rating"):
                tag_list = [t.replace("_", " ") for t in tag_data["rating"]]
                ctx_lines.append(f"Rating: {', '.join(tag_list)}")

    if ctx_lines:
        return f"{base_prompt}\n\nExisting tags:\n" + "\n".join(ctx_lines)
    return base_prompt


def run_worker_subprocess(
    gpu_id: str,
    rank: int,
    world_size: int,
    input_dir: Path,
    model: str,
    with_tags: bool,
    batch_size: int,
    max_tokens: int,
    max_model_len: int,
    temperature: float,
    max_samples: int | None,
    skip_existing: bool,
    gpu_memory_utilization: float,
    kv_cache_dtype: str,
    quantization: str,
    restart_after: int | None = None,
    recursive: bool = True,
) -> subprocess.Popen:
    """Spawn a worker subprocess with CUDA_VISIBLE_DEVICES set."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    cmd = [
        sys.executable,
        __file__,
        "-i",
        str(input_dir),
        "--gpus",
        gpu_id,  # Single GPU
        "--model",
        model,
        "--batch-size",
        str(batch_size),
        "--max-tokens",
        str(max_tokens),
        "--max-model-len",
        str(max_model_len),
        "--temperature",
        str(temperature),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--kv-cache-dtype",
        kv_cache_dtype,
        "--quantization",
        quantization,
        "--worker-rank",
        str(rank),
        "--worker-world-size",
        str(world_size),
    ]

    if with_tags:
        cmd.append("--with-tags")
    if not skip_existing:
        cmd.append("--no-skip-existing")
    if not recursive:
        cmd.append("--no-recursive")
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])
    if restart_after is not None:
        cmd.extend(["--restart-after", str(restart_after)])

    return subprocess.Popen(cmd, env=env)


def process_batch(
    image_items: list[tuple[str, Path, Path]],
    input_dir: Path,
    model_name: str,
    base_prompt: str,
    output_format,
    with_tags: bool,
    batch_size: int,
    max_tokens: int,
    max_model_len: int,
    temperature: float,
    rank: int,
    world_size: int,
    gpu_memory_utilization: float = 0.95,
    kv_cache_dtype: str = "fp8",
    quantization: str | None = None,
    restart_after: int | None = None,
) -> dict[str, int]:
    """
    Process images in batches using vLLM's batch inference.

    If world_size > 1, only processes items[rank::world_size].
    """
    global _interrupted
    from kohakucaption.local.vllm import VLLMModel, VLLMConfig

    # Shard data if multi-GPU
    if world_size > 1:
        my_items = image_items[rank::world_size]
    else:
        my_items = image_items

    if not my_items:
        return {"success": 0, "failed": 0}

    stats = {"success": 0, "failed": 0}
    speed = SpeedMonitor()
    total = len(my_items)

    # Load model
    config = VLLMConfig(
        model=model_name,
        tensor_parallel_size=1,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        temperature=temperature,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_dtype=kv_cache_dtype,
        quantization=quantization if quantization else None,
    )
    model = VLLMModel(config)

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    logger.info(f"[GPU {gpu_id}] Loading model: {model_name}")
    model.load()
    logger.info(f"[GPU {gpu_id}] Model loaded")

    pbar = tqdm(
        total=total,
        desc=f"[GPU {gpu_id}] Captioning",
        unit="img",
        position=rank,
    )

    def update_pbar():
        remaining = total - (stats["success"] + stats["failed"])
        pbar.set_postfix_str(
            f"{speed.get_speed():.2f}/s ETA:{speed.get_eta(remaining)} "
            f"| OK:{stats['success']} ERR:{stats['failed']}"
        )

    try:
        # Process in batches
        for batch_start in range(0, total, batch_size):
            if _interrupted:
                break

            batch_end = min(batch_start + batch_size, total)
            batch_items = my_items[batch_start:batch_end]

            # Load batch images and build prompts
            batch_images = []
            batch_prompts = []
            batch_ids = []
            batch_errors = []

            for image_id, image_path, tag_path in batch_items:
                try:
                    # Load image
                    image = Image.open(image_path)
                    image_rgb = pil_to_rgb(image)
                    batch_images.append(image_rgb)
                    batch_ids.append(image_id)

                    # Build prompt with tags if enabled
                    if with_tags and tag_path.exists():
                        try:
                            with open(tag_path, "r", encoding="utf-8") as f:
                                tag_data = json.load(f)
                            prompt = build_prompt_with_tags(base_prompt, tag_data)
                        except Exception:
                            prompt = base_prompt
                    else:
                        prompt = base_prompt

                    batch_prompts.append(prompt)
                    batch_errors.append(None)

                except Exception as e:
                    batch_errors.append(str(e))
                    batch_ids.append(image_id)

            # Filter out errors for batch inference
            valid_indices = [i for i, e in enumerate(batch_errors) if e is None]

            if valid_indices:
                valid_images = [batch_images[i] for i in range(len(batch_images))]
                valid_prompts = [batch_prompts[i] for i in range(len(batch_prompts))]

                # Batch inference with vLLM - with retry for non-CUDA errors
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        batch_output = model.generate_batch(valid_images, valid_prompts)
                        batch_success = True
                        break
                    except Exception as e:
                        error_msg = str(e)
                        is_cuda_error = "CUDA" in error_msg or "cuda" in error_msg

                        if is_cuda_error:
                            logger.error(
                                f"[GPU {gpu_id}] CUDA error during batch inference: {error_msg[:200]}"
                            )
                            # Mark all items in this batch as failed
                            stats["failed"] += len(batch_ids)
                            batch_success = False
                            logger.error(
                                f"[GPU {gpu_id}] Stopping due to CUDA error. "
                                f"Restart with --skip-existing to resume."
                            )
                            _interrupted = True
                            break
                        else:
                            if attempt < max_retries - 1:
                                logger.warning(
                                    f"[GPU {gpu_id}] Batch inference error (attempt {attempt + 1}/{max_retries}): {error_msg[:100]}"
                                )
                                continue
                            else:
                                logger.error(
                                    f"[GPU {gpu_id}] Batch inference failed after {max_retries} attempts: {error_msg[:200]}"
                                )
                                stats["failed"] += len(batch_ids)
                                batch_success = False

                if _interrupted:
                    break

                if batch_success:
                    # Process results - retry parse errors individually
                    valid_idx = 0
                    for i, (image_id, error) in enumerate(zip(batch_ids, batch_errors)):
                        if error:
                            stats["failed"] += 1
                            continue

                        caption_path = get_caption_path(input_dir, image_id)
                        gen_output = batch_output.outputs[valid_idx]
                        valid_idx += 1

                        parse_result = output_format.parse(gen_output.text)
                        if parse_result.success:
                            data = {
                                "image_id": image_id,
                                "caption": parse_result.data,
                            }
                            stats["success"] += 1
                            with open(caption_path, "w", encoding="utf-8") as f:
                                json.dump(data, f, indent=2, ensure_ascii=False)
                        else:
                            # Parse error - retry this single image
                            retry_success = False
                            last_raw_output = gen_output.text
                            for retry in range(2):
                                try:
                                    retry_output = model.generate_batch(
                                        [valid_images[i]], [valid_prompts[i]]
                                    )
                                    last_raw_output = retry_output.outputs[0].text
                                    retry_parse = output_format.parse(last_raw_output)
                                    if retry_parse.success:
                                        data = {
                                            "image_id": image_id,
                                            "caption": retry_parse.data,
                                        }
                                        stats["success"] += 1
                                        with open(
                                            caption_path, "w", encoding="utf-8"
                                        ) as f:
                                            json.dump(
                                                data, f, indent=2, ensure_ascii=False
                                            )
                                        retry_success = True
                                        break
                                except Exception:
                                    pass
                            if not retry_success:
                                stats["failed"] += 1
                                # Write raw output to .caption.fail for debugging
                                fail_path = input_dir / f"{image_id}.caption.fail"
                                fail_path.parent.mkdir(parents=True, exist_ok=True)
                                with open(fail_path, "w", encoding="utf-8") as f:
                                    f.write(last_raw_output)
                                logger.warning(
                                    f"[GPU {gpu_id}] Parse error for {image_id} after retries"
                                )
            else:
                # All items in batch had errors - don't write files
                stats["failed"] += len(batch_ids)

            speed.tick(len(batch_items))
            pbar.update(len(batch_items))
            update_pbar()

            # Check if we should restart to avoid memory leaks
            processed = stats["success"] + stats["failed"]
            if restart_after is not None and processed >= restart_after:
                logger.info(
                    f"[GPU {gpu_id}] Processed {processed} samples, "
                    f"restarting to avoid memory leaks..."
                )
                break

    except KeyboardInterrupt:
        logger.info(f"\n[GPU {gpu_id}] Interrupted!")
        _interrupted = True

    finally:
        pbar.close()
        model.unload()

    return stats


def process_tensor_parallel(
    image_items: list[tuple[str, Path, Path]],
    input_dir: Path,
    tensor_parallel_size: int,
    model_name: str,
    base_prompt: str,
    output_format,
    with_tags: bool,
    batch_size: int,
    max_tokens: int,
    max_model_len: int,
    temperature: float,
    gpu_memory_utilization: float = 0.95,
    kv_cache_dtype: str = "fp8",
    quantization: str | None = None,
) -> dict[str, int]:
    """
    Process with tensor parallelism (single model across GPUs).
    Uses batch inference for efficiency.
    """
    global _interrupted
    from kohakucaption.local.vllm import VLLMModel, VLLMConfig

    stats = {"success": 0, "failed": 0}
    speed = SpeedMonitor()
    total = len(image_items)

    # Load model with tensor parallelism
    config = VLLMConfig(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        temperature=temperature,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_dtype=kv_cache_dtype,
        quantization=quantization if quantization else None,
    )
    model = VLLMModel(config)

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
            batch_errors = []

            for image_id, image_path, tag_path in batch_items:
                try:
                    image = Image.open(image_path)
                    image_rgb = pil_to_rgb(image)
                    batch_images.append(image_rgb)
                    batch_ids.append(image_id)

                    if with_tags and tag_path.exists():
                        try:
                            with open(tag_path, "r", encoding="utf-8") as f:
                                tag_data = json.load(f)
                            prompt = build_prompt_with_tags(base_prompt, tag_data)
                        except Exception:
                            prompt = base_prompt
                    else:
                        prompt = base_prompt

                    batch_prompts.append(prompt)
                    batch_errors.append(None)

                except Exception as e:
                    batch_errors.append(str(e))
                    batch_ids.append(image_id)

            # Filter valid items
            valid_indices = [i for i, e in enumerate(batch_errors) if e is None]

            if valid_indices:
                valid_images = [batch_images[i] for i in range(len(batch_images))]
                valid_prompts = [batch_prompts[i] for i in range(len(batch_prompts))]

                # Batch inference with retry for non-CUDA errors
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        batch_output = model.generate_batch(valid_images, valid_prompts)
                        batch_success = True
                        break
                    except Exception as e:
                        error_msg = str(e)
                        is_cuda_error = "CUDA" in error_msg or "cuda" in error_msg

                        if is_cuda_error:
                            logger.error(
                                f"CUDA error during batch inference: {error_msg[:200]}"
                            )
                            stats["failed"] += len(batch_ids)
                            batch_success = False
                            logger.error(
                                "Stopping due to CUDA error. Restart with --skip-existing to resume."
                            )
                            _interrupted = True
                            break
                        else:
                            if attempt < max_retries - 1:
                                logger.warning(
                                    f"Batch inference error (attempt {attempt + 1}/{max_retries}): {error_msg[:100]}"
                                )
                                continue
                            else:
                                logger.error(
                                    f"Batch inference failed after {max_retries} attempts: {error_msg[:200]}"
                                )
                                stats["failed"] += len(batch_ids)
                                batch_success = False

                if _interrupted:
                    break

                if batch_success:
                    valid_idx = 0
                    for i, (image_id, error) in enumerate(zip(batch_ids, batch_errors)):
                        if error:
                            stats["failed"] += 1
                            continue

                        caption_path = get_caption_path(input_dir, image_id)
                        gen_output = batch_output.outputs[valid_idx]
                        valid_idx += 1

                        parse_result = output_format.parse(gen_output.text)
                        if parse_result.success:
                            data = {
                                "image_id": image_id,
                                "caption": parse_result.data,
                            }
                            stats["success"] += 1
                            with open(caption_path, "w", encoding="utf-8") as f:
                                json.dump(data, f, indent=2, ensure_ascii=False)
                        else:
                            # Parse error - retry this single image
                            retry_success = False
                            last_raw_output = gen_output.text
                            for retry in range(2):
                                try:
                                    retry_output = model.generate_batch(
                                        [valid_images[i]], [valid_prompts[i]]
                                    )
                                    last_raw_output = retry_output.outputs[0].text
                                    retry_parse = output_format.parse(last_raw_output)
                                    if retry_parse.success:
                                        data = {
                                            "image_id": image_id,
                                            "caption": retry_parse.data,
                                        }
                                        stats["success"] += 1
                                        with open(
                                            caption_path, "w", encoding="utf-8"
                                        ) as f:
                                            json.dump(
                                                data, f, indent=2, ensure_ascii=False
                                            )
                                        retry_success = True
                                        break
                                except Exception:
                                    pass
                            if not retry_success:
                                stats["failed"] += 1
                                # Write raw output to .caption.fail for debugging
                                fail_path = input_dir / f"{image_id}.caption.fail"
                                fail_path.parent.mkdir(parents=True, exist_ok=True)
                                with open(fail_path, "w", encoding="utf-8") as f:
                                    f.write(last_raw_output)
                                logger.warning(
                                    f"Parse error for {image_id} after retries"
                                )
            else:
                # All items in batch had errors - don't write files
                stats["failed"] += len(batch_ids)

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
    "--input-dir",
    "-i",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Input directory containing images (supports nested folders)",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    show_default=True,
    help="Recursively search for images in subdirectories",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default=None,
    help=(
        "Model name/path. Default: unsloth/gemma-3-4b-it-FP8-Dynamic. "
        "Gemma3 FP8: unsloth/gemma-3-4b-it-FP8-Dynamic, unsloth/gemma-3-12b-it-FP8-Dynamic. "
        "Gemma3 BF16: google/gemma-3-4b-it, google/gemma-3-12b-it, google/gemma-3-27b-it."
    ),
)
@click.option(
    "--gpus",
    type=str,
    default="0",
    show_default=True,
    help="Comma-separated GPU IDs (e.g., '0,3,4,5')",
)
@click.option(
    "--tensor-parallel",
    "-tp",
    type=int,
    default=None,
    help="Tensor parallel size (use instead of --gpus for large models)",
)
@click.option(
    "--with-tags/--no-tags",
    default=False,
    show_default=True,
    help="Include tags as context in the caption prompt",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=8,
    show_default=True,
    help="Batch size for inference",
)
@click.option(
    "--max-tokens",
    type=int,
    default=2048,
    show_default=True,
    help="Maximum tokens to generate",
)
@click.option(
    "--max-model-len",
    type=int,
    default=8192,
    show_default=True,
    help="Maximum model context length (default 8k, Gemma3 supports up to 128k)",
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    show_default=True,
    help="Sampling temperature",
)
@click.option(
    "--max-samples",
    "-n",
    type=int,
    default=None,
    help="Maximum number of samples to process",
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    show_default=True,
    help="Skip already processed samples",
)
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
    "--restart-after",
    type=int,
    default=None,
    help="Restart worker after processing N samples to avoid memory leaks (e.g., 50000)",
)
@click.option(
    "--worker-rank",
    type=int,
    default=None,
    hidden=True,
    help="Internal: worker rank for multi-GPU",
)
@click.option(
    "--worker-world-size",
    type=int,
    default=None,
    hidden=True,
    help="Internal: world size for multi-GPU",
)
def main(
    input_dir: Path,
    recursive: bool,
    model: str | None,
    gpus: str,
    tensor_parallel: int | None,
    with_tags: bool,
    batch_size: int,
    max_tokens: int,
    max_model_len: int,
    temperature: float,
    max_samples: int | None,
    skip_existing: bool,
    gpu_memory_utilization: float,
    kv_cache_dtype: str,
    quantization: str,
    restart_after: int | None,
    worker_rank: int | None,
    worker_world_size: int | None,
):
    """
    Batch caption images using local vLLM inference.

    Generates structured captions for images in a folder (supports nested subfolders).
    Uses the DefaultFormat system for structured output with:
    - aesthetic_score, nsfw_score, quality_score
    - title, brief, description

    \b
    Input structure (flat or nested):
        folder/
        ├── image1.jpg
        ├── image1.tag.json    (optional)
        ├── subfolder/
        │   └── image2.png
        └── ...

    Output:
        <relative_path>.caption.json - Generated caption (same location as image)

    \b
    Parallelism modes:
        Data Parallel (--gpus): Multiple model copies, one per GPU.
        Tensor Parallel (--tensor-parallel): Single model sharded across GPUs.

    \b
    Examples:
        # Single GPU (flat folder)
        python scripts/batch_caption_local.py -i ./images --gpus 0

        # Multi-GPU with nested folders
        python scripts/batch_caption_local.py -i ./dataset --gpus 0,1,2,3

        # With tags as context
        python scripts/batch_caption_local.py -i ./tagged_images --gpus 0,1 --with-tags

        # Tensor parallel (for large models like gemma-3-27b-it)
        python scripts/imagenet_captioning.py -i ./imagenet_data --tensor-parallel 4 -m google/gemma-3-27b-it
    """
    from kohakucaption.formats import DefaultFormat

    signal.signal(signal.SIGINT, _signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, _signal_handler)

    # Use default model if not specified
    if model is None:
        model = DEFAULT_MODEL

    # Create output format and base prompt
    output_format = DefaultFormat()
    base_prompt = f"""Provide a structured caption for this image.

{output_format.get_format_instruction()}"""

    # Parse GPU IDs
    gpu_ids = [g.strip() for g in gpus.split(",")]

    # Check if this is a worker subprocess or main process
    is_worker = worker_rank is not None and worker_world_size is not None

    if is_worker:
        # Worker subprocess: process our shard
        image_items = find_images(input_dir, recursive=recursive)

        if max_samples is not None:
            image_items = image_items[:max_samples]

        if skip_existing:
            image_items = [
                item
                for item in image_items
                if not (
                    input_dir / f"{item[0]}.caption.json"
                ).exists()  # item[0] is image_id
            ]

        if not image_items:
            return

        # Convert "none" to None for quantization
        quant = None if quantization == "none" else quantization

        process_batch(
            image_items=image_items,
            input_dir=input_dir,
            model_name=model,
            base_prompt=base_prompt,
            output_format=output_format,
            with_tags=with_tags,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
            rank=worker_rank,
            world_size=worker_world_size,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_dtype=kv_cache_dtype,
            quantization=quant,
            restart_after=restart_after,
        )

    elif tensor_parallel:
        # Tensor parallel mode
        # Convert "none" to None for quantization
        quant = None if quantization == "none" else quantization

        _run_tensor_parallel(
            input_dir=input_dir,
            model=model,
            tensor_parallel=tensor_parallel,
            base_prompt=base_prompt,
            output_format=output_format,
            with_tags=with_tags,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
            max_samples=max_samples,
            skip_existing=skip_existing,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_dtype=kv_cache_dtype,
            quantization=quant,
            recursive=recursive,
        )

    elif len(gpu_ids) > 1:
        # Multi-GPU data parallel: spawn subprocesses
        _run_multi_gpu(
            input_dir=input_dir,
            model=model,
            gpu_ids=gpu_ids,
            with_tags=with_tags,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
            max_samples=max_samples,
            skip_existing=skip_existing,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_dtype=kv_cache_dtype,
            quantization=quantization,
            restart_after=restart_after,
            recursive=recursive,
        )

    else:
        # Single GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]

        image_items = find_images(input_dir, recursive=recursive)

        if not image_items:
            click.echo("No images found in the input directory.")
            return

        if max_samples is not None:
            image_items = image_items[:max_samples]

        if skip_existing:
            original = len(image_items)
            image_items = [
                item
                for item in image_items
                if not (
                    input_dir / f"{item[0]}.caption.json"
                ).exists()  # item[0] is image_id
            ]
            skipped = original - len(image_items)
            if skipped > 0:
                logger.info(f"Skipping {skipped} already processed samples")

        if not image_items:
            click.echo("All samples already processed!")
            return

        # Convert "none" to None for quantization
        quant = None if quantization == "none" else quantization

        # Print config
        click.echo("=" * 60)
        click.echo("Batch Captioning")
        click.echo("=" * 60)
        click.echo(f"Input: {input_dir} | Samples: {len(image_items)}")
        click.echo(f"Model: {model}")
        click.echo(f"Mode: Single GPU | Batch size: {batch_size}")
        click.echo(f"With tags: {with_tags} | Max tokens: {max_tokens}")
        quant_display = (
            quantization if quantization != "none" else "none (use pre-quantized model)"
        )
        click.echo(
            f"GPU Mem: {gpu_memory_utilization:.0%} | KV Cache: {kv_cache_dtype} | Quant: {quant_display}"
        )
        click.echo("=" * 60)

        start_time = time.time()

        stats = process_batch(
            image_items=image_items,
            input_dir=input_dir,
            model_name=model,
            base_prompt=base_prompt,
            output_format=output_format,
            with_tags=with_tags,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
            rank=0,
            world_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_dtype=kv_cache_dtype,
            quantization=quant,
            restart_after=restart_after,
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


def _run_multi_gpu(
    input_dir: Path,
    model: str,
    gpu_ids: list[str],
    with_tags: bool,
    batch_size: int,
    max_tokens: int,
    max_model_len: int,
    temperature: float,
    max_samples: int | None,
    skip_existing: bool,
    gpu_memory_utilization: float,
    kv_cache_dtype: str,
    quantization: str,
    restart_after: int | None = None,
    recursive: bool = True,
):
    """Main process: spawn worker subprocesses for each GPU."""
    start_time = time.time()
    world_size = len(gpu_ids)

    # Find and filter images first (so all workers see same list)
    logger.info(f"Scanning {input_dir} for images...")
    image_items = find_images(input_dir, recursive=recursive)
    logger.info(f"Found {len(image_items)} images with tags")

    if not image_items:
        click.echo("No images found in the input directory.")
        return

    if max_samples is not None:
        image_items = image_items[:max_samples]

    if skip_existing:
        original = len(image_items)
        image_items = [
            item
            for item in image_items
            if not (
                input_dir / f"{item[0]}.caption.json"
            ).exists()  # item[0] is image_id
        ]
        skipped = original - len(image_items)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already processed samples")

    if not image_items:
        click.echo("All samples already processed!")
        return

    # Print config
    click.echo("=" * 60)
    click.echo("Batch Captioning")
    click.echo("=" * 60)
    click.echo(f"Input: {input_dir} | Samples: {len(image_items)}")
    click.echo(f"Model: {model}")
    click.echo(f"Mode: Data Parallel | GPUs: {gpu_ids} | Batch size: {batch_size}")
    click.echo(f"With tags: {with_tags} | Max tokens: {max_tokens}")
    quant_display = (
        quantization if quantization != "none" else "none (use pre-quantized model)"
    )
    click.echo(
        f"GPU Mem: {gpu_memory_utilization:.0%} | KV Cache: {kv_cache_dtype} | Quant: {quant_display}"
    )
    if restart_after:
        click.echo(
            f"Restart after: {restart_after} samples per worker (memory leak mitigation)"
        )
    click.echo("=" * 60)

    # Restart loop for memory leak mitigation
    iteration = 0
    try:
        while True:
            iteration += 1

            # Re-check remaining samples (with skip_existing always True in restart mode)
            if restart_after and iteration > 1:
                remaining_items = [
                    item
                    for item in image_items
                    if not (
                        input_dir / f"{item[0]}.caption.json"
                    ).exists()  # item[0] is image_id
                ]
                if not remaining_items:
                    logger.info("All samples processed!")
                    break
                logger.info(
                    f"Restart iteration {iteration}: {len(remaining_items)} samples remaining"
                )

            # Spawn worker subprocesses
            processes = []
            for rank, gpu_id in enumerate(gpu_ids):
                p = run_worker_subprocess(
                    gpu_id=gpu_id,
                    rank=rank,
                    world_size=world_size,
                    input_dir=input_dir,
                    model=model,
                    with_tags=with_tags,
                    batch_size=batch_size,
                    max_tokens=max_tokens,
                    max_model_len=max_model_len,
                    temperature=temperature,
                    max_samples=max_samples,
                    skip_existing=(
                        True if restart_after else skip_existing
                    ),  # Always skip in restart mode
                    gpu_memory_utilization=gpu_memory_utilization,
                    kv_cache_dtype=kv_cache_dtype,
                    quantization=quantization,
                    restart_after=restart_after,
                    recursive=recursive,
                )
                processes.append(p)
                if iteration == 1:
                    logger.info(
                        f"Spawned worker on GPU {gpu_id} (rank {rank}/{world_size})"
                    )

            # Wait for all workers
            for p in processes:
                p.wait()

            # If no restart_after, exit after one iteration
            if not restart_after:
                break

    except KeyboardInterrupt:
        logger.info("\nInterrupted! Terminating workers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait()

    elapsed = time.time() - start_time
    click.echo()
    click.echo("=" * 60)
    click.echo(f"Completed in {elapsed:.1f}s")
    click.echo("=" * 60)


def _run_tensor_parallel(
    input_dir: Path,
    model: str,
    tensor_parallel: int,
    base_prompt: str,
    output_format,
    with_tags: bool,
    batch_size: int,
    max_tokens: int,
    max_model_len: int,
    temperature: float,
    max_samples: int | None,
    skip_existing: bool,
    gpu_memory_utilization: float,
    kv_cache_dtype: str,
    quantization: str | None,
    recursive: bool = True,
):
    """Tensor parallel mode: single model across GPUs."""
    start_time = time.time()

    # Find images
    logger.info(f"Scanning {input_dir} for images...")
    image_items = find_images(input_dir, recursive=recursive)
    logger.info(f"Found {len(image_items)} images with tags")

    if not image_items:
        click.echo("No images found in the input directory.")
        return

    if max_samples is not None:
        image_items = image_items[:max_samples]

    if skip_existing:
        original = len(image_items)
        image_items = [
            item
            for item in image_items
            if not (
                input_dir / f"{item[0]}.caption.json"
            ).exists()  # item[0] is image_id
        ]
        skipped = original - len(image_items)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already processed samples")

    if not image_items:
        click.echo("All samples already processed!")
        return

    # Print config
    quant_str = quantization if quantization else "none"
    click.echo("=" * 60)
    click.echo("Batch Captioning")
    click.echo("=" * 60)
    click.echo(f"Input: {input_dir} | Samples: {len(image_items)}")
    click.echo(f"Model: {model}")
    click.echo(
        f"Mode: Tensor Parallel | TP Size: {tensor_parallel} | Batch size: {batch_size}"
    )
    click.echo(f"With tags: {with_tags} | Max tokens: {max_tokens}")
    quant_display = (
        quant_str if quant_str != "none" else "none (use pre-quantized model)"
    )
    click.echo(
        f"GPU Mem: {gpu_memory_utilization:.0%} | KV Cache: {kv_cache_dtype} | Quant: {quant_display}"
    )
    click.echo("=" * 60)

    # Process
    stats = process_tensor_parallel(
        image_items=image_items,
        input_dir=input_dir,
        tensor_parallel_size=tensor_parallel,
        model_name=model,
        base_prompt=base_prompt,
        output_format=output_format,
        with_tags=with_tags,
        batch_size=batch_size,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        temperature=temperature,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_dtype=kv_cache_dtype,
        quantization=quantization,
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
    main()
