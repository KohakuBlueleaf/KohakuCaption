#!/usr/bin/env python3
"""
Batch caption images with LLM API + optional AnimeTIMM Tagger.

Architecture:
  - PyTorch DataLoader: Multi-process image loading, preprocessing, base64 encoding
  - Main thread: Standard DataLoader loop for tagger inference (GPU)
  - Async caption worker: Concurrent LLM API calls

This design is simple and efficient:
  - DataLoader handles I/O parallelism via multiprocessing
  - Main thread does GPU inference (single-threaded, no GIL issues)
  - Async worker handles network I/O (LLM API calls)
"""

import asyncio
import base64
import io
import json
import logging
import os
import random
import signal
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import torch
from torch.utils.data import Dataset, DataLoader


# Global flag for interrupt handling
_interrupted = False


def _signal_handler(signum, frame):
    """Handle interrupt signal."""
    global _interrupted
    _interrupted = True
    raise KeyboardInterrupt

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import aiofiles
from PIL import Image
from tqdm import tqdm

from kohakucaption.clients import ClientConfig, OpenAIClient, OpenRouterClient
from kohakucaption.formats import DefaultFormat
from kohakucaption.tokenizer import TokenCounter
from kohakucaption.types import ImageInput


# Setup logging - only important startup events
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silence noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

DEFAULT_TAGGER_REPO = "animetimm/caformer_b36.dbv4-full"


@dataclass
class CaptionTask:
    """Task for the caption worker."""
    path: Path
    base64_image: str
    tags: dict[str, Any] | None = None  # Tags to send to LLM (None = don't send)
    tags_for_save: dict[str, Any] | None = None  # Tags to save in output (always save if available)


@dataclass
class CaptionResult:
    """Result from caption worker."""
    path: Path
    tags_for_save: dict[str, Any] | None  # Tags to save in output
    caption: dict[str, Any] | None
    success: bool
    error: str | None = None


class ImageDataset(Dataset):
    """
    PyTorch Dataset for image loading and preprocessing.

    Does in worker processes:
      - Load image from disk
      - Preprocess for tagger (tensor)
      - Encode to base64 for LLM API
    """

    def __init__(
        self,
        image_paths: list[Path],
        tagger_transform=None,
        base64_quality: int = 85,
    ):
        self.image_paths = image_paths
        self.tagger_transform = tagger_transform
        self.base64_quality = base64_quality

    def __len__(self):
        return len(self.image_paths)

    def _pil_to_rgb(self, image: Image.Image) -> Image.Image:
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

    def _encode_base64(self, image: Image.Image) -> str:
        """Encode PIL image to base64 JPEG."""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.base64_quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def __getitem__(self, idx: int) -> dict:
        path = self.image_paths[idx]

        try:
            image = Image.open(path)
            image_rgb = self._pil_to_rgb(image)

            # Encode for LLM API
            base64_image = self._encode_base64(image_rgb)

            # Preprocess for tagger (if transform provided)
            tagger_tensor = None
            if self.tagger_transform is not None:
                tagger_tensor = self.tagger_transform(image_rgb)

            return {
                "path": str(path),
                "base64_image": base64_image,
                "tagger_tensor": tagger_tensor,
                "error": None,
            }
        except Exception as e:
            return {
                "path": str(path),
                "base64_image": None,
                "tagger_tensor": None,
                "error": str(e),
            }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles None tensors."""
    paths = [item["path"] for item in batch]
    base64_images = [item["base64_image"] for item in batch]
    errors = [item["error"] for item in batch]

    # Stack valid tensors, keep track of which are valid
    valid_tensors = []
    valid_indices = []
    for i, item in enumerate(batch):
        if item["tagger_tensor"] is not None:
            valid_tensors.append(item["tagger_tensor"])
            valid_indices.append(i)

    tagger_batch = None
    if valid_tensors:
        tagger_batch = torch.stack(valid_tensors)

    return {
        "paths": paths,
        "base64_images": base64_images,
        "tagger_batch": tagger_batch,
        "valid_indices": valid_indices,
        "errors": errors,
    }


class SpeedMonitor:
    def __init__(self, window_size: int = 20):
        self.timestamps: deque[float] = deque(maxlen=window_size)
        self.start_time = time.time()
        self.total_count = 0

    def tick(self):
        self.timestamps.append(time.time())
        self.total_count += 1

    def get_speed(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0

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


def get_api_key(provider: str, api_key: str | None) -> str:
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
    return "openai/gpt-4o" if provider == "openrouter" else "gpt-4o"


def find_images(input_dir: Path, extensions: list[str]) -> list[Path]:
    images = []
    for ext in extensions:
        images.extend(input_dir.glob(f"*{ext}"))
        images.extend(input_dir.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def _convert_result(result) -> dict[str, Any]:
    return {
        "general_tags": list(result.general_tags.keys()),
        "character_tags": list(result.character_tags.keys()),
        "rating_tags": list(result.rating_tags.keys()),
        "general_scores": dict(result.general_tags),
        "character_scores": dict(result.character_tags),
        "rating_scores": dict(result.rating_tags),
    }


def build_prompt_with_tags(base_prompt: str, tags: dict[str, Any] | None) -> str:
    """Build prompt with optional tags context."""
    if not tags:
        return base_prompt

    ctx_lines = []
    if tags.get("general_tags"):
        tag_list = [t.replace("_", " ") for t in tags["general_tags"]]
        ctx_lines.append(f"Features: {', '.join(tag_list)}")
    if tags.get("character_tags"):
        tag_list = [t.replace("_", " ") for t in tags["character_tags"]]
        ctx_lines.append(f"Characters: {', '.join(tag_list)}")
    if tags.get("rating_tags"):
        tag_list = [t.replace("_", " ") for t in tags["rating_tags"]]
        ctx_lines.append(f"Rating: {', '.join(tag_list)}")

    if ctx_lines:
        return f"{base_prompt}\n\nExisting tags:\n" + "\n".join(ctx_lines)
    return base_prompt


async def write_result(path: Path, data: dict):
    """Write result to JSON file."""
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(data, indent=2, ensure_ascii=False))


class CaptionWorker:
    """
    Async caption worker that processes tasks from a queue.

    Runs in a separate thread, manages its own event loop.
    Main thread submits tasks, worker processes them concurrently.
    """

    def __init__(
        self,
        client,
        output_format: DefaultFormat,
        base_prompt: str,
        output_dir: Path,
        max_concurrent: int,
    ):
        self.client = client
        self.output_format = output_format
        self.base_prompt = base_prompt
        self.output_dir = output_dir
        self.max_concurrent = max_concurrent

        self.task_queue: asyncio.Queue[CaptionTask | None] = None
        self.result_queue: asyncio.Queue[CaptionResult] = None
        self.loop: asyncio.AbstractEventLoop = None
        self._thread: ThreadPoolExecutor = None
        self._started = False

    async def _caption_one(
        self,
        task: CaptionTask,
        semaphore: asyncio.Semaphore,
    ) -> CaptionResult:
        """Process a single caption task."""
        async with semaphore:
            # Random sleep to stagger requests (0.03 ~ 0.1 sec)
            await asyncio.sleep(random.uniform(0.03, 0.1))

            try:
                # task.tags is for LLM context (may be None in parallel mode)
                # task.tags_for_save is always saved to output
                final_prompt = build_prompt_with_tags(self.base_prompt, task.tags)
                image = ImageInput(source="", base64_data=task.base64_image, mime_type="image/jpeg")
                result = await self.client.caption(image=image, prompt=final_prompt)

                if result.success:
                    parse_result = self.output_format.parse(result.raw_response)
                    if parse_result.success:
                        return CaptionResult(
                            path=task.path,
                            tags_for_save=task.tags_for_save,
                            caption=parse_result.data,
                            success=True,
                        )
                    return CaptionResult(
                        path=task.path,
                        tags_for_save=task.tags_for_save,
                        caption=None,
                        success=False,
                        error=f"Parse: {parse_result.error}",
                    )
                return CaptionResult(
                    path=task.path,
                    tags_for_save=task.tags_for_save,
                    caption=None,
                    success=False,
                    error=result.error,
                )
            except Exception as e:
                return CaptionResult(
                    path=task.path,
                    tags_for_save=task.tags_for_save,
                    caption=None,
                    success=False,
                    error=str(e),
                )

    async def _worker_loop(self):
        """Main worker loop - processes tasks from queue."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        pending_tasks: list[asyncio.Task] = []

        while True:
            # Get next task (non-blocking if we have pending work)
            try:
                if pending_tasks:
                    # Non-blocking check
                    task = self.task_queue.get_nowait()
                else:
                    # Blocking wait
                    task = await self.task_queue.get()
            except asyncio.QueueEmpty:
                # Check if any pending tasks completed
                if pending_tasks:
                    done, pending_tasks_set = await asyncio.wait(
                        pending_tasks,
                        timeout=0.01,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    pending_tasks = list(pending_tasks_set)
                    for t in done:
                        result = t.result()
                        await self.result_queue.put(result)
                continue

            if task is None:  # Shutdown signal
                # Wait for all pending tasks
                if pending_tasks:
                    done, _ = await asyncio.wait(pending_tasks)
                    for t in done:
                        result = t.result()
                        await self.result_queue.put(result)
                await self.result_queue.put(None)  # Signal completion
                break

            # Start new task
            coro = self._caption_one(task, semaphore)
            pending_tasks.append(asyncio.create_task(coro))

            # Harvest completed tasks
            if len(pending_tasks) >= self.max_concurrent:
                done, pending_tasks_set = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                pending_tasks = list(pending_tasks_set)
                for t in done:
                    result = t.result()
                    await self.result_queue.put(result)

    def _run_loop(self):
        """Run the event loop in a thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._worker_loop())

    def start(self):
        """Start the worker thread."""
        if self._started:
            return

        self.loop = asyncio.new_event_loop()
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()

        self._thread = ThreadPoolExecutor(max_workers=1)
        self._thread.submit(self._run_loop)
        self._started = True

    def submit(self, task: CaptionTask):
        """Submit a task to the worker (thread-safe)."""
        asyncio.run_coroutine_threadsafe(
            self.task_queue.put(task),
            self.loop,
        )

    def shutdown(self):
        """Signal worker to shutdown."""
        asyncio.run_coroutine_threadsafe(
            self.task_queue.put(None),
            self.loop,
        )

    def get_result_blocking(self) -> CaptionResult | None:
        """Get a result, blocking until available."""
        future = asyncio.run_coroutine_threadsafe(
            self.result_queue.get(),
            self.loop,
        )
        return future.result()

    def close(self):
        """Clean up resources."""
        if self._thread:
            self._thread.shutdown(wait=True)
        if self.loop:
            self.loop.close()


def process_batch_caption(
    image_paths: list[Path],
    output_dir: Path,
    client,
    output_format: DefaultFormat,
    prompt: str,
    max_concurrent: int,
    with_tags: bool,
    mode: str,
    tagger_repo: str,
    tagger_batch_size: int,
    num_workers: int = 4,
) -> dict[str, Any]:
    """
    Main batch caption function using PyTorch DataLoader + async caption worker.

    Architecture:
      1. DataLoader (multiprocess): Load images, preprocess tensors, encode base64
      2. Main thread: Tagger inference (GPU) in standard loop
      3. Caption worker (async thread): Concurrent LLM API calls

    Args:
      mode: "context" = send tags to LLM as context; "parallel" = save tags but don't send to LLM

    This is simple and efficient:
      - DataLoader handles I/O parallelism
      - Main thread does sequential GPU work (no threading issues)
      - Async worker handles network I/O
    """
    send_tags_to_llm = (mode == "context")
    stats = {"success": 0, "failed": 0}
    speed = SpeedMonitor()
    total = len(image_paths)
    interrupted = False

    # Load tagger if needed
    tagger = None
    tagger_transform = None
    if with_tags:
        from kohakucaption.tagger import AnimeTimmTagger
        tagger = AnimeTimmTagger(repo_id=tagger_repo, use_fp16=True)
        tagger._load_model()
        tagger_transform = tagger._transform
        logger.info(f"Tagger loaded: {tagger_repo}")

    # Create dataset and dataloader
    dataset = ImageDataset(
        image_paths=image_paths,
        tagger_transform=tagger_transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=tagger_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Start caption worker
    worker = CaptionWorker(
        client=client,
        output_format=output_format,
        base_prompt=prompt,
        output_dir=output_dir,
        max_concurrent=max_concurrent,
    )
    worker.start()

    # Progress bar
    pbar = tqdm(total=total, desc="Processing", unit="img")
    num_batches = len(dataloader)
    submitted = 0
    completed = 0

    def update_pbar():
        pbar.set_postfix_str(
            f"B:{submitted}/{num_batches} "
            f"| {speed.get_speed():.1f}/s ETA:{speed.get_eta(total - completed)} "
            f"| OK:{stats['success']} ERR:{stats['failed']}"
        )

    try:
        # Main loop: iterate DataLoader, run tagger, submit to caption worker
        for batch_idx, batch in enumerate(dataloader):
            paths = [Path(p) for p in batch["paths"]]
            base64_images = batch["base64_images"]
            tagger_batch = batch["tagger_batch"]
            valid_indices = batch["valid_indices"]
            errors = batch["errors"]

            # Run tagger inference if enabled and we have valid tensors
            tags_list: list[dict | None] = [None] * len(paths)

            if with_tags and tagger_batch is not None:
                # GPU inference
                results = tagger.inference(tagger_batch)
                for i, res in zip(valid_indices, results):
                    tags_list[i] = _convert_result(res)

            # Submit tasks to caption worker
            for i, (path, base64_img, tags, error) in enumerate(
                zip(paths, base64_images, tags_list, errors)
            ):
                if error:
                    # Write error result directly
                    data = {"image": path.name, "error": error}
                    with open(output_dir / f"{path.stem}.json", "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    stats["failed"] += 1
                    speed.tick()
                    pbar.update(1)
                    completed += 1
                elif base64_img:
                    # In "parallel" mode, tags are saved but NOT sent to LLM
                    # In "context" mode, tags are sent to LLM as context
                    task = CaptionTask(
                        path=path,
                        base64_image=base64_img,
                        tags=tags if send_tags_to_llm else None,  # For LLM prompt
                        tags_for_save=tags,  # Always save tags in output
                    )
                    worker.submit(task)

            submitted = batch_idx + 1
            update_pbar()

        # Signal worker to finish and collect results
        worker.shutdown()

        while True:
            result = worker.get_result_blocking()
            if result is None:
                break

            # Write result
            data = {"image": result.path.name}
            if result.tags_for_save:
                data["tags"] = result.tags_for_save

            if result.success:
                data["caption"] = result.caption
                stats["success"] += 1
            else:
                data["caption_error"] = result.error
                stats["failed"] += 1

            with open(output_dir / f"{result.path.stem}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            speed.tick()
            completed += 1
            pbar.update(1)
            update_pbar()

    except KeyboardInterrupt:
        interrupted = True
        pbar.write("\n[!] Interrupted! Cleaning up...")

    finally:
        # Clean up worker
        if not interrupted:
            pass  # Already shutdown normally
        else:
            # Force shutdown on interrupt
            worker.shutdown()
        worker.close()
        pbar.close()

        # Clean up tagger GPU memory
        if tagger is not None:
            tagger.unload()

    if interrupted:
        click.echo(f"\nInterrupted. Completed: {stats['success']} OK, {stats['failed']} failed")
        raise KeyboardInterrupt

    return stats


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output-dir", "-o", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--provider", "-p", type=click.Choice(["openai", "openrouter"]), default="openai")
@click.option("--model", "-m", type=str, default=None)
@click.option("--api-key", type=str, default=None, envvar=["OPENAI_API_KEY", "OPENROUTER_API_KEY"])
@click.option("--max-concurrent", "-c", type=int, default=5, show_default=True)
@click.option("--with-tags/--no-tags", default=False, help="Enable tagger to generate tags")
@click.option("--mode", type=click.Choice(["parallel", "context"]), default="context",
              help="parallel: tags saved but NOT sent to LLM; context: tags sent to LLM as context")
@click.option("--tagger-repo", type=str, default=DEFAULT_TAGGER_REPO, show_default=True)
@click.option("--tagger-batch-size", type=int, default=8, show_default=True)
@click.option("--num-workers", "-w", type=int, default=4, show_default=True)
@click.option("--detail", type=click.Choice(["low", "high", "auto"]), default="auto", show_default=True)
@click.option("--timeout", type=float, default=120.0, show_default=True)
@click.option("--extensions", type=str, default=".jpg,.jpeg,.png,.webp,.gif", show_default=True)
@click.option("--skip-existing/--no-skip-existing", default=False)
def main(
    input_dir: Path,
    output_dir: Path | None,
    provider: str,
    model: str | None,
    api_key: str | None,
    max_concurrent: int,
    with_tags: bool,
    mode: str,
    tagger_repo: str,
    tagger_batch_size: int,
    num_workers: int,
    detail: str,
    timeout: float,
    extensions: str,
    skip_existing: bool,
):
    """
    Batch caption images with LLM API + optional tagger.

    \b
    Architecture:
      DataLoader (multiprocess) → Tagger (GPU) → Caption Worker (async)

    Simple and efficient:
      - DataLoader handles I/O parallelism via multiprocessing
      - Main thread does GPU inference (no threading issues)
      - Async worker handles concurrent LLM API calls
    """
    # Register signal handler for clean interrupt
    signal.signal(signal.SIGINT, _signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, _signal_handler)

    start = time.time()
    output_dir = output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ext_list = [e.strip() for e in extensions.split(",")]
    image_paths = find_images(input_dir, ext_list)

    if not image_paths:
        click.echo(f"No images found in {input_dir}")
        return

    if skip_existing:
        orig = len(image_paths)
        image_paths = [p for p in image_paths if not (output_dir / f"{p.stem}.json").exists()]
        if orig - len(image_paths) > 0:
            logger.info(f"Skipped {orig - len(image_paths)} existing")
        if not image_paths:
            click.echo("All done.")
            return

    click.echo("=" * 60)
    click.echo(f"Images: {len(image_paths)} | Provider: {provider} | Model: {model or get_default_model(provider)}")
    click.echo(f"Concurrent: {max_concurrent} | Tags: {with_tags} | Mode: {mode if with_tags else 'N/A'} | Workers: {num_workers}")
    if with_tags:
        click.echo(f"Tagger: {tagger_repo} | Batch: {tagger_batch_size}")
    click.echo("=" * 60)

    config = ClientConfig(
        api_key=get_api_key(provider, api_key),
        model=model or get_default_model(provider),
        timeout=timeout,
        detail=detail,
    )
    client = OpenRouterClient(config) if provider == "openrouter" else OpenAIClient(config)

    output_format = DefaultFormat()
    prompt = f"""Provide a structured caption for this image.

{output_format.get_format_instruction()}"""

    try:
        stats = process_batch_caption(
            image_paths=image_paths,
            output_dir=output_dir,
            client=client,
            output_format=output_format,
            prompt=prompt,
            max_concurrent=max_concurrent,
            with_tags=with_tags,
            mode=mode,
            tagger_repo=tagger_repo,
            tagger_batch_size=tagger_batch_size,
            num_workers=num_workers,
        )
        elapsed = time.time() - start
        click.echo()
        click.echo("=" * 60)
        click.echo(f"Done: {stats['success']} OK, {stats['failed']} failed | {elapsed:.1f}s | {len(image_paths)/elapsed:.2f} img/s")
        click.echo("=" * 60)
    except KeyboardInterrupt:
        elapsed = time.time() - start
        click.echo(f"\nAborted after {elapsed:.1f}s")
    finally:
        # Close client (need to run in event loop)
        asyncio.run(client.close())


if __name__ == "__main__":
    main()
