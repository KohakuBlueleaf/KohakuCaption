#!/usr/bin/env python3
"""
ImageNet-1k Tagging Script using HuggingFace Datasets.

Saves images and tags to output folder with structure:
    <id>.jpg        - The image
    <id>.tag.json   - Tags and metadata

Supports multi-GPU data parallelism for faster processing.

Usage:
    # Single GPU
    python scripts/imagenet_tagging.py -o ./imagenet_processed

    # Multi-GPU data parallel (4 GPUs)
    python scripts/imagenet_tagging.py -o ./imagenet_processed --gpus 0,1,2,3

    # Validation split only
    python scripts/imagenet_tagging.py -o ./imagenet_processed -s validation
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

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global interrupt flag
_interrupted = False


def _signal_handler(signum, frame):
    """Handle interrupt signal."""
    global _interrupted
    _interrupted = True


DEFAULT_TAGGER_REPO = "animetimm/caformer_b36.dbv4-full"


@dataclass
class TagTask:
    """Task for tagging worker."""
    index: int
    label: int
    image: Image.Image


@dataclass
class TagResult:
    """Result from tagging worker."""
    index: int
    label: int
    image: Image.Image
    tags: dict[str, Any] | None
    error: str | None = None


def convert_tag_result(result) -> dict[str, Any]:
    """Convert AnimeTimmTagResult to serializable dict."""
    return {
        "general_tags": list(result.general_tags.keys()),
        "character_tags": list(result.character_tags.keys()),
        "rating_tags": list(result.rating_tags.keys()),
        "general_scores": dict(result.general_tags),
        "character_scores": dict(result.character_tags),
        "rating_scores": dict(result.rating_tags),
    }


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


def worker_process(
    gpu_id: int,
    tagger_repo: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    batch_size: int,
):
    """
    Worker process for tagging on a single GPU.

    Collects tasks into batches for efficient inference.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        from kohakucaption.tagger import AnimeTimmTagger

        logger.info(f"[GPU {gpu_id}] Loading tagger: {tagger_repo}")
        tagger = AnimeTimmTagger(repo_id=tagger_repo, use_fp16=True)
        tagger._load_model()
        logger.info(f"[GPU {gpu_id}] Tagger loaded")

        batch_tasks: list[TagTask] = []

        def process_batch():
            """Process accumulated batch."""
            if not batch_tasks:
                return

            # Preprocess images
            images_rgb = [pil_to_rgb(t.image) for t in batch_tasks]
            batch_tensor = tagger.preprocess_batch(images_rgb)

            # Run inference
            tag_results = tagger.inference(batch_tensor)

            # Send results
            for task, tags in zip(batch_tasks, tag_results):
                result_queue.put(TagResult(
                    index=task.index,
                    label=task.label,
                    image=task.image,
                    tags=convert_tag_result(tags),
                ))

            batch_tasks.clear()

        while True:
            try:
                task = task_queue.get(timeout=0.1)
            except queue.Empty:
                # Process any remaining batch
                if batch_tasks:
                    process_batch()
                continue

            if task is None:  # Shutdown signal
                process_batch()  # Process remaining
                break

            batch_tasks.append(task)

            if len(batch_tasks) >= batch_size:
                process_batch()

        tagger.unload()
        logger.info(f"[GPU {gpu_id}] Worker finished")

    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Worker failed: {e}")
        import traceback
        traceback.print_exc()


def process_single_gpu(
    dataset,
    indices: list[int],
    output_dir: Path,
    tagger_repo: str,
    batch_size: int,
    num_workers: int,
    image_format: str,
    image_quality: int,
) -> dict[str, int]:
    """Process with single GPU using DataLoader."""
    global _interrupted

    from torch.utils.data import DataLoader, Dataset
    from kohakucaption.tagger import AnimeTimmTagger

    stats = {"success": 0, "failed": 0}
    speed = SpeedMonitor()
    total = len(indices)

    # Load tagger
    logger.info(f"Loading tagger: {tagger_repo}")
    tagger = AnimeTimmTagger(repo_id=tagger_repo, use_fp16=True)
    tagger._load_model()
    logger.info(f"Tagger loaded on {tagger.device}")

    class ImageNetDataset(Dataset):
        def __init__(self, hf_dataset, indices, transform):
            self.hf_dataset = hf_dataset
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            actual_idx = self.indices[idx]
            try:
                item = self.hf_dataset[actual_idx]
                image = item["image"]
                label = item["label"]
                image_rgb = pil_to_rgb(image)
                tensor = self.transform(image_rgb)
                return {
                    "index": actual_idx,
                    "label": label,
                    "image": image,  # Original for saving
                    "tensor": tensor,
                    "error": None,
                }
            except Exception as e:
                return {
                    "index": actual_idx,
                    "label": -1,
                    "image": None,
                    "tensor": None,
                    "error": str(e),
                }

    def collate_fn(batch):
        indices = [item["index"] for item in batch]
        labels = [item["label"] for item in batch]
        images = [item["image"] for item in batch]
        errors = [item["error"] for item in batch]

        valid_tensors = []
        valid_indices = []
        for i, item in enumerate(batch):
            if item["tensor"] is not None:
                valid_tensors.append(item["tensor"])
                valid_indices.append(i)

        tensor_batch = torch.stack(valid_tensors) if valid_tensors else None

        return {
            "indices": indices,
            "labels": labels,
            "images": images,
            "tensor_batch": tensor_batch,
            "valid_indices": valid_indices,
            "errors": errors,
        }

    torch_dataset = ImageNetDataset(dataset, indices, tagger._transform)
    dataloader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    pbar = tqdm(total=total, desc="Tagging", unit="img")

    def update_pbar():
        remaining = total - (stats["success"] + stats["failed"])
        pbar.set_postfix_str(
            f"{speed.get_speed():.1f}/s ETA:{speed.get_eta(remaining)} "
            f"| OK:{stats['success']} ERR:{stats['failed']}"
        )

    try:
        for batch in dataloader:
            if _interrupted:
                break

            batch_indices = batch["indices"]
            batch_labels = batch["labels"]
            batch_images = batch["images"]
            tensor_batch = batch["tensor_batch"]
            valid_indices = batch["valid_indices"]
            errors = batch["errors"]

            # Run inference on valid tensors
            tag_results = []
            if tensor_batch is not None:
                tag_results = tagger.inference(tensor_batch)

            # Map results back to batch positions
            tag_map = {valid_indices[i]: tag_results[i] for i in range(len(tag_results))}

            # Process each item in batch
            for i, (idx, label, image, error) in enumerate(
                zip(batch_indices, batch_labels, batch_images, errors)
            ):
                if error or image is None:
                    stats["failed"] += 1
                    continue

                try:
                    # Save image
                    image_path = output_dir / f"{idx:08d}.{image_format}"
                    if image_format == "jpg":
                        image.convert("RGB").save(image_path, "JPEG", quality=image_quality)
                    else:
                        image.save(image_path, image_format.upper())

                    # Save tags
                    tag_path = output_dir / f"{idx:08d}.tag.json"
                    tags = convert_tag_result(tag_map[i]) if i in tag_map else None

                    tag_data = {
                        "index": idx,
                        "label": label,
                        "tags": tags,
                    }

                    with open(tag_path, "w", encoding="utf-8") as f:
                        json.dump(tag_data, f, indent=2, ensure_ascii=False)

                    stats["success"] += 1

                except Exception as e:
                    logger.warning(f"Failed to save {idx}: {e}")
                    stats["failed"] += 1

            speed.tick(len(batch_indices))
            pbar.update(len(batch_indices))
            update_pbar()

    except KeyboardInterrupt:
        logger.info("\nInterrupted!")

    finally:
        pbar.close()
        tagger.unload()

    return stats


def process_multi_gpu(
    dataset,
    indices: list[int],
    output_dir: Path,
    gpu_ids: list[int],
    tagger_repo: str,
    batch_size: int,
    image_format: str,
    image_quality: int,
) -> dict[str, int]:
    """Process with multiple GPUs using data parallelism."""
    global _interrupted

    stats = {"success": 0, "failed": 0}
    speed = SpeedMonitor()
    total = len(indices)

    # Create queues
    task_queue = mp.Queue(maxsize=batch_size * len(gpu_ids) * 2)
    result_queue = mp.Queue()

    # Start workers
    workers = []
    for gpu_id in gpu_ids:
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, tagger_repo, task_queue, result_queue, batch_size),
        )
        p.start()
        workers.append(p)

    logger.info(f"Started {len(workers)} workers on GPUs: {gpu_ids}")

    pbar = tqdm(total=total, desc="Tagging", unit="img")
    submitted = 0
    completed = 0

    def update_pbar():
        remaining = total - completed
        pbar.set_postfix_str(
            f"{speed.get_speed():.1f}/s ETA:{speed.get_eta(remaining)} "
            f"| Q:{submitted - completed} | OK:{stats['success']} ERR:{stats['failed']}"
        )

    try:
        idx_iter = iter(indices)
        done_submitting = False

        while completed < total and not _interrupted:
            # Submit tasks
            while submitted < total and not done_submitting:
                try:
                    idx = next(idx_iter)
                    item = dataset[idx]
                    task = TagTask(
                        index=idx,
                        label=item["label"],
                        image=item["image"],
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

                # Save image
                try:
                    image_path = output_dir / f"{result.index:08d}.{image_format}"
                    if image_format == "jpg":
                        result.image.convert("RGB").save(image_path, "JPEG", quality=image_quality)
                    else:
                        result.image.save(image_path, image_format.upper())

                    # Save tags
                    tag_path = output_dir / f"{result.index:08d}.tag.json"
                    tag_data = {
                        "index": result.index,
                        "label": result.label,
                        "tags": result.tags,
                    }

                    with open(tag_path, "w", encoding="utf-8") as f:
                        json.dump(tag_data, f, indent=2, ensure_ascii=False)

                    stats["success"] += 1

                except Exception as e:
                    logger.warning(f"Failed to save {result.index}: {e}")
                    stats["failed"] += 1

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


@click.command()
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory for images and tags"
)
@click.option(
    "--split", "-s",
    type=click.Choice(["train", "validation", "test"]),
    default="train",
    show_default=True,
    help="Dataset split to process"
)
@click.option(
    "--gpus",
    type=str,
    default=None,
    help="Comma-separated GPU IDs for data parallelism (e.g., '0,1,2,3')"
)
@click.option(
    "--tagger-repo",
    type=str,
    default=DEFAULT_TAGGER_REPO,
    show_default=True,
    help="HuggingFace repo ID for AnimeTIMM tagger"
)
@click.option(
    "--batch-size", "-b",
    type=int,
    default=32,
    show_default=True,
    help="Batch size for tagger inference"
)
@click.option(
    "--num-workers", "-w",
    type=int,
    default=4,
    show_default=True,
    help="Number of DataLoader workers (single GPU mode)"
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
    "--image-format",
    type=click.Choice(["jpg", "png", "webp"]),
    default="jpg",
    show_default=True,
    help="Output image format"
)
@click.option(
    "--image-quality",
    type=int,
    default=95,
    show_default=True,
    help="JPEG quality (1-100)"
)
def main(
    output_dir: Path,
    split: str,
    gpus: str | None,
    tagger_repo: str,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    skip_existing: bool,
    image_format: str,
    image_quality: int,
):
    """
    Tag ImageNet-1k images and save to output folder.

    Output structure:
        <id>.jpg        - The image
        <id>.tag.json   - Tags and metadata

    \b
    Examples:
        # Single GPU
        python scripts/imagenet_tagging.py -o ./imagenet_data

        # Multi-GPU (4 GPUs)
        python scripts/imagenet_tagging.py -o ./imagenet_data --gpus 0,1,2,3

        # Validation split with limited samples
        python scripts/imagenet_tagging.py -o ./imagenet_data -s validation -n 1000
    """
    signal.signal(signal.SIGINT, _signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, _signal_handler)

    start_time = time.time()

    # Parse GPU IDs
    gpu_ids = None
    if gpus:
        gpu_ids = [int(x.strip()) for x in gpus.split(",")]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading ImageNet-1k dataset (split={split})...")
    dataset = load_dataset("ILSVRC/imagenet-1k", split=split, trust_remote_code=True)
    total_samples = len(dataset)
    logger.info(f"Dataset loaded: {total_samples} samples")

    # Determine indices
    if max_samples is not None:
        indices = list(range(min(max_samples, total_samples)))
    else:
        indices = list(range(total_samples))

    # Filter existing
    if skip_existing:
        original = len(indices)
        indices = [
            idx for idx in indices
            if not (output_dir / f"{idx:08d}.tag.json").exists()
        ]
        skipped = original - len(indices)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already processed samples")

    if not indices:
        click.echo("All samples already processed!")
        return

    # Print config
    click.echo("=" * 60)
    click.echo("ImageNet-1k Tagging")
    click.echo("=" * 60)
    click.echo(f"Split: {split} | Samples: {len(indices)}")
    click.echo(f"Tagger: {tagger_repo}")
    if gpu_ids:
        click.echo(f"Mode: Multi-GPU | GPUs: {gpu_ids}")
    else:
        click.echo(f"Mode: Single GPU | Workers: {num_workers}")
    click.echo(f"Batch size: {batch_size} | Format: {image_format}")
    click.echo("=" * 60)

    # Process
    if gpu_ids and len(gpu_ids) > 1:
        stats = process_multi_gpu(
            dataset=dataset,
            indices=indices,
            output_dir=output_dir,
            gpu_ids=gpu_ids,
            tagger_repo=tagger_repo,
            batch_size=batch_size,
            image_format=image_format,
            image_quality=image_quality,
        )
    else:
        stats = process_single_gpu(
            dataset=dataset,
            indices=indices,
            output_dir=output_dir,
            tagger_repo=tagger_repo,
            batch_size=batch_size,
            num_workers=num_workers,
            image_format=image_format,
            image_quality=image_quality,
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
