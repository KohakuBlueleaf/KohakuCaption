#!/usr/bin/env python3
"""
Batch Tagging Script for HuggingFace Datasets.

Downloads images from a HuggingFace dataset and tags them using AnimeTIMM tagger.
Saves images and tags to output folder.

Output structure:
    <prefix>_<id>.jpg        - The image
    <prefix>_<id>.tag.json   - Tags and metadata

Features:
    - Supports any HuggingFace dataset with image column
    - Configurable image column and ID column names
    - Multi-GPU via torchrun (DDP-style data sharding)
    - Automatic unique ID generation for datasets without ID column

Usage:
    # ImageNet-1k (default)
    python scripts/batch_tag_hf.py -o ./imagenet_data

    # Custom dataset with specified columns
    python scripts/batch_tag_hf.py -o ./output --dataset "username/my-dataset" \\
        --image-col "image" --id-col "file_name"

    # Dataset without ID column (uses index)
    python scripts/batch_tag_hf.py -o ./output --dataset "username/dataset" --image-col "img"

    # Multi-GPU (4 GPUs) via torchrun
    torchrun --nproc_per_node=4 scripts/batch_tag_hf.py -o ./output

    # Specific GPUs
    CUDA_VISIBLE_DEVICES=0,3,4,5 torchrun --nproc_per_node=4 scripts/batch_tag_hf.py -o ./output
"""

import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import click
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

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

# Split prefixes for unique IDs across all splits
SPLIT_PREFIXES = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


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


def make_image_id(split: str, index: int, id_value: str | None = None) -> str:
    """Create unique image ID.

    If id_value is provided, sanitize and use it directly.
    Otherwise, create ID with split prefix and zero-padded index.
    """
    if id_value is not None:
        # Sanitize: replace path separators and spaces with underscores
        sanitized = str(id_value).replace("/", "_").replace("\\", "_").replace(" ", "_")
        # Remove file extension if present
        if "." in sanitized:
            sanitized = sanitized.rsplit(".", 1)[0]
        return sanitized
    prefix = SPLIT_PREFIXES.get(split, split)
    return f"{prefix}_{index:08d}"


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


class HFTaggingDataset(Dataset):
    """Dataset for HuggingFace dataset tagging that returns preprocessed tensors."""

    def __init__(
        self,
        hf_dataset,
        indices: list[int],
        transform,
        split: str,
        image_col: str = "image",
        id_col: str | None = None,
    ):
        self.hf_dataset = hf_dataset
        self.indices = indices
        self.transform = transform
        self.split = split
        self.image_col = image_col
        self.id_col = id_col

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        try:
            item = self.hf_dataset[actual_idx]
            # Get ID from column or generate from index
            id_value = item.get(self.id_col) if self.id_col else None
            image_id = make_image_id(self.split, actual_idx, id_value)
            # Get image from configured column
            image = item[self.image_col]
            # Get label if available (for ImageNet compatibility)
            label = item.get("label", -1)
            image_rgb = pil_to_rgb(image)
            tensor = self.transform(image_rgb)
            return {
                "image_id": image_id,
                "index": actual_idx,
                "label": label,
                "image": image,  # Original for saving
                "tensor": tensor,
                "error": None,
            }
        except Exception as e:
            # Generate ID even on error for tracking
            id_value = None
            try:
                item = self.hf_dataset[actual_idx]
                id_value = item.get(self.id_col) if self.id_col else None
            except Exception:
                pass
            image_id = make_image_id(self.split, actual_idx, id_value)
            return {
                "image_id": image_id,
                "index": actual_idx,
                "label": -1,
                "image": None,
                "tensor": None,
                "error": str(e),
            }


def collate_fn(batch):
    """Custom collate function to handle mixed valid/invalid items."""
    image_ids = [item["image_id"] for item in batch]
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
        "image_ids": image_ids,
        "indices": indices,
        "labels": labels,
        "images": images,
        "tensor_batch": tensor_batch,
        "valid_indices": valid_indices,
        "errors": errors,
    }


def setup_distributed():
    """Setup distributed training if launched with torchrun."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    """Cleanup distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process."""
    return rank == 0


@click.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory for images and tags",
)
@click.option(
    "--dataset",
    "-d",
    type=str,
    default="ILSVRC/imagenet-1k",
    show_default=True,
    help="HuggingFace dataset repo ID",
)
@click.option(
    "--split",
    "-s",
    type=str,
    default="train",
    show_default=True,
    help="Dataset split to process (e.g., train, validation, test)",
)
@click.option(
    "--image-col",
    type=str,
    default="image",
    show_default=True,
    help="Name of the image column in the dataset",
)
@click.option(
    "--id-col",
    type=str,
    default=None,
    help="Name of the ID column (if not set, uses index with split prefix)",
)
@click.option(
    "--tagger-repo",
    type=str,
    default=DEFAULT_TAGGER_REPO,
    show_default=True,
    help="HuggingFace repo ID for AnimeTIMM tagger",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=64,
    show_default=True,
    help="Batch size per GPU for tagger inference",
)
@click.option(
    "--num-workers",
    "-w",
    type=int,
    default=4,
    show_default=True,
    help="Number of DataLoader workers per GPU",
)
@click.option(
    "--max-samples",
    "-n",
    type=int,
    default=None,
    help="Maximum number of samples to process (total across all GPUs)",
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    show_default=True,
    help="Skip already processed samples",
)
@click.option(
    "--image-format",
    type=click.Choice(["jpg", "png", "webp"]),
    default="jpg",
    show_default=True,
    help="Output image format",
)
@click.option(
    "--image-quality",
    type=int,
    default=95,
    show_default=True,
    help="JPEG quality (1-100)",
)
def main(
    output_dir: Path,
    dataset: str,
    split: str,
    image_col: str,
    id_col: str | None,
    tagger_repo: str,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    skip_existing: bool,
    image_format: str,
    image_quality: int,
):
    """
    Tag images from a HuggingFace dataset and save to output folder.

    Output structure:
        <prefix>_<id>.jpg        - The image (e.g., train_00000001.jpg)
        <prefix>_<id>.tag.json   - Tags and metadata

    If --id-col is not specified, generates IDs using: <split>_<index:08d>

    \b
    Examples:
        # ImageNet-1k (default) - single GPU
        python scripts/batch_tag_hf.py -o ./imagenet_data

        # Custom dataset with specified columns
        python scripts/batch_tag_hf.py -o ./output -d "username/my-dataset" \\
            --image-col "image" --id-col "file_name"

        # Multi-GPU (4 GPUs) via torchrun
        torchrun --nproc_per_node=4 scripts/batch_tag_hf.py -o ./imagenet_data

        # Specific GPUs
        CUDA_VISIBLE_DEVICES=0,3,4,5 torchrun --nproc_per_node=4 scripts/batch_tag_hf.py -o ./output
    """
    global _interrupted

    signal.signal(signal.SIGINT, _signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, _signal_handler)

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()

    try:
        _run_tagging(
            output_dir=output_dir,
            dataset_repo=dataset,
            split=split,
            image_col=image_col,
            id_col=id_col,
            tagger_repo=tagger_repo,
            batch_size=batch_size,
            num_workers=num_workers,
            max_samples=max_samples,
            skip_existing=skip_existing,
            image_format=image_format,
            image_quality=image_quality,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
        )
    finally:
        cleanup_distributed()


def _run_tagging(
    output_dir: Path,
    dataset_repo: str,
    split: str,
    image_col: str,
    id_col: str | None,
    tagger_repo: str,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    skip_existing: bool,
    image_format: str,
    image_quality: int,
    rank: int,
    local_rank: int,
    world_size: int,
):
    """Main tagging logic."""
    global _interrupted

    from kohakucaption.tagger import AnimeTimmTagger

    start_time = time.time()

    # Create output directory (only main process)
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)

    # Sync before proceeding
    if world_size > 1:
        dist.barrier()

    # Load dataset
    if is_main_process(rank):
        logger.info(f"Loading dataset: {dataset_repo} (split={split})...")
    hf_dataset = load_dataset(dataset_repo, split=split, trust_remote_code=True)
    total_samples = len(hf_dataset)
    if is_main_process(rank):
        logger.info(f"Dataset loaded: {total_samples} samples")

    # Determine indices
    if max_samples is not None:
        indices = list(range(min(max_samples, total_samples)))
    else:
        indices = list(range(total_samples))

    # Filter existing
    # Note: For datasets with custom ID columns, we need to check each item
    # For index-based IDs, we can check directly
    if skip_existing:
        original = len(indices)
        if id_col is None:
            # Index-based IDs - can check directly
            indices = [
                idx
                for idx in indices
                if not (output_dir / f"{make_image_id(split, idx)}.tag.json").exists()
            ]
        else:
            # Custom ID column - need to check each item's ID
            filtered_indices = []
            for idx in indices:
                try:
                    item = hf_dataset[idx]
                    id_value = item.get(id_col)
                    image_id = make_image_id(split, idx, id_value)
                    if not (output_dir / f"{image_id}.tag.json").exists():
                        filtered_indices.append(idx)
                except Exception:
                    filtered_indices.append(idx)  # Include if can't check
            indices = filtered_indices
        skipped = original - len(indices)
        if skipped > 0 and is_main_process(rank):
            logger.info(f"Skipping {skipped} already processed samples")

    if not indices:
        if is_main_process(rank):
            click.echo("All samples already processed!")
        return

    # Load tagger on correct GPU
    if is_main_process(rank):
        logger.info(f"Loading tagger: {tagger_repo}")
    tagger = AnimeTimmTagger(repo_id=tagger_repo, use_fp16=True)
    tagger._load_model()
    if is_main_process(rank):
        logger.info(f"Tagger loaded on {tagger.device}")

    # Create dataset and dataloader
    torch_dataset = HFTaggingDataset(
        hf_dataset, indices, tagger._transform, split, image_col, id_col
    )

    if world_size > 1:
        sampler = DistributedSampler(
            torch_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    else:
        sampler = None

    dataloader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Print config (only main process)
    if is_main_process(rank):
        click.echo("=" * 60)
        click.echo("HuggingFace Dataset Tagging")
        click.echo("=" * 60)
        click.echo(f"Dataset: {dataset_repo}")
        click.echo(f"Split: {split} | Total samples: {len(indices)}")
        click.echo(f"Image col: {image_col} | ID col: {id_col or '(auto-generated)'}")
        click.echo(f"GPUs: {world_size} | Batch size/GPU: {batch_size}")
        click.echo(f"Workers/GPU: {num_workers} | Tagger: {tagger_repo}")
        click.echo(f"Format: {image_format}")
        click.echo("=" * 60)

    # Process
    stats = {"success": 0, "failed": 0}
    speed = SpeedMonitor()
    local_total = len(torch_dataset) // world_size + (
        1 if rank < len(torch_dataset) % world_size else 0
    )

    pbar = tqdm(
        total=local_total,
        desc=f"[GPU {rank}] Tagging",
        unit="img",
        disable=not is_main_process(rank),  # Only show on main process
    )

    def update_pbar():
        remaining = local_total - (stats["success"] + stats["failed"])
        pbar.set_postfix_str(
            f"{speed.get_speed():.1f}/s ETA:{speed.get_eta(remaining)} "
            f"| OK:{stats['success']} ERR:{stats['failed']}"
        )

    try:
        for batch in dataloader:
            if _interrupted:
                break

            batch_image_ids = batch["image_ids"]
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
            tag_map = {
                valid_indices[i]: tag_results[i] for i in range(len(tag_results))
            }

            # Process each item in batch
            for i, (image_id, idx, label, image, error) in enumerate(
                zip(batch_image_ids, batch_indices, batch_labels, batch_images, errors)
            ):
                if error or image is None:
                    stats["failed"] += 1
                    continue

                try:
                    # Save image
                    image_path = output_dir / f"{image_id}.{image_format}"
                    if image_format == "jpg":
                        image.convert("RGB").save(
                            image_path, "JPEG", quality=image_quality
                        )
                    else:
                        image.save(image_path, image_format.upper())

                    # Save tags
                    tag_path = output_dir / f"{image_id}.tag.json"
                    tags = convert_tag_result(tag_map[i]) if i in tag_map else None

                    tag_data = {
                        "image_id": image_id,
                        "index": idx,
                        "label": label,
                        "tags": tags,
                    }

                    with open(tag_path, "w", encoding="utf-8") as f:
                        json.dump(tag_data, f, indent=2, ensure_ascii=False)

                    stats["success"] += 1

                except Exception as e:
                    logger.warning(f"[GPU {rank}] Failed to save {image_id}: {e}")
                    stats["failed"] += 1

            speed.tick(len(batch_indices))
            pbar.update(len(batch_indices))
            update_pbar()

    except KeyboardInterrupt:
        logger.info(f"\n[GPU {rank}] Interrupted!")

    finally:
        pbar.close()
        tagger.unload()

    # Gather stats from all processes
    if world_size > 1:
        # Convert to tensors for all_reduce
        success_tensor = torch.tensor([stats["success"]], device=f"cuda:{local_rank}")
        failed_tensor = torch.tensor([stats["failed"]], device=f"cuda:{local_rank}")
        dist.all_reduce(success_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(failed_tensor, op=dist.ReduceOp.SUM)
        total_success = success_tensor.item()
        total_failed = failed_tensor.item()
    else:
        total_success = stats["success"]
        total_failed = stats["failed"]

    elapsed = time.time() - start_time
    total_processed = total_success + total_failed

    if is_main_process(rank):
        click.echo()
        click.echo("=" * 60)
        click.echo(f"Completed in {elapsed:.1f}s")
        click.echo(f"Success: {total_success} | Failed: {total_failed}")
        if total_processed > 0:
            click.echo(f"Speed: {total_processed / elapsed:.2f} img/s")
        click.echo("=" * 60)


if __name__ == "__main__":
    main()
