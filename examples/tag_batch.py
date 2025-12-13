#!/usr/bin/env python3
"""
Batch tag images using AnimeTIMM tagger.

Output: xxx.tag.json for each xxx.jpg/png/etc.

This is a standalone tagging pipeline - run this first, then use
caption_batch_local.py with --load-tags to use tags as context.
"""

import json
import sys
import time
from pathlib import Path

import click
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PIL import Image

DEFAULT_TAGGER_REPO = "animetimm/caformer_b36.dbv4-full"


class ImageDataset(Dataset):
    """Dataset for batch image loading."""

    def __init__(self, image_paths: list[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def _to_rgb(self, image: Image.Image) -> Image.Image:
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            return bg
        elif image.mode == "P":
            rgba = image.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            if rgba.mode == "RGBA":
                bg.paste(rgba, mask=rgba.split()[3])
            return bg
        return image.convert("RGB")

    def __getitem__(self, idx: int) -> dict:
        path = self.image_paths[idx]
        try:
            image = Image.open(path)
            image_rgb = self._to_rgb(image)

            tensor = None
            if self.transform:
                tensor = self.transform(image_rgb)

            return {"path": str(path), "tensor": tensor, "error": None}
        except Exception as e:
            return {"path": str(path), "tensor": None, "error": str(e)}


def collate_fn(batch: list[dict]) -> dict:
    paths = [item["path"] for item in batch]
    errors = [item["error"] for item in batch]

    valid_tensors = []
    valid_indices = []
    for i, item in enumerate(batch):
        if item["tensor"] is not None:
            valid_tensors.append(item["tensor"])
            valid_indices.append(i)

    tensor_batch = None
    if valid_tensors:
        tensor_batch = torch.stack(valid_tensors)

    return {
        "paths": paths,
        "tensor_batch": tensor_batch,
        "valid_indices": valid_indices,
        "errors": errors,
    }


def find_images(input_dir: Path, extensions: list[str]) -> list[Path]:
    images = []
    for ext in extensions:
        images.extend(input_dir.glob(f"*{ext}"))
        images.extend(input_dir.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def format_tags_result(result, include_scores: bool = True) -> dict:
    """Convert tagger result to serializable dict."""
    data = {
        "general": list(result.general_tags.keys()),
        "character": list(result.character_tags.keys()),
        "rating": list(result.rating_tags.keys()),
    }
    if include_scores:
        data["scores"] = {
            "general": {k: round(v, 4) for k, v in result.general_tags.items()},
            "character": {k: round(v, 4) for k, v in result.character_tags.items()},
            "rating": {k: round(v, 4) for k, v in result.rating_tags.items()},
        }
    return data


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
    "--tagger-repo",
    type=str,
    default=DEFAULT_TAGGER_REPO,
    show_default=True,
    help="HuggingFace repo ID for tagger model",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=16,
    show_default=True,
    help="Batch size for tagger inference",
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
    "--extensions",
    type=str,
    default=".jpg,.jpeg,.png,.webp",
    show_default=True,
    help="Image file extensions to process",
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=False,
    help="Skip images that already have .tag.json files",
)
@click.option(
    "--include-scores/--no-scores",
    default=True,
    help="Include confidence scores in output",
)
def main(
    input_dir: Path,
    output_dir: Path | None,
    tagger_repo: str,
    batch_size: int,
    num_workers: int,
    extensions: str,
    skip_existing: bool,
    include_scores: bool,
):
    """
    Batch tag images using AnimeTIMM tagger.

    \b
    Output format: xxx.tag.json for each image
    {
        "general": ["tag1", "tag2", ...],
        "character": ["char1", ...],
        "rating": ["general"],
        "scores": {...}  // optional
    }

    \b
    Examples:
        # Tag all images
        python tag_batch.py ./images

        # Skip already tagged
        python tag_batch.py ./images --skip-existing

        # Without scores (smaller files)
        python tag_batch.py ./images --no-scores

    \b
    Workflow:
        1. python tag_batch.py ./images
        2. python caption_batch_local.py ./images --load-tags
    """
    from kohakucaption.tagger import AnimeTimmTagger

    output_dir = output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ext_list = [e.strip() for e in extensions.split(",")]
    image_paths = find_images(input_dir, ext_list)

    if not image_paths:
        click.echo(f"No images found in {input_dir}")
        return

    if skip_existing:
        orig = len(image_paths)
        image_paths = [
            p for p in image_paths if not (output_dir / f"{p.stem}.tag.json").exists()
        ]
        skipped = orig - len(image_paths)
        if skipped > 0:
            click.echo(f"Skipped {skipped} existing .tag.json files")
        if not image_paths:
            click.echo("All done.")
            return

    click.echo("=" * 60)
    click.echo(f"Images: {len(image_paths)} | Batch: {batch_size}")
    click.echo(f"Tagger: {tagger_repo}")
    click.echo(f"Output: {{name}}.tag.json")
    click.echo("=" * 60)

    # Load tagger
    tagger = AnimeTimmTagger(repo_id=tagger_repo, use_fp16=True)
    tagger._load_model()

    # Create dataloader
    dataset = ImageDataset(image_paths, transform=tagger._transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    stats = {"success": 0, "failed": 0}
    start_time = time.time()

    try:
        with tqdm(total=len(image_paths), desc="Tagging", unit="img") as pbar:
            for batch in dataloader:
                paths = [Path(p) for p in batch["paths"]]
                tensor_batch = batch["tensor_batch"]
                valid_indices = batch["valid_indices"]
                errors = batch["errors"]

                # Run tagger inference
                tag_results: list = [None] * len(paths)
                if tensor_batch is not None:
                    results = tagger.inference(tensor_batch)
                    for i, res in zip(valid_indices, results):
                        tag_results[i] = res

                # Write outputs
                for path, result, error in zip(paths, tag_results, errors):
                    out_path = output_dir / f"{path.stem}.tag.json"

                    if error or result is None:
                        data = {"error": error or "Failed to load image"}
                        stats["failed"] += 1
                    else:
                        data = format_tags_result(result, include_scores)
                        stats["success"] += 1

                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)

                    pbar.update(1)

    finally:
        tagger.unload()

    elapsed = time.time() - start_time
    click.echo()
    click.echo("=" * 60)
    click.echo(f"Done: {stats['success']} OK, {stats['failed']} failed")
    click.echo(
        f"Time: {elapsed:.1f}s | Throughput: {len(image_paths)/elapsed:.1f} img/s"
    )
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
