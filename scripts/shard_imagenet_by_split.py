#!/usr/bin/env python3
"""
Organize ImageNet data by split (train/val/test).

Moves class directories from:
    imagenet_data/<class_id>/train_*.jpg
    imagenet_data/<class_id>/val_*.jpg
    imagenet_data/-1/test_*.jpg

To:
    imagenet_data/train/<class_id>/train_*.jpg
    imagenet_data/val/<class_id>/val_*.jpg
    imagenet_data/test/test_*.jpg  (no class subdirs)

Usage:
    # Dry run
    python scripts/shard_imagenet_by_split.py ./imagenet_data --dry-run

    # Actually move
    python scripts/shard_imagenet_by_split.py ./imagenet_data
"""

import os
import shutil
from pathlib import Path

import click
from tqdm import tqdm


def get_split_from_filename(filename: str) -> str:
    """Extract split from filename prefix."""
    if filename.startswith("train_"):
        return "train"
    elif filename.startswith("val_"):
        return "val"
    elif filename.startswith("test_"):
        return "test"
    return "unknown"


@click.command()
@click.argument(
    "data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--dry-run", is_flag=True, help="Show what would be moved without actually moving"
)
def main(data_dir: Path, dry_run: bool):
    """
    Organize class-sharded ImageNet data by split.

    Expects input structure:
        imagenet_data/<class_id>/*.jpg

    Output structure:
        imagenet_data/train/<class_id>/*.jpg
        imagenet_data/val/<class_id>/*.jpg
        imagenet_data/test/*.jpg (flat, no class subdirs)
    """
    # Find all class directories (numeric names)
    class_dirs = []
    for entry in os.scandir(data_dir):
        if entry.is_dir():
            # Check if directory name is numeric (class ID) or -1
            try:
                int(entry.name)
                class_dirs.append(Path(entry.path))
            except ValueError:
                pass

    click.echo(f"Found {len(class_dirs)} class directories")

    # Count files by split
    split_counts = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    files_to_move = []  # (src_path, dst_path)

    click.echo("Scanning files...")
    for class_dir in tqdm(class_dirs, desc="Scanning", unit="class"):
        class_id = class_dir.name

        for entry in os.scandir(class_dir):
            if not entry.is_file():
                continue

            split = get_split_from_filename(entry.name)
            split_counts[split] += 1

            if split == "test" or class_id == "-1":
                # Test files go flat under test/
                dst = data_dir / "test" / entry.name
            elif split in ("train", "val"):
                dst = data_dir / split / class_id / entry.name
            else:
                continue  # Skip unknown

            files_to_move.append((Path(entry.path), dst))

    click.echo(f"\nFiles by split:")
    for split, count in split_counts.items():
        if count > 0:
            click.echo(f"  {split}: {count:,}")

    click.echo(f"\nTotal files to move: {len(files_to_move):,}")

    if dry_run:
        click.echo("\n[DRY RUN] Would move files as follows:")
        for src, dst in files_to_move[:20]:
            rel_src = src.relative_to(data_dir)
            rel_dst = dst.relative_to(data_dir)
            click.echo(f"  {rel_src} -> {rel_dst}")
        if len(files_to_move) > 20:
            click.echo(f"  ... and {len(files_to_move) - 20} more files")
        return

    # Move files
    click.echo("\nMoving files...")
    moved = 0
    errors = 0

    for src, dst in tqdm(files_to_move, desc="Moving", unit="file"):
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dst)
            moved += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                click.echo(f"Error moving {src}: {e}")

    # Clean up empty class directories
    click.echo("\nCleaning up empty directories...")
    removed_dirs = 0
    for class_dir in class_dirs:
        try:
            if class_dir.exists() and not any(class_dir.iterdir()):
                class_dir.rmdir()
                removed_dirs += 1
        except Exception:
            pass

    click.echo(f"\nDone! Moved {moved:,} files, {errors} errors")
    click.echo(f"Removed {removed_dirs} empty directories")


if __name__ == "__main__":
    main()
