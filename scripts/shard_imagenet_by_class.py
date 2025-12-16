#!/usr/bin/env python3
"""
Shard ImageNet data by class ID.

Reads the ImageNet dataset to get image_id <-> class_id mapping,
then moves files from imagenet_data/*.{jpg,tag.json,caption.json}
to imagenet_data/<class_id>/*.

Usage:
    # Dry run (show what would be moved)
    python scripts/shard_imagenet_by_class.py ./imagenet_data --dry-run

    # Actually move files
    python scripts/shard_imagenet_by_class.py ./imagenet_data

    # Use symlinks instead of moving
    python scripts/shard_imagenet_by_class.py ./imagenet_data --symlink
"""

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import click
from datasets import load_dataset
from tqdm import tqdm


# Split prefixes used in our tagging script
SPLIT_PREFIXES = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


def build_id_to_class_mapping(splits: list[str] = None) -> dict[str, int]:
    """
    Build mapping from image_id to class_id by loading ImageNet dataset.

    Returns dict like {"train_00000001": 123, "val_00000001": 456, ...}
    """
    if splits is None:
        splits = ["train", "validation", "test"]

    mapping = {}

    for split in splits:
        click.echo(f"Loading ImageNet {split} split (labels only)...")
        try:
            # Load only the label column, skip images for speed
            dataset = load_dataset(
                "ILSVRC/imagenet-1k",
                split=split,
                trust_remote_code=True,
            ).select_columns(["label"])
        except Exception as e:
            click.echo(f"  Skipping {split}: {e}")
            continue

        prefix = SPLIT_PREFIXES.get(split, split)
        click.echo(f"  Building mapping for {len(dataset)} images...")

        # Fast iteration - labels are already in memory
        labels = dataset["label"]
        for idx, label in enumerate(tqdm(labels, desc=f"  {split}", unit="img")):
            image_id = f"{prefix}_{idx:08d}"
            mapping[image_id] = label

    return mapping


def find_files_to_move(data_dir: Path) -> dict[str, list[Path]]:
    """
    Find all files in data_dir that match our naming convention.

    Returns dict mapping image_id to list of associated files.
    e.g. {"train_00000001": [train_00000001.jpg, train_00000001.tag.json, ...]}
    """
    files_by_id = defaultdict(list)

    click.echo(f"Scanning {data_dir} for files...")

    with os.scandir(data_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue

            name = entry.name

            # Extract image_id from filename
            if name.endswith(".tag.json"):
                image_id = name[:-9]
            elif name.endswith(".caption.json"):
                image_id = name[:-13]
            elif name.endswith(".jpg"):
                image_id = name[:-4]
            elif name.endswith(".jpeg"):
                image_id = name[:-5]
            elif name.endswith(".png"):
                image_id = name[:-4]
            elif name.endswith(".webp"):
                image_id = name[:-5]
            else:
                continue

            files_by_id[image_id].append(Path(entry.path))

    return dict(files_by_id)


@click.command()
@click.argument(
    "data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--dry-run", is_flag=True, help="Show what would be moved without actually moving"
)
@click.option("--symlink", is_flag=True, help="Create symlinks instead of moving files")
@click.option("--copy", is_flag=True, help="Copy files instead of moving")
def main(data_dir: Path, dry_run: bool, symlink: bool, copy: bool):
    """
    Shard ImageNet data by class ID.

    Moves files from:
        imagenet_data/train_00000001.jpg
        imagenet_data/train_00000001.tag.json

    To:
        imagenet_data/123/train_00000001.jpg
        imagenet_data/123/train_00000001.tag.json

    Where 123 is the class ID from ImageNet.
    """
    if symlink and copy:
        raise click.UsageError("Cannot use both --symlink and --copy")

    # Build mapping
    id_to_class = build_id_to_class_mapping()
    click.echo(f"Built mapping for {len(id_to_class)} images")

    # Find files
    files_by_id = find_files_to_move(data_dir)
    click.echo(f"Found {len(files_by_id)} image IDs with files")

    # Count files per class
    class_counts = defaultdict(int)
    missing_ids = []

    for image_id in files_by_id:
        if image_id in id_to_class:
            class_counts[id_to_class[image_id]] += 1
        else:
            missing_ids.append(image_id)

    if missing_ids:
        click.echo(
            f"Warning: {len(missing_ids)} image IDs not found in ImageNet dataset"
        )
        if len(missing_ids) <= 10:
            for mid in missing_ids:
                click.echo(f"  - {mid}")

    click.echo(f"Will create {len(class_counts)} class directories")

    if dry_run:
        click.echo("\n[DRY RUN] Would move files as follows:")
        sample_count = 0
        for image_id, files in files_by_id.items():
            if image_id not in id_to_class:
                continue
            class_id = id_to_class[image_id]
            for f in files:
                click.echo(f"  {f.name} -> {class_id}/{f.name}")
                sample_count += 1
                if sample_count >= 20:
                    click.echo(
                        f"  ... and {sum(len(f) for f in files_by_id.values()) - 20} more files"
                    )
                    return
        return

    # Actually move/copy/symlink files
    action = "Symlinking" if symlink else ("Copying" if copy else "Moving")
    click.echo(f"\n{action} files...")

    moved = 0
    errors = 0

    for image_id, files in tqdm(files_by_id.items(), desc=action, unit="img"):
        if image_id not in id_to_class:
            continue

        class_id = id_to_class[image_id]
        class_dir = data_dir / str(class_id)
        class_dir.mkdir(exist_ok=True)

        for src_path in files:
            dst_path = class_dir / src_path.name

            try:
                if symlink:
                    if dst_path.exists():
                        dst_path.unlink()
                    dst_path.symlink_to(src_path.resolve())
                elif copy:
                    shutil.copy2(src_path, dst_path)
                else:
                    shutil.move(src_path, dst_path)
                moved += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    click.echo(f"Error {action.lower()} {src_path}: {e}")

    click.echo(f"\nDone! {action} {moved} files, {errors} errors")
    click.echo(f"Created {len(class_counts)} class directories")


if __name__ == "__main__":
    main()
