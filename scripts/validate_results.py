#!/usr/bin/env python3
"""
Validation Script for Tagging/Captioning Results.

Scans output directories for images and checks corresponding tag/caption files.
Uses os.scandir for efficient traversal of large directories (1M+ files).

Usage:
    python scripts/validate_results.py ./imagenet_data
    python scripts/validate_results.py ./imagenet_data --quick 10000
    python scripts/validate_results.py ./imagenet_data --min-tags 3 --min-caption-len 200
"""

import json
import os
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def scan_images_and_check(
    directory: Path,
    validate_type: str,
    quick_limit: int | None = None,
    recursive: bool = True,
    pbar: tqdm | None = None,
    min_tags: int = 1,
    min_caption_len: int = 100,
    max_caption_len: int = 2000,
) -> dict:
    """Scan directory for images and check corresponding tag/caption files."""
    # Counters
    images_found = 0
    images_with_tags = 0
    images_without_tags = 0
    images_with_captions = 0
    images_without_captions = 0

    # Tag stats
    tags_analyzed = 0
    tags_success = 0
    tags_error = 0
    tags_too_few = 0
    tag_counts = []

    # Caption stats
    captions_analyzed = 0
    captions_success = 0
    captions_parse_error = 0
    captions_error = 0
    captions_too_short = 0
    captions_too_long = 0
    caption_lengths = []  # description length
    brief_lengths = []
    aesthetic_scores = []
    nsfw_scores = []
    quality_scores = []

    # Error file lists (just filenames)
    tag_error_files = []
    tag_too_few_files = []
    caption_error_files = []
    caption_parse_error_files = []
    caption_too_short_files = []
    caption_too_long_files = []

    check_tags = validate_type in ("tags", "all")
    check_captions = validate_type in ("captions", "all")

    def analyze_tag_file(filepath: str):
        nonlocal tags_analyzed, tags_success, tags_error, tags_too_few
        tags_analyzed += 1
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "error" in data:
                tags_error += 1
                tag_error_files.append(filepath)
            else:
                tags = data.get("tags", {})
                general = tags.get("general_tags", [])
                character = tags.get("character_tags", [])
                count = len(general) + len(character)
                tag_counts.append(count)
                if count < min_tags:
                    tags_too_few += 1
                    tag_too_few_files.append(filepath)
                else:
                    tags_success += 1
        except Exception:
            tags_error += 1
            tag_error_files.append(filepath)

    def analyze_caption_file(filepath: str):
        nonlocal captions_analyzed, captions_success, captions_parse_error, captions_error
        nonlocal captions_too_short, captions_too_long
        captions_analyzed += 1
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "error" in data:
                captions_error += 1
                caption_error_files.append(filepath)
            elif "parse_error" in data:
                captions_parse_error += 1
                caption_parse_error_files.append(filepath)
            else:
                caption = data.get("caption", {})
                desc = caption.get("description", "")
                brief = caption.get("brief", "")
                desc_len = len(desc)
                caption_lengths.append(desc_len)
                brief_lengths.append(len(brief))
                # Collect scores
                if "aesthetic_score" in caption:
                    aesthetic_scores.append(caption["aesthetic_score"])
                if "nsfw_score" in caption:
                    nsfw_scores.append(caption["nsfw_score"])
                if "quality_score" in caption:
                    quality_scores.append(caption["quality_score"])
                # Check length limits
                if desc_len < min_caption_len:
                    captions_too_short += 1
                    caption_too_short_files.append(filepath)
                elif desc_len > max_caption_len:
                    captions_too_long += 1
                    caption_too_long_files.append(filepath)
                else:
                    captions_success += 1
        except Exception:
            captions_error += 1
            caption_error_files.append(filepath)

    def process_image(dirpath: str, filename: str):
        nonlocal images_found, images_with_tags, images_without_tags
        nonlocal images_with_captions, images_without_captions

        images_found += 1
        stem = os.path.splitext(filename)[0]

        if check_tags:
            tag_path = os.path.join(dirpath, f"{stem}.tag.json")
            if os.path.exists(tag_path):
                images_with_tags += 1
                analyze_tag_file(tag_path)
            else:
                images_without_tags += 1

        if check_captions:
            caption_path = os.path.join(dirpath, f"{stem}.caption.json")
            cap_path = os.path.join(dirpath, f"{stem}.cap.json")
            if os.path.exists(caption_path):
                images_with_captions += 1
                analyze_caption_file(caption_path)
            elif os.path.exists(cap_path):
                images_with_captions += 1
                analyze_caption_file(cap_path)
            else:
                images_without_captions += 1

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str(f"tags:{images_with_tags} caps:{images_with_captions}")

    def scan_dir(path: str) -> bool:
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    if quick_limit and images_found >= quick_limit:
                        return True
                    if entry.is_file():
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in IMAGE_EXTENSIONS:
                            process_image(path, entry.name)
                    elif entry.is_dir() and recursive:
                        if scan_dir(entry.path):
                            return True
        except PermissionError:
            pass
        return False

    scan_dir(str(directory))

    return {
        "images_found": images_found,
        "images_with_tags": images_with_tags,
        "images_without_tags": images_without_tags,
        "images_with_captions": images_with_captions,
        "images_without_captions": images_without_captions,
        # Tag stats
        "tags_analyzed": tags_analyzed,
        "tags_success": tags_success,
        "tags_error": tags_error,
        "tags_too_few": tags_too_few,
        "tag_counts": np.array(tag_counts) if tag_counts else np.array([]),
        "tag_error_files": tag_error_files,
        "tag_too_few_files": tag_too_few_files,
        # Caption stats
        "captions_analyzed": captions_analyzed,
        "captions_success": captions_success,
        "captions_parse_error": captions_parse_error,
        "captions_error": captions_error,
        "captions_too_short": captions_too_short,
        "captions_too_long": captions_too_long,
        "caption_lengths": (
            np.array(caption_lengths) if caption_lengths else np.array([])
        ),
        "brief_lengths": np.array(brief_lengths) if brief_lengths else np.array([]),
        "aesthetic_scores": (
            np.array(aesthetic_scores) if aesthetic_scores else np.array([])
        ),
        "nsfw_scores": np.array(nsfw_scores) if nsfw_scores else np.array([]),
        "quality_scores": np.array(quality_scores) if quality_scores else np.array([]),
        "caption_error_files": caption_error_files,
        "caption_parse_error_files": caption_parse_error_files,
        "caption_too_short_files": caption_too_short_files,
        "caption_too_long_files": caption_too_long_files,
    }


@click.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--type",
    "-t",
    "validate_type",
    type=click.Choice(["tags", "captions", "all"]),
    default="all",
)
@click.option("--quick", "-q", type=int, default=None, help="Stop after N images")
@click.option("--recursive/--no-recursive", default=True)
@click.option(
    "--min-tags", type=int, default=1, show_default=True, help="Min tags to be valid"
)
@click.option(
    "--min-caption-len",
    type=int,
    default=100,
    show_default=True,
    help="Min description length",
)
@click.option(
    "--max-caption-len",
    type=int,
    default=2000,
    show_default=True,
    help="Max description length",
)
def main(
    directory: Path,
    validate_type: str,
    quick: int | None,
    recursive: bool,
    min_tags: int,
    min_caption_len: int,
    max_caption_len: int,
):
    """Validate tagging/captioning results."""
    click.echo("=" * 70)
    click.echo(f"Validating: {directory}")
    click.echo(f"Type: {validate_type} | Quick: {quick or 'full'}")
    click.echo(
        f"Min tags: {min_tags} | Caption len: {min_caption_len}-{max_caption_len}"
    )
    click.echo("=" * 70)

    pbar = tqdm(total=quick, desc="Scanning", unit="img", dynamic_ncols=True)

    try:
        stats = scan_images_and_check(
            directory,
            validate_type=validate_type,
            quick_limit=quick,
            recursive=recursive,
            pbar=pbar,
            min_tags=min_tags,
            min_caption_len=min_caption_len,
            max_caption_len=max_caption_len,
        )
    finally:
        pbar.close()

    click.echo(f"\nImages found: {stats['images_found']:,}")

    # === TAGS ===
    if validate_type in ("tags", "all"):
        click.echo(f"\n{'='*30} TAGS {'='*30}")
        click.echo(f"With tags:    {stats['images_with_tags']:,}")
        click.echo(f"Without tags: {stats['images_without_tags']:,}")

        if stats["tags_analyzed"] > 0:
            click.echo(f"\nAnalysis ({stats['tags_analyzed']:,} files):")
            click.echo(f"  Success:  {stats['tags_success']:,}")
            click.echo(f"  Too few:  {stats['tags_too_few']:,} (<{min_tags} tags)")
            click.echo(f"  Errors:   {stats['tags_error']:,}")

            if len(stats["tag_counts"]) > 0:
                tc = stats["tag_counts"]
                click.echo(
                    f"\nTag counts: min={tc.min()} max={tc.max()} mean={tc.mean():.1f} median={np.median(tc):.1f}"
                )

            # Print ALL error files
            if stats["tag_error_files"]:
                click.echo(f"\nTag error files ({len(stats['tag_error_files'])}):")
                for f in stats["tag_error_files"]:
                    click.echo(f)

            if stats["tag_too_few_files"]:
                click.echo(f"\nToo few tags files ({len(stats['tag_too_few_files'])}):")
                for f in stats["tag_too_few_files"]:
                    click.echo(f)

    # === CAPTIONS ===
    if validate_type in ("captions", "all"):
        click.echo(f"\n{'='*28} CAPTIONS {'='*28}")
        click.echo(f"With captions:    {stats['images_with_captions']:,}")
        click.echo(f"Without captions: {stats['images_without_captions']:,}")

        if stats["captions_analyzed"] > 0:
            click.echo(f"\nAnalysis ({stats['captions_analyzed']:,} files):")
            click.echo(f"  Success:      {stats['captions_success']:,}")
            click.echo(
                f"  Too short:    {stats['captions_too_short']:,} (<{min_caption_len} chars)"
            )
            click.echo(
                f"  Too long:     {stats['captions_too_long']:,} (>{max_caption_len} chars)"
            )
            click.echo(f"  Parse errors: {stats['captions_parse_error']:,}")
            click.echo(f"  Other errors: {stats['captions_error']:,}")

            if len(stats["caption_lengths"]) > 0:
                cl = stats["caption_lengths"]
                click.echo(
                    f"\nDescription lengths: min={cl.min()} max={cl.max()} mean={cl.mean():.1f} median={np.median(cl):.1f}"
                )

            if len(stats["brief_lengths"]) > 0:
                bl = stats["brief_lengths"]
                click.echo(
                    f"Brief lengths: min={bl.min()} max={bl.max()} mean={bl.mean():.1f} median={np.median(bl):.1f}"
                )

            if len(stats["aesthetic_scores"]) > 0:
                aes = stats["aesthetic_scores"]
                click.echo(
                    f"\nAesthetic scores: min={aes.min():.2f} max={aes.max():.2f} mean={aes.mean():.2f} median={np.median(aes):.2f}"
                )

            if len(stats["nsfw_scores"]) > 0:
                nsfw = stats["nsfw_scores"]
                click.echo(
                    f"NSFW scores: min={nsfw.min():.2f} max={nsfw.max():.2f} mean={nsfw.mean():.2f} median={np.median(nsfw):.2f}"
                )

            if len(stats["quality_scores"]) > 0:
                qual = stats["quality_scores"]
                click.echo(
                    f"Quality scores: min={qual.min():.2f} max={qual.max():.2f} mean={qual.mean():.2f} median={np.median(qual):.2f}"
                )

            # Print ALL error files
            if stats["caption_parse_error_files"]:
                click.echo(
                    f"\nParse error files ({len(stats['caption_parse_error_files'])}):"
                )
                for f in stats["caption_parse_error_files"]:
                    click.echo(f)

            if stats["caption_error_files"]:
                click.echo(
                    f"\nCaption error files ({len(stats['caption_error_files'])}):"
                )
                for f in stats["caption_error_files"]:
                    click.echo(f)

            if stats["caption_too_short_files"]:
                click.echo(
                    f"\nToo short files ({len(stats['caption_too_short_files'])}):"
                )
                for f in stats["caption_too_short_files"]:
                    click.echo(f)

            if stats["caption_too_long_files"]:
                click.echo(
                    f"\nToo long files ({len(stats['caption_too_long_files'])}):"
                )
                for f in stats["caption_too_long_files"]:
                    click.echo(f)

    click.echo("\n" + "=" * 70)


if __name__ == "__main__":
    main()
