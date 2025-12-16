#!/usr/bin/env python3
"""
Fix out-of-range scores in caption files.

Scores should be in [0.0, 1.0] range, but model sometimes outputs 1-10 scale.
This script detects and fixes these issues.

Usage:
    # Dry run - show what would be fixed
    python scripts/fix_scores.py ./imagenet_data --dry-run

    # Actually fix files
    python scripts/fix_scores.py ./imagenet_data
"""

import json
import os
from pathlib import Path

import click
from tqdm import tqdm

SCORE_FIELDS = ["aesthetic_score", "nsfw_score", "quality_score"]


def scan_and_fix_scores(
    directory: Path,
    dry_run: bool = True,
    recursive: bool = True,
) -> dict:
    """Scan caption files and fix out-of-range scores."""
    stats = {
        "total_files": 0,
        "files_with_issues": 0,
        "files_fixed": 0,
        "scores_below_0": 0,
        "scores_above_1": 0,
        "scores_above_10": 0,  # These can't be fixed by /10
        "issues_by_field": {
            f: {"below_0": 0, "above_1": 0, "above_10": 0} for f in SCORE_FIELDS
        },
        "problem_files": [],
    }

    # Find all caption files
    if recursive:
        caption_files = list(directory.rglob("*.caption.json"))
    else:
        caption_files = list(directory.glob("*.caption.json"))

    for caption_path in tqdm(caption_files, desc="Scanning", unit="file"):
        stats["total_files"] += 1

        try:
            with open(caption_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        caption = data.get("caption", {})
        if not caption:
            continue

        file_has_issue = False
        file_modified = False

        for field in SCORE_FIELDS:
            if field not in caption:
                continue

            value = caption[field]
            if not isinstance(value, (int, float)):
                continue

            if value < 0.0:
                stats["scores_below_0"] += 1
                stats["issues_by_field"][field]["below_0"] += 1
                file_has_issue = True
                # Can't auto-fix negative scores
                stats["problem_files"].append(
                    (str(caption_path), field, value, "below_0")
                )

            elif value > 10.0:
                stats["scores_above_10"] += 1
                stats["issues_by_field"][field]["above_10"] += 1
                file_has_issue = True
                # Can't auto-fix scores > 10
                stats["problem_files"].append(
                    (str(caption_path), field, value, "above_10")
                )

            elif value > 1.0:
                stats["scores_above_1"] += 1
                stats["issues_by_field"][field]["above_1"] += 1
                file_has_issue = True

                if not dry_run:
                    # Fix by dividing by 10
                    caption[field] = value / 10.0
                    file_modified = True

        if file_has_issue:
            stats["files_with_issues"] += 1

        if file_modified:
            stats["files_fixed"] += 1
            with open(caption_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    return stats


@click.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--dry-run", is_flag=True, help="Show what would be fixed without modifying files"
)
@click.option(
    "--recursive/--no-recursive", default=True, help="Recursively scan subdirectories"
)
def main(directory: Path, dry_run: bool, recursive: bool):
    """Fix out-of-range scores in caption files."""
    click.echo("=" * 60)
    click.echo(f"Score Validation/Fix: {directory}")
    click.echo(f"Mode: {'DRY RUN' if dry_run else 'FIX'}")
    click.echo("=" * 60)

    stats = scan_and_fix_scores(directory, dry_run=dry_run, recursive=recursive)

    click.echo(f"\nTotal caption files: {stats['total_files']:,}")
    click.echo(f"Files with issues: {stats['files_with_issues']:,}")

    click.echo(f"\nOut-of-range scores:")
    click.echo(f"  Below 0.0: {stats['scores_below_0']:,}")
    click.echo(f"  Above 1.0 (fixable): {stats['scores_above_1']:,}")
    click.echo(f"  Above 10.0 (unfixable): {stats['scores_above_10']:,}")

    click.echo(f"\nBy field:")
    for field in SCORE_FIELDS:
        field_stats = stats["issues_by_field"][field]
        total = (
            field_stats["below_0"] + field_stats["above_1"] + field_stats["above_10"]
        )
        if total > 0:
            click.echo(
                f"  {field}: {total} issues (below_0={field_stats['below_0']}, above_1={field_stats['above_1']}, above_10={field_stats['above_10']})"
            )

    if not dry_run:
        click.echo(f"\nFiles fixed: {stats['files_fixed']:,}")
    else:
        click.echo(
            f"\n[DRY RUN] Would fix {stats['scores_above_1']:,} scores in {stats['files_with_issues'] - len([p for p in stats['problem_files'] if p[3] != 'above_1']):,} files"
        )

    # Show unfixable problem files
    unfixable = [p for p in stats["problem_files"] if p[3] != "above_1"]
    if unfixable:
        click.echo(f"\nUnfixable files ({len(unfixable)}):")
        for path, field, value, issue in unfixable[:20]:
            click.echo(f"  {path}: {field}={value} ({issue})")
        if len(unfixable) > 20:
            click.echo(f"  ... and {len(unfixable) - 20} more")

    click.echo("\n" + "=" * 60)


if __name__ == "__main__":
    main()
