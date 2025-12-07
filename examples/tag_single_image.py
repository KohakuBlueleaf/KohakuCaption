#!/usr/bin/env python3
"""
Tag a single image using PixAI Tagger v0.9.

Generates anime-style tags including features, characters, and IP/series information.
"""

import json
import sys
from pathlib import Path

import click

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kohakucaption.tagger import PixAITagger


def format_default(result) -> str:
    """Format as # key / value."""
    return f"""# feature_tags
{', '.join(result.feature_tags)}

# character_tags
{', '.join(result.character_tags)}

# ip_tags
{', '.join(result.ip_tags)}
""".strip()


def format_json(result) -> str:
    """Format as JSON."""
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)


def format_csv(result) -> str:
    """Format as comma-separated tags."""
    return ', '.join(result.all_tags())


@click.command()
@click.argument(
    "image",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--tagger-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path("pixai-tagger"),
    show_default=True,
    help="Directory containing PixAI tagger model files (model_v0.9.pth, tags_v0.9_13k.json, char_ip_map.json).",
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output file path. If not specified, prints to stdout.",
)
@click.option(
    "--general-threshold",
    type=float,
    default=0.3,
    show_default=True,
    help="Confidence threshold for general/feature tags. Lower = more tags.",
)
@click.option(
    "--character-threshold",
    type=float,
    default=0.85,
    show_default=True,
    help="Confidence threshold for character tags. Higher = more confident matches only.",
)
@click.option(
    "--format", "-f",
    "output_format",
    type=click.Choice(["default", "json", "csv"]),
    default="default",
    show_default=True,
    help="Output format: 'default' (# key/value), 'json' (structured), 'csv' (comma-separated).",
)
@click.option(
    "--no-fp16",
    is_flag=True,
    default=False,
    help="Disable FP16 inference. Use if you encounter precision issues on CPU.",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    default=False,
    help="Only output tags, suppress stats and info messages.",
)
def main(
    image: Path,
    tagger_dir: Path,
    output: Path | None,
    general_threshold: float,
    character_threshold: float,
    output_format: str,
    no_fp16: bool,
    quiet: bool,
):
    """
    Tag a single image using PixAI Tagger v0.9.

    IMAGE is the path to an image file (PNG, JPG, WebP, etc.).

    \b
    Examples:
        # Basic usage
        python tag_single_image.py image.png

        # Use custom tagger directory
        python tag_single_image.py image.png --tagger-dir ./models/pixai-tagger

        # Output as JSON
        python tag_single_image.py image.png -f json

        # Lower threshold for more tags
        python tag_single_image.py image.png --general-threshold 0.2

        # Save to file
        python tag_single_image.py image.png -o tags.txt

        # Quiet mode for scripting
        python tag_single_image.py image.png -q -f csv > tags.csv

    \b
    Output categories:
        - feature_tags: Visual features (hair color, clothing, pose, etc.)
        - character_tags: Identified anime characters
        - ip_tags: Identified intellectual properties/series

    \b
    Requirements:
        Download model files first:
        python scripts/download_pixai_tagger.py --output-dir ./pixai-tagger
    """
    # Validate tagger directory
    if not (tagger_dir / "model_v0.9.pth").exists():
        raise click.ClickException(
            f"Tagger model not found in {tagger_dir}.\n"
            "Run: python scripts/download_pixai_tagger.py"
        )

    if not quiet:
        click.echo(f"Tagging image: {image}", err=True)
        click.echo(f"Tagger directory: {tagger_dir}", err=True)

    # Load tagger
    tagger = PixAITagger(
        model_dir=tagger_dir,
        use_fp16=not no_fp16,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
    )

    # Tag image
    result = tagger.tag(image)

    # Format output
    if output_format == "json":
        output_str = format_json(result)
    elif output_format == "csv":
        output_str = format_csv(result)
    else:
        output_str = format_default(result)

    if quiet:
        click.echo(output_str)
    else:
        # Print results
        click.echo("=" * 60)
        click.echo("TAGS:")
        click.echo("=" * 60)
        click.echo(output_str)

        # Print stats
        click.echo()
        click.echo("=" * 60)
        click.echo("STATS:")
        click.echo("=" * 60)
        click.echo(f"  Feature tags: {len(result.feature_tags)}")
        click.echo(f"  Character tags: {len(result.character_tags)}")
        click.echo(f"  IP tags: {len(result.ip_tags)}")
        click.echo(f"  Total tags: {len(result.all_tags())}")

    # Save to file if specified
    if output:
        output.write_text(output_str, encoding="utf-8")
        if not quiet:
            click.echo(f"\nTags saved to: {output}")


if __name__ == "__main__":
    main()
