#!/usr/bin/env python3
"""
Tag a single image using AnimeTIMM tagger.

Downloads models automatically from HuggingFace Hub and generates anime-style tags
with per-tag confidence scores.
"""

import json
import sys
from pathlib import Path

import click

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kohakucaption.tagger import AnimeTimmTagger


def format_default(result) -> str:
    """Format as # key / value."""
    general = ", ".join(result.general_tags.keys())
    character = ", ".join(result.character_tags.keys())
    rating = ", ".join(result.rating_tags.keys())
    return f"""# general_tags
{general}

# character_tags
{character}

# rating_tags
{rating}
""".strip()


def format_json(result) -> str:
    """Format as JSON."""
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)


def format_csv(result) -> str:
    """Format as comma-separated tags."""
    return ", ".join(result.all_tags())


def format_scores(result) -> str:
    """Format with confidence scores."""
    lines = []

    if result.general_tags:
        lines.append("# general_tags")
        for tag, score in result.general_tags.items():
            lines.append(f"  {tag}: {score:.3f}")

    if result.character_tags:
        lines.append("\n# character_tags")
        for tag, score in result.character_tags.items():
            lines.append(f"  {tag}: {score:.3f}")

    if result.rating_tags:
        lines.append("\n# rating_tags")
        for tag, score in result.rating_tags.items():
            lines.append(f"  {tag}: {score:.3f}")

    return "\n".join(lines)


@click.command()
@click.argument(
    "image",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--repo-id",
    type=str,
    default="animetimm/caformer_b36.dbv4-full",
    show_default=True,
    help="HuggingFace Hub repository ID for the model.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output file path. If not specified, prints to stdout.",
)
@click.option(
    "--general-threshold",
    type=float,
    default=None,
    help="Override threshold for general tags (default: use per-tag best threshold).",
)
@click.option(
    "--character-threshold",
    type=float,
    default=None,
    help="Override threshold for character tags (default: use per-tag best threshold).",
)
@click.option(
    "--rating-threshold",
    type=float,
    default=None,
    help="Override threshold for rating tags (default: use per-tag best threshold).",
)
@click.option(
    "--no-best-threshold",
    is_flag=True,
    default=False,
    help="Disable per-tag optimal thresholds. Use fixed thresholds instead.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["default", "json", "csv", "scores"]),
    default="default",
    show_default=True,
    help="Output format: 'default' (tags only), 'json' (with scores), 'csv' (comma-separated), 'scores' (verbose with confidence).",
)
@click.option(
    "--no-fp16",
    is_flag=True,
    default=False,
    help="Disable FP16 inference. Use if you encounter precision issues on CPU.",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Only output tags, suppress stats and info messages.",
)
def main(
    image: Path,
    repo_id: str,
    output: Path | None,
    general_threshold: float | None,
    character_threshold: float | None,
    rating_threshold: float | None,
    no_best_threshold: bool,
    output_format: str,
    no_fp16: bool,
    quiet: bool,
):
    """
    Tag a single image using AnimeTIMM tagger.

    IMAGE is the path to an image file (PNG, JPG, WebP, etc.).

    Models are automatically downloaded from HuggingFace Hub on first use.

    \b
    Examples:
        # Basic usage (downloads model automatically)
        python tag_single_image_animetimm.py image.png

        # Use a different model
        python tag_single_image_animetimm.py image.png --repo-id animetimm/swinv2_v3

        # Output with confidence scores
        python tag_single_image_animetimm.py image.png -f scores

        # Output as JSON
        python tag_single_image_animetimm.py image.png -f json

        # Use fixed thresholds instead of per-tag optimal
        python tag_single_image_animetimm.py image.png --no-best-threshold --general-threshold 0.4

        # Save to file
        python tag_single_image_animetimm.py image.png -o tags.txt

        # Quiet mode for scripting
        python tag_single_image_animetimm.py image.png -q -f csv > tags.csv

    \b
    Output categories:
        - general_tags: Visual features (with confidence scores in JSON/scores format)
        - character_tags: Identified anime characters
        - rating_tags: Content rating (general, sensitive, questionable, explicit)

    \b
    Available models:
        - animetimm/caformer_b36.dbv4-full (default, most accurate)
        - animetimm/swinv2_v3
        - See https://huggingface.co/animetimm for more
    """
    if not quiet:
        click.echo(f"Tagging image: {image}", err=True)
        click.echo(f"Model repo: {repo_id}", err=True)

    # Load tagger
    tagger = AnimeTimmTagger(
        repo_id=repo_id,
        use_fp16=not no_fp16,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        rating_threshold=rating_threshold,
        use_best_threshold=not no_best_threshold,
    )

    # Tag image
    result = tagger.tag(image)

    # Format output
    if output_format == "json":
        output_str = format_json(result)
    elif output_format == "csv":
        output_str = format_csv(result)
    elif output_format == "scores":
        output_str = format_scores(result)
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
        click.echo(f"  General tags: {len(result.general_tags)}")
        click.echo(f"  Character tags: {len(result.character_tags)}")
        click.echo(f"  Rating tags: {len(result.rating_tags)}")
        click.echo(f"  Total tags: {len(result.all_tags())}")

    # Save to file if specified
    if output:
        output.write_text(output_str, encoding="utf-8")
        if not quiet:
            click.echo(f"\nTags saved to: {output}")


if __name__ == "__main__":
    main()
