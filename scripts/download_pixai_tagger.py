#!/usr/bin/env python3
"""
Download PixAI Tagger v0.9 model files from HuggingFace Hub.

This is a gated model that requires HuggingFace account and license acceptance.
"""

import sys
from pathlib import Path

import click


@click.command()
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("pixai-tagger"),
    show_default=True,
    help="Output directory for downloaded model files.",
)
@click.option(
    "--repo-id",
    type=str,
    default="pixai-labs/pixai-tagger-v0.9",
    show_default=True,
    help="HuggingFace Hub repository ID.",
)
@click.option(
    "--force/--no-force",
    default=False,
    help="Force re-download even if files already exist.",
)
def main(output_dir: Path, repo_id: str, force: bool):
    """
    Download PixAI Tagger v0.9 model files from HuggingFace Hub.

    \b
    Examples:
        # Download to default directory
        python download_pixai_tagger.py

        # Download to custom directory
        python download_pixai_tagger.py -o ./models/pixai-tagger

        # Force re-download
        python download_pixai_tagger.py --force

    \b
    Requirements:
        This is a gated model. Before downloading, you must:
        1. Create a HuggingFace account at https://huggingface.co
        2. Accept the model license at https://huggingface.co/pixai-labs/pixai-tagger-v0.9
        3. Login with: huggingface-cli login

    \b
    Downloaded files:
        - model_v0.9.pth: Model weights (~1.2GB)
        - tags_v0.9_13k.json: Tag definitions
        - char_ip_map.json: Character to IP mapping
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise click.ClickException(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Downloading PixAI Tagger v0.9 to: {output_dir.absolute()}")
    click.echo(f"Repository: {repo_id}")
    click.echo()

    # Required files
    files = [
        ("model_v0.9.pth", "Model weights (~1.2GB)"),
        ("tags_v0.9_13k.json", "Tag definitions"),
        ("char_ip_map.json", "Character to IP mapping"),
    ]

    # Check existing files
    if not force:
        existing = [f for f, _ in files if (output_dir / f).exists()]
        if len(existing) == len(files):
            click.echo("All files already exist. Use --force to re-download.")
            click.echo()
            _print_usage(output_dir)
            return

    for filename, description in files:
        target_path = output_dir / filename

        if target_path.exists() and not force:
            click.echo(f"  {click.style('✓', fg='green')} {filename} (already exists)")
            continue

        click.echo(f"Downloading {filename} ({description})...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=output_dir,
                local_dir_use_symlinks=False,
            )
            click.echo(f"  {click.style('✓', fg='green')} {filename}")
        except Exception as e:
            click.echo(f"  {click.style('✗', fg='red')} {filename}: {e}")
            click.echo()
            click.echo(click.style("Note: This is a gated model.", fg="yellow"))
            click.echo("Make sure you have:")
            click.echo("  1. A HuggingFace account")
            click.echo(f"  2. Accepted the license at: https://huggingface.co/{repo_id}")
            click.echo("  3. Logged in with: huggingface-cli login")
            sys.exit(1)

    click.echo()
    click.echo(click.style("Download complete!", fg="green"))
    click.echo(f"Model files saved to: {output_dir.absolute()}")
    click.echo()
    _print_usage(output_dir)


def _print_usage(output_dir: Path):
    """Print usage instructions."""
    click.echo("Usage:")
    click.echo("  from kohakucaption.tagger import PixAITagger")
    click.echo(f'  tagger = PixAITagger("{output_dir}")')
    click.echo('  result = tagger.tag("image.png")')
    click.echo()
    click.echo("Or via CLI:")
    click.echo(f"  python examples/tag_single_image.py image.png --tagger-dir {output_dir}")


if __name__ == "__main__":
    main()
