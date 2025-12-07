#!/usr/bin/env python3
"""
Download images from URL lists.

Each .txt file contains URLs (one per line) and images are saved to a folder
with the same name as the txt file.

Example:
    datas/kaka_ozisan.txt -> datas/kaka_ozisan/001.jpg, 002.png, ...
"""

import asyncio
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import click
import httpx


async def download_image(
    client: httpx.AsyncClient,
    url: str,
    output_path: Path,
    semaphore: asyncio.Semaphore,
) -> tuple[bool, str]:
    """Download a single image."""
    async with semaphore:
        try:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

            output_path.write_bytes(response.content)
            return True, str(output_path.name)
        except Exception as e:
            return False, f"{url}: {e}"


def get_extension_from_url(url: str) -> str:
    """Extract file extension from URL."""
    parsed = urlparse(url)
    path = unquote(parsed.path)

    # Try to get extension from path
    if "." in path.split("/")[-1]:
        ext = path.split(".")[-1].lower()
        # Clean up query params if any
        ext = ext.split("?")[0].split("&")[0]
        if ext in ("jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff", "avif"):
            return ext if ext != "jpeg" else "jpg"

    return "jpg"  # Default


def get_filename_from_url(url: str, index: int) -> str:
    """Extract or generate filename from URL."""
    parsed = urlparse(url)
    path = unquote(parsed.path)

    # Try to get original filename
    if "/" in path:
        filename = path.split("/")[-1]
        # Clean up and validate
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        if filename and "." in filename:
            return filename

    # Fallback to numbered filename
    ext = get_extension_from_url(url)
    return f"{index:04d}.{ext}"


async def process_url_file(
    url_file: Path,
    output_dir: Path,
    max_concurrent: int,
    timeout: float,
    use_original_names: bool,
    on_progress,
) -> dict[str, int]:
    """Process a single URL file."""
    # Read URLs
    urls = [
        line.strip()
        for line in url_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not urls:
        return {"total": 0, "success": 0, "failed": 0}

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(urls), "success": 0, "failed": 0}
    semaphore = asyncio.Semaphore(max_concurrent)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
    ) as client:
        tasks = []
        for i, url in enumerate(urls, 1):
            if use_original_names:
                filename = get_filename_from_url(url, i)
            else:
                ext = get_extension_from_url(url)
                filename = f"{i:04d}.{ext}"

            output_path = output_dir / filename

            # Skip if exists
            if output_path.exists():
                stats["success"] += 1
                on_progress(url_file.stem, filename, True, "skipped (exists)")
                continue

            task = asyncio.create_task(
                download_image(client, url, output_path, semaphore)
            )
            tasks.append((url, filename, task))

        for url, filename, task in tasks:
            success, msg = await task
            if success:
                stats["success"] += 1
                on_progress(url_file.stem, msg, True, None)
            else:
                stats["failed"] += 1
                on_progress(url_file.stem, filename, False, msg)

    return stats


@click.command()
@click.argument(
    "input_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Base output directory. Defaults to same directory as input.",
)
@click.option(
    "--max-concurrent", "-c",
    type=int,
    default=5,
    show_default=True,
    help="Maximum concurrent downloads.",
)
@click.option(
    "--timeout",
    type=float,
    default=30.0,
    show_default=True,
    help="Request timeout in seconds.",
)
@click.option(
    "--original-names/--numbered",
    default=False,
    help="Use original filenames from URLs instead of numbered (0001.jpg, etc.).",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Only show summary, not individual downloads.",
)
def main(
    input_path: Path,
    output_dir: Path | None,
    max_concurrent: int,
    timeout: float,
    original_names: bool,
    quiet: bool,
):
    """
    Download images from URL list files.

    INPUT_PATH can be a single .txt file or a directory containing .txt files.

    Each .txt file should contain one URL per line. Lines starting with # are
    ignored. Images are saved to a folder with the same name as the txt file.

    \b
    Examples:
        # Single file: datas/artist.txt -> datas/artist/*.jpg
        python download_images.py datas/artist.txt

        # All txt files in directory
        python download_images.py datas/

        # Custom output directory
        python download_images.py datas/artist.txt -o ./downloads

        # Faster downloads with more concurrency
        python download_images.py datas/ -c 10

        # Keep original filenames from URLs
        python download_images.py datas/artist.txt --original-names

    \b
    URL file format (artist.txt):
        # Comments are ignored
        https://example.com/image1.jpg
        https://example.com/image2.png
        https://example.com/path/to/image3.webp
    """
    # Find all URL files
    if input_path.is_file():
        if not input_path.suffix == ".txt":
            raise click.ClickException(f"Input file must be .txt: {input_path}")
        url_files = [input_path]
        base_dir = output_dir or input_path.parent
    else:
        url_files = sorted(input_path.glob("*.txt"))
        if not url_files:
            raise click.ClickException(f"No .txt files found in {input_path}")
        base_dir = output_dir or input_path

    click.echo(f"Found {len(url_files)} URL file(s)")
    click.echo(f"Output base: {base_dir}")
    click.echo(f"Max concurrent: {max_concurrent}")
    click.echo()

    total_stats = {"files": 0, "total": 0, "success": 0, "failed": 0}

    def on_progress(folder: str, filename: str, success: bool, error: str | None):
        if quiet:
            return
        status = click.style("✓", fg="green") if success else click.style("✗", fg="red")
        msg = f"  [{folder}] {status} {filename}"
        if error and not success:
            msg += f" - {error[:60]}"
        click.echo(msg)

    for url_file in url_files:
        folder_name = url_file.stem
        folder_output = base_dir / folder_name

        click.echo(f"Processing: {url_file.name} -> {folder_output}/")

        stats = asyncio.run(
            process_url_file(
                url_file=url_file,
                output_dir=folder_output,
                max_concurrent=max_concurrent,
                timeout=timeout,
                use_original_names=original_names,
                on_progress=on_progress,
            )
        )

        total_stats["files"] += 1
        total_stats["total"] += stats["total"]
        total_stats["success"] += stats["success"]
        total_stats["failed"] += stats["failed"]

        if not quiet:
            click.echo(f"  Done: {stats['success']}/{stats['total']} succeeded")
        click.echo()

    # Summary
    click.echo("=" * 60)
    click.echo("SUMMARY")
    click.echo("=" * 60)
    click.echo(f"  Files processed: {total_stats['files']}")
    click.echo(f"  Total images: {total_stats['total']}")
    click.echo(f"  Downloaded: {total_stats['success']}")
    click.echo(f"  Failed: {total_stats['failed']}")


if __name__ == "__main__":
    main()
