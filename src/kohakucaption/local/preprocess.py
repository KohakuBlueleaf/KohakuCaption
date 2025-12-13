"""
Preprocessing pipeline for high-throughput local VLM inference.

Features:
- Async image loading with prefetching
- Multi-process preprocessing
- Queue-based batching ahead of inference
- Memory-efficient streaming
"""

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedBatch:
    """A batch of preprocessed images ready for inference."""

    paths: list[Path]
    images: list[Image.Image]  # PIL images for VLM
    tagger_tensors: torch.Tensor | None  # Batched tensors for tagger
    valid_indices: list[int]  # Indices of successfully loaded images
    errors: list[str | None]  # Error messages for failed loads
    metadata: list[dict[str, Any]]  # Optional metadata per image


class ImageDataset(Dataset):
    """PyTorch Dataset for parallel image loading."""

    def __init__(
        self,
        image_paths: list[Path],
        tagger_transform: Callable | None = None,
        load_metadata: bool = False,
    ):
        self.image_paths = image_paths
        self.tagger_transform = tagger_transform
        self.load_metadata = load_metadata

    def __len__(self) -> int:
        return len(self.image_paths)

    def _to_rgb(self, image: Image.Image) -> Image.Image:
        """Convert image to RGB mode."""
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

    def _get_metadata(self, image: Image.Image, path: Path) -> dict[str, Any]:
        """Extract image metadata."""
        meta = {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode,
            "filename": path.name,
        }
        # EXIF data if available
        if hasattr(image, "_getexif") and image._getexif():
            meta["exif"] = dict(image._getexif())
        return meta

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.image_paths[idx]

        try:
            image = Image.open(path)
            image_rgb = self._to_rgb(image)

            # Metadata
            metadata = {}
            if self.load_metadata:
                metadata = self._get_metadata(image, path)

            # Tagger preprocessing
            tagger_tensor = None
            if self.tagger_transform is not None:
                tagger_tensor = self.tagger_transform(image_rgb)

            return {
                "path": str(path),
                "image": image_rgb,
                "tagger_tensor": tagger_tensor,
                "metadata": metadata,
                "error": None,
            }

        except Exception as e:
            return {
                "path": str(path),
                "image": None,
                "tagger_tensor": None,
                "metadata": {},
                "error": str(e),
            }


def collate_preprocessed(batch: list[dict]) -> PreprocessedBatch:
    """Collate function that creates PreprocessedBatch."""
    paths = [Path(item["path"]) for item in batch]
    images = [item["image"] for item in batch]
    errors = [item["error"] for item in batch]
    metadata = [item["metadata"] for item in batch]

    # Stack valid tagger tensors
    valid_tensors = []
    valid_indices = []
    for i, item in enumerate(batch):
        if item["tagger_tensor"] is not None and item["image"] is not None:
            valid_tensors.append(item["tagger_tensor"])
            valid_indices.append(i)

    tagger_batch = None
    if valid_tensors:
        tagger_batch = torch.stack(valid_tensors)

    return PreprocessedBatch(
        paths=paths,
        images=images,
        tagger_tensors=tagger_batch,
        valid_indices=valid_indices,
        errors=errors,
        metadata=metadata,
    )


class PreprocessPipeline:
    """
    Preprocessing pipeline that loads and preprocesses images ahead of inference.

    Uses a background thread to load batches from DataLoader and queue them,
    hiding I/O latency from the inference loop.

    Example:
        pipeline = PreprocessPipeline(
            image_paths=paths,
            batch_size=8,
            num_workers=4,
            prefetch_batches=3,
        )

        with pipeline:
            for batch in pipeline:
                # batch is PreprocessedBatch
                results = model.generate_batch(batch.images, prompts)
    """

    def __init__(
        self,
        image_paths: list[Path],
        batch_size: int = 8,
        num_workers: int = 4,
        prefetch_batches: int = 3,
        tagger_transform: Callable | None = None,
        load_metadata: bool = False,
        pin_memory: bool = True,
    ):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self.tagger_transform = tagger_transform
        self.load_metadata = load_metadata
        self.pin_memory = pin_memory and torch.cuda.is_available()

        self._queue: queue.Queue[PreprocessedBatch | None] = None
        self._loader_thread: threading.Thread | None = None
        self._stop_event: threading.Event = None
        self._started = False

    def _create_dataloader(self) -> DataLoader:
        """Create the underlying DataLoader."""
        dataset = ImageDataset(
            image_paths=self.image_paths,
            tagger_transform=self.tagger_transform,
            load_metadata=self.load_metadata,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_preprocessed,
            pin_memory=self.pin_memory,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )

    def _loader_worker(self):
        """Background thread that loads batches into queue."""
        try:
            dataloader = self._create_dataloader()
            for batch in dataloader:
                if self._stop_event.is_set():
                    break
                self._queue.put(batch)
        except Exception as e:
            logger.error(f"PreprocessPipeline loader error: {e}")
        finally:
            self._queue.put(None)  # Signal end of data

    def start(self) -> None:
        """Start the preprocessing pipeline."""
        if self._started:
            return

        self._queue = queue.Queue(maxsize=self.prefetch_batches)
        self._stop_event = threading.Event()
        self._loader_thread = threading.Thread(target=self._loader_worker, daemon=True)
        self._loader_thread.start()
        self._started = True

        logger.debug(
            f"PreprocessPipeline started: batch_size={self.batch_size}, "
            f"workers={self.num_workers}, prefetch={self.prefetch_batches}"
        )

    def stop(self) -> None:
        """Stop the preprocessing pipeline."""
        if not self._started:
            return

        self._stop_event.set()

        # Drain the queue
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        if self._loader_thread:
            self._loader_thread.join(timeout=5.0)

        self._started = False
        logger.debug("PreprocessPipeline stopped")

    def __iter__(self) -> Iterator[PreprocessedBatch]:
        """Iterate over preprocessed batches."""
        if not self._started:
            self.start()

        while True:
            batch = self._queue.get()
            if batch is None:
                break
            yield batch

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __len__(self) -> int:
        """Return number of batches."""
        return (len(self.image_paths) + self.batch_size - 1) // self.batch_size


class StreamingPreprocessor:
    """
    Memory-efficient streaming preprocessor for very large datasets.

    Instead of loading all paths upfront, streams paths from a generator.

    Example:
        def path_generator():
            for root, dirs, files in os.walk(image_dir):
                for f in files:
                    if f.endswith(('.jpg', '.png')):
                        yield Path(root) / f

        preprocessor = StreamingPreprocessor(
            path_generator=path_generator,
            batch_size=8,
        )

        for batch in preprocessor:
            results = model.generate_batch(batch.images, prompts)
    """

    def __init__(
        self,
        path_generator: Callable[[], Iterator[Path]],
        batch_size: int = 8,
        num_threads: int = 4,
        prefetch_images: int = 16,
        tagger_transform: Callable | None = None,
    ):
        self.path_generator = path_generator
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.prefetch_images = prefetch_images
        self.tagger_transform = tagger_transform

    def _load_image(self, path: Path) -> dict[str, Any]:
        """Load and preprocess a single image."""
        try:
            image = Image.open(path)

            # Convert to RGB
            if image.mode == "RGBA":
                bg = Image.new("RGB", image.size, (255, 255, 255))
                bg.paste(image, mask=image.split()[3])
                image = bg
            elif image.mode != "RGB":
                image = image.convert("RGB")

            # Tagger tensor
            tagger_tensor = None
            if self.tagger_transform:
                tagger_tensor = self.tagger_transform(image)

            return {
                "path": path,
                "image": image,
                "tagger_tensor": tagger_tensor,
                "error": None,
            }

        except Exception as e:
            return {
                "path": path,
                "image": None,
                "tagger_tensor": None,
                "error": str(e),
            }

    def __iter__(self) -> Iterator[PreprocessedBatch]:
        """Stream preprocessed batches."""
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit initial batch of loads
            futures = []
            path_iter = iter(self.path_generator())

            # Fill prefetch buffer
            for _ in range(self.prefetch_images):
                try:
                    path = next(path_iter)
                    futures.append(executor.submit(self._load_image, path))
                except StopIteration:
                    break

            # Process in batches
            current_batch = []

            while futures:
                # Get next completed result
                future = futures.pop(0)
                result = future.result()
                current_batch.append(result)

                # Submit next load to keep prefetch buffer full
                try:
                    path = next(path_iter)
                    futures.append(executor.submit(self._load_image, path))
                except StopIteration:
                    pass

                # Yield batch when full
                if len(current_batch) >= self.batch_size:
                    yield self._make_batch(current_batch)
                    current_batch = []

            # Yield remaining items
            if current_batch:
                yield self._make_batch(current_batch)

    def _make_batch(self, items: list[dict]) -> PreprocessedBatch:
        """Create PreprocessedBatch from list of loaded items."""
        paths = [item["path"] for item in items]
        images = [item["image"] for item in items]
        errors = [item["error"] for item in items]

        valid_tensors = []
        valid_indices = []
        for i, item in enumerate(items):
            if item["tagger_tensor"] is not None:
                valid_tensors.append(item["tagger_tensor"])
                valid_indices.append(i)

        tagger_batch = None
        if valid_tensors:
            tagger_batch = torch.stack(valid_tensors)

        return PreprocessedBatch(
            paths=paths,
            images=images,
            tagger_tensors=tagger_batch,
            valid_indices=valid_indices,
            errors=errors,
            metadata=[{}] * len(items),
        )
