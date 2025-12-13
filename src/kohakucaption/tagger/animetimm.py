"""
AnimeTIMM Tagger wrapper for batch image tagging.
Supports models from https://huggingface.co/animetimm
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from PIL import Image
from timm import create_model

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "animetimm/caformer_b36.dbv4-full"


@dataclass
class AnimeTimmTagResult:
    """Result of tagging a single image with AnimeTIMM."""

    general_tags: dict[str, float]
    character_tags: dict[str, float]
    rating_tags: dict[str, float]

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            "general": self.general_tags,
            "character": self.character_tags,
            "rating": self.rating_tags,
        }

    def all_tags(self) -> list[str]:
        """Get all tag names as a single list."""
        return (
            list(self.general_tags.keys())
            + list(self.character_tags.keys())
            + list(self.rating_tags.keys())
        )

    def all_tags_with_scores(self) -> dict[str, float]:
        """Get all tags with their scores."""
        result = {}
        result.update(self.general_tags)
        result.update(self.character_tags)
        result.update(self.rating_tags)
        return result

    def format_for_prompt(self) -> str:
        """Format tags for inclusion in LLM prompt. Replaces underscores with spaces."""
        parts = []
        if self.general_tags:
            tags = ", ".join(tag.replace("_", " ") for tag in self.general_tags.keys())
            parts.append(f"Features: {tags}")
        if self.character_tags:
            tags = ", ".join(
                tag.replace("_", " ") for tag in self.character_tags.keys()
            )
            parts.append(f"Characters: {tags}")
        if self.rating_tags:
            rating = max(self.rating_tags.keys(), key=lambda k: self.rating_tags[k])
            parts.append(f"Rating: {rating.replace('_', ' ')}")
        return "\n".join(parts)


class PadToSize:
    """Pad image to target size with background color."""

    def __init__(
        self,
        size: tuple[int, int],
        interpolation: str = "bilinear",
        background_color: str = "white",
    ):
        self.size = size
        self.background_color = background_color

    def __call__(self, img: Image.Image) -> Image.Image:
        target_w, target_h = self.size
        orig_w, orig_h = img.size

        # Calculate scale to fit within target
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Create background and paste
        bg_color = (255, 255, 255) if self.background_color == "white" else (0, 0, 0)
        background = Image.new("RGB", self.size, bg_color)
        offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
        background.paste(img, offset)

        return background


class AnimeTimmTagger:
    """
    AnimeTIMM Tagger for anime image tagging.

    Supports batch inference with fp16 and no_grad for efficiency.
    Downloads model from HuggingFace Hub automatically.

    Example:
        tagger = AnimeTimmTagger("animetimm/caformer_b36.dbv4-full")
        results = tagger.tag_batch(images)
    """

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        device: str | None = None,
        use_fp16: bool = True,
        general_threshold: float | None = None,
        character_threshold: float | None = None,
        rating_threshold: float | None = None,
        use_best_threshold: bool = True,
    ):
        """
        Initialize the tagger.

        Args:
            repo_id: HuggingFace Hub repo ID (e.g., "animetimm/caformer_b36.dbv4-full")
            device: Device to use (auto-detect if None)
            use_fp16: Use fp16 for inference
            general_threshold: Override threshold for general tags (None = use best)
            character_threshold: Override threshold for character tags (None = use best)
            rating_threshold: Override threshold for rating tags (None = use best)
            use_best_threshold: Use per-tag best thresholds from model (default True)
        """
        self.repo_id = repo_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.rating_threshold = rating_threshold
        self.use_best_threshold = use_best_threshold

        self._model: nn.Module | None = None
        self._transform: T.Compose | None = None
        self._tags_df: pd.DataFrame | None = None
        self._categories: dict[int, str] | None = None
        self._loaded = False

    def _download_file(self, filename: str) -> str:
        """Download a file from the repo."""
        return hf_hub_download(
            repo_id=self.repo_id, repo_type="model", filename=filename
        )

    def _load_model(self) -> None:
        """Load model and metadata lazily."""
        if self._loaded:
            return

        # Load categories
        categories_path = self._download_file("categories.json")
        with open(categories_path, "r", encoding="utf-8") as f:
            categories_list = json.load(f)
        self._categories = {c["category"]: c["name"] for c in categories_list}

        # Reverse map: category name -> category id
        self._category_name_to_id = {c["name"]: c["category"] for c in categories_list}

        # Load tags
        tags_path = self._download_file("selected_tags.csv")
        self._tags_df = pd.read_csv(tags_path, keep_default_na=False)

        # Precompute numpy arrays for vectorized processing
        self._tag_names = self._tags_df["name"].values  # np.ndarray of strings
        self._tag_categories = self._tags_df["category"].values  # np.ndarray of ints

        # Build category masks (boolean arrays)
        general_cat_id = self._category_name_to_id.get("general", -1)
        character_cat_id = self._category_name_to_id.get("character", -1)
        rating_cat_id = self._category_name_to_id.get("rating", -1)

        self._general_mask = self._tag_categories == general_cat_id
        self._character_mask = self._tag_categories == character_cat_id
        self._rating_mask = self._tag_categories == rating_cat_id

        # Precompute thresholds array
        if self.use_best_threshold:
            self._thresholds = self._tags_df["best_threshold"].values.astype(np.float32)
        else:
            # Build threshold array based on category
            self._thresholds = np.full(len(self._tags_df), 0.4, dtype=np.float32)
            self._thresholds[self._general_mask] = self.general_threshold or 0.4
            self._thresholds[self._character_mask] = self.character_threshold or 0.5
            self._thresholds[self._rating_mask] = self.rating_threshold or 0.4

        # Load preprocessing config
        preprocess_path = self._download_file("preprocess.json")
        with open(preprocess_path, "r", encoding="utf-8") as f:
            preprocess_config = json.load(f)

        # Build transform from config
        self._transform = self._build_transform(preprocess_config["test"])

        # Load model
        self._model = create_model(f"hf-hub:{self.repo_id}", pretrained=True)
        self._model.to(self.device)
        self._model.eval()

        if self.use_fp16:
            self._model = self._model.half()

        self._loaded = True
        logger.info(
            f"AnimeTIMM Tagger loaded: {self.repo_id} on {self.device} (fp16={self.use_fp16})"
        )

    def _build_transform(self, config: list[dict]) -> T.Compose:
        """Build torchvision transforms from preprocess config."""
        transforms = []

        for step in config:
            step_type = step["type"]

            if step_type == "pad_to_size":
                transforms.append(
                    PadToSize(
                        size=tuple(step["size"]),
                        interpolation=step.get("interpolation", "bilinear"),
                        background_color=step.get("background_color", "white"),
                    )
                )

            elif step_type == "resize":
                size = step["size"]
                interpolation_map = {
                    "bilinear": T.InterpolationMode.BILINEAR,
                    "bicubic": T.InterpolationMode.BICUBIC,
                    "nearest": T.InterpolationMode.NEAREST,
                }
                interpolation = interpolation_map.get(
                    step.get("interpolation", "bilinear"),
                    T.InterpolationMode.BILINEAR,
                )
                transforms.append(
                    T.Resize(
                        size,
                        interpolation=interpolation,
                        antialias=step.get("antialias", True),
                    )
                )

            elif step_type == "center_crop":
                transforms.append(T.CenterCrop(step["size"]))

            elif step_type == "maybe_to_tensor":
                transforms.append(T.ToTensor())

            elif step_type == "normalize":
                transforms.append(T.Normalize(mean=step["mean"], std=step["std"]))

        return T.Compose(transforms)

    def _pil_to_rgb(self, image: Image.Image) -> Image.Image:
        """Convert image to RGB, handling RGBA and palette modes."""
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            return background
        elif image.mode == "P":
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            if rgba.mode == "RGBA":
                background.paste(rgba, mask=rgba.split()[3])
            return background
        else:
            return image.convert("RGB")

    def _process_probs(self, probs: torch.Tensor) -> AnimeTimmTagResult:
        """Process probability tensor into tag result (vectorized, single image)."""
        probs_np = probs.cpu().numpy()

        # Vectorized threshold comparison: [num_tags] > [num_tags] -> [num_tags] bool
        above_threshold = probs_np >= self._thresholds

        # Extract tags for each category using precomputed masks
        general_indices = np.where(above_threshold & self._general_mask)[0]
        character_indices = np.where(above_threshold & self._character_mask)[0]
        rating_indices = np.where(above_threshold & self._rating_mask)[0]

        # Sort indices by probability descending
        if len(general_indices) > 0:
            general_indices = general_indices[np.argsort(-probs_np[general_indices])]
        if len(character_indices) > 0:
            character_indices = character_indices[
                np.argsort(-probs_np[character_indices])
            ]
        if len(rating_indices) > 0:
            rating_indices = rating_indices[np.argsort(-probs_np[rating_indices])]

        # Build dicts from sorted indices (minimal Python loop over selected tags only)
        general_tags = {self._tag_names[i]: float(probs_np[i]) for i in general_indices}
        character_tags = {
            self._tag_names[i]: float(probs_np[i]) for i in character_indices
        }
        rating_tags = {self._tag_names[i]: float(probs_np[i]) for i in rating_indices}

        return AnimeTimmTagResult(
            general_tags=general_tags,
            character_tags=character_tags,
            rating_tags=rating_tags,
        )

    def _process_probs_batch(
        self, batch_probs: torch.Tensor
    ) -> list[AnimeTimmTagResult]:
        """Process batch of probability tensors (vectorized).

        Args:
            batch_probs: [batch_size, num_tags] tensor

        Returns:
            List of AnimeTimmTagResult
        """
        # [B, num_tags]
        probs_np = batch_probs.cpu().numpy()
        batch_size = probs_np.shape[0]

        # Vectorized threshold comparison: [B, num_tags] >= [num_tags] -> [B, num_tags]
        above_threshold = probs_np >= self._thresholds  # broadcasts

        # Compute masks for all categories at once: [B, num_tags]
        general_selected = above_threshold & self._general_mask  # [B, num_tags]
        character_selected = above_threshold & self._character_mask
        rating_selected = above_threshold & self._rating_mask

        # For sorting: set non-selected probs to -inf so they sort last
        NEG_INF = -np.inf

        # Create masked prob arrays for sorting (all at once)
        general_probs_masked = np.where(general_selected, probs_np, NEG_INF)
        character_probs_masked = np.where(character_selected, probs_np, NEG_INF)
        rating_probs_masked = np.where(rating_selected, probs_np, NEG_INF)

        # Get sorted indices (descending) for each category: [B, num_tags]
        general_sorted_idx = np.argsort(-general_probs_masked, axis=1)
        character_sorted_idx = np.argsort(-character_probs_masked, axis=1)
        rating_sorted_idx = np.argsort(-rating_probs_masked, axis=1)

        # Count how many tags are selected per image per category: [B]
        general_counts = general_selected.sum(axis=1)
        character_counts = character_selected.sum(axis=1)
        rating_counts = rating_selected.sum(axis=1)

        # Only loop for final dict construction (unavoidable for variable-length dicts)
        results = []
        for b in range(batch_size):
            # Slice sorted indices up to count (already sorted by prob descending)
            g_idx = general_sorted_idx[b, : general_counts[b]]
            c_idx = character_sorted_idx[b, : character_counts[b]]
            r_idx = rating_sorted_idx[b, : rating_counts[b]]

            # Build dicts using numpy advanced indexing
            probs_b = probs_np[b]
            general_tags = dict(
                zip(self._tag_names[g_idx], probs_b[g_idx].astype(float))
            )
            character_tags = dict(
                zip(self._tag_names[c_idx], probs_b[c_idx].astype(float))
            )
            rating_tags = dict(
                zip(self._tag_names[r_idx], probs_b[r_idx].astype(float))
            )

            results.append(
                AnimeTimmTagResult(
                    general_tags=general_tags,
                    character_tags=character_tags,
                    rating_tags=rating_tags,
                )
            )

        return results

    def tag(
        self,
        image: Image.Image | str | Path,
    ) -> AnimeTimmTagResult:
        """
        Tag a single image.

        Args:
            image: PIL Image or path to image

        Returns:
            AnimeTimmTagResult with general, character, and rating tags
        """
        self._load_model()

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        image = self._pil_to_rgb(image)
        tensor = self._transform(image).unsqueeze(0)

        if self.use_fp16:
            tensor = tensor.half()

        tensor = tensor.to(self.device)

        with torch.no_grad():
            output = self._model(tensor)
            probs = torch.sigmoid(output)[0]

        return self._process_probs(probs)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess a single PIL image to tensor.

        Args:
            image: PIL Image (already loaded)

        Returns:
            Preprocessed tensor (not batched, not on device)
        """
        self._load_model()
        image = self._pil_to_rgb(image)
        return self._transform(image)

    def preprocess_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """
        Preprocess multiple PIL images to a batched tensor.

        Args:
            images: List of PIL Images (already loaded)

        Returns:
            Batched tensor ready for inference (not on device yet)
        """
        self._load_model()
        tensors = [self._transform(self._pil_to_rgb(img)) for img in images]
        return torch.stack(tensors)

    def inference(self, batch_tensor: torch.Tensor) -> list[AnimeTimmTagResult]:
        """
        Run inference on preprocessed tensor batch.

        Args:
            batch_tensor: Preprocessed batched tensor from preprocess_batch()

        Returns:
            List of AnimeTimmTagResults
        """
        self._load_model()

        if self.use_fp16:
            batch_tensor = batch_tensor.half()

        batch_tensor = batch_tensor.to(self.device)

        with torch.no_grad():
            output = self._model(batch_tensor)
            batch_probs = torch.sigmoid(output)

        # Use vectorized batch processing
        return self._process_probs_batch(batch_probs)

    def tag_batch(
        self,
        images: list[Image.Image | str | Path],
        batch_size: int = 8,
    ) -> list[AnimeTimmTagResult]:
        """
        Tag multiple images in batches.

        Args:
            images: List of PIL Images or paths
            batch_size: Batch size for inference

        Returns:
            List of AnimeTimmTagResults in same order as input
        """
        self._load_model()

        results = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]

            # Load if paths
            pil_images = []
            for img in batch_images:
                if isinstance(img, (str, Path)):
                    img = Image.open(img)
                pil_images.append(img)

            # Preprocess and inference
            batch_tensor = self.preprocess_batch(pil_images)
            batch_results = self.inference(batch_tensor)
            results.extend(batch_results)

        return results

    def unload(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("AnimeTIMM Tagger unloaded")
