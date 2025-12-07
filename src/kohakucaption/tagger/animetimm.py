"""
AnimeTIMM Tagger wrapper for batch image tagging.
Supports models from https://huggingface.co/animetimm
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

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
            tags = ", ".join(tag.replace("_", " ") for tag in self.character_tags.keys())
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
        return hf_hub_download(repo_id=self.repo_id, repo_type="model", filename=filename)

    def _load_model(self) -> None:
        """Load model and metadata lazily."""
        if self._loaded:
            return

        # Load categories
        categories_path = self._download_file("categories.json")
        with open(categories_path, "r", encoding="utf-8") as f:
            categories_list = json.load(f)
        self._categories = {c["category"]: c["name"] for c in categories_list}

        # Load tags
        tags_path = self._download_file("selected_tags.csv")
        self._tags_df = pd.read_csv(tags_path, keep_default_na=False)

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
        logger.info(f"AnimeTIMM Tagger loaded: {self.repo_id} on {self.device} (fp16={self.use_fp16})")

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
        """Process probability tensor into tag result."""
        probs_np = probs.cpu().numpy()

        general_tags = {}
        character_tags = {}
        rating_tags = {}

        for idx, row in self._tags_df.iterrows():
            tag_name = row["name"]
            category = row["category"]
            prob = float(probs_np[idx])

            # Determine threshold
            if self.use_best_threshold:
                threshold = row["best_threshold"]
            else:
                category_name = self._categories.get(category, "")
                if category_name == "general":
                    threshold = self.general_threshold or 0.4
                elif category_name == "character":
                    threshold = self.character_threshold or 0.5
                elif category_name == "rating":
                    threshold = self.rating_threshold or 0.4
                else:
                    threshold = 0.4

            if prob >= threshold:
                category_name = self._categories.get(category, "")
                if category_name == "general":
                    general_tags[tag_name] = prob
                elif category_name == "character":
                    character_tags[tag_name] = prob
                elif category_name == "rating":
                    rating_tags[tag_name] = prob

        # Sort by probability descending
        general_tags = dict(sorted(general_tags.items(), key=lambda x: -x[1]))
        character_tags = dict(sorted(character_tags.items(), key=lambda x: -x[1]))
        rating_tags = dict(sorted(rating_tags.items(), key=lambda x: -x[1]))

        return AnimeTimmTagResult(
            general_tags=general_tags,
            character_tags=character_tags,
            rating_tags=rating_tags,
        )

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

            # Preprocess
            tensors = []
            for img in batch_images:
                if isinstance(img, (str, Path)):
                    img = Image.open(img)
                img = self._pil_to_rgb(img)
                tensors.append(self._transform(img))

            batch_tensor = torch.stack(tensors)

            if self.use_fp16:
                batch_tensor = batch_tensor.half()

            batch_tensor = batch_tensor.to(self.device)

            with torch.no_grad():
                output = self._model(batch_tensor)
                batch_probs = torch.sigmoid(output)

            for probs in batch_probs:
                results.append(self._process_probs(probs))

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
