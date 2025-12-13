"""
PixAI Tagger v0.9 wrapper for batch image tagging.
Based on: https://huggingface.co/pixai-labs/pixai-tagger-v0.9
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class TagResult:
    """Result of tagging a single image."""

    feature_tags: list[str]
    character_tags: list[str]
    ip_tags: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "feature": self.feature_tags,
            "character": self.character_tags,
            "ip": self.ip_tags,
        }

    def all_tags(self) -> list[str]:
        """Get all tags as a single list."""
        return self.feature_tags + self.character_tags + self.ip_tags

    def format_for_prompt(self) -> str:
        """Format tags for inclusion in LLM prompt. Replaces underscores with spaces."""
        parts = []
        if self.feature_tags:
            tags = [tag.replace("_", " ") for tag in self.feature_tags]
            parts.append(f"Features: {', '.join(tags)}")
        if self.character_tags:
            tags = [tag.replace("_", " ") for tag in self.character_tags]
            parts.append(f"Characters: {', '.join(tags)}")
        if self.ip_tags:
            tags = [tag.replace("_", " ") for tag in self.ip_tags]
            parts.append(f"IPs/Series: {', '.join(tags)}")
        return "\n".join(parts)


class TaggingHead(nn.Module):
    """Classification head for the tagger."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(x)
        return torch.sigmoid(logits)


class PixAITagger:
    """
    PixAI Tagger v0.9 for anime image tagging.

    Supports batch inference with fp16 and no_grad for efficiency.

    Example:
        tagger = PixAITagger("pixai-tagger/")
        results = tagger.tag_batch(images)
    """

    MODEL_FILE = "model_v0.9.pth"
    TAGS_FILE = "tags_v0.9_13k.json"
    CHAR_IP_FILE = "char_ip_map.json"

    def __init__(
        self,
        model_dir: str | Path,
        device: str | None = None,
        use_fp16: bool = True,
        general_threshold: float = 0.3,
        character_threshold: float = 0.85,
    ):
        """
        Initialize the tagger.

        Args:
            model_dir: Directory containing model files
            device: Device to use (auto-detect if None)
            use_fp16: Use fp16 for inference
            general_threshold: Threshold for general tags
            character_threshold: Threshold for character tags
        """
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold

        self._model: nn.Module | None = None
        self._transform: T.Compose | None = None
        self._index_to_tag: dict[int, str] | None = None
        self._char_ip_map: dict[str, list[str]] | None = None
        self._gen_tag_count: int = 0
        self._loaded = False

    def _load_model(self) -> None:
        """Load model and metadata lazily."""
        if self._loaded:
            return

        weights_file = self.model_dir / self.MODEL_FILE
        tags_file = self.model_dir / self.TAGS_FILE
        char_ip_file = self.model_dir / self.CHAR_IP_FILE

        if not weights_file.exists():
            raise FileNotFoundError(f"Model file not found: {weights_file}")
        if not tags_file.exists():
            raise FileNotFoundError(f"Tags file not found: {tags_file}")

        # Load tags
        with open(tags_file, "r", encoding="utf-8") as f:
            tag_info = json.load(f)
        tag_map = tag_info["tag_map"]
        tag_split = tag_info["tag_split"]
        self._gen_tag_count = tag_split["gen_tag_count"]
        self._index_to_tag = {v: k for k, v in tag_map.items()}

        # Load character IP mapping
        if char_ip_file.exists():
            with open(char_ip_file, "r", encoding="utf-8") as f:
                self._char_ip_map = json.load(f)
        else:
            self._char_ip_map = {}

        # Build model
        encoder = timm.create_model(
            "hf_hub:SmilingWolf/wd-eva02-large-tagger-v3",
            pretrained=False,
        )
        encoder.reset_classifier(0)
        decoder = TaggingHead(1024, 13461)
        model = nn.Sequential(encoder, decoder)

        # Load weights
        state_dict = torch.load(
            weights_file, map_location=self.device, weights_only=True
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        # Convert to fp16 if enabled
        if self.use_fp16:
            model = model.half()

        self._model = model

        # Transform
        self._transform = T.Compose(
            [
                T.Resize((448, 448)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self._loaded = True
        logger.info(f"PixAI Tagger loaded on {self.device} (fp16={self.use_fp16})")

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

    def _process_probs(
        self,
        probs: torch.Tensor,
        general_threshold: float | None = None,
        character_threshold: float | None = None,
    ) -> TagResult:
        """Process probability tensor into tag result."""
        gen_thresh = general_threshold or self.general_threshold
        char_thresh = character_threshold or self.character_threshold

        # Apply thresholds
        general_mask = probs[: self._gen_tag_count] > gen_thresh
        character_mask = probs[self._gen_tag_count :] > char_thresh

        # Get indices
        general_indices = general_mask.nonzero(as_tuple=True)[0]
        character_indices = (
            character_mask.nonzero(as_tuple=True)[0] + self._gen_tag_count
        )

        combined_indices = torch.cat((general_indices, character_indices)).cpu()

        # Map to tags
        feature_tags = []
        character_tags = []

        for idx in combined_indices:
            idx_val = idx.item()
            tag = self._index_to_tag[idx_val]
            if idx_val < self._gen_tag_count:
                feature_tags.append(tag)
            else:
                character_tags.append(tag)

        # Get IP tags
        ip_tags = []
        for tag in character_tags:
            if tag in self._char_ip_map:
                ip_tags.extend(self._char_ip_map[tag])
        ip_tags = sorted(set(ip_tags))

        return TagResult(
            feature_tags=feature_tags,
            character_tags=character_tags,
            ip_tags=ip_tags,
        )

    def tag(
        self,
        image: Image.Image | str | Path,
        general_threshold: float | None = None,
        character_threshold: float | None = None,
    ) -> TagResult:
        """
        Tag a single image.

        Args:
            image: PIL Image or path to image
            general_threshold: Override default general threshold
            character_threshold: Override default character threshold

        Returns:
            TagResult with feature, character, and IP tags
        """
        self._load_model()

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        image = self._pil_to_rgb(image)
        tensor = self._transform(image).unsqueeze(0)

        if self.use_fp16:
            tensor = tensor.half()

        tensor = tensor.to(self.device)

        with torch.no_grad(), torch.amp.autocast(self.device, enabled=self.use_fp16):
            probs = self._model(tensor)[0]

        return self._process_probs(probs, general_threshold, character_threshold)

    def tag_batch(
        self,
        images: list[Image.Image | str | Path],
        batch_size: int = 8,
        general_threshold: float | None = None,
        character_threshold: float | None = None,
    ) -> list[TagResult]:
        """
        Tag multiple images in batches.

        Args:
            images: List of PIL Images or paths
            batch_size: Batch size for inference
            general_threshold: Override default general threshold
            character_threshold: Override default character threshold

        Returns:
            List of TagResults in same order as input
        """
        self._load_model()

        results = []

        # Process in batches
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

            with (
                torch.no_grad(),
                torch.amp.autocast(self.device, enabled=self.use_fp16),
            ):
                batch_probs = self._model(batch_tensor)

            # Process each result
            for probs in batch_probs:
                results.append(
                    self._process_probs(probs, general_threshold, character_threshold)
                )

        return results

    def unload(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("PixAI Tagger unloaded")
