# KohakuCaption

A comprehensive image captioning pipeline for text-to-image (T2I) projects. Generates structured captions using multimodal LLM APIs (OpenAI, OpenRouter) with integrated anime image tagging support.

## Features

- **MLLM Captioning**: Generate structured captions via OpenAI or OpenRouter APIs
- **Structured Output**: Pydantic model validation, JSON Schema, regex patterns
- **Image Tagging**: PixAI Tagger v0.9 and AnimeTIMM for anime-style images
- **Template Engine**: Flexible context formatting with variable substitution
- **Batch Processing**: Async concurrent processing with rate limiting
- **Token Counting**: Count tokens using HuggingFace tokenizers

## Installation

```bash
pip install kohaku-caption
```

Or install from source:

```bash
git clone https://github.com/KohakuBlueleaf/KohakuCaption.git
cd KohakuCaption
pip install -e .
```

### Optional Dependencies

```bash
# For JSON Schema validation
pip install kohaku-caption[jsonschema]

# For development
pip install kohaku-caption[dev]
```

## Quick Start

### Caption a Single Image

```python
import asyncio
from kohakucaption import OpenAIClient, ClientConfig, CaptionPipeline, BasicCaption

async def main():
    config = ClientConfig(
        api_key="sk-...",
        model="gpt-4o",
    )

    async with OpenAIClient(config) as client:
        pipeline = CaptionPipeline(client)

        # Simple caption
        result = await pipeline.caption("image.png")
        print(result.content)

        # Structured caption with Pydantic validation
        result = await pipeline.caption_structured(
            "image.png",
            prompt=None,  # Uses default template
            output_model=BasicCaption,
        )
        if result.success:
            print(f"Title: {result.content.title}")
            print(f"Description: {result.content.description}")

asyncio.run(main())
```

### Using OpenRouter

```python
from kohakucaption import OpenRouterClient, ClientConfig

config = ClientConfig(
    api_key="sk-or-...",  # OpenRouter API key
    model="openai/gpt-4o",  # Provider/model format
)

client = OpenRouterClient(config)
```

### Tag Images with PixAI Tagger

```python
from kohakucaption import PixAITagger

# Download model first: python scripts/download_pixai_tagger.py
tagger = PixAITagger("pixai-tagger/")

result = tagger.tag("anime_image.png")
print(f"Features: {result.feature_tags}")
print(f"Characters: {result.character_tags}")
print(f"IPs/Series: {result.ip_tags}")
```

### Tag Images with AnimeTIMM

```python
from kohakucaption import AnimeTimmTagger

# Downloads model automatically from HuggingFace Hub
tagger = AnimeTimmTagger("animetimm/caformer_b36.dbv4-full")

result = tagger.tag("anime_image.png")
print(f"General: {list(result.general_tags.keys())}")
print(f"Characters: {list(result.character_tags.keys())}")
print(f"Rating: {list(result.rating_tags.keys())}")
```

## Command Line Usage

All CLI tools use Click with comprehensive help. Run any command with `--help` for details.

### Caption Single Image

```bash
# Basic usage with OpenAI
export OPENAI_API_KEY="sk-..."
python examples/caption_single_image.py image.png

# Using OpenRouter with a specific model
python examples/caption_single_image.py image.png -p openrouter -m anthropic/claude-sonnet-4

# Save output to file with high detail
python examples/caption_single_image.py image.png -o caption.txt --detail high

# See all options
python examples/caption_single_image.py --help
```

### Tag Single Image

```bash
# PixAI Tagger
python examples/tag_single_image.py image.png --tagger-dir ./pixai-tagger

# With JSON output and lower threshold for more tags
python examples/tag_single_image.py image.png -f json --general-threshold 0.2

# AnimeTIMM Tagger (auto-downloads model)
python examples/tag_single_image_animetimm.py image.png

# With confidence scores
python examples/tag_single_image_animetimm.py image.png -f scores

# See all options
python examples/tag_single_image.py --help
python examples/tag_single_image_animetimm.py --help
```

### Batch Processing

The batch processor supports two tagging modes with full parallelization:

```bash
# Caption only
python scripts/batch_caption.py ./images

# Caption + tags in parallel mode (fastest)
# - Tagger and LLM API run simultaneously
# - Both results merged into single JSON per image
python scripts/batch_caption.py ./images --with-tags --mode parallel

# Caption with tags as context
# - Tagger runs first
# - Tags sent to LLM as context for better captions
# - Both results stored in output JSON
python scripts/batch_caption.py ./images --with-tags --mode context

# Using OpenRouter with high concurrency
python scripts/batch_caption.py ./images \
    -p openrouter \
    -m openai/gpt-4o \
    --max-concurrent 10 \
    --with-tags \
    --skip-existing

# See all options
python scripts/batch_caption.py --help
```

**Output format** (JSON per image):
```json
{
  "image": "filename.png",
  "tags": {
    "feature_tags": ["1girl", "blue_hair", "school_uniform"],
    "character_tags": ["hatsune_miku"],
    "ip_tags": ["vocaloid"]
  },
  "caption": {
    "aesthetic_score": 0.85,
    "nsfw_score": 0.0,
    "quality_score": 0.9,
    "title": "Anime schoolgirl portrait",
    "brief": "A blue-haired anime girl in school uniform.",
    "description": "A digital illustration featuring..."
  }
}
```

### Download Tagger Models

```bash
# Download PixAI Tagger (gated model, requires HF login)
python scripts/download_pixai_tagger.py -o ./pixai-tagger

# Force re-download
python scripts/download_pixai_tagger.py --force

# See all options
python scripts/download_pixai_tagger.py --help
```

## Output Formats

### Default Format

```
# aesthetic_score
0.85

# nsfw_score
0.0

# quality_score
0.9

# title
Sunset over mountains

# brief
A scenic mountain landscape at sunset.

# description
Golden sunlight illuminates snow-capped peaks...
```

### JSON Format

```json
{
  "aesthetic_score": 0.85,
  "nsfw_score": 0.0,
  "quality_score": 0.9,
  "title": "Sunset over mountains",
  "brief": "A scenic mountain landscape at sunset.",
  "description": "Golden sunlight illuminates snow-capped peaks..."
}
```

## Architecture

```
kohakucaption/
├── clients/           # MLLM API clients
│   ├── base.py       # Base client with retry logic
│   ├── openai.py     # OpenAI client
│   └── openrouter.py # OpenRouter client
├── context/           # Context formatting
│   ├── providers.py  # Context providers (metadata, bbox, tags, etc.)
│   └── template.py   # Template engine
├── formats/           # Output formats
│   ├── default.py    # # key / value format
│   └── json.py       # JSON format
├── pipeline/          # Captioning pipeline
│   └── caption.py    # Main pipeline with validation
├── tagger/            # Image taggers
│   ├── pixai.py      # PixAI Tagger v0.9
│   └── animetimm.py  # AnimeTIMM tagger
├── validation/        # Output validators
│   └── validators.py # Pydantic, JSON Schema, regex validators
├── tokenizer.py       # Token counting utilities
└── types.py           # Core types and schemas
```

## Context Providers

Add structured context to captioning requests:

```python
from kohakucaption import CaptionRequest, ContextType, ImageInput

request = CaptionRequest(image=ImageInput(source="image.png"))

# Add metadata
request.add_context(ContextType.METADATA, {
    "width": 1920,
    "height": 1080,
    "format": "PNG",
})

# Add detected objects
request.add_context(ContextType.OBJECT_BBOX, {
    "boxes": [
        {"x": 100, "y": 200, "width": 50, "height": 80, "label": "person"},
    ]
})

# Add existing tags
request.add_context(ContextType.TAGS, {
    "tags": ["landscape", "sunset", "mountains"],
})
```

## Validators

### Pydantic Validator

```python
from pydantic import BaseModel, Field
from kohakucaption import PydanticValidator

class CustomCaption(BaseModel):
    title: str = Field(description="Image title")
    tags: list[str] = Field(description="Relevant tags")

validator = PydanticValidator(CustomCaption)
result = validator.validate(llm_output)
```

### JSON Schema Validator

```python
from kohakucaption import JsonSchemaValidator

schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["title", "score"],
}

validator = JsonSchemaValidator(schema)
```

### Regex Validator

```python
from kohakucaption import RegexValidator

validator = RegexValidator(
    pattern=r"Title: (.+)",
    extract_group=1,
)
```

## Token Counting

```python
from kohakucaption import TokenCounter, count_tokens

# Simple count
tokens = count_tokens("Hello, world!")

# Count caption fields
counter = TokenCounter("Qwen/Qwen3-0.6B")
caption = {"title": "Test", "description": "A test image"}
counts = counter.count_fields(caption)
print(f"Title: {counts['title']} tokens")
print(f"Description: {counts['description']} tokens")
```

## Downloading Tagger Models

### PixAI Tagger v0.9

```bash
python scripts/download_pixai_tagger.py -o ./pixai-tagger
```

Note: PixAI Tagger is a gated model. You need to:
1. Create a HuggingFace account
2. Accept the license at https://huggingface.co/pixai-labs/pixai-tagger-v0.9
3. Login with `huggingface-cli login`

### AnimeTIMM

AnimeTIMM models are downloaded automatically from HuggingFace Hub on first use.

## Requirements

- Python 3.10+
- PyTorch
- transformers
- timm
- openai
- pydantic
- pillow
- pandas
- huggingface_hub
- aiofiles (for batch processing)
- click (for CLI)

## License

Apache-2.0

## Author

Shih-Ying Yeh (KohakuBlueLeaf) - apolloyeh0123@gmail.com
