# InternVL Precomputed Features

This guide shows how to use precomputed vision features with InternVL models in SGLang, which can save computation time when processing the same images multiple times.

## Basic Usage

### 1. Regular Image Processing (Existing Functionality)

```python
from sglang import Engine
from PIL import Image
import requests
from io import BytesIO

# Load model
engine = Engine(
    model_path="OpenGVLab/InternVL2_5-2B",
    chat_template="internvl-2-5",
    trust_remote_code=True,
)

# Load image
image = Image.open(
    BytesIO(
        requests.get(
            "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
        ).content
    )
)

# Generate response
output = engine.generate(
    prompt="What's in this image? <IMG_CONTEXT>",
    image_data=[image],
    sampling_params={"temperature": 0.0, "max_new_tokens": 100},
)
print(output["text"])
```

### 2. Using Precomputed Features

```python
from sglang import Engine
from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image
import requests
from io import BytesIO

# Load model and processor
model_path = "OpenGVLab/InternVL2_5-2B"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
hf_model = AutoModel.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
).eval()

# Load image
image = Image.open(
    BytesIO(
        requests.get(
            "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
        ).content
    )
)

# Precompute vision features using HuggingFace
processed = processor(
    images=[image], 
    text="What's in this image? <IMG_CONTEXT>", 
    return_tensors="pt"
)

with torch.inference_mode():
    # Extract vision features (this matches InternVL's extract_feature method)
    vit_embeds = hf_model.vision_model(processed["pixel_values"])
    precomputed_features = hf_model.mlp1(vit_embeds)

# Create multimodal item with precomputed features
mm_item = {
    "modality": "IMAGE",
    "precomputed_features": precomputed_features,
}

# Use with SGLang engine
engine = Engine(
    model_path=model_path,
    chat_template="internvl-2-5",
    trust_remote_code=True,
)

output = engine.generate(
    input_ids=processed["input_ids"][0].tolist(),
    image_data=[mm_item],
    sampling_params={"temperature": 0.0, "max_new_tokens": 100},
)
print(output["text"])
```

## Benefits

1. **Performance**: Skip vision processing in SGLang when features are already computed
2. **Caching**: Precompute features once and reuse them for multiple queries
3. **Disaggregated Setup**: Compute vision features on different hardware/processes
4. **Flexibility**: Mix precomputed and regular images in the same request

## Technical Details

The implementation supports:
- **Automatic Detection**: SGLang automatically detects when images are preprocessed (dictionary format)
- **Mixed Inputs**: All images in a request must be either preprocessed or regular (no mixing)
- **Backward Compatibility**: Existing code continues to work without changes
- **Error Handling**: Clear error messages for invalid input combinations

## Format Requirements

Precomputed features must be provided as dictionaries with:
- `"modality": "IMAGE"` (required)
- `"precomputed_features": torch.Tensor` (the computed vision features)
- Optional: `"pixel_values": torch.Tensor` (raw pixel values as fallback)

The precomputed features should match the output of InternVL's vision processing pipeline:
1. Vision model forward pass: `vit_embeds = vision_model(pixel_values)`
2. MLP projection: `features = mlp1(vit_embeds)`
