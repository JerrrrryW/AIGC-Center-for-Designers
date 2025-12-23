import io
from typing import Dict, List, Optional

from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_ID = "Salesforce/blip-image-captioning-base"

_caption_processor = None
_caption_model = None
_caption_device = None


def get_caption_model(device: str):
    global _caption_processor, _caption_model, _caption_device
    if _caption_processor is None or _caption_model is None:
        _caption_processor = BlipProcessor.from_pretrained(MODEL_ID)
        _caption_model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
        _caption_model.eval()
        _caption_model.to(device)
        _caption_device = device
    elif _caption_device != device:
        _caption_model.to(device)
        _caption_device = device
    return _caption_processor, _caption_model


def caption_images(
    images: List[Dict[str, bytes]],
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    status_updater: Optional[dict] = None,
) -> Dict[str, str]:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor, model = get_caption_model(device)

    total = len(images)
    results: Dict[str, str] = {}

    for index, item in enumerate(images, start=1):
        if status_updater:
            status_updater.update(
                {
                    "status": "processing",
                    "progress": round((index - 1) / total * 100, 2) if total else 0,
                    "message": f"Captioning {index}/{total}...",
                    "results": results,
                }
            )
        image = Image.open(io.BytesIO(item["content"])).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()

        if prefix:
            caption = f"{prefix.strip()} {caption}".strip()
        if suffix:
            caption = f"{caption} {suffix.strip()}".strip()

        results[item["filename"]] = caption

    if status_updater:
        status_updater.update(
            {
                "status": "completed",
                "progress": 100,
                "message": "Captioning complete.",
                "results": results,
            }
        )

    return results
