import io
import logging

import requests
from PIL import Image, ImageFile

# PIL safety: prevent crashes on large/truncated images
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


def load_image(src) -> Image.Image:
    """Load image from various sources and return PIL Image in RGB."""
    # bytes -> PIL
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(src)).convert("RGB")
    # URL -> PIL
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        resp = requests.get(
            src, stream=True, headers={"User-Agent": "leap-finetune"}, timeout=15
        )
        resp.raise_for_status()
        return Image.open(resp.raw).convert("RGB")
    # file path -> PIL
    return Image.open(src).convert("RGB")


def is_image_loadable(src: str) -> bool:
    """Check if an image source can be loaded without error."""
    try:
        img = load_image(src)
        img.close()
        return True
    except Exception:
        return False
