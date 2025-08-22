from PIL import Image
import io
import requests


def load_image(src):
    """Load image from various sources and return PIL Image."""
    # bytes -> PIL
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(src))
    # URL -> PIL
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        return Image.open(requests.get(src, stream=True).raw)
    # file path -> PIL
    return Image.open(src)
