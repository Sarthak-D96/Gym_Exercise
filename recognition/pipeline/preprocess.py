"""4.1 Preprocess Frames — input: video frames, output: processed frames."""

import base64
import binascii
import io

from PIL import Image, UnidentifiedImageError


class FrameDecodeError(ValueError):
    pass


def decode_data_url(data: str) -> bytes:
    """Accept a `data:image/jpeg;base64,...` URL (as produced by canvas.toDataURL) or raw base64."""
    if "," in data and data.startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FrameDecodeError("Frame is not valid base64") from exc


def preprocess_frame(raw: bytes, size: int = 256) -> Image.Image:
    """
    Decode, convert to RGB and letterbox-resize to `size` x `size`.

    Pose models (MoveNet, MediaPipe, YOLO-pose) expect a fixed square RGB input,
    so this output can be fed to them directly once a real detector is plugged in.
    """
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise FrameDecodeError("Frame is not a decodable image") from exc
    img = img.convert("RGB")
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size))
    left, top = (size - img.width) // 2, (size - img.height) // 2
    canvas.paste(img, (left, top))
    # Where the real picture sits inside the square, so a pose detector can map
    # keypoints back to 0..1 of the *original* frame (the contract in ml_models/README.md).
    canvas.info["content_box"] = (left, top, img.width, img.height)
    return canvas
