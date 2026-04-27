"""
Step 4: Render a block letter (default "S") as a grayscale mask for the selection-bias meme.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _candidate_bold_font_paths() -> list[str]:
    return [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        r"C:\Windows\Fonts\timesbd.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
    ]


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _candidate_bold_font_paths():
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_fits(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_w: int,
    max_h: int,
) -> bool:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    return tw <= max_w and th <= max_h


def create_block_letter_s(
    height: int,
    width: int,
    letter: str = "S",
    font_size_ratio: float = 0.9,
) -> np.ndarray:
    """
    Draw a single block letter on a white canvas and return it as a float grayscale image.

    Parameters
    ----------
    height, width : int
        Output array shape (rows, columns).
    letter : str
        Character to render (default ``"S"``).
    font_size_ratio : float
        Target scale: the glyph is sized to fit within this fraction of the image
        width and height (with a small margin).

    Returns
    -------
    np.ndarray
        2D array of shape (height, width) with values in [0, 1]; letter pixels are 0,
        background is 1.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    text = letter[:1] if letter else "S"

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    margin = max(2, min(width, height) // 64)
    max_w = max(1, int(width * font_size_ratio) - 2 * margin)
    max_h = max(1, int(height * font_size_ratio) - 2 * margin)

    upper = max(1, int(min(width, height) * font_size_ratio))
    start_size = min(upper, 2048)
    font = _load_font(8)
    for size in range(start_size, 0, -1):
        candidate = _load_font(size)
        if _text_fits(draw, text, candidate, max_w, max_h):
            font = candidate
            break

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (width - tw) / 2.0 - bbox[0]
    y = (height - th) / 2.0 - bbox[1]

    draw.text((x, y), text, fill=(0, 0, 0), font=font)

    gray = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
    return gray
