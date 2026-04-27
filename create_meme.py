"""
Assemble the four-panel statistics meme (selection bias visualization).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def _to_gray_float(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=2)
    elif arr.ndim != 2:
        raise ValueError("Expected a 2D grayscale image or a 3D array")
    return np.clip(arr, 0.0, 1.0)


def _resize_to_match(img: np.ndarray, height: int, width: int) -> np.ndarray:
    g = _to_gray_float(img)
    if g.shape == (height, width):
        return g
    pil = Image.fromarray((g * 255.0).astype(np.uint8), mode="L")
    pil = pil.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.float64) / 255.0


def create_statistics_meme(
    original_img: np.ndarray,
    stipple_img: np.ndarray,
    block_letter_img: np.ndarray,
    masked_stipple_img: np.ndarray,
    output_path: str,
    dpi: int = 150,
    background_color: str = "white",
) -> None:
    """
    Save a 1×4 figure labeled Reality | Your Model | Selection Bias | Estimate.

    All panels are resized to match ``original_img`` height and width so layouts align.
    """
    titles = ["Reality", "Your Model", "Selection Bias", "Estimate"]
    base = _to_gray_float(original_img)
    h, w = base.shape
    panels = [
        base,
        _resize_to_match(stipple_img, h, w),
        _resize_to_match(block_letter_img, h, w),
        _resize_to_match(masked_stipple_img, h, w),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), facecolor=background_color)
    fig.patch.set_facecolor(background_color)

    for ax, img, title in zip(np.ravel(axes), panels, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=13, fontweight="bold", color="#222222", pad=12)
        ax.axis("off")

    plt.tight_layout(pad=1.0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=background_color)
    plt.close(fig)
