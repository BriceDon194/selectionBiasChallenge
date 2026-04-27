"""
Step 5: Apply the block-letter mask to the stippled image (biased estimate).
"""

from __future__ import annotations

import numpy as np


def create_masked_stipple(
    stipple_img: np.ndarray,
    mask_img: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Where the mask is dark (below ``threshold``), remove stipples by setting pixels to white.

    Parameters
    ----------
    stipple_img, mask_img : np.ndarray
        Same 2D shape; ``mask_img`` uses [0, 1] with 0 = masked region, 1 = keep region.
    threshold : float
        Pixels with ``mask_img < threshold`` are treated as mask (cleared to 1.0).

    Returns
    -------
    np.ndarray
        Stipple image with masked regions filled with 1.0 (white).
    """
    if stipple_img.shape != mask_img.shape:
        raise ValueError("stipple_img and mask_img must have the same shape")
    return np.where(mask_img >= threshold, stipple_img, 1.0).astype(np.float64)
