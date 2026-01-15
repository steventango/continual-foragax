"""Rendering utilities for Foragax environments."""

from typing import Tuple

import jax
import jax.numpy as jnp

from foragax.colors import hsv_to_rgb_255


def apply_true_borders(
    base_img: jax.Array,
    true_grid: jax.Array,
    grid_size: Tuple[int, int],
    num_objects: int,
) -> jax.Array:
    """Apply true object borders by overlaying HSV border colors on border pixels.

    Args:
        base_img: Base image with object colors
        true_grid: Grid of object IDs for determining border colors
        grid_size: (height, width) of the grid
        num_objects: Number of object types

    Returns:
        Image with HSV borders overlaid on border pixels
    """
    # Create HSV border colors for each object type
    hues = jnp.linspace(0, 1, num_objects, endpoint=False)

    # Convert HSV to RGB for border colors
    border_colors = hsv_to_rgb_255(hues[true_grid])

    # Resize border colors to match rendered image size
    border_img = jax.image.resize(
        border_colors,
        (grid_size[0] * 24, grid_size[1] * 24, 3),
        jax.image.ResizeMethod.NEAREST,
    )

    # Create border mask (2-pixel thick borders) using vectorized modulo operations
    img_height, img_width = grid_size[0] * 24, grid_size[1] * 24
    y_idx = jnp.arange(img_height) % 24
    x_idx = jnp.arange(img_width) % 24

    # Border pixels are those with offset 0, 1, 22, or 23 within each 24x24 cell
    is_border_row = (y_idx < 2) | (y_idx >= 22)
    is_border_col = (x_idx < 2) | (x_idx >= 22)
    border_mask = is_border_row[:, None] | is_border_col[None, :]

    # Apply border mask: use HSV border colors for border pixels, base colors elsewhere
    result_img = jnp.where(border_mask[..., None], border_img, base_img)
    return result_img
