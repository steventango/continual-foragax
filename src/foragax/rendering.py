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

    # Create border mask (2-pixel thick borders) - vectorized like grid lines
    height, width = grid_size
    img_height, img_width = height * 24, width * 24

    border_mask = jnp.zeros((img_height, img_width), dtype=bool)

    # Create border row and column indices for all cells at once
    cell_rows = jnp.arange(height)
    cell_cols = jnp.arange(width)

    # Top border rows: 2 rows per cell
    top_border_rows = cell_rows[:, None] * 24 + jnp.arange(2)[None, :]
    top_border_rows_flat = top_border_rows.flatten()

    # Bottom border rows: 2 rows per cell
    bottom_border_rows = cell_rows[:, None] * 24 + 22 + jnp.arange(2)[None, :]
    bottom_border_rows_flat = bottom_border_rows.flatten()

    # Left border columns: 2 columns per cell
    left_border_cols = cell_cols[:, None] * 24 + jnp.arange(2)[None, :]
    left_border_cols_flat = left_border_cols.flatten()

    # Right border columns: 2 columns per cell
    right_border_cols = cell_cols[:, None] * 24 + 22 + jnp.arange(2)[None, :]
    right_border_cols_flat = right_border_cols.flatten()

    # Set top and bottom borders (full width rectangles)
    all_border_rows = jnp.concatenate([top_border_rows_flat, bottom_border_rows_flat])
    border_mask = border_mask.at[all_border_rows, :].set(True)

    # Set left and right borders (full height rectangles)
    all_border_cols = jnp.concatenate([left_border_cols_flat, right_border_cols_flat])
    border_mask = border_mask.at[:, all_border_cols].set(True)

    # Apply border mask: use HSV border colors for border pixels, base colors elsewhere
    result_img = jnp.where(border_mask[..., None], border_img, base_img)
    return result_img
