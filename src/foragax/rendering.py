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


def reward_to_color(reward: jax.Array) -> jax.Array:
    """Convert reward value to RGB color using diverging gradient.

    Args:
        reward: Reward value (typically -1 to +1)

    Returns:
        RGB color array with shape (..., 3) and dtype uint8
    """
    # Diverging gradient: +1 = green (0, 255, 0), 0 = white (255, 255, 255), -1 = magenta (255, 0, 255)
    # Clamp reward to [-1, 1] range for color mapping
    reward_clamped = jnp.clip(reward, -1.0, 1.0)

    # For positive rewards: interpolate from white to green
    # For negative rewards: interpolate from white to magenta
    # At reward = 0: white (255, 255, 255)
    # At reward = +1: green (0, 255, 0)
    # At reward = -1: magenta (255, 0, 255)

    red_component = jnp.where(
        reward_clamped >= 0,
        (1 - reward_clamped) * 255,  # Fade from white to green: 255 -> 0
        255,  # Stay at 255 for all negative rewards
    )

    green_component = jnp.where(
        reward_clamped >= 0,
        255,  # Stay at 255 for all positive rewards
        (1 + reward_clamped) * 255,  # Fade from white to magenta: 255 -> 0
    )

    blue_component = jnp.where(
        reward_clamped >= 0,
        (1 - reward_clamped) * 255,  # Fade from white to green: 255 -> 0
        255,  # Stay at 255 for all negative rewards
    )

    return jnp.stack([red_component, green_component, blue_component], axis=-1).astype(
        jnp.uint8
    )


def get_base_image(
    object_id: jax.Array,
    color_state: jax.Array,
    object_colors: jax.Array,
    dynamic_biomes: bool,
) -> jax.Array:
    """Construct base RGB image from object IDs or colors."""
    if dynamic_biomes:
        # Use per-instance colors from state
        img = color_state.copy()
        # Mask empty cells (object_id == 0) to white
        empty_mask = object_id == 0
        white_color = jnp.array([255, 255, 255], dtype=jnp.uint8)
        img = jnp.where(empty_mask[..., None], white_color, img)
    else:
        # Map object IDs to colors
        img = object_colors[object_id]

    return img.astype(jnp.uint8)


def apply_grid_lines(
    img: jax.Array,
    grid_size: Tuple[int, int],
    grid_color: jax.Array,
    cell_size: int = 24,
) -> jax.Array:
    """Apply grid lines to the image."""
    row_grid = (jnp.arange(grid_size[0] * cell_size) % cell_size) == 0
    col_grid = (jnp.arange(grid_size[1] * cell_size) % cell_size) == 0
    # skip first rows/cols as they are borders or managed by caller
    row_grid = row_grid.at[0].set(False)
    col_grid = col_grid.at[0].set(False)
    grid_mask = row_grid[:, None] | col_grid[None, :]
    return jnp.where(grid_mask[..., None], grid_color, img)


def apply_reward_overlay(
    base_img: jax.Array,
    reward_colors: jax.Array,
    reward_grid: jax.Array,
    grid_size: Tuple[int, int],
) -> jax.Array:
    """Apply reward visualization overlay (center dots) to the image.

    Only applies dots where the reward is non-zero (abs > 1e-5).

    Args:
        base_img: Base image at 3x scale (each cell is 3x3)
        reward_colors: Array of RGB colors for rewards
        reward_grid: Grid of reward values
        grid_size: (height, width) of the grid

    Returns:
        Image with reward dots overlaid
    """
    # Create a 3x3 pattern mask for center pixels
    cell_mask = jnp.array(
        [[False, False, False], [False, True, False], [False, False, False]]
    )
    grid_reward_mask = jnp.tile(cell_mask, grid_size)

    # Only show reward where reward is meaningfully non-zero
    reward_nonzero = jnp.abs(reward_grid) > 1e-5
    # Expand to 3x scale
    reward_nonzero_x3 = jnp.repeat(jnp.repeat(reward_nonzero, 3, axis=0), 3, axis=1)

    # Final mask: center pixel of a cell AND cell has a non-zero reward
    composite_mask = grid_reward_mask & reward_nonzero_x3

    # Repeat reward colors to 3x to match image scale
    reward_colors_x3 = jnp.repeat(jnp.repeat(reward_colors, 3, axis=0), 3, axis=1)

    return jnp.where(composite_mask[..., None], reward_colors_x3, base_img)


def apply_hint_bottom_bar(
    img: jax.Array,
    hint_vector: jax.Array,
    bar_height: int = 12,
    separator_height: int = 2,
) -> jax.Array:
    """Append a black separator and a bottom bar showing the binary hint vector.

    Args:
        img: RGB image array of shape (H, W, 3).
        hint_vector: Binary vector of shape (N,) to visualize.
        bar_height: Height of the hint bar segments.
        separator_height: Height of the black line separator.

    Returns:
        New image of shape (H + separator_height + bar_height, W, 3) with the hint overlay.
    """
    H, W, C = img.shape
    num_bins = hint_vector.shape[0]

    # Create the black separator line
    separator = jnp.zeros((separator_height, W, C), dtype=jnp.uint8)

    # Calculate widths for each hint segment
    segment_width = W // num_bins

    # Expand hint vector to create blocks of colors:
    # 0 -> Black (0,0,0), 1 -> White (255,255,255)
    hint_colors = hint_vector[:, None] * 255
    hint_colors = jnp.tile(hint_colors, (1, 3)).astype(jnp.uint8)  # (N, 3)

    # Repeat colors across width
    hint_bar = jnp.repeat(hint_colors, segment_width, axis=0)  # (segment_width * N, 3)

    # Handle remainder if W is not perfectly divisible by num_bins
    remainder = W - (segment_width * num_bins)
    if remainder > 0:
        last_color = jnp.expand_dims(hint_colors[-1], axis=0)
        padding = jnp.repeat(last_color, remainder, axis=0)
        hint_bar = jnp.concatenate([hint_bar, padding], axis=0)

    # Expand to height
    hint_bar = jnp.expand_dims(hint_bar, axis=0)  # (1, W, 3)
    hint_bar = jnp.repeat(hint_bar, bar_height, axis=0)  # (bar_height, W, 3)

    # Concatenate vertically
    return jnp.concatenate([img, separator, hint_bar], axis=0)
