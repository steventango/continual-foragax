"""Color utility functions for Foragax."""

import jax
import jax.numpy as jnp


def hsv_to_rgb(h: jax.Array, s: float = 1.0, v: float = 1.0) -> jax.Array:
    """Convert HSV color values to RGB.

    Args:
        h: Hue values in range [0, 1]
        s: Saturation value in range [0, 1], default 1.0
        v: Value (brightness) in range [0, 1], default 1.0

    Returns:
        RGB values as array of shape (..., 3) with values in range [0, 1]
    """
    c = v * s
    x = c * (1 - jnp.abs(jnp.mod(h * 6, 2) - 1))
    m = v - c

    # Create RGB arrays
    r = jnp.zeros_like(h)
    g = jnp.zeros_like(h)
    b = jnp.zeros_like(h)

    # Sector 0: 0-60 degrees (red to yellow)
    mask0 = h < 1 / 6
    r = jnp.where(mask0, c, r)
    g = jnp.where(mask0, x, g)

    # Sector 1: 60-120 degrees (yellow to green)
    mask1 = (h >= 1 / 6) & (h < 2 / 6)
    r = jnp.where(mask1, x, r)
    g = jnp.where(mask1, c, g)

    # Sector 2: 120-180 degrees (green to cyan)
    mask2 = (h >= 2 / 6) & (h < 3 / 6)
    g = jnp.where(mask2, c, g)
    b = jnp.where(mask2, x, b)

    # Sector 3: 180-240 degrees (cyan to blue)
    mask3 = (h >= 3 / 6) & (h < 4 / 6)
    g = jnp.where(mask3, x, g)
    b = jnp.where(mask3, c, b)

    # Sector 4: 240-300 degrees (blue to magenta)
    mask4 = (h >= 4 / 6) & (h < 5 / 6)
    r = jnp.where(mask4, x, r)
    b = jnp.where(mask4, c, b)

    # Sector 5: 300-360 degrees (magenta to red)
    mask5 = h >= 5 / 6
    r = jnp.where(mask5, c, r)
    b = jnp.where(mask5, x, b)

    # Add value offset
    rgb = jnp.stack([r + m, g + m, b + m], axis=-1)
    return rgb


def hsv_to_rgb_255(h: jax.Array, s: float = 0.9, v: float = 0.8) -> jax.Array:
    """Convert HSV to RGB with values scaled to 0-255 range for image rendering.

    Args:
        h: Hue values in range [0, 1]
        s: Saturation value, default 0.9
        v: Value (brightness), default 0.8

    Returns:
        RGB values as uint8 array of shape (..., 3) with values in range [0, 255]
    """
    rgb = hsv_to_rgb(h, s, v)
    return (rgb * 255).astype(jnp.uint8)
