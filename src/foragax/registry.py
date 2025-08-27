"""Factory functions for creating Foragax environment variants."""

from typing import Any, Dict, Optional, Tuple

from foragax.env import (
    Biome,
    ForagaxEnv,
    ForagaxObjectEnv,
    ForagaxRGBEnv,
    ForagaxWorldEnv,
)
from foragax.objects import LARGE_MOREL, LARGE_OYSTER

ENV_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ForagaxTwoBiomeSmall": {
        "size": (16, 8),
        "aperture_size": (5, 5),
        "objects": (LARGE_MOREL, LARGE_OYSTER),
        "biomes": (
            # Morel biome
            Biome(start=(2, 2), stop=(6, 6), object_frequencies=(1.0, 0.0)),
            # Oyster biome
            Biome(start=(10, 2), stop=(14, 6), object_frequencies=(0.0, 1.0)),
        ),
    }
}


def make(
    env_id: str,
    observation_type: str = "object",
    aperture_size: Optional[Tuple[int, int]] = None,
) -> ForagaxEnv:
    """Create a Foragax environment.

    Args:
        env_id: The ID of the environment to create.
        observation_type: The type of observation to use. One of "object", "rgb", or "world".
        aperture_size: The size of the agent's observation aperture. If None, the default
            for the environment is used.

    Returns:
        A Foragax environment instance.
    """
    if env_id not in ENV_CONFIGS:
        raise ValueError(f"Unknown env_id: {env_id}")

    config = ENV_CONFIGS[env_id].copy()

    if aperture_size is not None:
        config["aperture_size"] = aperture_size

    env_class_map = {
        "object": ForagaxObjectEnv,
        "rgb": ForagaxRGBEnv,
        "world": ForagaxWorldEnv,
    }

    if observation_type not in env_class_map:
        raise ValueError(f"Unknown observation type: {observation_type}")

    env_class = env_class_map[observation_type]

    return env_class(**config)
