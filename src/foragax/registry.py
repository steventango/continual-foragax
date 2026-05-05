"""Factory functions for creating Foragax environment variants."""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

from foragax.env import (
    Biome,
    ForagaxEnv,
)
from foragax.objects import (
    WALL,
    create_fourier_objects,
    create_shift_square_wave_biome_objects,
)

BIG_WIDTH = 9
BIG_GAP = 5
BIG_OFFSET = BIG_GAP // 2 + 1
BIG_WALL_WIDTH = 1

ENV_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ForagaxBig-v5": {
        "size": (2 * (BIG_WIDTH + BIG_GAP), 2 * (BIG_WIDTH + BIG_GAP)),
        "aperture_size": None,
        "objects": None,
        "biomes": (
            Biome(
                start=(BIG_OFFSET, BIG_OFFSET),
                stop=(BIG_OFFSET + BIG_WIDTH, BIG_OFFSET + BIG_WIDTH),
                object_frequencies=(0.2, 0.0),
            ),
            Biome(
                start=(BIG_OFFSET + BIG_WIDTH + BIG_GAP, BIG_OFFSET),
                stop=(
                    BIG_OFFSET + 2 * BIG_WIDTH + BIG_GAP,
                    BIG_OFFSET + BIG_WIDTH,
                ),
                object_frequencies=(0.2, 0.0),
            ),
            Biome(
                start=(BIG_OFFSET, BIG_OFFSET + BIG_WIDTH + BIG_GAP),
                stop=(
                    BIG_OFFSET + BIG_WIDTH,
                    BIG_OFFSET + 2 * BIG_WIDTH + BIG_GAP,
                ),
                object_frequencies=(0.2, 0.0),
            ),
            Biome(
                start=(
                    BIG_OFFSET + BIG_WIDTH + BIG_GAP,
                    BIG_OFFSET + BIG_WIDTH + BIG_GAP,
                ),
                stop=(
                    BIG_OFFSET + 2 * BIG_WIDTH + BIG_GAP,
                    BIG_OFFSET + 2 * BIG_WIDTH + BIG_GAP,
                ),
                object_frequencies=(0.2, 0.0),
            ),
            Biome(
                start=(
                    BIG_OFFSET + BIG_WIDTH // 2 - BIG_WALL_WIDTH,
                    BIG_OFFSET + BIG_WIDTH // 2 - BIG_WALL_WIDTH,
                ),
                stop=(
                    BIG_OFFSET + BIG_WIDTH // 2 + 1 + BIG_WALL_WIDTH,
                    BIG_OFFSET + BIG_WIDTH // 2 + 1 + BIG_WALL_WIDTH,
                ),
                object_frequencies=(0.0, 0.6),
            ),
            Biome(
                start=(
                    BIG_OFFSET + BIG_WIDTH // 2 - BIG_WALL_WIDTH,
                    BIG_OFFSET + BIG_WIDTH + BIG_GAP + BIG_WIDTH // 2 - BIG_WALL_WIDTH,
                ),
                stop=(
                    BIG_OFFSET + BIG_WIDTH // 2 + 1 + BIG_WALL_WIDTH,
                    BIG_OFFSET
                    + BIG_WIDTH
                    + BIG_GAP
                    + BIG_WIDTH // 2
                    + 1
                    + BIG_WALL_WIDTH,
                ),
                object_frequencies=(0.0, 0.6),
            ),
            Biome(
                start=(
                    BIG_OFFSET + BIG_WIDTH + BIG_GAP + BIG_WIDTH // 2 - BIG_WALL_WIDTH,
                    BIG_OFFSET + BIG_WIDTH // 2 - BIG_WALL_WIDTH,
                ),
                stop=(
                    BIG_OFFSET
                    + BIG_WIDTH
                    + BIG_GAP
                    + BIG_WIDTH // 2
                    + 1
                    + BIG_WALL_WIDTH,
                    BIG_OFFSET + BIG_WIDTH // 2 + 1 + BIG_WALL_WIDTH,
                ),
                object_frequencies=(0.0, 0.6),
            ),
            Biome(
                start=(
                    BIG_OFFSET + BIG_WIDTH + BIG_GAP + BIG_WIDTH // 2 - BIG_WALL_WIDTH,
                    BIG_OFFSET + BIG_WIDTH + BIG_GAP + BIG_WIDTH // 2 - BIG_WALL_WIDTH,
                ),
                stop=(
                    BIG_OFFSET
                    + BIG_WIDTH
                    + BIG_GAP
                    + BIG_WIDTH // 2
                    + 1
                    + BIG_WALL_WIDTH,
                    BIG_OFFSET
                    + BIG_WIDTH
                    + BIG_GAP
                    + BIG_WIDTH // 2
                    + 1
                    + BIG_WALL_WIDTH,
                ),
                object_frequencies=(0.0, 0.6),
            ),
            Biome(
                start=(0, 0),
                stop=(2 * (BIG_WIDTH + BIG_GAP), 2 * (BIG_WIDTH + BIG_GAP)),
                object_frequencies=(0.0, 0.1),
            ),
        ),
        "nowrap": False,
        "deterministic_spawn": True,
        "dynamic_biomes": True,
        "biome_consumption_threshold": 10000,
        "dynamic_biome_spawn_empty": 1.0,
        "center_reward": True,
        "return_hint": True,
        "hint_duration": 10,
    },
    "ForagaxSquareWaveTwoBiome-v11": {
        "size": (24, 15),
        "aperture_size": None,
        "objects": None,
        "biomes": (
            # Void
            Biome(
                start=(0, 0),
                stop=(24, 15),
                object_frequencies=(0.0, 0.0, 0.0, 0.0, 4 / 60),
            ),
            # Biome 1 (left): Oyster +10, Death Cap -10 with square wave
            Biome(
                start=(4, 0),
                stop=(8, 15),
                object_frequencies=(15 / 60, 15 / 60, 0.0, 0.0, 4 / 60),
            ),
            # Biome 2 (right): Oyster -10, Death Cap +10 with inverted square wave
            Biome(
                start=(16, 0),
                stop=(20, 15),
                object_frequencies=(0.0, 0.0, 15 / 60, 15 / 60, 4 / 60),
            ),
        ),
        "nowrap": False,
        "deterministic_spawn": True,
    },
}


def make(
    env_id: str,
    observation_type: str = "color",
    aperture_size: Optional[Union[Tuple[int, int], int]] = (5, 5),
    reward_delay: int = 0,
    random_shift_max_steps: int = 0,
    **kwargs: Any,
) -> ForagaxEnv:
    """Create a Foragax environment.

    Args:
        env_id: The ID of the environment to create.
        observation_type: The type of observation to use. One of "object", "rgb", or "color".
        aperture_size: The size of the agent's observation aperture. If -1, full world observation.
            If None, the default for the environment is used.
        reward_delay: Number of steps required to digest food items.
        **kwargs: Additional keyword arguments to pass to the ForagaxEnv constructor.

    Returns:
        A Foragax environment instance.
    """
    if env_id not in ENV_CONFIGS:
        raise ValueError(f"Unknown env_id: {env_id}")

    config = ENV_CONFIGS[env_id].copy()
    if isinstance(aperture_size, int):
        if aperture_size == -1:
            aperture_size = -1  # keep as -1
        else:
            aperture_size = (aperture_size, aperture_size)
    config["aperture_size"] = aperture_size
    config["random_shift_max_steps"] = random_shift_max_steps

    if env_id == "ForagaxBig-v5":
        config["objects"] = (
            create_fourier_objects(
                num_fourier_terms=10,
                reward_delay=reward_delay,
                regen_delay=(9, 11),
                reward_repeat=1000,
            )[0],
            WALL,
        )

    if env_id == "ForagaxSquareWaveTwoBiome-v11":
        biome1_oyster, biome1_chanterelle, biome2_oyster, biome2_chanterelle = (
            create_shift_square_wave_biome_objects(
                period=500000,
                amplitude_big=9.0,
                amplitude_small=3.0,
                base_oyster_reward=-5.0,
                base_chanterelle_reward=-5.0,
                regen_delay=(9, 11),
                reward_delay=reward_delay,
                expiry_time=500,
                expiry_regen_delay=(9, 11),
            )
        )
        config["objects"] = (
            biome1_oyster,
            biome1_chanterelle,
            biome2_oyster,
            biome2_chanterelle,
            WALL,
        )

    # Backward compatibility: map "world" to "object" with full world
    if observation_type == "world":
        # add deprecation warning
        warnings.warn(
            "'world' observation_type is deprecated. Use 'object' with aperture_size=-1 instead.",
            DeprecationWarning,
        )
        observation_type = "object"
        config["aperture_size"] = -1

    if observation_type not in ("object", "rgb", "color"):
        raise ValueError(f"Unknown observation_type: {observation_type}")

    config["name"] = env_id
    config["observation_type"] = observation_type

    return ForagaxEnv(**{**config, **kwargs})
