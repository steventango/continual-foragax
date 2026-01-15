"""Factory functions for creating Foragax environment variants."""

import warnings
from typing import Any, Dict, Optional, Tuple

from foragax.env import (
    Biome,
    ForagaxEnv,
)
from foragax.objects import (
    BROWN_MOREL,
    BROWN_MOREL_2,
    BROWN_MOREL_UNIFORM,
    BROWN_MOREL_UNIFORM_RANDOM,
    BROWN_MOREL_UNIFORM_RANDOM_EXPIRY,
    BROWN_OYSTER,
    BROWN_OYSTER_UNIFORM,
    BROWN_OYSTER_UNIFORM_RANDOM,
    BROWN_OYSTER_UNIFORM_RANDOM_EXPIRY,
    GREEN_DEATHCAP,
    GREEN_DEATHCAP_2,
    GREEN_DEATHCAP_3,
    GREEN_DEATHCAP_UNIFORM,
    GREEN_DEATHCAP_UNIFORM_RANDOM,
    GREEN_DEATHCAP_UNIFORM_RANDOM_EXPIRY,
    GREEN_FAKE,
    GREEN_FAKE_2,
    GREEN_FAKE_UNIFORM,
    GREEN_FAKE_UNIFORM_RANDOM,
    GREEN_FAKE_UNIFORM_RANDOM_EXPIRY,
    LARGE_MOREL,
    LARGE_OYSTER,
    MEDIUM_MOREL,
    create_fourier_objects,
    create_sine_biome_objects,
    create_weather_objects,
)

ENV_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ForagaxWeather-v1": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": None,
        "biomes": (
            # Hot biome
            Biome(start=(0, 3), stop=(15, 5), object_frequencies=(0.5, 0.0)),
            # Cold biome
            Biome(start=(0, 10), stop=(15, 12), object_frequencies=(0.0, 0.5)),
        ),
        "nowrap": False,
    },
    "ForagaxWeather-v2": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": None,
        "biomes": (
            # Hot biome
            Biome(start=(0, 3), stop=(15, 5), object_frequencies=(0.5, 0.0)),
            # Cold biome
            Biome(start=(0, 10), stop=(15, 12), object_frequencies=(0.0, 0.5)),
        ),
        "nowrap": True,
    },
    "ForagaxWeather-v3": {
        "size": None,
        "aperture_size": None,
        "objects": None,
        "biomes": None,
        "nowrap": True,
    },
    "ForagaxWeather-v4": {
        "size": None,
        "aperture_size": None,
        "objects": None,
        "biomes": None,
        "nowrap": True,
        "deterministic_spawn": True,
    },
    "ForagaxWeather-v5": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": None,
        "biomes": (
            # Hot biome
            Biome(start=(0, 3), stop=(15, 5), object_frequencies=(0.5, 0.0)),
            # Cold biome
            Biome(start=(0, 10), stop=(15, 12), object_frequencies=(0.0, 0.5)),
        ),
        "nowrap": False,
        "deterministic_spawn": True,
    },
    "ForagaxDiwali-v1": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": None,
        "biomes": (
            # Hot biome
            Biome(start=(0, 3), stop=(15, 5), object_frequencies=(0.5, 0.0)),
            # Cold biome
            Biome(start=(0, 10), stop=(15, 12), object_frequencies=(0.0, 0.5)),
        ),
        "nowrap": False,
        "deterministic_spawn": True,
        "dynamic_biomes": True,
        "biome_consumption_threshold": 0.9,
    },
    "ForagaxDiwali-v2": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": None,
        "biomes": (
            # Hot biome
            Biome(start=(0, 2), stop=(15, 6), object_frequencies=(0.5, 0.0)),
            # Cold biome
            Biome(start=(0, 9), stop=(15, 13), object_frequencies=(0.0, 0.5)),
        ),
        "nowrap": False,
        "deterministic_spawn": True,
        "dynamic_biomes": True,
        "biome_consumption_threshold": 200,
    },
    "ForagaxDiwali-v3": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": None,
        "biomes": (Biome(start=(0, 0), stop=(15, 15), object_frequencies=(0.5,)),),
        "nowrap": False,
        "deterministic_spawn": True,
        "dynamic_biomes": True,
    },
    "ForagaxDiwali-v4": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": None,
        "biomes": (
            # Hot biome
            Biome(start=(0, 2), stop=(15, 6), object_frequencies=(0.5, 0.0)),
            # Cold biome
            Biome(start=(0, 9), stop=(15, 13), object_frequencies=(0.0, 0.5)),
        ),
        "nowrap": False,
        "deterministic_spawn": True,
        "dynamic_biomes": True,
        "biome_consumption_threshold": 1000,
    },
    "ForagaxDiwali-v5": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": None,
        "biomes": (Biome(start=(0, 0), stop=(15, 15), object_frequencies=(0.4,)),),
        "nowrap": False,
        "deterministic_spawn": True,
        "dynamic_biomes": True,
        "biome_consumption_threshold": 1000,
        "dynamic_biome_spawn_empty": 0.4,
    },
    "ForagaxTwoBiome-v1": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": (BROWN_MOREL, BROWN_OYSTER, GREEN_DEATHCAP, GREEN_FAKE),
        "biomes": (
            # Morel biome
            Biome(start=(3, 0), stop=(5, 15), object_frequencies=(0.5, 0.0, 0.25, 0.0)),
            # Oyster biome
            Biome(
                start=(10, 0), stop=(12, 15), object_frequencies=(0.0, 0.5, 0.0, 0.25)
            ),
        ),
        "nowrap": False,
    },
    "ForagaxTwoBiome-v2": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": (BROWN_MOREL, BROWN_OYSTER, GREEN_DEATHCAP, GREEN_FAKE),
        "biomes": (
            # Morel biome
            Biome(start=(3, 0), stop=(5, 15), object_frequencies=(0.5, 0.0, 0.25, 0.0)),
            # Oyster biome
            Biome(
                start=(10, 0), stop=(12, 15), object_frequencies=(0.0, 0.5, 0.0, 0.25)
            ),
        ),
        "nowrap": True,
    },
    "ForagaxTwoBiome-v3": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": (BROWN_MOREL, BROWN_OYSTER, GREEN_DEATHCAP_2, GREEN_FAKE),
        "biomes": (
            # Morel biome
            Biome(start=(3, 0), stop=(5, 15), object_frequencies=(0.5, 0.0, 0.25, 0.0)),
            # Oyster biome
            Biome(
                start=(10, 0), stop=(12, 15), object_frequencies=(0.0, 0.5, 0.0, 0.25)
            ),
        ),
        "nowrap": True,
    },
    "ForagaxTwoBiome-v4": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": (BROWN_MOREL_2, BROWN_OYSTER, GREEN_DEATHCAP_3, GREEN_FAKE_2),
        "biomes": (
            # Morel biome
            Biome(start=(3, 0), stop=(5, 15), object_frequencies=(0.5, 0.0, 0.25, 0.0)),
            # Oyster biome
            Biome(
                start=(10, 0), stop=(12, 15), object_frequencies=(0.0, 0.5, 0.0, 0.25)
            ),
        ),
        "nowrap": True,
    },
    "ForagaxTwoBiome-v5": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": (BROWN_MOREL_2, BROWN_OYSTER, GREEN_DEATHCAP_3, GREEN_FAKE_2),
        "biomes": (
            # Morel biome
            Biome(start=(3, 0), stop=(5, 15), object_frequencies=(0.25, 0.0, 0.5, 0.0)),
            # Oyster biome
            Biome(
                start=(10, 0), stop=(12, 15), object_frequencies=(0.0, 0.5, 0.0, 0.25)
            ),
        ),
        "nowrap": True,
    },
    "ForagaxTwoBiome-v6": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": (BROWN_MOREL_2, BROWN_OYSTER, GREEN_DEATHCAP_3, GREEN_FAKE_2),
        "biomes": (
            # Morel biome
            Biome(start=(3, 0), stop=(5, 15), object_frequencies=(0.25, 0.0, 0.5, 0.0)),
            # Oyster biome
            Biome(
                start=(10, 0), stop=(12, 15), object_frequencies=(0.0, 0.25, 0.0, 0.5)
            ),
        ),
        "nowrap": True,
    },
    "ForagaxTwoBiome-v7": {
        "size": None,
        "aperture_size": None,
        "objects": (BROWN_MOREL_2, BROWN_OYSTER, GREEN_DEATHCAP_3, GREEN_FAKE_2),
        "biomes": None,
        "nowrap": True,
    },
    "ForagaxTwoBiome-v8": {
        "size": None,
        "aperture_size": None,
        "objects": (
            BROWN_MOREL_UNIFORM,
            BROWN_OYSTER_UNIFORM,
            GREEN_DEATHCAP_UNIFORM,
            GREEN_FAKE_UNIFORM,
        ),
        "biomes": None,
        "nowrap": True,
    },
    "ForagaxTwoBiome-v9": {
        "size": None,
        "aperture_size": None,
        "objects": (
            BROWN_MOREL_UNIFORM,
            BROWN_OYSTER_UNIFORM,
            GREEN_DEATHCAP_UNIFORM,
            GREEN_FAKE_UNIFORM,
        ),
        "biomes": None,
        "nowrap": True,
        "deterministic_spawn": True,
    },
    "ForagaxTwoBiome-v10": {
        "size": None,
        "aperture_size": None,
        "objects": (
            BROWN_MOREL_UNIFORM_RANDOM,
            BROWN_OYSTER_UNIFORM_RANDOM,
            GREEN_DEATHCAP_UNIFORM_RANDOM,
            GREEN_FAKE_UNIFORM_RANDOM,
        ),
        "biomes": None,
        "nowrap": True,
        "deterministic_spawn": True,
    },
    "ForagaxTwoBiome-v11": {
        "size": None,
        "aperture_size": None,
        "objects": (
            BROWN_MOREL_UNIFORM,
            BROWN_OYSTER_UNIFORM,
            GREEN_DEATHCAP_UNIFORM,
            GREEN_FAKE_UNIFORM,
        ),
        "biomes": None,
        "nowrap": True,
        "deterministic_spawn": True,
    },
    "ForagaxTwoBiome-v12": {
        "size": None,
        "aperture_size": None,
        "objects": (
            BROWN_MOREL_UNIFORM_RANDOM,
            BROWN_OYSTER_UNIFORM_RANDOM,
            GREEN_DEATHCAP_UNIFORM_RANDOM,
            GREEN_FAKE_UNIFORM_RANDOM,
        ),
        "biomes": None,
        "nowrap": True,
        "deterministic_spawn": True,
    },
    "ForagaxTwoBiome-v13": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": (
            BROWN_MOREL_UNIFORM_RANDOM,
            BROWN_OYSTER_UNIFORM_RANDOM,
            GREEN_DEATHCAP_UNIFORM_RANDOM,
            GREEN_FAKE_UNIFORM_RANDOM,
        ),
        "biomes": (
            # Morel biome
            Biome(start=(3, 0), stop=(5, 15), object_frequencies=(0.25, 0.0, 0.5, 0.0)),
            # Oyster biome
            Biome(
                start=(10, 0), stop=(12, 15), object_frequencies=(0.0, 0.25, 0.0, 0.5)
            ),
        ),
        "nowrap": False,
        "deterministic_spawn": True,
    },
    "ForagaxTwoBiome-v14": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": (
            BROWN_MOREL_UNIFORM,
            BROWN_OYSTER_UNIFORM,
            GREEN_DEATHCAP_UNIFORM,
            GREEN_FAKE_UNIFORM,
        ),
        "biomes": (
            # Morel biome
            Biome(start=(3, 0), stop=(5, 15), object_frequencies=(0.25, 0.0, 0.5, 0.0)),
            # Oyster biome
            Biome(
                start=(10, 0), stop=(12, 15), object_frequencies=(0.0, 0.25, 0.0, 0.5)
            ),
        ),
        "nowrap": False,
        "deterministic_spawn": True,
    },
    "ForagaxTwoBiome-v15": {
        "size": None,
        "aperture_size": None,
        "objects": (
            BROWN_MOREL_UNIFORM,
            BROWN_OYSTER_UNIFORM,
            GREEN_DEATHCAP_UNIFORM,
            GREEN_FAKE_UNIFORM,
        ),
        "biomes": None,
        "nowrap": True,
        "deterministic_spawn": True,
    },
    "ForagaxTwoBiome-v16": {
        "size": None,
        "aperture_size": None,
        "objects": (
            BROWN_MOREL_UNIFORM_RANDOM,
            BROWN_OYSTER_UNIFORM_RANDOM,
            GREEN_DEATHCAP_UNIFORM_RANDOM,
            GREEN_FAKE_UNIFORM_RANDOM,
        ),
        "biomes": None,
        "nowrap": True,
        "deterministic_spawn": True,
    },
    "ForagaxTwoBiome-v17": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": (
            BROWN_MOREL_UNIFORM_RANDOM_EXPIRY,
            BROWN_OYSTER_UNIFORM_RANDOM_EXPIRY,
            GREEN_DEATHCAP_UNIFORM_RANDOM_EXPIRY,
            GREEN_FAKE_UNIFORM_RANDOM_EXPIRY,
        ),
        "biomes": (
            # Morel biome
            Biome(start=(3, 0), stop=(5, 15), object_frequencies=(0.25, 0.0, 0.5, 0.0)),
            # Oyster biome
            Biome(
                start=(10, 0), stop=(12, 15), object_frequencies=(0.0, 0.25, 0.0, 0.5)
            ),
        ),
        "nowrap": False,
        "deterministic_spawn": True,
    },
    "ForagaxTwoBiomeSmall-v1": {
        "size": (16, 8),
        "aperture_size": None,
        "objects": (LARGE_MOREL, LARGE_OYSTER),
        "biomes": (
            # Morel biome
            Biome(start=(2, 2), stop=(6, 6), object_frequencies=(1.0, 0.0)),
            # Oyster biome
            Biome(start=(10, 2), stop=(14, 6), object_frequencies=(0.0, 1.0)),
        ),
        "nowrap": False,
    },
    "ForagaxTwoBiomeSmall-v2": {
        "size": (16, 8),
        "aperture_size": None,
        "objects": (MEDIUM_MOREL, LARGE_OYSTER),
        "biomes": (
            # Morel biome
            Biome(start=(3, 3), stop=(6, 6), object_frequencies=(1.0, 0.0)),
            # Oyster biome
            Biome(start=(11, 3), stop=(14, 6), object_frequencies=(0.0, 1.0)),
        ),
        "nowrap": False,
    },
    "ForagaxTwoBiomeSmall-v3": {
        "size": (16, 8),
        "aperture_size": None,
        "objects": (MEDIUM_MOREL, LARGE_OYSTER),
        "biomes": (
            # Morel biome
            Biome(start=(3, 3), stop=(6, 6), object_frequencies=(1.0, 0.0)),
            # Oyster biome
            Biome(start=(11, 3), stop=(14, 6), object_frequencies=(0.0, 1.0)),
        ),
        "nowrap": True,
    },
    "ForagaxSineTwoBiome-v1": {
        "size": (15, 15),
        "aperture_size": None,
        "objects": None,
        "biomes": (
            # Biome 1 (left): Oyster +10, Death Cap -10 with sine
            Biome(
                start=(3, 0),
                stop=(5, 15),
                object_frequencies=(8 / 30, 8 / 30, 0.0, 0.0),
            ),
            # Biome 2 (right): Oyster -10, Death Cap +10 with inverted sine
            Biome(
                start=(10, 0),
                stop=(12, 15),
                object_frequencies=(0.0, 0.0, 8 / 30, 8 / 30),
            ),
        ),
        "nowrap": False,
        "deterministic_spawn": True,
    },
}


def make(
    env_id: str,
    observation_type: str = "color",
    aperture_size: Optional[Tuple[int, int]] = (5, 5),
    file_index: int = 0,
    repeat: int = 500,
    reward_delay: int = 0,
    **kwargs: Any,
) -> ForagaxEnv:
    """Create a Foragax environment.

    Args:
        env_id: The ID of the environment to create.
        observation_type: The type of observation to use. One of "object", "rgb", or "color".
        aperture_size: The size of the agent's observation aperture. If -1, full world observation.
            If None, the default for the environment is used.
        file_index: File index for weather objects.
        repeat: How many steps each temperature value repeats for (weather environments).
        reward_delay: Number of steps required to digest food items (weather environments).
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

    # Handle special size and biome configurations
    if env_id in (
        "ForagaxTwoBiome-v7",
        "ForagaxTwoBiome-v8",
        "ForagaxTwoBiome-v9",
        "ForagaxTwoBiome-v10",
        "ForagaxTwoBiome-v15",
        "ForagaxTwoBiome-v16",
    ):
        if aperture_size == -1:
            margin = 0  # for world view, no margin needed
        else:
            margin = aperture_size[1] // 2 + 1
        width = 2 * margin + 9
        config["size"] = (width, 15)
        config["biomes"] = (
            # Morel biome
            Biome(
                start=(margin, 0),
                stop=(margin + 2, 15),
                object_frequencies=(0.25, 0.0, 0.5, 0.0),
            ),
            # Oyster biome
            Biome(
                start=(margin + 7, 0),
                stop=(margin + 9, 15),
                object_frequencies=(0.0, 0.25, 0.0, 0.5),
            ),
        )

    if env_id in ("ForagaxTwoBiome-v11", "ForagaxTwoBiome-v12"):
        if aperture_size == -1:
            margin = 0
        else:
            margin = aperture_size[1] // 2 + 1
        width = 2 * margin + 9
        config["size"] = (width, 15)
        config["biomes"] = (
            # Morel biome
            Biome(
                start=(margin, 0),
                stop=(margin + 2, 15),
                object_frequencies=(0.5, 0.0, 0.25, 0.0),
            ),
            # Oyster biome
            Biome(
                start=(margin + 7, 0),
                stop=(margin + 9, 15),
                object_frequencies=(0.0, 0.5, 0.0, 0.25),
            ),
        )

    if env_id in ("ForagaxWeather-v3", "ForagaxWeather-v4"):
        if aperture_size == -1:
            margin = 0
        else:
            margin = aperture_size[1] // 2 + 1
        width = 2 * margin + 9
        config["size"] = (15, width)
        config["biomes"] = (
            # Hot biome
            Biome(
                start=(0, margin),
                stop=(15, margin + 2),
                object_frequencies=(0.5, 0.0),
            ),
            # Cold biome
            Biome(
                start=(0, margin + 7),
                stop=(15, margin + 9),
                object_frequencies=(0.0, 0.5),
            ),
        )

    if env_id.startswith("ForagaxWeather"):
        same_color = env_id in (
            "ForagaxWeather-v2",
            "ForagaxWeather-v3",
            "ForagaxWeather-v4",
            "ForagaxWeather-v5",
        )
        random_respawn = env_id in (
            "ForagaxWeather-v4",
            "ForagaxWeather-v5",
        )
        hot, cold = create_weather_objects(
            file_index=file_index,
            repeat=repeat,
            same_color=same_color,
            random_respawn=random_respawn,
            reward_delay=reward_delay,
        )
        config["objects"] = (hot, cold)

    if env_id == "ForagaxDiwali-v1":
        config["objects"] = create_fourier_objects(
            num_fourier_terms=10,
            reward_delay=reward_delay,
        )
    if env_id == "ForagaxDiwali-v2":
        config["objects"] = create_fourier_objects(
            num_fourier_terms=10,
            reward_delay=reward_delay,
            regen_delay=(9, 11),
        )
    if env_id == "ForagaxDiwali-v3":
        config["objects"] = create_fourier_objects(
            num_fourier_terms=10,
            reward_delay=reward_delay,
            regen_delay=(9, 11),
        )[:1]
    if env_id == "ForagaxDiwali-v4":
        config["objects"] = create_fourier_objects(
            num_fourier_terms=10,
            reward_delay=reward_delay,
            regen_delay=(9, 11),
            reward_repeat=100,
        )
    if env_id == "ForagaxDiwali-v5":
        config["objects"] = create_fourier_objects(
            num_fourier_terms=10,
            reward_delay=reward_delay,
            regen_delay=(9, 11),
            reward_repeat=100,
        )[:1]

    if env_id == "ForagaxSineTwoBiome-v1":
        biome1_oyster, biome1_deathcap, biome2_oyster, biome2_deathcap = (
            create_sine_biome_objects(
                period=1000,
                amplitude=20.0,
                base_oyster_reward=10.0,
                base_deathcap_reward=-10.0,
                regen_delay=(9, 11),
                reward_delay=reward_delay,
                expiry_time=500,
                expiry_regen_delay=(9, 11),
            )
        )
        config["objects"] = (
            biome1_oyster,
            biome1_deathcap,
            biome2_oyster,
            biome2_deathcap,
        )

    if env_id == "ForagaxTwoBiome-v16":
        config["teleport_interval"] = 10000

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

    return ForagaxEnv(**config, **kwargs)
