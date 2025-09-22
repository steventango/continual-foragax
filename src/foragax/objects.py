import abc
from typing import Tuple

import jax
import jax.numpy as jnp

from foragax.weather import FILE_PATHS, get_temperature, load_data


class BaseForagaxObject:
    """Base class for objects in the Foragax environment."""

    def __init__(
        self,
        name: str = "empty",
        blocking: bool = False,
        collectable: bool = False,
        color: Tuple[int, int, int] = (0, 0, 0),
    ):
        self.name = name
        self.blocking = blocking
        self.collectable = collectable
        self.color = color

    @abc.abstractmethod
    def reward(self, clock: int, rng: jax.Array) -> float:
        """Reward function."""
        raise NotImplementedError

    @abc.abstractmethod
    def regen_delay(self, clock: int, rng: jax.Array) -> int:
        """Regeneration delay function."""
        raise NotImplementedError


class DefaultForagaxObject(BaseForagaxObject):
    """Base class for default objects in the Foragax environment."""

    def __init__(
        self,
        name: str = "empty",
        reward: float = 0.0,
        blocking: bool = False,
        collectable: bool = False,
        regen_delay: Tuple[int, int] = (10, 100),
        color: Tuple[int, int, int] = (255, 255, 255),
    ):
        super().__init__(name, blocking, collectable, color)
        self.reward_val = reward
        self.regen_delay_range = regen_delay

    def reward(self, clock: int, rng: jax.Array) -> float:
        """Default reward function."""
        return self.reward_val

    def regen_delay(self, clock: int, rng: jax.Array) -> int:
        """Default regeneration delay function."""
        min_delay, max_delay = self.regen_delay_range
        return jax.random.randint(rng, (), min_delay, max_delay)


class NormalRegenForagaxObject(DefaultForagaxObject):
    """Object with regeneration delay from a normal distribution."""

    def __init__(
        self,
        name: str = "empty",
        reward: float = 0.0,
        collectable: bool = False,
        mean_regen_delay: int = 10,
        std_regen_delay: int = 1,
        color: Tuple[int, int, int] = (0, 0, 0),
    ):
        super().__init__(
            name=name,
            reward=reward,
            collectable=collectable,
            regen_delay=(mean_regen_delay, mean_regen_delay),
            color=color,
        )
        self.mean_regen_delay = mean_regen_delay
        self.std_regen_delay = std_regen_delay

    def regen_delay(self, clock: int, rng: jax.Array) -> int:
        """Regeneration delay from a normal distribution."""
        delay = self.mean_regen_delay + jax.random.normal(rng) * self.std_regen_delay
        return jnp.maximum(0, delay).astype(int)


class WeatherObject(NormalRegenForagaxObject):
    """Object with reward based on temperature data."""

    def __init__(
        self,
        name: str,
        rewards: jnp.ndarray,
        repeat: int,
        multiplier: float = 1.0,
        mean_regen_delay: int = 10,
        std_regen_delay: int = 1,
        color: Tuple[int, int, int] = (0, 0, 0),
    ):
        super().__init__(
            name=name,
            collectable=True,
            mean_regen_delay=mean_regen_delay,
            std_regen_delay=std_regen_delay,
            color=color,
        )
        self.rewards = rewards
        self.repeat = repeat
        self.multiplier = multiplier

    def reward(self, clock: int, rng: jax.Array) -> float:
        """Reward is based on temperature."""
        return get_temperature(self.rewards, clock, self.repeat) * self.multiplier


EMPTY = DefaultForagaxObject()
WALL = DefaultForagaxObject(name="wall", blocking=True, color=(127, 127, 127))
FLOWER = DefaultForagaxObject(
    name="flower",
    reward=1.0,
    collectable=True,
    color=(0, 255, 0),
)

THORNS = DefaultForagaxObject(
    name="thorns",
    reward=-1.0,
    collectable=True,
    color=(255, 0, 0),
)

MOREL = DefaultForagaxObject(
    name="morel",
    reward=10.0,
    collectable=True,
    regen_delay=(100, 100),
    color=(63, 30, 25),
)
OYSTER = DefaultForagaxObject(
    name="oyster",
    reward=1.0,
    collectable=True,
    regen_delay=(10, 10),
    color=(124, 61, 81),
)
LARGE_MOREL = NormalRegenForagaxObject(
    name="large_morel",
    reward=30.0,
    collectable=True,
    mean_regen_delay=300,
    std_regen_delay=30,
    color=(63, 30, 25),
)
MEDIUM_MOREL = NormalRegenForagaxObject(
    name="medium_morel",
    reward=10.0,
    collectable=True,
    mean_regen_delay=100,
    std_regen_delay=10,
    color=(63, 30, 25),
)
LARGE_OYSTER = NormalRegenForagaxObject(
    name="large_oyster",
    reward=1.0,
    collectable=True,
    mean_regen_delay=10,
    std_regen_delay=1,
    color=(124, 61, 81),
)
DEATHCAP = DefaultForagaxObject(
    name="deathcap",
    reward=-1.0,
    collectable=True,
    regen_delay=(10, 10),
    color=(193, 178, 30),
)
AGENT = DefaultForagaxObject(name="agent", blocking=True, color=(0, 0, 255))

BROWN_MOREL = NormalRegenForagaxObject(
    name="brown_morel",
    reward=30.0,
    collectable=True,
    color=(63, 30, 25),
    mean_regen_delay=300,
    std_regen_delay=30,
)
BROWN_OYSTER = NormalRegenForagaxObject(
    name="brown_oyster",
    reward=1.0,
    collectable=True,
    color=(63, 30, 25),
    mean_regen_delay=10,
    std_regen_delay=1,
)
GREEN_DEATHCAP = DefaultForagaxObject(
    name="green_deathcap",
    reward=-1.0,
    collectable=True,
    color=(0, 255, 0),
    regen_delay=(10, 10),
)
GREEN_FAKE = DefaultForagaxObject(
    name="green_fake",
    reward=0.0,
    collectable=True,
    color=(0, 255, 0),
    regen_delay=(10, 10),
)


def create_weather_objects(
    file_index: int = 0, repeat: int = 500, multiplier: float = 1.0
):
    """Create HOT and COLD WeatherObject instances using the specified file.

    Args:
        file_index: Index into `FILE_PATHS` to select the temperature file.
        repeat: How many steps each temperature value repeats for.
        multiplier: Base multiplier applied to HOT; COLD will use -multiplier.

    Returns:
        A tuple (HOT, COLD) of WeatherObject instances.
    """
    # Clamp file_index
    if file_index < 0 or file_index >= len(FILE_PATHS):
        raise IndexError(
            f"file_index {file_index} out of range (0..{len(FILE_PATHS) - 1})"
        )

    rewards = load_data(FILE_PATHS[file_index])

    hot = WeatherObject(
        name="hot",
        rewards=rewards,
        repeat=repeat,
        multiplier=multiplier,
        color=(255, 0, 255),
    )

    cold = WeatherObject(
        name="cold",
        rewards=rewards,
        repeat=repeat,
        multiplier=-multiplier,
        color=(0, 255, 255),
    )

    return hot, cold
