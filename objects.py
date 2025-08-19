import abc
from typing import Tuple

import jax


class BaseForagerObject:
    """Base class for objects in the Forager environment."""

    def __init__(
        self,
        name: str = "empty",
        blocking: bool = False,
        collectable: bool = False,
        color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
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


class DefaultForagerObject(BaseForagerObject):
    """Base class for default objects in the Forager environment."""

    def __init__(
        self,
        name: str = "empty",
        reward: float = 0.0,
        blocking: bool = False,
        collectable: bool = False,
        regen_delay: Tuple[int, int] = (10, 100),
        color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
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


EMPTY = DefaultForagerObject()
WALL = DefaultForagerObject(name="wall", blocking=True, color=(0.5, 0.5, 0.5))
FLOWER = DefaultForagerObject(
    name="flower",
    reward=1.0,
    collectable=True,
    color=(0.0, 1.0, 0.0),
)

THORNS = DefaultForagerObject(
    name="thorns",
    reward=-1.0,
    collectable=True,
    color=(1.0, 0.0, 0.0),
)

MOREL = DefaultForagerObject(
    name="morel",
    reward=10.0,
    collectable=True,
    regen_delay=(100, 100),
    color=(0.25, 0.12, 0.1),
)
OYSTER = DefaultForagerObject(
    name="oyster",
    reward=1.0,
    collectable=True,
    regen_delay=(10, 10),
    color=(0.49, 0.24, 0.32),
)
DEATHCAP = DefaultForagerObject(
    name="deathcap",
    reward=-1.0,
    collectable=True,
    regen_delay=(10, 10),
    color=(0.76, 0.7, 0.12),
)
AGENT = DefaultForagerObject(name="agent", blocking=True, color=(0.0, 0.0, 1.0))
