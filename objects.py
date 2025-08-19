from typing import Tuple

import jax


class ForagerObject:
    """Base class for objects in the Forager environment."""

    def __init__(
        self,
        name: str = "empty",
        reward: float = 0.0,
        blocking: bool = False,
        collectable: bool = False,
        regen_delay: Tuple[int, int] = (0, 0),
        color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.name = name
        self.reward_val = reward
        self.blocking = blocking
        self.collectable = collectable
        self.regen_delay_range = regen_delay
        self.color = color

    def reward(self, clock: int, rng: jax.Array) -> float:
        """Default reward function."""
        return self.reward_val

    def regen_delay(self, clock: int, rng: jax.Array) -> int:
        """Default regeneration delay function."""
        min_delay, max_delay = self.regen_delay_range
        return jax.random.randint(rng, (), min_delay, max_delay)


EMPTY = ForagerObject()
WALL = ForagerObject(name="wall", blocking=True, color=(0.5, 0.5, 0.5))
FLOWER = ForagerObject(
    name="flower",
    reward=1.0,
    collectable=True,
    regen_delay=(10, 100),
    color=(0.0, 1.0, 0.0),
)

THORNS = ForagerObject(
    name="thorns",
    reward=-1.0,
    collectable=True,
    regen_delay=(10, 100),
    color=(1.0, 0.0, 0.0),
)

MOREL = ForagerObject(
    name="morel",
    reward=10.0,
    collectable=True,
    regen_delay=(100, 100),
    color=(0.25, 0.12, 0.1),
)
OYSTER = ForagerObject(
    name="oyster",
    reward=1.0,
    collectable=True,
    regen_delay=(10, 10),
    color=(0.49, 0.24, 0.32),
)
AGENT = ForagerObject(name="agent", blocking=True, color=(0.0, 0.0, 1.0))
DEATHCAP = ForagerObject(
    name="deathcap",
    reward=-1.0,
    collectable=True,
    regen_delay=(10, 10),
    color=(0.76, 0.7, 0.12),
)
