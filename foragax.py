"""JAX implementation of Forager environment.

Source: https://github.com/andnp/forager
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class ObjectType:
    reward: float = 0.0
    blocking: bool = False
    collectable: bool = False
    regen_delay: Tuple[int, int] = (0, 0)
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)


# Default object types
EMPTY = ObjectType()
WALL = ObjectType(blocking=True, color=(0.5, 0.5, 0.5))
FLOWER = ObjectType(
    reward=1.0, collectable=True, regen_delay=(10, 100), color=(0.0, 1.0, 0.0)
)
THORNS = ObjectType(
    reward=-1.0, collectable=True, regen_delay=(10, 100), color=(1.0, 0.0, 0.0)
)
MOREL = ObjectType(
    reward=10.0, collectable=True, regen_delay=(100, 100), color=(0.25, 0.12, 0.1)
)
OYSTER = ObjectType(
    reward=1.0, collectable=True, regen_delay=(10, 10), color=(0.49, 0.24, 0.32)
)
DEATHCAP = ObjectType(
    reward=-1.0, collectable=True, regen_delay=(10, 10), color=(0.76, 0.7, 0.12)
)
AGENT = ObjectType(blocking=True, color=(0.0, 0.0, 1.0))


class Actions(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


DIRECTIONS = jnp.array(
    [
        [0, 1],
        [1, 0],
        [0, -1],
        [-1, 0],
    ]
)


@dataclass
class Biome:
    # Object generation frequencies for this biome
    object_frequencies: Tuple[float, ...] = ()
    start: Tuple[int, int] | None = None
    stop: Tuple[int, int] | None = None


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int


@struct.dataclass
class EnvState(environment.EnvState):
    pos: jax.Array
    object_grid: jax.Array
    original_object_grid: jax.Array
    respawn_timers: jax.Array
    time: int
    key: jax.Array


class ForagerEnv(environment.Environment[EnvState, EnvParams]):
    """JAX implementation of Forager environment."""

    def __init__(
        self,
        size: Tuple[int, int] | int = (10, 10),
        aperture_size: Tuple[int, int] | int = (5, 5),
        object_types: Tuple[ObjectType, ...] = (EMPTY,),
        biomes: Tuple[Biome, ...] = (Biome(object_frequencies=(1.0,)),),
    ):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size

        if isinstance(aperture_size, int):
            aperture_size = (aperture_size, aperture_size)
        self.aperture_size = aperture_size

        # JIT-compatible versions of object and biome properties
        self.object_ids = jnp.arange(len(object_types))
        self.object_rewards = jnp.array([o.reward for o in object_types])
        self.object_blocking = jnp.array([o.blocking for o in object_types])
        self.object_collectable = jnp.array([o.collectable for o in object_types])
        self.object_regen_delays = jnp.array([o.regen_delay for o in object_types])
        self.object_colors = jnp.array([o.color for o in object_types])

        self.biome_object_frequencies = jnp.array(
            [b.object_frequencies for b in biomes]
        )
        self.biome_starts = jnp.array(
            [b.start if b.start is not None else (-1, -1) for b in biomes]
        )
        self.biome_stops = jnp.array(
            [b.stop if b.stop is not None else (-1, -1) for b in biomes]
        )

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(
            max_steps_in_episode=500,
        )

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition."""
        # 1. UPDATE AGENT POSITION
        direction = DIRECTIONS[action]
        new_pos = state.pos + direction

        # Wrap around boundaries
        new_pos = jnp.mod(new_pos, jnp.array(self.size))

        # Check for blocking objects
        obj_at_new_pos = state.object_grid[new_pos[0], new_pos[1]]
        is_blocking = self.object_blocking[obj_at_new_pos]
        pos = jax.lax.select(is_blocking, state.pos, new_pos)

        # 2. HANDLE COLLISIONS AND REWARDS
        obj_at_pos = state.object_grid[pos[0], pos[1]]
        reward = self.object_rewards[obj_at_pos]
        is_collectable = self.object_collectable[obj_at_pos]

        # 3. HANDLE OBJECT COLLECTION AND RESPAWNING
        key, subkey = jax.random.split(state.key)

        # Update respawn timers
        respawn_timers = state.respawn_timers - 1

        # Respawn objects
        respawn_candidates = respawn_timers == 0
        object_grid = jnp.where(
            respawn_candidates, state.original_object_grid, state.object_grid
        )
        respawn_timers = jnp.where(respawn_candidates, -1, respawn_timers)

        # Collect object
        new_object_grid = jax.lax.select(
            is_collectable,
            object_grid.at[pos[0], pos[1]].set(0),
            object_grid,
        )

        # Set respawn timer for collected object
        def get_regen_delay(obj_type):
            key_regen, _ = jax.random.split(subkey)
            regen_delays = self.object_regen_delays
            min_delay = regen_delays[obj_type, 0]
            max_delay = regen_delays[obj_type, 1]
            return jax.random.randint(key_regen, (), min_delay, max_delay)

        regen_delay = get_regen_delay(obj_at_pos)

        new_respawn_timers = jax.lax.select(
            is_collectable,
            respawn_timers.at[pos[0], pos[1]].set(regen_delay),
            respawn_timers,
        )

        # 4. UPDATE STATE
        state = EnvState(
            pos=pos,
            object_grid=new_object_grid,
            original_object_grid=state.original_object_grid,
            respawn_timers=new_respawn_timers,
            time=state.time + 1,
            key=key,
        )

        done = self.is_terminal(state, params)
        return (
            jax.lax.stop_gradient(self.get_obs(state, params)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state."""
        key, subkey = jax.random.split(key)

        object_grid = jnp.zeros(self.size, dtype=jnp.int32)

        for i in range(self.biome_object_frequencies.shape[0]):
            key, biome_key = jax.random.split(key)
            # Generate random layout
            grid_rand = jax.random.uniform(biome_key, self.size)

            # Generate objects for this biome
            biome_grid = jnp.zeros(self.size, dtype=jnp.int32)

            cumulative_freq = 0.0
            for j, freq in enumerate(self.biome_object_frequencies[i]):
                obj_id = self.object_ids[j]
                biome_grid = jnp.where(
                    (grid_rand >= cumulative_freq)
                    & (grid_rand < cumulative_freq + freq),
                    obj_id,
                    biome_grid,
                )
                cumulative_freq += freq

            # Create mask for the biome
            start = jax.lax.select(
                self.biome_starts[i, 0] == -1,
                jnp.array([0, 0]),
                self.biome_starts[i],
            )
            stop = jax.lax.select(
                self.biome_stops[i, 0] == -1,
                jnp.array(self.size),
                self.biome_stops[i],
            )

            rows = jnp.arange(self.size[0])[:, None]
            cols = jnp.arange(self.size[1])

            mask = (
                (rows >= start[0])
                & (rows < stop[0])
                & (cols >= start[1])
                & (cols < stop[1])
            )

            # Update the main grid with the biome's objects
            object_grid = jnp.where(mask, biome_grid, object_grid)

        # Place agent in the center of the world and ensure the cell is empty.
        agent_pos = jnp.array([self.size[0] // 2, self.size[1] // 2])
        object_grid = object_grid.at[agent_pos[0], agent_pos[1]].set(0)

        state = EnvState(
            pos=agent_pos,
            object_grid=object_grid,
            original_object_grid=object_grid,
            respawn_timers=jnp.full(self.size, -1, dtype=jnp.int32),
            time=0,
            key=key,
        )

        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        """Return observation from raw state trafo."""
        raise NotImplementedError

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Forager is a continuing environment."""
        return False

    @property
    def name(self) -> str:
        """Environment name."""
        return "Forager-v0"

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        raise NotImplementedError

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "pos": spaces.Box(0, max(self.size), (2,), jnp.int32),
                "object_grid": spaces.Box(
                    0,
                    len(self.object_ids),
                    self.size,
                    jnp.int32,
                ),
                "original_object_grid": spaces.Box(
                    0,
                    len(self.object_ids),
                    self.size,
                    jnp.int32,
                ),
                "respawn_timers": spaces.Box(-1, 1000, self.size, jnp.int32),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "key": spaces.PRNGKey(),
            }
        )

    def render(self, state: EnvState, params: EnvParams):
        """Render the environment state."""
        fig, ax = plt.subplots()

        # Create an RGB image from the object grid
        img = jnp.zeros((self.size[0], self.size[1], 3))
        for i, obj_id in enumerate(self.object_ids):
            color = self.object_colors[i]
            img = img.at[state.object_grid == obj_id].set(jnp.array(color))

        # Agent color
        agent_color = self.object_colors[-1]
        img = img.at[state.pos[0], state.pos[1]].set(jnp.array(agent_color))

        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax


class ForagerObject(ForagerEnv):
    """Forager environment with object-based aperture observation."""

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        num_obj_types = len(self.object_ids)

        # Roll the grid to center the agent's position
        # The agent should be at the center of the aperture, which is aperture_size // 2
        # We want to move the agent's position (state.pos) to the center.
        # The amount to roll is -(state.pos - aperture_size // 2)
        # Note: jnp.roll is on (row, col) but pos is (x, y), so we swap them.
        roll_amount = -(state.pos - jnp.array(self.aperture_size) // 2)
        roll_amount = jnp.array([roll_amount[1], roll_amount[0]])
        rolled_grid = jnp.roll(state.object_grid, shift=roll_amount, axis=(0, 1))

        # Extract the aperture
        aperture = jax.lax.dynamic_slice(
            rolled_grid,
            (0, 0),
            self.aperture_size,
        )

        # flip rows to match the agent's perspective
        aperture = jnp.flip(aperture, axis=0)

        obs = jax.nn.one_hot(aperture, num_obj_types)

        obs = obs[:, :, 1:]

        return obs

    def observation_space(self, params: EnvParams) -> spaces.Box:
        num_obj_types = len(self.object_ids)
        obs_shape = (
            self.aperture_size[0],
            self.aperture_size[1],
            num_obj_types - 1,
        )
        return spaces.Box(0, 1, obs_shape, jnp.float32)


class ForagerRGB(ForagerEnv):
    """Forager environment with color-based aperture observation."""

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        num_obj_types = len(self.object_ids)
        padded_grid = jnp.pad(
            state.object_grid,
            (
                (self.aperture_size[0] // 2, self.aperture_size[0] // 2),
                (self.aperture_size[1] // 2, self.aperture_size[1] // 2),
            ),
            "constant",
            constant_values=0,
        )

        aperture = jax.lax.dynamic_slice(
            padded_grid,
            (state.pos[0], state.pos[1]),
            self.aperture_size,
        )

        aperture_one_hot = jax.nn.one_hot(aperture, num_obj_types)

        # Agent position is always at the center of the aperture
        center = (self.aperture_size[0] // 2, self.aperture_size[1] // 2)
        aperture_one_hot = aperture_one_hot.at[center[0], center[1], -1].set(1)

        colors = self.object_colors
        obs = jnp.tensordot(aperture_one_hot, colors, axes=1)
        return obs

    def observation_space(self, params: EnvParams) -> spaces.Box:
        obs_shape = (self.aperture_size[0], self.aperture_size[1], 3)
        return spaces.Box(0, 1, obs_shape, jnp.float32)


class ForagerWorld(ForagerEnv):
    """Forager environment with world observation."""

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        num_obj_types = len(self.object_ids)
        obs = jax.nn.one_hot(state.object_grid, num_obj_types)
        obs = obs.at[state.pos[0], state.pos[1], -1].set(1)
        return obs

    def observation_space(self, params: EnvParams) -> spaces.Box:
        num_obj_types = len(self.object_ids)
        obs_shape = (self.size[0], self.size[1], num_obj_types)
        return spaces.Box(0, 1, obs_shape, jnp.float32)
