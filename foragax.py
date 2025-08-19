"""JAX implementation of Foragax environment.

Source: https://github.com/andnp/Foragax
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from objects import AGENT, EMPTY, BaseForagaxObject


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
    max_steps_in_episode: int | None


@struct.dataclass
class EnvState(environment.EnvState):
    pos: jax.Array
    object_grid: jax.Array
    time: int


class ForagaxEnv(environment.Environment[EnvState, EnvParams]):
    """JAX implementation of Foragax environment."""

    def __init__(
        self,
        size: Tuple[int, int] | int = (10, 10),
        aperture_size: Tuple[int, int] | int = (5, 5),
        objects: Tuple[BaseForagaxObject, ...] = (EMPTY,),
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
        self.object_ids = jnp.arange(len(objects))
        self.object_blocking = jnp.array([o.blocking for o in objects])
        self.object_collectable = jnp.array([o.collectable for o in objects])
        self.object_colors = jnp.array([o.color for o in objects])

        self.reward_fns = [o.reward for o in objects]
        self.regen_delay_fns = [o.regen_delay for o in objects]

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
            max_steps_in_episode=None,
        )

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition."""
        num_obj_types = len(self.object_ids)
        # Decode the object grid: positive values are objects, negative are timers (treat as empty)
        current_objects = jnp.maximum(0, state.object_grid)

        # 1. UPDATE AGENT POSITION
        direction = DIRECTIONS[action]
        new_pos = state.pos + direction

        # Wrap around boundaries
        new_pos = jnp.mod(new_pos, jnp.array(self.size))

        # Check for blocking objects
        obj_at_new_pos = current_objects[new_pos[1], new_pos[0]]
        is_blocking = self.object_blocking[obj_at_new_pos]
        pos = jax.lax.select(is_blocking, state.pos, new_pos)

        # 2. HANDLE COLLISIONS AND REWARDS
        obj_at_pos = current_objects[pos[1], pos[0]]
        key, subkey = jax.random.split(key)
        reward = jax.lax.switch(obj_at_pos, self.reward_fns, state.time, subkey)
        is_collectable = self.object_collectable[obj_at_pos]

        # 3. HANDLE OBJECT COLLECTION AND RESPAWNING
        key, subkey = jax.random.split(key)

        # Decrement timers (stored as negative values)
        is_timer = state.object_grid < 0
        object_grid = jnp.where(
            is_timer, state.object_grid + num_obj_types, state.object_grid
        )

        # Collect object: set a timer
        regen_delay = jax.lax.switch(
            obj_at_pos, self.regen_delay_fns, state.time, subkey
        )
        encoded_timer = -((regen_delay * num_obj_types) + obj_at_pos)

        # If collected, replace object with timer; otherwise, keep it
        val_at_pos = object_grid[pos[1], pos[0]]
        new_val_at_pos = jax.lax.select(is_collectable, encoded_timer, val_at_pos)
        object_grid = object_grid.at[pos[1], pos[0]].set(new_val_at_pos)

        # 4. UPDATE STATE
        state = EnvState(
            pos=pos,
            object_grid=object_grid,
            time=state.time + 1,
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

        object_grid = jnp.zeros((self.size[1], self.size[0]), dtype=jnp.int32)

        iter_key = subkey
        for i in range(self.biome_object_frequencies.shape[0]):
            iter_key, biome_key = jax.random.split(iter_key)
            # Generate random layout
            grid_rand = jax.random.uniform(biome_key, (self.size[1], self.size[0]))

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

            rows = jnp.arange(self.size[1])[:, None]
            cols = jnp.arange(self.size[0])

            mask = (
                (rows >= start[1])
                & (rows < stop[1])
                & (cols >= start[0])
                & (cols < stop[0])
            )

            # Generate objects for this biome and update the main grid
            cumulative_freq = 0.0
            for j, freq in enumerate(self.biome_object_frequencies[i]):
                obj_id = self.object_ids[j]
                object_grid = jnp.where(
                    mask
                    & (grid_rand >= cumulative_freq)
                    & (grid_rand < cumulative_freq + freq),
                    obj_id,
                    object_grid,
                )
                cumulative_freq += freq

        # Place agent in the center of the world and ensure the cell is empty.
        agent_pos = jnp.array([self.size[0] // 2, self.size[1] // 2])
        object_grid = object_grid.at[agent_pos[1], agent_pos[0]].set(0)

        state = EnvState(
            pos=agent_pos,
            object_grid=object_grid,
            time=0,
        )

        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        """Return observation from raw state trafo."""
        raise NotImplementedError

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Foragax is a continuing environment."""
        return False

    @property
    def name(self) -> str:
        """Environment name."""
        return "Foragax-v0"

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
                    -1000 * len(self.object_ids),
                    len(self.object_ids),
                    (self.size[1], self.size[0]),
                    jnp.int32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "key": spaces.PRNGKey(),
            }
        )

    def render(self, state: EnvState, params: EnvParams):
        """Render the environment state."""
        # Create an RGB image from the object grid
        img = jnp.zeros((self.size[1], self.size[0], 3))
        # Decode grid for rendering: non-negative are objects, negative are empty
        render_grid = jnp.maximum(0, state.object_grid)
        for i, obj_id in enumerate(self.object_ids):
            color = self.object_colors[i]
            img = img.at[render_grid == obj_id].set(jnp.array(color))

        # Agent color
        img = img.at[state.pos[1], state.pos[0]].set(jnp.array(AGENT.color))

        img = img * 255
        img = jax.image.resize(img, (self.size[1] * 24, self.size[0] * 24, 3), jax.image.ResizeMethod.NEAREST)

        grid_color = jnp.array([0, 0, 0])
        row_indices = jnp.arange(1, self.size[1]) * 24
        col_indices = jnp.arange(1, self.size[0]) * 24
        img = img.at[row_indices, :].set(grid_color)
        img = img.at[:, col_indices].set(grid_color)

        return img


class ForagaxObjectEnv(ForagaxEnv):
    """Foragax environment with object-based aperture observation."""

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        num_obj_types = len(self.object_ids)
        # Decode grid for observation
        obs_grid = jnp.maximum(0, state.object_grid)

        # Roll the grid to center the agent's position
        # The agent should be at the center of the aperture, which is aperture_size // 2
        # We want to move the agent's position (state.pos) to the center.
        # The amount to roll is -(state.pos - aperture_size // 2)
        # Note: jnp.roll is on (row, col) but pos is (x, y), so we swap them.
        roll_amount = -(state.pos - jnp.array(self.aperture_size) // 2)
        roll_amount = jnp.array([roll_amount[1], roll_amount[0]])
        rolled_grid = jnp.roll(obs_grid, shift=roll_amount, axis=(0, 1))

        # Extract the aperture
        aperture = jax.lax.dynamic_slice(
            rolled_grid,
            (0, 0),
            self.aperture_size,
        )

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


class ForagaxRGBEnv(ForagaxEnv):
    """Foragax environment with color-based aperture observation."""

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        num_obj_types = len(self.object_ids)
        # Decode grid for observation
        obs_grid = jnp.maximum(0, state.object_grid)
        padded_grid = jnp.pad(
            obs_grid,
            (
                (self.aperture_size[1] // 2, self.aperture_size[1] // 2),
                (self.aperture_size[0] // 2, self.aperture_size[0] // 2),
            ),
            "constant",
            constant_values=0,
        )

        aperture = jax.lax.dynamic_slice(
            padded_grid,
            (state.pos[1], state.pos[0]),
            (self.aperture_size[1], self.aperture_size[0]),
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


class ForagaxWorldEnv(ForagaxEnv):
    """Foragax environment with world observation."""

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        num_obj_types = len(self.object_ids)
        # Decode grid for observation
        obs_grid = jnp.maximum(0, state.object_grid)
        obs = jax.nn.one_hot(obs_grid, num_obj_types)
        obs = obs.at[state.pos[1], state.pos[0], -1].set(1)
        return obs

    def observation_space(self, params: EnvParams) -> spaces.Box:
        num_obj_types = len(self.object_ids)
        obs_shape = (self.size[1], self.size[0], num_obj_types)
        return spaces.Box(0, 1, obs_shape, jnp.float32)
