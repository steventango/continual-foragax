"""JAX implementation of Foragax environment.

Source: https://github.com/andnp/Foragax
"""

from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from foragax.objects import AGENT, EMPTY, BaseForagaxObject


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
        objects: Tuple[BaseForagaxObject, ...] = (),
        biomes: Tuple[Biome, ...] = (Biome(object_frequencies=()),),
    ):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size

        if isinstance(aperture_size, int):
            aperture_size = (aperture_size, aperture_size)
        self.aperture_size = aperture_size

        objects = (EMPTY,) + objects

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
        encoded_timer = obj_at_pos - ((regen_delay + 1) * num_obj_types)

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

        object_grid = jnp.zeros((self.size[1], self.size[0]), dtype=jnp.int_)

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
            biome_freqs = self.biome_object_frequencies[i]
            empty_freq = 1.0 - jnp.sum(biome_freqs)
            all_freqs = jnp.concatenate([jnp.array([empty_freq]), biome_freqs])

            cumulative_freqs = jnp.cumsum(
                jnp.concatenate([jnp.array([0.0]), all_freqs])
            )

            # Determine which object to place in each cell
            # The last object ID will be used for any value of grid_rand >= cumulative_freqs[-1]
            # so we don't need to cap grid_rand
            obj_ids_for_biome = jnp.arange(len(all_freqs))
            cell_obj_ids = (
                jnp.searchsorted(cumulative_freqs, grid_rand, side="right") - 1
            )
            biome_objects = obj_ids_for_biome[cell_obj_ids]

            object_grid = jnp.where(mask, biome_objects, object_grid)

        # Place agent in the center of the world and ensure the cell is empty.
        agent_pos = jnp.array([self.size[0] // 2, self.size[1] // 2])
        object_grid = object_grid.at[agent_pos[1], agent_pos[0]].set(0)

        state = EnvState(
            pos=agent_pos,
            object_grid=object_grid,
            time=0,
        )

        return self.get_obs(state, params), state

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
        action_space = spaces.Discrete(self.num_actions)
        # NOTE: workaround for https://github.com/RobertTLange/gymnax/issues/58
        action_space.dtype = jnp.int_
        return action_space

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "pos": spaces.Box(0, max(self.size), (2,), jnp.int_),
                "object_grid": spaces.Box(
                    -1000 * len(self.object_ids),
                    len(self.object_ids),
                    (self.size[1], self.size[0]),
                    jnp.int_,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    def _get_aperture(self, object_grid: jax.Array, pos: jax.Array) -> jax.Array:
        """Extract the aperture view from the object grid."""
        ap_h, ap_w = self.aperture_size
        start_y = pos[1] - ap_h // 2
        start_x = pos[0] - ap_w // 2

        y_offsets = jnp.arange(ap_h)
        x_offsets = jnp.arange(ap_w)
        y_coords = jnp.mod(start_y + y_offsets[:, None], self.size[1])
        x_coords = jnp.mod(start_x + x_offsets, self.size[0])

        return object_grid[y_coords, x_coords]

    @partial(jax.jit, static_argnames=("self", "render_mode"))
    def render(self, state: EnvState, params: EnvParams, render_mode: str = "world"):
        """Render the environment state."""
        if render_mode == "world":
            # Create an RGB image from the object grid
            img = jnp.zeros((self.size[1], self.size[0], 3))
            # Decode grid for rendering: non-negative are objects, negative are empty
            render_grid = jnp.maximum(0, state.object_grid)

            def update_image(i, img):
                color = self.object_colors[i]
                mask = render_grid == i
                img = jnp.where(mask[..., None], color, img)
                return img

            img = jax.lax.fori_loop(0, len(self.object_ids), update_image, img)

            # Tint the agent's aperture
            ap_h, ap_w = self.aperture_size
            start_y = state.pos[1] - ap_h // 2
            start_x = state.pos[0] - ap_w // 2

            alpha = 0.2
            agent_color = jnp.array(AGENT.color)

            # Create indices for the aperture
            y_offsets = jnp.arange(ap_h)
            x_offsets = jnp.arange(ap_w)
            y_coords = jnp.mod(start_y + y_offsets[:, None], self.size[1])
            x_coords = jnp.mod(start_x + x_offsets, self.size[0])

            # Get original colors from the aperture area
            original_colors = img[y_coords, x_coords]

            # Calculate tinted colors
            tinted_colors = (1 - alpha) * original_colors + alpha * agent_color

            # Update the image with tinted colors
            img = img.at[y_coords, x_coords].set(tinted_colors)

            # Agent color
            img = img.at[state.pos[1], state.pos[0]].set(jnp.array(AGENT.color))

            img = jax.image.resize(
                img,
                (self.size[1] * 24, self.size[0] * 24, 3),
                jax.image.ResizeMethod.NEAREST,
            )

            grid_color = jnp.zeros(3, dtype=jnp.uint8)
            row_indices = jnp.arange(1, self.size[1]) * 24
            col_indices = jnp.arange(1, self.size[0]) * 24
            img = img.at[row_indices, :].set(grid_color)
            img = img.at[:, col_indices].set(grid_color)

            return img

        elif render_mode == "aperture":
            obs_grid = jnp.maximum(0, state.object_grid)
            aperture = self._get_aperture(obs_grid, state.pos)
            aperture_one_hot = jax.nn.one_hot(aperture, len(self.object_ids))
            img = jnp.tensordot(aperture_one_hot, self.object_colors, axes=1)

            # Draw agent in the center
            center_y, center_x = self.aperture_size[1] // 2, self.aperture_size[0] // 2
            img = img.at[center_y, center_x].set(jnp.array(AGENT.color))

            img = img.astype(jnp.uint8)
            img = jax.image.resize(
                img,
                (self.aperture_size[0] * 24, self.aperture_size[1] * 24, 3),
                jax.image.ResizeMethod.NEAREST,
            )

            grid_color = jnp.zeros(3, dtype=jnp.uint8)
            row_indices = jnp.arange(1, self.aperture_size[0]) * 24
            col_indices = jnp.arange(1, self.aperture_size[1]) * 24
            img = img.at[row_indices, :].set(grid_color)
            img = img.at[:, col_indices].set(grid_color)

            return img
        else:
            raise ValueError(f"Unknown render_mode: {render_mode}")


class ForagaxObjectEnv(ForagaxEnv):
    """Foragax environment with object-based aperture observation."""

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        num_obj_types = len(self.object_ids)
        # Decode grid for observation
        obs_grid = jnp.maximum(0, state.object_grid)
        aperture = self._get_aperture(obs_grid, state.pos)
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
        return spaces.Box(0, 1, obs_shape, jnp.float_)


class ForagaxRGBEnv(ForagaxEnv):
    """Foragax environment with color-based aperture observation."""

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        num_obj_types = len(self.object_ids)
        # Decode grid for observation
        obs_grid = jnp.maximum(0, state.object_grid)
        aperture = self._get_aperture(obs_grid, state.pos)
        aperture_one_hot = jax.nn.one_hot(aperture, num_obj_types)

        # Agent position is always at the center of the aperture
        center = (self.aperture_size[1] // 2, self.aperture_size[0] // 2)
        aperture_one_hot = aperture_one_hot.at[center[0], center[1], :].set(0)
        aperture_one_hot = aperture_one_hot.at[center[0], center[1], -1].set(1)

        colors = self.object_colors / 255.0
        obs = jnp.tensordot(aperture_one_hot, colors, axes=1)
        return obs

    def observation_space(self, params: EnvParams) -> spaces.Box:
        obs_shape = (self.aperture_size[0], self.aperture_size[1], 3)
        return spaces.Box(0, 1, obs_shape, jnp.float_)


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
        return spaces.Box(0, 1, obs_shape, jnp.float_)
