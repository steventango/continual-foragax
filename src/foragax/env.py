"""JAX implementation of Foragax environment.

Source: https://github.com/andnp/Foragax
"""

from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from typing import Any, Dict, Tuple, Union

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from foragax.objects import (
    AGENT,
    EMPTY,
    PADDING,
    BaseForagaxObject,
    WeatherObject,
)
from foragax.rendering import apply_true_borders
from foragax.weather import get_temperature


class Actions(IntEnum):
    DOWN = 0
    RIGHT = 1
    UP = 2
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
    start: Union[Tuple[int, int], None] = None
    stop: Union[Tuple[int, int], None] = None


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: Union[int, None]


@struct.dataclass
class EnvState(environment.EnvState):
    pos: jax.Array
    object_grid: jax.Array
    time: int


class ForagaxEnv(environment.Environment):
    """JAX implementation of Foragax environment."""

    def __init__(
        self,
        size: Union[Tuple[int, int], int] = (10, 10),
        aperture_size: Union[Tuple[int, int], int] = (5, 5),
        objects: Tuple[BaseForagaxObject, ...] = (),
        biomes: Tuple[Biome, ...] = (Biome(object_frequencies=()),),
        nowrap: bool = False,
    ):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size

        if isinstance(aperture_size, int):
            aperture_size = (aperture_size, aperture_size)
        self.aperture_size = aperture_size
        self.nowrap = nowrap
        objects = (EMPTY,) + objects
        if self.nowrap:
            objects = objects + (PADDING,)
        self.objects = objects
        self.weather_object = None
        for o in objects:
            if isinstance(o, WeatherObject):
                self.weather_object = o
                break

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
        action: Union[int, float, jax.Array],
        params: EnvParams,
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, Dict[Any, Any]]:
        """Perform single timestep state transition."""
        num_obj_types = len(self.object_ids)
        # Decode the object grid: positive values are objects, negative are timers (treat as empty)
        current_objects = jnp.maximum(0, state.object_grid)

        # 1. UPDATE AGENT POSITION
        direction = DIRECTIONS[action]
        new_pos = state.pos + direction

        if self.nowrap:
            in_bounds = jnp.all((new_pos >= 0) & (new_pos < jnp.array(self.size)))
            new_pos = jnp.where(in_bounds, new_pos, state.pos)
        else:
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

        info = {"discount": self.discount(state, params)}
        if self.weather_object is not None:
            info["temperature"] = get_temperature(
                self.weather_object.rewards, state.time, self.weather_object.repeat
            )

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
            info,
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        """Reset environment state."""
        key, subkey = jax.random.split(key)

        object_grid = jnp.zeros((self.size[1], self.size[0]), dtype=int)

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
        action_space.dtype = int
        return action_space

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "pos": spaces.Box(0, max(self.size), (2,), int),
                "object_grid": spaces.Box(
                    -1000 * len(self.object_ids),
                    len(self.object_ids),
                    (self.size[1], self.size[0]),
                    int,
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
        y_coords = start_y + y_offsets[:, None]
        x_coords = start_x + x_offsets

        if self.nowrap:
            # Clamp coordinates to bounds
            y_coords_clamped = jnp.clip(y_coords, 0, self.size[1] - 1)
            x_coords_clamped = jnp.clip(x_coords, 0, self.size[0] - 1)
            values = object_grid[y_coords_clamped, x_coords_clamped]
            # Mark out-of-bounds positions with -1
            y_out = (y_coords < 0) | (y_coords >= self.size[1])
            x_out = (x_coords < 0) | (x_coords >= self.size[0])
            out_of_bounds = y_out | x_out
            padding_index = self.object_ids[-1]
            aperture = jnp.where(out_of_bounds, padding_index, values)
        else:
            y_coords_mod = jnp.mod(y_coords, self.size[1])
            x_coords_mod = jnp.mod(x_coords, self.size[0])
            aperture = object_grid[y_coords_mod, x_coords_mod]

        return aperture

    @partial(jax.jit, static_argnames=("self", "render_mode"))
    def render(self, state: EnvState, params: EnvParams, render_mode: str = "world"):
        """Render the environment state."""
        is_world_mode = render_mode in ("world", "world_true")
        is_aperture_mode = render_mode in ("aperture", "aperture_true")
        is_true_mode = render_mode in ("world_true", "aperture_true")

        if is_world_mode:
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
            y_coords_original = start_y + y_offsets[:, None]
            x_coords_original = start_x + x_offsets

            if self.nowrap:
                y_coords = jnp.clip(y_coords_original, 0, self.size[1] - 1)
                x_coords = jnp.clip(x_coords_original, 0, self.size[0] - 1)
                # Create tint mask: any in-bounds original position maps to a cell makes it tinted
                tint_mask = jnp.zeros((self.size[1], self.size[0]), dtype=int)
                tint_mask = tint_mask.at[y_coords, x_coords].set(1)
                # Apply tint to masked positions
                original_colors = img
                tinted_colors = (1 - alpha) * original_colors + alpha * agent_color
                img = jnp.where(tint_mask[..., None], tinted_colors, img)
            else:
                y_coords = jnp.mod(y_coords_original, self.size[1])
                x_coords = jnp.mod(x_coords_original, self.size[0])
                original_colors = img[y_coords, x_coords]
                tinted_colors = (1 - alpha) * original_colors + alpha * agent_color
                img = img.at[y_coords, x_coords].set(tinted_colors)

            # Agent color
            img = img.at[state.pos[1], state.pos[0]].set(jnp.array(AGENT.color))

            img = jax.image.resize(
                img,
                (self.size[1] * 24, self.size[0] * 24, 3),
                jax.image.ResizeMethod.NEAREST,
            )

            if is_true_mode:
                # Apply true object borders by overlaying true colors on border pixels
                img = apply_true_borders(
                    img, render_grid, self.size, len(self.object_ids)
                )

            # Add grid lines for world mode
            grid_color = jnp.zeros(3, dtype=jnp.uint8)
            row_indices = jnp.arange(1, self.size[1]) * 24
            col_indices = jnp.arange(1, self.size[0]) * 24
            img = img.at[row_indices, :].set(grid_color)
            img = img.at[:, col_indices].set(grid_color)

        elif is_aperture_mode:
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

            if is_true_mode:
                # Apply true object borders by overlaying true colors on border pixels
                img = apply_true_borders(
                    img, aperture, self.aperture_size, len(self.object_ids)
                )

            # Add grid lines for aperture mode
            grid_color = jnp.zeros(3, dtype=jnp.uint8)
            row_indices = jnp.arange(1, self.aperture_size[0]) * 24
            col_indices = jnp.arange(1, self.aperture_size[1]) * 24
            img = img.at[row_indices, :].set(grid_color)
            img = img.at[:, col_indices].set(grid_color)

        else:
            raise ValueError(f"Unknown render_mode: {render_mode}")

        return img


class ForagaxObjectEnv(ForagaxEnv):
    """Foragax environment with object-based aperture observation."""

    def __init__(
        self,
        size: Union[Tuple[int, int], int] = (10, 10),
        aperture_size: Union[Tuple[int, int], int] = (5, 5),
        objects: Tuple[BaseForagaxObject, ...] = (),
        biomes: Tuple[Biome, ...] = (Biome(object_frequencies=()),),
        nowrap: bool = False,
    ):
        super().__init__(size, aperture_size, objects, biomes, nowrap)

        # Compute unique colors and mapping for partial observability
        # Exclude EMPTY (index 0) from color channels
        object_colors_no_empty = self.object_colors[1:]

        # Find unique colors in order of first appearance
        unique_colors = []
        color_indices = jnp.zeros(len(object_colors_no_empty), dtype=int)
        color_map = {}
        next_channel = 0

        for i, color in enumerate(object_colors_no_empty):
            color_tuple = tuple(color.tolist())
            if color_tuple not in color_map:
                color_map[color_tuple] = next_channel
                unique_colors.append(color)
                next_channel += 1
            color_indices = color_indices.at[i].set(color_map[color_tuple])

        self.unique_colors = jnp.array(unique_colors)
        self.num_color_channels = len(unique_colors)
        # color_indices maps from object_id-1 to color_channel_index
        self.object_to_color_map = color_indices

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        # Decode grid for observation
        obs_grid = jnp.maximum(0, state.object_grid)
        aperture = self._get_aperture(obs_grid, state.pos)

        # Handle case with no objects (only EMPTY)
        if self.num_color_channels == 0:
            return jnp.zeros(aperture.shape + (0,), dtype=jnp.float32)

        # Map object IDs to color channel indices
        # aperture contains object IDs (0 = EMPTY, 1+ = objects)
        # For EMPTY (0), we want no color channel activated
        # For objects (1+), map to color channel using object_to_color_map
        color_channels = jnp.where(
            aperture == 0,
            -1,  # Special value for EMPTY
            jnp.take(self.object_to_color_map, aperture - 1, axis=0),
        )

        # Create one-hot encoding for color channels
        # jax.nn.one_hot produces all zeros for -1 (EMPTY positions)
        obs = jax.nn.one_hot(color_channels, self.num_color_channels, axis=-1)

        return obs

    def observation_space(self, params: EnvParams) -> spaces.Box:
        obs_shape = (
            self.aperture_size[0],
            self.aperture_size[1],
            self.num_color_channels,
        )
        return spaces.Box(0, 1, obs_shape, float)


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
        return spaces.Box(0, 1, obs_shape, float)


class ForagaxWorldEnv(ForagaxEnv):
    """Foragax environment with world observation."""

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        num_obj_types = len(self.object_ids)
        # Decode grid for observation
        obs_grid = jnp.maximum(0, state.object_grid)
        obs = jnp.zeros((self.size[1], self.size[0], num_obj_types), dtype=jnp.float32)

        num_object_channels = num_obj_types - 1
        # create masks for all objects at once
        object_ids = jnp.arange(1, num_obj_types)
        object_masks = obs_grid[..., None] == object_ids[None, None, :]
        obs = obs.at[:, :, :num_object_channels].set(object_masks.astype(float))

        obs = obs.at[state.pos[1], state.pos[0], -1].set(1)
        return obs

    def observation_space(self, params: EnvParams) -> spaces.Box:
        num_obj_types = len(self.object_ids)
        obs_shape = (self.size[1], self.size[0], num_obj_types)
        return spaces.Box(0, 1, obs_shape, float)
