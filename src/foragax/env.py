"""JAX implementation of Forager environment.

Source: https://github.com/andnp/Forager
"""

from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces

from foragax.objects import (
    AGENT,
    EMPTY,
    PADDING,
    BaseForagaxObject,
    FourierObject,
    SineObject,
    WeatherObject,
    WeatherWaveObject,
)
from foragax.rendering import (
    apply_grid_lines,
    apply_reward_overlay,
    apply_true_borders,
    get_base_image,
    reward_to_color,
    apply_hint_bottom_bar,
)
from foragax.weather import get_temperature


ID_DTYPE = jnp.int32  # Object type ID (0 = empty, >0 = object type)
TIMER_DTYPE = jnp.int32  # Respawn countdown (0 = no timer, >0 = countdown)
TIME_DTYPE = jnp.int32  # Timesteps (spawn time, current time)
PARAM_DTYPE = jnp.float16  # Per-instance object parameters
COLOR_DTYPE = jnp.uint8  # RGB color channels (0-255)
BIOME_ID_DTYPE = jnp.int16  # Biome assignment for each cell
REWARD_DTYPE = jnp.float32  # Reward values


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


@struct.dataclass
class ObjectState:
    """Per-cell object state information.

    This struct encapsulates all state information about objects in the grid:
    - object_id: (H, W) The object type (0 for empty, positive for object type)
    - respawn_timer: (H, W) Countdown timer for respawning (0 = no timer, positive = countdown remaining)
    - respawn_object_id: (H, W) What object type will spawn when timer reaches 0
    - spawn_time: (H, W) When each object was spawned (for expiry tracking)
    - color: (H, W, 3) RGB color for each object instance (for dynamic biomes)
    - generation: (H, W) Which biome generation each object belongs to
    - state_params: (H, W, N) Per-instance parameters (e.g., Fourier coefficients)
    - biome_id: (H, W) Which biome each cell belongs to
    """

    object_id: jax.Array  # (H, W) - Object type ID (0 = empty, >0 = object type)
    respawn_timer: (
        jax.Array
    )  # (H, W) - Respawn countdown (0 = no timer, >0 = countdown)
    respawn_object_id: jax.Array  # (H, W) - Object type to spawn when timer reaches 0
    spawn_time: jax.Array  # (H, W) - Timestep when object spawned
    color: jax.Array  # (H, W, 3) - RGB color per instance
    generation: jax.Array  # (H, W) - Biome generation number
    state_params: jax.Array  # (H, W, N) - Per-instance parameters
    biome_id: jax.Array  # (H, W) - Biome assignment for each cell

    @classmethod
    def create_empty(cls, size: Tuple[int, int], num_params: int) -> "ObjectState":
        """Create an empty ObjectState for the given grid size."""
        h, w = size[1], size[0]
        return cls(
            object_id=jnp.zeros((h, w), dtype=ID_DTYPE),
            respawn_timer=jnp.zeros((h, w), dtype=TIMER_DTYPE),
            respawn_object_id=jnp.zeros((h, w), dtype=ID_DTYPE),
            spawn_time=jnp.zeros((h, w), dtype=TIME_DTYPE),
            color=jnp.full((h, w, 3), 255, dtype=COLOR_DTYPE),
            generation=jnp.zeros((h, w), dtype=ID_DTYPE),
            state_params=jnp.zeros((h, w, num_params), dtype=PARAM_DTYPE),
            biome_id=jnp.full((h, w), -1, dtype=BIOME_ID_DTYPE),
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
class BiomeState:
    """Biome-level tracking state (num_biomes,)."""

    consumption_count: jax.Array  # objects consumed per biome
    total_objects: jax.Array  # total objects spawned per biome
    generation: jax.Array  # current generation per biome


@struct.dataclass
class EnvState(environment.EnvState):
    pos: jax.Array
    time: int
    offset: int
    digestion_buffer: jax.Array
    object_state: ObjectState
    biome_state: BiomeState


class ForagaxEnv(environment.Environment):
    """JAX implementation of Foragax environment."""

    def __init__(
        self,
        name: str = "Foragax-v0",
        size: Union[Tuple[int, int], int] = (10, 10),
        aperture_size: Union[Tuple[int, int], int] = (5, 5),
        objects: Tuple[BaseForagaxObject, ...] = (),
        biomes: Tuple[Biome, ...] = (Biome(object_frequencies=()),),
        nowrap: bool = False,
        deterministic_spawn: bool = False,
        observation_type: str = "object",
        dynamic_biomes: bool = False,
        biome_consumption_threshold: float = 0.9,
        dynamic_biome_spawn_empty: float = 0.0,
        max_expiries_per_step: int = 1,
        center_reward: bool = False,
        return_hint: bool = False,
        hint_every: int = 100,
        hint_duration: int = 4,
        random_shift_max_steps: int = 0,
        random_teleport_period: int = 0,
        random_teleport_offset: int = 0,
        deterministic_teleport_period: int = 0,
        deterministic_teleport_offset: int = 0,
    ):
        super().__init__()
        self._name = name
        if isinstance(size, int):
            size = (size, size)
        self.size = size

        # Handle aperture_size = -1 for world view
        if isinstance(aperture_size, int) and aperture_size == -1:
            self.full_world = True
            # size is (W, H), aperture_size is (H, W)
            self.aperture_size = (self.size[1], self.size[0])
        else:
            self.full_world = False
            if isinstance(aperture_size, int):
                aperture_size = (aperture_size, aperture_size)
            self.aperture_size = aperture_size

        self.observation_type = observation_type
        self.nowrap = nowrap
        self.deterministic_spawn = deterministic_spawn
        self.dynamic_biomes = dynamic_biomes
        self.biome_consumption_threshold = biome_consumption_threshold
        self.dynamic_biome_spawn_empty = dynamic_biome_spawn_empty
        if max_expiries_per_step < 1:
            raise ValueError("max_expiries_per_step must be at least 1")
        self.max_expiries_per_step = max_expiries_per_step
        self.center_reward = center_reward
        self.return_hint = return_hint
        self.hint_every = hint_every
        self.hint_duration = hint_duration
        self.random_shift_max_steps = random_shift_max_steps
        self.random_teleport_period = random_teleport_period
        self.random_teleport_offset = random_teleport_offset
        self.deterministic_teleport_period = deterministic_teleport_period
        self.deterministic_teleport_offset = deterministic_teleport_offset

        objects = (EMPTY,) + objects
        if self.nowrap and not self.full_world:
            objects = objects + (PADDING,)
        self.objects = objects

        # Identify real objects (index 0 is EMPTY, last might be PADDING)
        self.real_object_start = 1
        num_objects = len(self.objects)
        self.real_object_end = num_objects - (
            1 if self.nowrap and not self.full_world else 0
        )
        self.real_object_indices = jnp.arange(
            self.real_object_start, self.real_object_end
        )

        # Infer num_fourier_terms and record which objects are Fourier types
        self.num_fourier_terms = max(
            (
                obj.num_fourier_terms
                for obj in self.objects
                if isinstance(obj, FourierObject)
            ),
            default=0,
        )
        self.object_is_fourier = jnp.array(
            [isinstance(obj, (FourierObject, SineObject)) for obj in self.objects]
        )

        # JIT-compatible versions of object and biome properties
        self.object_ids = jnp.arange(len(objects))
        self.object_blocking = jnp.array([o.blocking for o in objects])
        self.object_collectable = jnp.array([o.collectable for o in objects])
        self.object_colors = jnp.array([o.color for o in objects])
        self.object_random_respawn = jnp.array([o.random_respawn for o in objects])

        self.reward_fns = [o.reward for o in objects]
        self.regen_delay_fns = [o.regen_delay for o in objects]
        self.reward_delay_fns = [o.reward_delay for o in objects]
        self.expiry_regen_delay_fns = [o.expiry_regen_delay for o in objects]

        # Expiry times per object (None becomes -1 for no expiry)
        self.object_expiry_time = jnp.array(
            [o.expiry_time if o.expiry_time is not None else -1 for o in objects]
        )

        # Check if any objects can expire
        self.has_expiring_objects = jnp.any(self.object_expiry_time >= 0)

        # Compute reward steps per object (using max_reward_delay attribute)
        object_max_reward_delay = jnp.array([o.max_reward_delay for o in objects])
        self.max_reward_delay = (
            int(jnp.max(object_max_reward_delay)) + 1 if len(objects) > 0 else 0
        )

        self.biome_object_frequencies = jnp.array(
            [b.object_frequencies for b in biomes]
        )
        self.biome_starts = np.array(
            [b.start if b.start is not None else (-1, -1) for b in biomes]
        )
        self.biome_stops = np.array(
            [b.stop if b.stop is not None else (-1, -1) for b in biomes]
        )

        # Precompute whether each biome contains ANY Fourier objects (for static reset_env branching)
        self.is_fourier_biome = []
        for b in biomes:
            is_fourier = any(
                isinstance(self.objects[j + 1], (FourierObject, SineObject))
                for j, freq in enumerate(b.object_frequencies)
                if freq > 0
            )
            self.is_fourier_biome.append(is_fourier)
        self.biome_sizes = np.prod(self.biome_stops - self.biome_starts, axis=1)
        # Precompute the order to apply biomes: largest to smallest, ensuring smaller biomes overwrite correctly.
        self.biome_order = np.argsort(self.biome_sizes)[::-1].tolist()
        self.biome_sizes_jax = jnp.array(self.biome_sizes)  # JAX version for indexing
        self.biome_starts_jax = jnp.array(self.biome_starts)
        self.biome_stops_jax = jnp.array(self.biome_stops)
        biome_centers = []
        for i in range(len(self.biome_starts)):
            start = self.biome_starts[i]
            stop = self.biome_stops[i]
            center_x = (start[0] + stop[0] - 1) // 2
            center_y = (start[1] + stop[1] - 1) // 2
            biome_centers.append((center_x, center_y))
        self.biome_centers_jax = jnp.array(biome_centers)
        self.biome_masks = []
        biome_masks_array = []
        for i in range(self.biome_object_frequencies.shape[0]):
            # Create mask for the biome
            start = jax.lax.select(
                self.biome_starts[i, 0] == -1,
                jnp.array([0, 0], dtype=jnp.int32),
                self.biome_starts[i].astype(jnp.int32),
            )
            stop = jax.lax.select(
                self.biome_stops[i, 0] == -1,
                jnp.array(self.size, dtype=jnp.int32),
                self.biome_stops[i].astype(jnp.int32),
            )
            rows = jnp.arange(self.size[1])[:, None]
            cols = jnp.arange(self.size[0])
            mask = (
                (rows >= start[1])
                & (rows < stop[1])
                & (cols >= start[0])
                & (cols < stop[0])
            )
            self.biome_masks.append(mask)
            biome_masks_array.append(mask)

        # Convert to JAX array for indexing in JIT-compiled code
        self.biome_masks_array = jnp.array(biome_masks_array)

        # Identify "food biomes" (those defined with non-blocking objects)
        # Handle cases where biome_object_frequencies might have fewer columns than objects
        num_freq_cols = self.biome_object_frequencies.shape[1]
        is_food_mask = ~self.object_blocking[1 : 1 + num_freq_cols]
        is_food_biome = jnp.any(
            (jnp.array(self.biome_object_frequencies) > 0)
            & jnp.array(is_food_mask)[None, :],
            axis=1,
        )
        self.is_food_biome = is_food_biome
        self.food_biome_indices = jnp.where(is_food_biome)[0]
        self.num_food_biomes = len(self.food_biome_indices)

        # Pre-calculate effective metrics index grid
        # This grid maps each cell to its index in the [food_biomes] + [void] metrics array.
        # Length of metrics array: num_food_biomes + 1
        metrics_idx_grid = np.full(
            (self.size[1], self.size[0]), self.num_food_biomes, dtype=np.int32
        )
        # Fill food biomes in order (later biomes overwrite earlier ones if overlapping)
        for i, original_idx in enumerate(self.food_biome_indices.tolist()):
            mask = np.array(self.biome_masks[original_idx])
            metrics_idx_grid[mask] = i

        self.cell_to_metrics_idx_grid = jnp.array(metrics_idx_grid)

        # Compute unique colors and mapping for partial observability (for 'color' observation_type)
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

        # Rendering constants
        self.agent_color_jax = jnp.array(AGENT.color, dtype=jnp.uint8)
        self.white_color_jax = jnp.array([255, 255, 255], dtype=jnp.uint8)
        self.grid_color_jax = jnp.zeros(3, dtype=jnp.uint8)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(
            max_steps_in_episode=None,
        )

    @partial(jax.named_call, name="_place_timer")
    def _place_timer(
        self,
        object_state: ObjectState,
        y: int,
        x: int,
        object_type: int,
        timer_val: int,
        random_respawn: bool,
        rand_key: jax.Array,
    ) -> ObjectState:
        """Place a timer at position or randomly within the same biome.

        Args:
            object_state: Current object state
            y, x: Original position
            object_type: The object type ID that will respawn (0 for permanent removal)
            timer_val: Timer countdown value (0 for permanent removal, positive for countdown)
            random_respawn: If True, place at random location in same biome
            rand_key: Random key for random placement

        Returns:
            Updated object_state with timer placed
        """
        # Ensure inputs match ObjectState dtypes
        object_type = jnp.array(object_type, dtype=ID_DTYPE)
        timer_val = jnp.array(timer_val, dtype=TIMER_DTYPE)
        y = jnp.array(y, dtype=jnp.int32)
        x = jnp.array(x, dtype=jnp.int32)

        # Handle permanent removal (timer_val == 0)
        def place_empty():
            return object_state.replace(
                object_id=object_state.object_id.at[y, x].set(
                    jnp.array(0, dtype=ID_DTYPE)
                ),
                respawn_timer=object_state.respawn_timer.at[y, x].set(
                    jnp.array(0, dtype=TIMER_DTYPE)
                ),
                respawn_object_id=object_state.respawn_object_id.at[y, x].set(
                    jnp.array(0, dtype=ID_DTYPE)
                ),
            )

        # Handle timer placement
        def place_timer():
            # Non-random: place at original position
            def place_at_position():
                return object_state.replace(
                    object_id=object_state.object_id.at[y, x].set(
                        jnp.array(0, dtype=ID_DTYPE)
                    ),
                    respawn_timer=object_state.respawn_timer.at[y, x].set(timer_val),
                    respawn_object_id=object_state.respawn_object_id.at[y, x].set(
                        object_type
                    ),
                )

            # Random: place at random location in same biome
            def place_randomly():
                # Clear the collected object's position
                new_object_id = object_state.object_id.at[y, x].set(
                    jnp.array(0, dtype=ID_DTYPE)
                )
                new_respawn_timer = object_state.respawn_timer.at[y, x].set(
                    jnp.array(0, dtype=TIMER_DTYPE)
                )
                new_respawn_object_id = object_state.respawn_object_id.at[y, x].set(
                    jnp.array(0, dtype=ID_DTYPE)
                )

                # Extract the actual state to move
                obj_color = object_state.color[y, x]
                obj_params = object_state.state_params[y, x]
                obj_gen = object_state.generation[y, x]

                # Clear visuals at old position
                new_color = object_state.color.at[y, x].set(
                    jnp.zeros(3, dtype=COLOR_DTYPE)
                )
                new_params = object_state.state_params.at[y, x].set(
                    jnp.zeros_like(obj_params)
                )

                # Find valid spawn locations in the same biome
                biome_id = object_state.biome_id[y, x]
                biome_mask = object_state.biome_id == biome_id
                empty_mask = new_object_id == 0
                no_timer_mask = new_respawn_timer == 0
                valid_spawn_mask = biome_mask & empty_mask & no_timer_mask
                num_valid_spawns = jnp.sum(valid_spawn_mask, dtype=jnp.int32)

                y_indices, x_indices = jnp.nonzero(
                    valid_spawn_mask, size=self.size[0] * self.size[1], fill_value=-1
                )
                valid_spawn_indices = jnp.stack([y_indices, x_indices], axis=1)
                random_idx = jax.random.randint(
                    rand_key, (), jnp.array(0, dtype=jnp.int32), num_valid_spawns
                )
                new_spawn_pos = valid_spawn_indices[random_idx]

                # Place timer and move properties at the new random position
                new_respawn_timer = new_respawn_timer.at[
                    new_spawn_pos[0], new_spawn_pos[1]
                ].set(timer_val)
                new_respawn_object_id = new_respawn_object_id.at[
                    new_spawn_pos[0], new_spawn_pos[1]
                ].set(object_type)

                # Move properties to new position
                new_color = new_color.at[new_spawn_pos[0], new_spawn_pos[1]].set(
                    obj_color
                )
                new_params = new_params.at[new_spawn_pos[0], new_spawn_pos[1]].set(
                    obj_params
                )
                new_generation = object_state.generation.at[
                    new_spawn_pos[0], new_spawn_pos[1]
                ].set(obj_gen)

                return object_state.replace(
                    object_id=new_object_id,
                    respawn_timer=new_respawn_timer,
                    respawn_object_id=new_respawn_object_id,
                    color=new_color,
                    state_params=new_params,
                    generation=new_generation,
                )

            return jax.lax.cond(random_respawn, place_randomly, place_at_position)

        return jax.lax.cond(timer_val == 0, place_empty, place_timer)

    @partial(jax.named_call, name="move_agent")
    def _move_agent(
        self, state: EnvState, action: Union[int, float, jax.Array]
    ) -> Tuple[jax.Array, jax.Array]:
        current_objects = state.object_state.object_id
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
        is_blocking = self.object_blocking[obj_at_new_pos.astype(jnp.int32)]
        pos = jnp.where(
            is_blocking[..., None],
            state.pos.astype(jnp.int32),
            new_pos.astype(jnp.int32),
        )
        return pos, new_pos

    def _get_global_mean_reward(self, state: EnvState) -> jax.Array:
        """Calculate mean reward of all active objects in the world."""
        if not self.center_reward:
            return jnp.array(0.0)

        fixed_key = jax.random.key(0)  # Use fixed key for baseline rewards

        def compute_reward(obj_id, params, timer):
            reward = jax.lax.switch(
                obj_id.astype(jnp.int32),
                self.reward_fns,
                state.time,
                fixed_key,
                params.astype(jnp.float32),
            )
            # Mask for "real" active objects (not EMPTY/PADDING and not respawning)
            is_real = (obj_id >= self.real_object_start) & (
                obj_id < self.real_object_end
            )
            active = timer == 0
            # Ignore walls (blocking objects)
            is_not_wall = ~self.object_blocking[obj_id.astype(jnp.int32)]
            mask = is_real & active & is_not_wall
            return jnp.where(mask, reward, 0.0), mask.astype(jnp.float32)

        rewards, masks = jax.vmap(jax.vmap(compute_reward))(
            state.object_state.object_id,
            state.object_state.state_params,
            state.object_state.respawn_timer,
        )
        total_reward = jnp.sum(rewards)
        count = jnp.sum(masks)
        return jnp.where(count > 0, total_reward / count, 0.0)

    def _get_biome_mean_rewards(self, state: EnvState) -> Tuple[jax.Array, jax.Array]:
        """Calculate mean reward for each effective metrics region (food biomes + void)."""
        fixed_key = jax.random.key(0)  # Use fixed key for baseline rewards

        def compute_reward(obj_id, params, timer):
            reward = jax.lax.switch(
                obj_id.astype(jnp.int32),
                self.reward_fns,
                state.time,
                fixed_key,
                params.astype(jnp.float32),
            )
            # Mask for "real" active objects (not EMPTY/PADDING and not respawning)
            is_real = (obj_id >= self.real_object_start) & (
                obj_id < self.real_object_end
            )
            active = timer == 0
            # Ignore walls (blocking objects)
            is_not_wall = ~self.object_blocking[obj_id.astype(jnp.int32)]

            mask = is_real & active & is_not_wall
            return jnp.where(mask, reward, 0.0), mask.astype(jnp.float32)

        rewards, masks = jax.vmap(jax.vmap(compute_reward))(
            state.object_state.object_id,
            state.object_state.state_params,
            state.object_state.respawn_timer,
        )

        # Aggregate using the pre-calculated metrics index grid
        # This ensures wall biomes are ignored and overlaps are resolved to food biomes.
        num_metrics_regions = self.num_food_biomes + 1

        def get_region_mask(i):
            return self.cell_to_metrics_idx_grid == i

        region_masks = jax.vmap(get_region_mask)(jnp.arange(num_metrics_regions))

        biome_total_rewards = jnp.sum(region_masks * rewards[None, ...], axis=(1, 2))
        biome_counts = jnp.sum(region_masks * masks[None, ...], axis=(1, 2))

        means = jnp.where(biome_counts > 0, biome_total_rewards / biome_counts, 0.0)
        return means, biome_counts

    def _apply_centering(
        self,
        reward: jax.Array,
        obj_id: jax.Array,
        mean_reward: jax.Array,
    ) -> jax.Array:
        """Subtract global mean reward if centering is enabled."""
        if not self.center_reward:
            return reward

        # Subtract mean if the object is a real object and not a wall
        is_real = (obj_id >= self.real_object_start) & (obj_id < self.real_object_end)
        is_not_wall = ~self.object_blocking[obj_id.astype(jnp.int32)]

        return jnp.where(
            is_real & is_not_wall,
            reward - mean_reward,
            reward,
        )

    @partial(jax.named_call, name="random_teleport")
    def _random_teleport(self, state: EnvState, key: jax.Array) -> jax.Array:
        """Randomly teleport agent to an empty position within a random biome.

        Args:
            state: Current environment state
            key: Random key for biome and position selection

        Returns:
            New agent position (x, y). Returns current position if no empty cell found.
        """
        num_biomes = self.biome_object_frequencies.shape[0]

        # Split key for biome selection and position selection
        key, biome_key, pos_key = jax.random.split(key, 3)

        # Randomly select a biome
        biome_idx = jax.random.randint(biome_key, (), 0, num_biomes)

        # Get biome mask for the selected biome
        biome_mask = self.biome_masks_array[biome_idx]

        # Find empty cells (object_id == 0) within the biome
        empty_mask = state.object_state.object_id == 0
        valid_spawn_mask = biome_mask & empty_mask

        # Count valid spawn locations
        num_valid_spawns = jnp.sum(valid_spawn_mask, dtype=jnp.int32)

        # Get all valid positions using nonzero with fixed size
        y_indices, x_indices = jnp.nonzero(
            valid_spawn_mask, size=self.size[0] * self.size[1], fill_value=-1
        )
        valid_spawn_indices = jnp.stack([x_indices, y_indices], axis=1)  # (x, y) format

        # Randomly select from valid positions
        random_idx = jax.random.randint(
            pos_key, (), jnp.array(0, dtype=jnp.int32), jnp.maximum(num_valid_spawns, 1)
        )
        new_pos = valid_spawn_indices[random_idx]

        # If no valid spawn found, keep current position
        new_pos = jax.lax.select(
            num_valid_spawns > 0,
            new_pos.astype(jnp.int32),
            state.pos.astype(jnp.int32),
        )

        return new_pos

    @partial(jax.named_call, name="deterministic_teleport")
    def _deterministic_teleport(
        self, state: EnvState, teleport_index: jax.Array, key: jax.Array
    ) -> jax.Array:
        """Deterministically teleport agent to a random empty cell within a deterministically selected biome.

        Cycles through biomes in order: biome 0 → biome 1 → ... → biome N-1 → biome 0 → ...
        The agent is placed at a random empty cell within the selected biome.
        Falls back to current position if no empty cell exists.

        Args:
            state: Current environment state
            teleport_index: The cumulative teleport count used to select the biome
            key: Random key for position selection

        Returns:
            New agent position (x, y) within the selected biome.
        """
        num_biomes = self.biome_object_frequencies.shape[0]

        # Deterministically select biome by cycling through them
        biome_idx = teleport_index % num_biomes

        # Get biome mask for the selected biome
        biome_mask = self.biome_masks_array[biome_idx]

        # Find empty cells (object_id == 0) within the biome
        empty_mask = state.object_state.object_id == 0
        valid_mask = biome_mask & empty_mask

        # Count valid spawn locations
        num_valid = jnp.sum(valid_mask, dtype=jnp.int32)

        # Get all valid positions using nonzero with fixed size
        y_indices, x_indices = jnp.nonzero(
            valid_mask, size=self.size[0] * self.size[1], fill_value=-1
        )
        valid_positions = jnp.stack(
            [x_indices, y_indices], axis=1
        )  # (N, 2) in (x, y) format

        # Randomly select from valid positions
        random_idx = jax.random.randint(
            key, (), jnp.array(0, dtype=jnp.int32), jnp.maximum(num_valid, 1)
        )
        new_pos = valid_positions[random_idx]

        # If no valid cell found, keep current position
        new_pos = jax.lax.select(
            num_valid > 0,
            new_pos.astype(jnp.int32),
            state.pos.astype(jnp.int32),
        )

        return new_pos

    @partial(jax.named_call, name="compute_reward")
    def _compute_reward(
        self,
        state: EnvState,
        pos: jax.Array,
        key: jax.Array,
        should_collect: jax.Array,
        digestion_buffer: jax.Array,
        obj_at_pos: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        key, reward_subkey = jax.random.split(key)

        object_params = state.object_state.state_params[pos[1], pos[0]]
        object_reward = jax.lax.switch(
            obj_at_pos.astype(jnp.int32),
            self.reward_fns,
            state.time + state.offset,
            reward_subkey,
            object_params.astype(jnp.float32),
        )

        global_mean = self._get_global_mean_reward(state)
        object_reward = self._apply_centering(object_reward, obj_at_pos, global_mean)

        key, digestion_subkey = jax.random.split(key)
        reward_delay = jax.lax.switch(
            obj_at_pos,
            self.reward_delay_fns,
            state.time + state.offset,
            digestion_subkey,
        )
        reward = jnp.where(
            should_collect & (reward_delay == jnp.array(0, dtype=jnp.int32)),
            object_reward,
            0.0,
        )
        if self.max_reward_delay > 0:
            # Add delayed rewards to buffer
            digestion_buffer = jax.lax.cond(
                should_collect & (reward_delay > 0),
                lambda: digestion_buffer.at[
                    (state.time + reward_delay) % self.max_reward_delay
                ].add(object_reward),
                lambda: digestion_buffer,
            )
            # Deliver current rewards
            current_index = state.time % self.max_reward_delay
            reward += digestion_buffer[current_index]
            digestion_buffer = digestion_buffer.at[current_index].set(0.0)
        return reward, digestion_buffer

    @partial(jax.named_call, name="respawn_logic")
    def _respawn_logic(
        self,
        state: EnvState,
        pos: jax.Array,
        key: jax.Array,
        current_objects: jax.Array,
        is_collectable: jax.Array,
        obj_at_pos: jax.Array,
    ) -> ObjectState:
        # 3. HANDLE OBJECT COLLECTION AND RESPAWNING
        key, regen_subkey, rand_key = jax.random.split(key, 3)

        # Decrement respawn timers
        has_timer = state.object_state.respawn_timer > jnp.array(0, dtype=TIMER_DTYPE)
        new_respawn_timer = jnp.where(
            has_timer,
            state.object_state.respawn_timer - jnp.array(1, dtype=TIMER_DTYPE),
            state.object_state.respawn_timer,
        )

        # Track which cells have timers that just reached 0
        just_respawned = has_timer & (
            new_respawn_timer == jnp.array(0, dtype=TIMER_DTYPE)
        )

        # Respawn objects where timer reached 0
        new_object_id = jnp.where(
            just_respawned,
            state.object_state.respawn_object_id,
            state.object_state.object_id,
        )

        # Clear respawn_object_id for cells that just respawned
        new_respawn_object_id = jnp.where(
            just_respawned,
            jnp.array(0, dtype=ID_DTYPE),
            state.object_state.respawn_object_id,
        )

        # Update spawn times for objects that just respawned
        spawn_time = jnp.where(
            just_respawned, state.time, state.object_state.spawn_time
        )

        # Collect object: set a timer
        regen_delay = jax.lax.switch(
            obj_at_pos, self.regen_delay_fns, state.time + state.offset, regen_subkey
        )
        # Cast timer_countdown to match ObjectState.respawn_timer dtype
        timer_countdown = jax.lax.cond(
            regen_delay == jnp.iinfo(jnp.int32).max,
            lambda: jnp.array(0, dtype=TIMER_DTYPE),  # No timer (permanent removal)
            lambda: (regen_delay + 1).astype(TIMER_DTYPE),  # Timer countdown
        )

        # If collected, replace object with timer; otherwise, keep it
        val_at_pos = current_objects[pos[1], pos[0]]
        # Use original should_collect for consumption tracking
        should_collect_now = is_collectable & (
            val_at_pos > jnp.array(0, dtype=ID_DTYPE)
        )

        # Create updated object state with new respawn_timer, object_id, and spawn_time
        object_state = state.object_state.replace(
            object_id=new_object_id,
            respawn_timer=new_respawn_timer,
            respawn_object_id=new_respawn_object_id,
            spawn_time=spawn_time,
        )

        # Place timer on collection
        object_state = jax.lax.cond(
            should_collect_now,
            lambda: self._place_timer(
                object_state,
                pos[1],
                pos[0],
                obj_at_pos,  # object type
                timer_countdown,  # timer value
                self.object_random_respawn[obj_at_pos],
                rand_key,
            ),
            lambda: object_state,
        )
        return object_state

    @partial(jax.named_call, name="dynamic_biomes")
    def _dynamic_biomes(
        self,
        state: EnvState,
        pos: jax.Array,
        key: jax.Array,
        object_state: ObjectState,
        should_collect: jax.Array,
    ) -> Tuple[ObjectState, BiomeState]:
        if self.dynamic_biomes:
            # Update consumption count if an object was collected
            # Only count if the object belongs to the current generation of its biome
            collected_biome_id = object_state.biome_id[pos[1], pos[0]]
            object_gen_at_pos = object_state.generation[pos[1], pos[0]]
            current_biome_gen = state.biome_state.generation[collected_biome_id]
            is_current_generation = object_gen_at_pos == current_biome_gen

            biome_consumption_count = state.biome_state.consumption_count
            biome_consumption_count = jax.lax.cond(
                should_collect & is_current_generation,
                lambda: biome_consumption_count.at[collected_biome_id].add(
                    jnp.array(1, dtype=ID_DTYPE)
                ),
                lambda: biome_consumption_count,
            )

            # Check each biome for threshold crossing and respawn if needed
            key, respawn_key = jax.random.split(key)
            biome_state = BiomeState(
                consumption_count=biome_consumption_count,
                total_objects=state.biome_state.total_objects,
                generation=state.biome_state.generation,
            )
            object_state, biome_state, respawn_key = self._check_and_respawn_biomes(
                object_state,
                biome_state,
                state.time,
                respawn_key,
            )
            return object_state, biome_state
        else:
            return object_state, state.biome_state

    @partial(jax.named_call, name="reward_grid")
    def _reward_grid(self, state: EnvState, object_state: ObjectState) -> jax.Array:
        # Compute reward at each grid position
        fixed_key = jax.random.key(0)  # Fixed key for deterministic reward computation
        global_mean = self._get_global_mean_reward(state)

        def compute_reward(obj_id, params, timer):
            reward = jax.lax.switch(
                obj_id.astype(jnp.int32),
                self.reward_fns,
                state.time + state.offset,
                fixed_key,
                params.astype(jnp.float32),
            )

            reward = self._apply_centering(reward, obj_id, global_mean)

            # Only show reward for objects that are fully present (no timer)
            mask = (obj_id > 0) & (timer == 0)
            return jnp.where(mask, reward, 0.0)

        reward_grid = jax.vmap(jax.vmap(compute_reward))(
            object_state.object_id.astype(ID_DTYPE),
            object_state.state_params.astype(PARAM_DTYPE),
            object_state.respawn_timer.astype(TIMER_DTYPE),
        )
        return reward_grid

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: Union[int, float, jax.Array],
        params: Optional[EnvParams] = None,
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        if params is None:
            params = self.default_params

        # Step
        key_step, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key_step, state, action, params
        )

        # No auto-reset (Foragax is a continuing environment).

        return obs_st, state_st, reward, done, info

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: Union[int, float, jax.Array],
        params: EnvParams,
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, Dict[Any, Any]]:
        """Perform single timestep state transition."""
        current_objects = state.object_state.object_id
        pos, new_pos = self._move_agent(state, action)

        with jax.named_scope("compute_reward"):
            # 2. HANDLE COLLISIONS AND REWARDS
            obj_at_pos = current_objects[pos[1], pos[0]]
            is_collectable = self.object_collectable[obj_at_pos]
            should_collect = is_collectable & (
                obj_at_pos > jnp.array(0, dtype=ID_DTYPE)
            )

            # Handle digestion: add reward to buffer if collected
            digestion_buffer = state.digestion_buffer
            key, reward_subkey = jax.random.split(key)

            reward, digestion_buffer = self._compute_reward(
                state, pos, key, should_collect, digestion_buffer, obj_at_pos
            )

        object_state = self._respawn_logic(
            state, pos, key, current_objects, is_collectable, obj_at_pos
        )

        # 3.5. HANDLE OBJECT EXPIRY
        # Only process expiry if there are objects that can expire
        key, object_state = self.expire_objects(key, state, object_state)

        # 3.6. HANDLE DYNAMIC BIOME CONSUMPTION AND RESPAWNING
        object_state, biome_state = self._dynamic_biomes(
            state, pos, key, object_state, should_collect
        )

        info = {"discount": self.discount(state, params)}
        temperatures = jnp.zeros(len(self.objects))
        for obj_index, obj in enumerate(self.objects):
            if isinstance(obj, (WeatherObject, WeatherWaveObject)):
                temperatures = temperatures.at[obj_index].set(
                    get_temperature(obj.rewards, state.time + state.offset, obj.repeat)
                )
        info["temperatures"] = temperatures
        info["biome_id"] = object_state.biome_id[pos[1], pos[0]]
        info["object_collected_id"] = jnp.where(
            should_collect,
            obj_at_pos.astype(ID_DTYPE),
            jnp.array(-1, dtype=ID_DTYPE),
        )

        # Biome regret metrics
        biome_means, biome_counts = self._get_biome_mean_rewards(state)

        # Use the effective metrics index for the agent's current position
        # biome_means has shape (num_food_biomes + 1,)
        metrics_idx = self.cell_to_metrics_idx_grid[pos[1], pos[0]]
        current_biome_mean = biome_means[metrics_idx]

        # Max mean over all consolidated regions (food biomes + void)
        # This ensures max_biome_mean is at least 0.0 (from an empty region)
        # and guarantees max_biome_mean >= current_biome_mean.
        max_biome_mean = jnp.max(biome_means)

        biome_regret = max_biome_mean - current_biome_mean

        # Rank strictly among food biomes and void region
        ranks = jnp.argsort(jnp.argsort(-biome_means)) + 1
        biome_rank = ranks[metrics_idx]

        info["current_biome_mean"] = current_biome_mean
        info["max_biome_mean"] = max_biome_mean
        info["biome_regret"] = biome_regret
        info["biome_rank"] = biome_rank

        # 4. UPDATE STATE
        # Ensure all fields have canonical dtypes for consistency (e.g., for gymnax step selection)
        object_state = object_state.replace(
            object_id=object_state.object_id.astype(ID_DTYPE),
            respawn_timer=object_state.respawn_timer.astype(TIMER_DTYPE),
            respawn_object_id=object_state.respawn_object_id.astype(ID_DTYPE),
            spawn_time=object_state.spawn_time.astype(TIME_DTYPE),
            color=object_state.color.astype(COLOR_DTYPE),
            generation=object_state.generation.astype(ID_DTYPE),
            state_params=object_state.state_params.astype(PARAM_DTYPE),
            biome_id=object_state.biome_id.astype(BIOME_ID_DTYPE),
        )
        biome_state = biome_state.replace(
            consumption_count=biome_state.consumption_count.astype(ID_DTYPE),
            total_objects=biome_state.total_objects.astype(ID_DTYPE),
            generation=biome_state.generation.astype(ID_DTYPE),
        )

        state = EnvState(
            pos=pos.astype(jnp.int32),
            time=jnp.array(state.time + 1, dtype=jnp.int32),
            offset=state.offset,
            digestion_buffer=digestion_buffer.astype(REWARD_DTYPE),
            object_state=object_state,
            biome_state=biome_state,
        )

        # 5. HANDLE RANDOM TELEPORT
        # Teleport agent to a random empty cell in a random biome at 1/4 and 3/4 of the period.
        # The effective time is (state.time + state.offset + random_teleport_offset) to sync
        # with the square wave timing. Teleport triggers when:
        #   effective_time % period == period // 4  (1/4 of period)
        #   effective_time % period == 3 * period // 4  (3/4 of period)
        if self.random_teleport_period > 0:
            key, teleport_key = jax.random.split(key)
            effective_time = state.time + state.offset + self.random_teleport_offset
            time_in_period = effective_time % self.random_teleport_period
            quarter_period = self.random_teleport_period // 4
            three_quarter_period = 3 * self.random_teleport_period // 4
            should_teleport = (time_in_period == quarter_period) | (
                time_in_period == three_quarter_period
            )
            new_pos = self._random_teleport(state, teleport_key)
            state = state.replace(
                pos=jax.lax.select(should_teleport, new_pos, state.pos)
            )

        # 6. HANDLE DETERMINISTIC TELEPORT
        # Teleport agent to a random empty cell within a deterministically selected biome,
        # following a deterministic cycle at the beginning (0) and halfway (1/2) of the period.
        # The biome is selected by cycling through
        # biomes in order based on the cumulative teleport count.
        if self.deterministic_teleport_period > 0:
            key, teleport_key = jax.random.split(key)
            effective_time = (
                state.time + state.offset + self.deterministic_teleport_offset
            )
            time_in_period = effective_time % self.deterministic_teleport_period
            half_period = self.deterministic_teleport_period // 2
            should_teleport = (time_in_period == 0) | (time_in_period == half_period)
            # Compute cumulative teleport index from time
            # Each full period has 2 teleport events (at 0 and 1/2)
            full_periods = effective_time // self.deterministic_teleport_period
            past_half = (time_in_period >= half_period).astype(jnp.int32)
            teleport_index = 2 * full_periods + past_half
            new_pos = self._deterministic_teleport(state, teleport_index, teleport_key)
            state = state.replace(
                pos=jax.lax.select(should_teleport, new_pos, state.pos)
            )

        reward_grid = self._reward_grid(state, object_state)
        if self.full_world:
            info["rewards"] = reward_grid.astype(jnp.float16)
        else:
            info["rewards"] = self._get_aperture(reward_grid, state.pos).astype(
                jnp.float16
            )

        done = self.is_terminal(state, params)
        return (
            jax.lax.stop_gradient(self.get_obs(state, params)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    @partial(jax.named_call, name="expire_objects")
    def expire_objects(
        self, key, state, object_state: ObjectState
    ) -> Tuple[jax.Array, ObjectState]:
        if self.has_expiring_objects:
            # Check each cell for objects that have exceeded their expiry time
            current_objects_for_expiry = object_state.object_id

            # Calculate age of each object (current_time - spawn_time)
            object_ages = state.time - object_state.spawn_time

            # Get expiry time for each object type in the grid
            expiry_times = self.object_expiry_time[current_objects_for_expiry]

            # Check if object should expire (age >= expiry_time and expiry_time >= 0)
            should_expire = (
                (object_ages >= expiry_times)
                & (expiry_times >= jnp.array(0, dtype=TIME_DTYPE))
                & (current_objects_for_expiry > jnp.array(0, dtype=ID_DTYPE))
            )

            # Only process expiry if there are actually objects to expire
            has_expiring = jnp.any(should_expire)

            # Precompute the first expiring index in flat space so the work inside cond is minimal.
            overage = jnp.where(
                should_expire,
                object_ages - expiry_times,
                -jnp.inf,
            ).reshape(-1)
            sorted_flat_indices = jnp.argsort(overage)[::-1]
            selected_flat_indices = jnp.where(
                overage[sorted_flat_indices] > -jnp.inf,
                sorted_flat_indices,
                -jnp.ones_like(sorted_flat_indices),
            )[: self.max_expiries_per_step]

            def process_expiries():
                key_local, expiry_key = jax.random.split(key)

                def body_fn(i, obj_state):
                    flat_idx = selected_flat_indices[i]

                    def expire_at(obj_state):
                        y = flat_idx // self.size[0]
                        x = flat_idx % self.size[0]
                        obj_id = current_objects_for_expiry[y, x]
                        exp_key = jax.random.fold_in(expiry_key, flat_idx)
                        exp_delay = jax.lax.switch(
                            obj_id,
                            self.expiry_regen_delay_fns,
                            state.time + state.offset,
                            exp_key,
                        )
                        timer_countdown = jax.lax.cond(
                            exp_delay == jnp.iinfo(jnp.int32).max,
                            lambda: jnp.array(0, dtype=TIMER_DTYPE),
                            lambda: (exp_delay + 1).astype(TIMER_DTYPE),
                        )

                        respawn_random = self.object_random_respawn[obj_id]
                        rand_key = jax.random.fold_in(exp_key, 1)
                        return self._place_timer(
                            obj_state,
                            y,
                            x,
                            obj_id,
                            timer_countdown,
                            respawn_random,
                            rand_key,
                        )

                    return jax.lax.cond(
                        flat_idx >= 0,
                        expire_at,
                        lambda obj_state: obj_state,
                        obj_state,
                    )

                new_object_state = jax.lax.fori_loop(
                    0, self.max_expiries_per_step, body_fn, object_state
                )
                return key_local, new_object_state

            def no_expiries():
                return key, object_state

            key, object_state = jax.lax.cond(
                has_expiring,
                process_expiries,
                no_expiries,
            )

        return key, object_state

    def _check_and_respawn_biomes(
        self,
        object_state: ObjectState,
        biome_state: BiomeState,
        current_time: int,
        key: jax.Array,
    ) -> Tuple[ObjectState, BiomeState, jax.Array]:
        """Check all biomes for consumption threshold and respawn if needed."""

        num_biomes = self.biome_object_frequencies.shape[0]

        if isinstance(self.biome_consumption_threshold, float):
            # Compute consumption rates for all biomes
            consumption_rates = biome_state.consumption_count / jnp.maximum(
                1.0, biome_state.total_objects.astype(float)
            )
            should_respawn = consumption_rates >= self.biome_consumption_threshold
        else:
            should_respawn = (
                biome_state.consumption_count >= self.biome_consumption_threshold
            )

        any_respawn = jnp.any(should_respawn)

        def do_respawn(args):
            (
                object_state,
                biome_state,
                should_respawn,
                key,
            ) = args
            # Split key for all biomes in parallel
            key, subkey = jax.random.split(key)
            biome_keys = jax.random.split(subkey, num_biomes)

            # Prepare existing state with dropout
            is_curr_fourier = self.object_is_fourier[object_state.object_id]
            is_static = (object_state.object_id > 0) & (~is_curr_fourier)
            is_free_mask = ~is_static

            # Compute all new spawns in parallel using vmap for random, switch for deterministic
            if self.deterministic_spawn:
                # Deterministic spawn needs static loop for per-biome sizes
                def make_spawn_fn(biome_idx):
                    def spawn_fn(key):
                        return self._spawn_biome_objects(
                            biome_idx,
                            key,
                            deterministic=True,
                            is_free_mask=is_free_mask,
                        )

                    return spawn_fn

                spawn_fns = [make_spawn_fn(idx) for idx in range(num_biomes)]

                # Apply switch for each biome
                all_new_objects_list = []
                all_new_colors_list = []
                all_new_params_list = []
                for i in range(num_biomes):
                    obj, col, par = jax.lax.switch(i, spawn_fns, biome_keys[i])
                    all_new_objects_list.append(obj)
                    all_new_colors_list.append(col)
                    all_new_params_list.append(par)

                all_new_objects = jnp.stack(all_new_objects_list)
                all_new_colors = jnp.stack(all_new_colors_list)
                all_new_params = jnp.stack(all_new_params_list)
            else:
                # Random spawn works with vmap
                all_new_objects, all_new_colors, all_new_params = jax.vmap(
                    lambda i, k: self._spawn_biome_objects(
                        i, k, deterministic=False, is_free_mask=is_free_mask
                    )
                )(jnp.arange(num_biomes), biome_keys)

            # Initialize updated grids
            new_obj_id = object_state.object_id
            new_color = object_state.color
            new_params = object_state.state_params
            new_spawn = object_state.spawn_time
            new_gen = object_state.generation

            # Update biome state
            new_consumption_count = jnp.where(
                should_respawn,
                jnp.array(0, dtype=ID_DTYPE),
                biome_state.consumption_count,
            )
            new_generation = biome_state.generation + should_respawn.astype(ID_DTYPE)

            # Compute new total objects for respawning biomes
            def count_objects(i):
                return jnp.sum(
                    (all_new_objects[i] > 0) & self.biome_masks_array[i], dtype=ID_DTYPE
                )

            new_object_counts = jax.vmap(count_objects)(jnp.arange(num_biomes))
            new_total_objects = jnp.where(
                should_respawn, new_object_counts, biome_state.total_objects
            )

            new_biome_state = BiomeState(
                consumption_count=new_consumption_count,
                total_objects=new_total_objects,
                generation=new_generation,
            )

            # Update grids for respawning biomes
            for i in range(num_biomes):
                biome_mask = self.biome_masks_array[i]
                new_gen_value = new_biome_state.generation[i]

                # Update mask: biome area AND needs respawn
                should_update = biome_mask & should_respawn[i][..., None]

                # 1. Prepare existing state with dropout
                is_curr_fourier = self.object_is_fourier[new_obj_id]
                is_static = (new_obj_id > 0) & (~is_curr_fourier)

                if self.dynamic_biome_spawn_empty > 0:
                    key, dropout_key = jax.random.split(key)
                    # Dropout only applies to CURRENT Fourier objects in the biome
                    keep_mask_fourier = jax.random.bernoulli(
                        dropout_key,
                        1.0 - self.dynamic_biome_spawn_empty,
                        new_obj_id.shape,
                    )
                    # We keep it if it's static OR if it's fourier and not dropped
                    keep_mask = is_static | (is_curr_fourier & keep_mask_fourier)

                    # Apply dropout to the current state (only within the update area)
                    # We only clear if it was a fourier object and we decided to drop it
                    dropped_obj_id = jnp.where(
                        keep_mask, new_obj_id, jnp.array(0, dtype=ID_DTYPE)
                    )
                    dropped_color = jnp.where(
                        keep_mask[..., None], new_color, jnp.array(0, dtype=COLOR_DTYPE)
                    )
                    dropped_params = jnp.where(
                        keep_mask[..., None],
                        new_params,
                        jnp.array(0, dtype=PARAM_DTYPE),
                    )
                    dropped_gen = jnp.where(
                        keep_mask, new_gen, jnp.array(0, dtype=ID_DTYPE)
                    )
                    dropped_spawn = jnp.where(
                        keep_mask, new_spawn, jnp.array(0, dtype=TIME_DTYPE)
                    )
                else:
                    dropped_obj_id = new_obj_id
                    dropped_color = new_color
                    dropped_params = new_params
                    dropped_gen = new_gen
                    dropped_spawn = new_spawn

                # 2. Merge with new state
                # Only allow Fourier objects to be spawned during regeneration
                new_is_fourier = self.object_is_fourier[all_new_objects[i]]
                regeneration_objs = jnp.where(
                    new_is_fourier, all_new_objects[i], jnp.array(0, dtype=ID_DTYPE)
                )

                new_spawn_valid = regeneration_objs > 0
                # Protect static objects from being overwritten by new spawns
                take_new = new_spawn_valid & (~is_static)

                final_objs = jnp.where(take_new, regeneration_objs, dropped_obj_id)
                final_colors = jnp.where(
                    take_new[..., None], all_new_colors[i], dropped_color
                )
                final_params = jnp.where(
                    take_new[..., None], all_new_params[i], dropped_params
                )

                # Update generation and spawn time only for new spawns
                final_gen = jnp.where(take_new, new_gen_value, dropped_gen)
                final_spawn = jnp.where(take_new, current_time, dropped_spawn)

                new_obj_id = jnp.where(should_update, final_objs, new_obj_id)
                new_color = jnp.where(should_update[..., None], final_colors, new_color)
                new_params = jnp.where(
                    should_update[..., None], final_params, new_params
                )
                new_gen = jnp.where(should_update, final_gen, new_gen)
                new_spawn = jnp.where(should_update, final_spawn, new_spawn)

            # Clear timers in respawning biomes
            new_respawn_timer = object_state.respawn_timer
            new_respawn_object_id = object_state.respawn_object_id
            for i in range(num_biomes):
                biome_mask = self.biome_masks_array[i]
                should_clear = biome_mask & should_respawn[i][..., None]
                new_respawn_timer = jnp.where(
                    should_clear, jnp.array(0, dtype=TIMER_DTYPE), new_respawn_timer
                )
                new_respawn_object_id = jnp.where(
                    should_clear, jnp.array(0, dtype=ID_DTYPE), new_respawn_object_id
                )

            new_object_state = object_state.replace(
                object_id=new_obj_id,
                respawn_timer=new_respawn_timer,
                respawn_object_id=new_respawn_object_id,
                color=new_color,
                state_params=new_params,
            )
            return new_object_state, new_biome_state, key

        def no_respawn(args):
            object_state, biome_state, _, key = args
            return object_state, biome_state, key

        object_state, biome_state, key = jax.lax.cond(
            any_respawn,
            do_respawn,
            no_respawn,
            (object_state, biome_state, should_respawn, key),
        )

        return object_state, biome_state, key

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        """Reset environment state."""
        key, offset_key = jax.random.split(key)
        offset = jax.random.randint(offset_key, (), 0, self.random_shift_max_steps + 1)

        num_object_params = 3 + 2 * self.num_fourier_terms
        object_state = ObjectState.create_empty(self.size, num_object_params)

        num_biomes = self.biome_object_frequencies.shape[0]
        key, iter_key = jax.random.split(key)

        # Pass 1: Set Biome IDs and spawn Static Objects (non-Fourier)
        # Apply biomes from largest to smallest so that nested/overlapping biomes correctly overwrite background biomes.
        for i in self.biome_order:
            mask = self.biome_masks[i]

            # Set biome_id
            object_state = object_state.replace(
                biome_id=jnp.where(
                    mask, jnp.array(i, dtype=BIOME_ID_DTYPE), object_state.biome_id
                )
            )

            if not self.is_fourier_biome[i]:
                iter_key, biome_key = jax.random.split(iter_key)
                # Use unified spawn method
                biome_objects, biome_colors, biome_object_params = (
                    self._spawn_biome_objects(i, biome_key, self.deterministic_spawn)
                )

                # Merge biome objects/colors/params into object_state
                object_state = object_state.replace(
                    object_id=jnp.where(mask, biome_objects, object_state.object_id),
                    color=jnp.where(mask[..., None], biome_colors, object_state.color),
                    state_params=jnp.where(
                        mask[..., None], biome_object_params, object_state.state_params
                    ),
                )

        # Pass 2: Spawn Dynamic Objects (Fourier) using is_free_mask to avoid static objects
        is_free_mask = ~(object_state.object_id > 0)
        for i in self.biome_order:
            if self.is_fourier_biome[i]:
                iter_key, biome_key = jax.random.split(iter_key)
                mask = self.biome_masks[i]

                # Use unified spawn method with is_free_mask
                biome_objects, biome_colors, biome_object_params = (
                    self._spawn_biome_objects(
                        i,
                        biome_key,
                        self.deterministic_spawn,
                        is_free_mask=is_free_mask,
                    )
                )

                # Merge biome objects/colors/params into object_state
                # Only overwrite where a new object was actually spawned (biome_objects > 0)
                # to preserve static objects from Pass 1.
                take_new = mask & (biome_objects > 0)
                object_state = object_state.replace(
                    object_id=jnp.where(
                        take_new, biome_objects, object_state.object_id
                    ),
                    color=jnp.where(
                        take_new[..., None], biome_colors, object_state.color
                    ),
                    state_params=jnp.where(
                        take_new[..., None],
                        biome_object_params,
                        object_state.state_params,
                    ),
                )

        # Place agent in the center of the world
        agent_pos = jnp.array([self.size[0] // 2, self.size[1] // 2])

        # Initialize biome consumption tracking
        num_biomes = self.biome_object_frequencies.shape[0]
        biome_consumption_count = jnp.zeros(num_biomes, dtype=ID_DTYPE)
        biome_total_objects = jnp.zeros(num_biomes, dtype=ID_DTYPE)

        # Count objects in each biome
        for i in range(num_biomes):
            mask = self.biome_masks[i]
            # Count non-empty objects (object_id > 0)
            total = jnp.sum((object_state.object_id > 0) & mask, dtype=ID_DTYPE)
            biome_total_objects = biome_total_objects.at[i].set(total)

        biome_generation = jnp.zeros(num_biomes, dtype=ID_DTYPE)

        # Final state cleanup to ensure type consistency
        object_state = object_state.replace(
            object_id=object_state.object_id.astype(ID_DTYPE),
            respawn_timer=object_state.respawn_timer.astype(TIMER_DTYPE),
            respawn_object_id=object_state.respawn_object_id.astype(ID_DTYPE),
            spawn_time=object_state.spawn_time.astype(TIME_DTYPE),
            color=object_state.color.astype(COLOR_DTYPE),
            generation=object_state.generation.astype(ID_DTYPE),
            state_params=object_state.state_params.astype(PARAM_DTYPE),
            biome_id=object_state.biome_id.astype(BIOME_ID_DTYPE),
        )

        state = EnvState(
            pos=agent_pos.astype(jnp.int32),
            time=jnp.array(0, dtype=jnp.int32),
            offset=offset.astype(jnp.int32),
            digestion_buffer=jnp.zeros((self.max_reward_delay,), dtype=REWARD_DTYPE),
            object_state=object_state,
            biome_state=BiomeState(
                consumption_count=biome_consumption_count.astype(ID_DTYPE),
                total_objects=biome_total_objects.astype(ID_DTYPE),
                generation=biome_generation.astype(ID_DTYPE),
            ),
        )

        # 6. HANDLE DETERMINISTIC TELEPORT (Initial Placement)
        # Place agent in the correct biome at the start of the episode.
        if self.deterministic_teleport_period > 0:
            effective_time = (
                state.time + state.offset + self.deterministic_teleport_offset
            )
            time_in_period = effective_time % self.deterministic_teleport_period
            half_period = self.deterministic_teleport_period // 2

            full_periods = effective_time // self.deterministic_teleport_period
            past_half = (time_in_period >= half_period).astype(jnp.int32)
            teleport_index = 2 * full_periods + past_half

            # Use available randomness from `key` for initial placement
            new_pos = self._deterministic_teleport(state, teleport_index, key)
            state = state.replace(pos=new_pos)

        return jax.lax.stop_gradient(
            self.get_obs(state, params)
        ), jax.lax.stop_gradient(state)

    def _spawn_biome_objects(
        self,
        biome_idx: int,
        key: jax.Array,
        deterministic: bool = False,
        is_free_mask: Optional[jax.Array] = None,
        biome_freqs: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Spawn objects in a biome.

        Returns:
            object_grid: (H, W) array of object IDs
            color_grid: (H, W, 3) array of RGB colors
            state_grid: (H, W, num_state_params) array of object state parameters
        """
        if biome_freqs is None:
            biome_freqs = self.biome_object_frequencies[biome_idx]
        biome_mask = self.biome_masks_array[biome_idx]

        key, spawn_key, color_key, params_key = jax.random.split(key, 4)

        # Generate object IDs using deterministic or random spawn
        if deterministic:
            # Deterministic spawn: exact number of each object type
            # NOTE: Requires concrete biome_idx to compute size at trace time
            biome_start = self.biome_starts[biome_idx]
            biome_stop = self.biome_stops[biome_idx]
            biome_height = biome_stop[1] - biome_start[1]
            biome_width = biome_stop[0] - biome_start[0]
            biome_size = int(self.biome_sizes[biome_idx])

            # Handle is_free_mask for the biome area
            if is_free_mask is None:
                biome_free_mask = jnp.ones((biome_height, biome_width), dtype=jnp.bool_)
            else:
                biome_free_mask = is_free_mask[
                    biome_start[1] : biome_stop[1], biome_start[0] : biome_stop[0]
                ]

            grid = jnp.linspace(0, 1, biome_size, endpoint=False)
            biome_objects_flat = (
                len(biome_freqs)
                - jnp.searchsorted(jnp.cumsum(biome_freqs[::-1]), grid, side="right")
            ).astype(ID_DTYPE)

            # Priority-based placement to avoid blocked cells
            # 1. Sort objects so non-zero IDs are at the front
            sorted_objects = jnp.sort(biome_objects_flat)[::-1]

            # 2. Assign high priority to free cells, extremely low to blocked ones
            priorities = jax.random.uniform(spawn_key, (biome_height, biome_width))
            priorities = jnp.where(biome_free_mask, priorities, -10.0)

            # 3. Get order of cells by priority (highest priority first)
            # Flatten to map objects to the ranked cells
            order = jnp.argsort(priorities.flatten())[::-1]

            # 4. Place objects in the ranked order
            # The top N ranked slots (all free if possible) get the top N objects from sorted_objects
            final_biome_grid_flat = jnp.zeros(
                biome_height * biome_width, dtype=ID_DTYPE
            )
            final_biome_grid_flat = final_biome_grid_flat.at[order].set(sorted_objects)
            biome_objects = final_biome_grid_flat.reshape(biome_height, biome_width)

            # Place in full grid using slicing with static bounds
            object_grid = jnp.zeros((self.size[1], self.size[0]), dtype=ID_DTYPE)
            object_grid = object_grid.at[
                biome_start[1] : biome_stop[1], biome_start[0] : biome_stop[0]
            ].set(biome_objects)
        else:
            # Random spawn: probabilistic placement (works with traced biome_idx)
            grid_rand = jax.random.uniform(spawn_key, (self.size[1], self.size[0]))
            empty_freq = 1.0 - jnp.sum(biome_freqs)
            all_freqs = jnp.concatenate([jnp.array([empty_freq]), biome_freqs])
            cumulative_freqs = jnp.cumsum(
                jnp.concatenate([jnp.array([0.0]), all_freqs])
            )
            object_grid = (
                jnp.searchsorted(cumulative_freqs, grid_rand, side="right") - 1
            ).astype(ID_DTYPE)

            # Apply occupancy mask if provided
            if is_free_mask is not None:
                object_grid = jnp.where(
                    is_free_mask, object_grid, jnp.array(0, dtype=ID_DTYPE)
                )

        # Initialize color grid
        color_grid = jnp.full((self.size[1], self.size[0], 3), 255, dtype=COLOR_DTYPE)

        # Sample ONE color per object type in this biome (not per instance)
        # This gives objects of the same type the same color within a biome generation
        # Skip index 0 (EMPTY object) - only sample colors for actual objects
        num_object_types = len(self.objects)
        num_actual_objects = num_object_types - 1  # Exclude EMPTY

        if num_actual_objects > 0:
            biome_object_colors = jax.random.randint(
                color_key,
                (num_actual_objects, 3),
                minval=jnp.array(0, dtype=COLOR_DTYPE),
                maxval=jnp.array(256, dtype=jnp.int32),  # randint range is often int32
                dtype=COLOR_DTYPE,
            )

            # Assign colors based on object type (starting from index 1)
            for obj_idx in range(1, num_object_types):
                obj_mask = (object_grid == obj_idx) & biome_mask

                obj_color = jax.lax.cond(
                    self.object_is_fourier[obj_idx],
                    lambda: biome_object_colors[obj_idx - 1],
                    lambda: self.object_colors[obj_idx].astype(COLOR_DTYPE),
                )

                color_grid = jnp.where(obj_mask[..., None], obj_color, color_grid)

        # Initialize parameters grid
        num_object_params = 3 + 2 * self.num_fourier_terms
        params_grid = jnp.zeros(
            (self.size[1], self.size[0], num_object_params), dtype=PARAM_DTYPE
        )

        # Generate per-object parameters for each object type
        for obj_idx in range(num_object_types):
            # Get params for this object type - this happens at trace time
            params_key, obj_key = jax.random.split(params_key)
            obj_params = self.objects[obj_idx].get_state(obj_key)

            # Skip if no params (e.g., for EMPTY or default objects)
            if len(obj_params) == 0:
                continue

            # Ensure params match expected size
            if len(obj_params) != num_object_params:
                if len(obj_params) < num_object_params:
                    obj_params = jnp.pad(
                        obj_params,
                        (0, num_object_params - len(obj_params)),
                        constant_values=0.0,
                    )
                else:
                    # Truncate if too long
                    obj_params = obj_params[:num_object_params]

            # Assign to all objects of this type in this biome
            obj_mask = (object_grid == obj_idx) & biome_mask
            params_grid = jnp.where(
                obj_mask[..., None], obj_params.astype(PARAM_DTYPE), params_grid
            )

        return object_grid, color_grid, params_grid

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Foragax is a continuing environment."""
        return False

    @property
    def name(self) -> str:
        """Environment name."""
        return self._name

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
        num_object_params = 3 + 2 * self.num_fourier_terms
        return spaces.Dict(
            {
                "pos": spaces.Box(0, max(self.size), (2,), int),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "offset": spaces.Discrete(self.random_shift_max_steps + 1),
                "digestion_buffer": spaces.Box(
                    -jnp.inf,
                    jnp.inf,
                    (self.max_reward_delay,),
                    float,
                ),
                "object_state": spaces.Dict(
                    {
                        "object_id": spaces.Box(
                            -1000 * len(self.object_ids),
                            len(self.object_ids),
                            (self.size[1], self.size[0]),
                            int,
                        ),
                        "spawn_time": spaces.Box(
                            0,
                            jnp.inf,
                            (self.size[1], self.size[0]),
                            int,
                        ),
                        "color": spaces.Box(
                            0,
                            255,
                            (self.size[1], self.size[0], 3),
                            int,
                        ),
                        "generation": spaces.Box(
                            0,
                            jnp.inf,
                            (self.size[1], self.size[0]),
                            int,
                        ),
                        "state_params": spaces.Box(
                            -jnp.inf,
                            jnp.inf,
                            (self.size[1], self.size[0], num_object_params),
                            float,
                        ),
                        "biome_id": spaces.Box(
                            -1,
                            self.biome_object_frequencies.shape[0],
                            (self.size[1], self.size[0]),
                            int,
                        ),
                    }
                ),
                "biome_state": spaces.Dict(
                    {
                        "consumption_count": spaces.Box(
                            0,
                            jnp.inf,
                            (self.biome_object_frequencies.shape[0],),
                            int,
                        ),
                        "total_objects": spaces.Box(
                            0,
                            jnp.inf,
                            (self.biome_object_frequencies.shape[0],),
                            int,
                        ),
                        "generation": spaces.Box(
                            0,
                            jnp.inf,
                            (self.biome_object_frequencies.shape[0],),
                            int,
                        ),
                    }
                ),
            }
        )

    def _compute_aperture_coordinates(
        self, pos: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Compute aperture coordinates for the given position.

        Returns:
            (y_coords, x_coords, y_coords_clamped/mod, x_coords_clamped/mod)
        """
        ap_h, ap_w = self.aperture_size
        start_y = pos[1] - ap_h // 2
        start_x = pos[0] - ap_w // 2

        y_offsets = jnp.arange(ap_h)
        x_offsets = jnp.arange(ap_w)
        y_coords = start_y + y_offsets[:, None]
        x_coords = start_x + x_offsets

        if self.nowrap:
            y_coords_adj = jnp.clip(y_coords, 0, self.size[1] - 1)
            x_coords_adj = jnp.clip(x_coords, 0, self.size[0] - 1)
        else:
            y_coords_adj = jnp.mod(y_coords, self.size[1])
            x_coords_adj = jnp.mod(x_coords, self.size[0])

        return y_coords, x_coords, y_coords_adj, x_coords_adj

    def _get_aperture(
        self, grid: jax.Array, pos: jax.Array, fill_value: Optional[Any] = None
    ) -> jax.Array:
        """Extract the aperture view from the grid."""
        y_coords, x_coords, y_coords_adj, x_coords_adj = (
            self._compute_aperture_coordinates(pos)
        )

        values = grid[y_coords_adj, x_coords_adj]

        if self.nowrap:
            # Mark out-of-bounds positions with padding
            y_out = (y_coords < 0) | (y_coords >= self.size[1])
            x_out = (x_coords < 0) | (x_coords >= self.size[0])
            out_of_bounds = y_out | x_out

            if fill_value is not None:
                if len(values.shape) == 3:
                    aperture = jnp.where(out_of_bounds[..., None], fill_value, values)
                else:
                    aperture = jnp.where(out_of_bounds, fill_value, values)
            else:
                # Handle both object_id grids (2D) and color grids (3D)
                if len(values.shape) == 3:
                    # Color grid: use PADDING color (0, 0, 0)
                    padding_value = jnp.array([0, 0, 0], dtype=values.dtype)
                    aperture = jnp.where(
                        out_of_bounds[..., None], padding_value, values
                    )
                else:
                    # Object ID grid: use PADDING index
                    padding_index = self.object_ids[-1].astype(values.dtype)
                    aperture = jnp.where(out_of_bounds, padding_index, values)
        else:
            aperture = values

        return aperture

    def get_obs(
        self, state: EnvState, params: EnvParams, key=None
    ) -> Union[jax.Array, Dict[str, jax.Array]]:
        """Get observation based on observation_type and full_world."""
        obs_grid = state.object_state.object_id
        color_grid = state.object_state.color

        if self.full_world:
            obs = self._get_world_obs(obs_grid, state)
        else:
            grid = self._get_aperture(obs_grid, state.pos)
            color_grid = self._get_aperture(color_grid, state.pos)
            obs = self._get_aperture_obs(grid, color_grid, state)

        if self.return_hint:
            biome_means, _ = self._get_biome_mean_rewards(state)
            # Max over food biomes (first self.num_food_biomes elements)
            food_biome_means = biome_means[: self.num_food_biomes]
            best_biome = jnp.argmax(food_biome_means)
            hint = jax.nn.one_hot(best_biome, self.num_food_biomes)
            show_hint = (state.time % self.hint_every) < self.hint_duration
            hint = jnp.where(show_hint, hint, jnp.zeros_like(hint))
            return {"image": obs, "hint": hint}

        return obs

    def _get_world_obs(self, obs_grid: jax.Array, state: EnvState) -> jax.Array:
        """Get world observation."""
        num_obj_types = len(self.object_ids)
        if self.observation_type == "object":
            obs = jnp.zeros(
                (self.size[1], self.size[0], num_obj_types), dtype=jnp.float32
            )
            num_object_channels = num_obj_types - 1
            # create masks for all objects at once
            object_ids = jnp.arange(1, num_obj_types)
            object_masks = obs_grid[..., None] == object_ids[None, None, :]
            obs = obs.at[:, :, :num_object_channels].set(object_masks.astype(float))
            obs = obs.at[state.pos[1], state.pos[0], -1].set(1)
            return obs
        elif self.observation_type == "rgb":
            # Use state colors directly (supports dynamic biomes)
            colors = state.object_state.color / 255.0

            # Mask empty cells (object_id == 0) to white
            empty_mask = obs_grid == 0
            white_color = jnp.ones((self.size[1], self.size[0], 3), dtype=jnp.float32)
            obs = jnp.where(empty_mask[..., None], white_color, colors)

            return obs
        elif self.observation_type == "color":
            # Handle case with no objects (only EMPTY)
            if self.num_color_channels == 0:
                return jnp.zeros(obs_grid.shape + (0,), dtype=jnp.float32)
            # Map object IDs to color channel indices
            color_channels = jnp.where(
                obs_grid == jnp.array(0, dtype=obs_grid.dtype),
                jnp.array(-1, dtype=jnp.int32),
                jnp.take(
                    self.object_to_color_map, obs_grid.astype(jnp.int32) - 1, axis=0
                ),
            )
            obs = jax.nn.one_hot(color_channels, self.num_color_channels, axis=-1)
            return obs
        else:
            raise ValueError(f"Unknown observation_type: {self.observation_type}")

    def _get_aperture_obs(
        self, aperture: jax.Array, color_aperture: jax.Array, state: EnvState
    ) -> jax.Array:
        """Get aperture observation."""
        if self.observation_type == "object":
            num_obj_types = len(self.object_ids)
            obs = jax.nn.one_hot(aperture, num_obj_types, axis=-1)
            return obs
        elif self.observation_type == "rgb":
            # Use the color aperture that was passed in
            aperture_colors = color_aperture / 255.0

            # Mask empty cells (object_id == 0) to white
            empty_mask = aperture == jnp.array(0, dtype=aperture.dtype)
            white_color = jnp.ones(aperture_colors.shape, dtype=jnp.float32)
            obs = jnp.where(empty_mask[..., None], white_color, aperture_colors)

            return obs
        elif self.observation_type == "color":
            # Handle case with no objects (only EMPTY)
            if self.num_color_channels == 0:
                return jnp.zeros(aperture.shape + (0,), dtype=jnp.float32)
            # Map object IDs to color channel indices
            color_channels = jnp.where(
                aperture == 0,
                -1,  # Special value for EMPTY
                jnp.take(self.object_to_color_map, aperture - 1, axis=0),
            )
            obs = jax.nn.one_hot(color_channels, self.num_color_channels, axis=-1)
            return obs
        else:
            raise ValueError(f"Unknown observation_type: {self.observation_type}")

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space based on observation_type and full_world."""
        if self.full_world:
            size = tuple(reversed(self.size))
        else:
            size = self.aperture_size

        if self.observation_type == "rgb":
            channels = 3
        elif self.observation_type == "object":
            num_obj_types = len(self.objects)
            channels = num_obj_types
        elif self.observation_type == "color":
            channels = self.num_color_channels
        else:
            raise ValueError(f"Unknown observation_type: {self.observation_type}")

        obs_shape = (*size, channels)

        obs_space = spaces.Box(0, 1, obs_shape, float)
        if self.return_hint:
            return spaces.Dict(
                {
                    "image": obs_space,
                    "hint": spaces.Box(0, 1, (self.num_food_biomes,), float),
                }
            )

        return obs_space

    def _compute_reward_grid(
        self, state: EnvState, object_id=None, state_params=None, respawn_timer=None
    ) -> jax.Array:
        """Compute rewards for given positions. If no grid provided, uses full world."""
        if object_id is None:
            object_id = state.object_state.object_id
        if state_params is None:
            state_params = state.object_state.state_params
        if respawn_timer is None:
            respawn_timer = state.object_state.respawn_timer

        fixed_key = jax.random.key(0)  # Fixed key for deterministic reward computation
        global_mean = self._get_global_mean_reward(state)

        def compute_reward(obj_id, params, timer):
            reward = jax.lax.switch(
                obj_id.astype(jnp.int32),
                self.reward_fns,
                state.time + state.offset,
                fixed_key,
                params.astype(jnp.float32),
            )

            reward = self._apply_centering(reward, obj_id, global_mean)

            # Only show reward for objects that are fully present (no timer)
            mask = (obj_id > 0) & (timer == 0)
            return jnp.where(mask, reward, 0.0)

        reward_grid = jax.vmap(jax.vmap(compute_reward))(
            object_id, state_params, respawn_timer
        )
        return reward_grid

    @partial(jax.jit, static_argnames=("self", "render_mode"))
    def render(
        self,
        state: EnvState,
        params: EnvParams,
        render_mode: str = "world",
    ):
        """Render the environment state."""
        is_world_mode = render_mode in ("world", "world_true", "world_reward")
        is_aperture_mode = render_mode in (
            "aperture",
            "aperture_true",
            "aperture_reward",
        )
        is_true_mode = render_mode in ("world_true", "aperture_true")
        is_reward_mode = render_mode in ("world_reward", "aperture_reward")

        if is_world_mode:
            img = get_base_image(
                state.object_state.object_id,
                state.object_state.color,
                self.object_colors,
                self.dynamic_biomes,
            )

            # Define constants for all world modes
            alpha = 0.2
            y_coords, x_coords, y_coords_adj, x_coords_adj = (
                self._compute_aperture_coordinates(state.pos)
            )

            if is_reward_mode:
                # Construct 3x intermediate image
                reward_grid = self._compute_reward_grid(state)
                reward_colors = reward_to_color(reward_grid)

                # Repeat base colors to 3x scale
                base_img_x3 = jnp.repeat(jnp.repeat(img, 3, axis=0), 3, axis=1)

                # Composite base and reward colors using helper
                img = apply_reward_overlay(
                    base_img_x3,
                    reward_colors,
                    reward_grid,
                    (self.size[1], self.size[0]),
                )

                # Tint the aperture region at 3x scale
                aperture_mask = jnp.zeros((self.size[1], self.size[0]), dtype=bool)
                aperture_mask = aperture_mask.at[y_coords_adj, x_coords_adj].set(True)
                aperture_mask_x3 = jnp.repeat(
                    jnp.repeat(aperture_mask, 3, axis=0), 3, axis=1
                )

                tinted_img = (
                    (1.0 - alpha) * img.astype(jnp.float32)
                    + alpha * self.agent_color_jax.astype(jnp.float32)
                ).astype(jnp.uint8)
                img = jnp.where(aperture_mask_x3[..., None], tinted_img, img)

                # Set agent center block
                agent_mask = jnp.zeros((self.size[1], self.size[0]), dtype=bool)
                agent_mask = agent_mask.at[state.pos[1], state.pos[0]].set(True)
                agent_mask_x3 = jnp.repeat(jnp.repeat(agent_mask, 3, axis=0), 3, axis=1)
                img = jnp.where(agent_mask_x3[..., None], self.agent_color_jax, img)

                # Final scale by 3 to get 9x
                img = jnp.repeat(jnp.repeat(img, 3, axis=0), 3, axis=1)
            else:
                # Standard rendering without reward visualization
                aperture_mask = jnp.zeros((self.size[1], self.size[0]), dtype=bool)
                aperture_mask = aperture_mask.at[y_coords_adj, x_coords_adj].set(True)

                tinted_img = (
                    (1.0 - alpha) * img.astype(jnp.float32)
                    + alpha * self.agent_color_jax.astype(jnp.float32)
                ).astype(jnp.uint8)
                img = jnp.where(aperture_mask[..., None], tinted_img, img)

                # Set agent
                img = img.at[state.pos[1], state.pos[0]].set(self.agent_color_jax)
                # Scale by 9
                img = jnp.repeat(jnp.repeat(img, 9, axis=0), 9, axis=1)

            if is_true_mode:
                # Apply true object borders
                img = apply_true_borders(
                    img,
                    state.object_state.object_id,
                    (self.size[1], self.size[0]),
                    len(self.object_ids),
                )

            # Add grid lines
            img = apply_grid_lines(
                img, (self.size[1], self.size[0]), self.grid_color_jax, 9
            )

        elif is_aperture_mode:
            obs_grid = state.object_state.object_id
            aperture = self._get_aperture(obs_grid, state.pos)

            y_coords, x_coords, y_coords_adj, x_coords_adj = (
                self._compute_aperture_coordinates(state.pos)
            )
            color_state = state.object_state.color[y_coords_adj, x_coords_adj]

            img = get_base_image(
                aperture,
                color_state,
                self.object_colors,
                self.dynamic_biomes,
            )

            if self.dynamic_biomes and self.nowrap:
                # For out-of-bounds, use padding object color
                y_out = (y_coords < 0) | (y_coords >= self.size[1])
                x_out = (x_coords < 0) | (x_coords >= self.size[0])
                out_of_bounds = y_out | x_out
                padding_color = jnp.array(self.objects[-1].color, dtype=jnp.float32)
                img = jnp.where(out_of_bounds[..., None], padding_color, img)

            if is_reward_mode:
                # Scale image by 3 to create space for reward visualization
                img = img.astype(jnp.uint8)
                img = jax.image.resize(
                    img,
                    (
                        self.aperture_size[0] * 3,
                        self.aperture_size[1] * 3,
                        3,
                    ),
                    jax.image.ResizeMethod.NEAREST,
                )

                # Compute rewards for aperture region
                aperture_params = state.object_state.state_params[
                    y_coords_adj, x_coords_adj
                ]
                aperture_timer = self._get_aperture(
                    state.object_state.respawn_timer, state.pos
                )
                aperture_rewards = self._compute_reward_grid(
                    state, aperture, aperture_params, aperture_timer
                )

                # Convert rewards to colors
                reward_colors = reward_to_color(aperture_rewards)

                # Apply reward overlay using helper
                img = apply_reward_overlay(
                    img,
                    reward_colors,
                    aperture_rewards,
                    self.aperture_size,
                )

                # Draw agent in the center (all 9 cells of the 3x3 block)
                center_y, center_x = (
                    self.aperture_size[0] // 2,
                    self.aperture_size[1] // 2,
                )
                agent_offsets = jnp.array(
                    [[dy, dx] for dy in range(3) for dx in range(3)]
                )
                agent_y_cells = center_y * 3 + agent_offsets[:, 0]
                agent_x_cells = center_x * 3 + agent_offsets[:, 1]
                img = img.at[agent_y_cells, agent_x_cells].set(
                    jnp.array(AGENT.color, dtype=jnp.uint8)
                )

                # Scale by 3 to final size
                img = jax.image.resize(
                    img,
                    (self.aperture_size[0] * 3, self.aperture_size[1] * 3, 3),
                    jax.image.ResizeMethod.NEAREST,
                )
            else:
                # Standard rendering without reward visualization
                # Draw agent in the center
                center_y, center_x = (
                    self.aperture_size[1] // 2,
                    self.aperture_size[0] // 2,
                )
                img = img.at[center_y, center_x].set(jnp.array(AGENT.color))

                img = img.astype(jnp.uint8)
                img = jax.image.resize(
                    img,
                    (self.aperture_size[0] * 9, self.aperture_size[1] * 9, 3),
                    jax.image.ResizeMethod.NEAREST,
                )

            if is_true_mode:
                # Apply true object borders
                img = apply_true_borders(
                    img, aperture, self.aperture_size, len(self.object_ids)
                )

            # Add grid lines
            img = apply_grid_lines(img, self.aperture_size, self.grid_color_jax, 9)

        else:
            raise ValueError(f"Unknown render_mode: {render_mode}")

        if self.return_hint:
            obs = self.get_obs(state, params)
            if isinstance(obs, dict) and "hint" in obs:
                hint = obs["hint"]
                img = apply_hint_bottom_bar(img, hint, bar_height=9, separator_height=9)

        return img
