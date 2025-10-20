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
    biome_grid: jax.Array
    time: int
    digestion_buffer: jax.Array
    object_spawn_time_grid: jax.Array
    biome_objects_collected: jax.Array  # Number of objects collected per biome
    biome_total_spawned: jax.Array  # Total objects spawned per biome
    biome_regenerated: jax.Array  # Whether each biome has been regenerated (bool)
    object_colors: jax.Array  # (num_objects, 3) - RGB colors for each object type
    fourier_params: jax.Array  # (num_objects, 2*num_harmonics + 1) - [period, a_1, b_1, a_2, b_2, ..., a_n, b_n]


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
        teleport_interval: Optional[int] = None,
        observation_type: str = "object",
        biome_regen_threshold: float = 0.0,
        alternate_objects: Optional[Tuple[BaseForagaxObject, ...]] = None,
    ):
        super().__init__()
        self._name = name
        if isinstance(size, int):
            size = (size, size)
        self.size = size

        # Handle aperture_size = -1 for world view
        if isinstance(aperture_size, int) and aperture_size == -1:
            self.full_world = True
            self.aperture_size = self.size  # Use full size for consistency
        else:
            self.full_world = False
            if isinstance(aperture_size, int):
                aperture_size = (aperture_size, aperture_size)
            self.aperture_size = aperture_size

        self.observation_type = observation_type
        self.nowrap = nowrap
        self.deterministic_spawn = deterministic_spawn
        self.teleport_interval = teleport_interval
        self.biome_regen_threshold = (
            biome_regen_threshold  # 0.0 means disabled, 0.9 means 90%
        )
        self.num_fourier_harmonics = 50  # Number of harmonics for Fourier series

        # Combine primary and alternate objects into a single list
        # Primary objects: indices 0 to len(objects)-1
        # Alternate objects: indices len(objects) to len(objects)+len(alternate_objects)-1
        objects = (EMPTY,) + objects
        if alternate_objects is not None:
            alternate_objects_tuple = (EMPTY,) + alternate_objects
            if self.nowrap and not self.full_world:
                # Add padding to both sets
                objects = objects + (PADDING,)
                alternate_objects_tuple = alternate_objects_tuple + (PADDING,)
            # Combine all objects
            all_objects = (
                objects + alternate_objects_tuple[1:]
            )  # Skip EMPTY from alternates to avoid duplicate
            self.num_primary_objects = len(objects)
            self.has_alternate_objects = True
        else:
            if self.nowrap and not self.full_world:
                objects = objects + (PADDING,)
            all_objects = objects
            self.num_primary_objects = len(objects)
            self.has_alternate_objects = False

        self.objects = all_objects

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

        # Compute reward steps per object (using max_reward_delay attribute)
        object_max_reward_delay = jnp.array([o.max_reward_delay for o in objects])
        self.max_reward_delay = (
            int(jnp.max(object_max_reward_delay)) + 1 if len(objects) > 0 else 0
        )

        # Store initial object colors (will be copied to state)
        self.initial_object_colors = self.object_colors.copy()

        # Initialize Fourier parameters (will be copied to state and potentially modified)
        # Shape: (num_objects, 2*num_harmonics + 1) where first element is period, rest are [a_1, b_1, a_2, b_2, ...]
        # Initially set to zeros - will be populated if needed during biome regeneration
        self.initial_fourier_params = jnp.zeros(
            (len(objects), 2 * self.num_fourier_harmonics + 1)
        )

        # Setup alternate objects if biome regeneration is enabled
        if self.has_alternate_objects:
            self.alt_object_ids = jnp.arange(len(self.alternate_objects))
            self.alt_object_blocking = jnp.array(
                [o.blocking for o in self.alternate_objects]
            )
            self.alt_object_collectable = jnp.array(
                [o.collectable for o in self.alternate_objects]
            )
            self.alt_object_colors = jnp.array(
                [o.color for o in self.alternate_objects]
            )
            self.alt_object_random_respawn = jnp.array(
                [o.random_respawn for o in self.alternate_objects]
            )

            self.alt_reward_fns = [o.reward for o in self.alternate_objects]
            self.alt_regen_delay_fns = [o.regen_delay for o in self.alternate_objects]
            self.alt_reward_delay_fns = [o.reward_delay for o in self.alternate_objects]
            self.alt_expiry_regen_delay_fns = [
                o.expiry_regen_delay for o in self.alternate_objects
            ]

            self.alt_object_expiry_time = jnp.array(
                [
                    o.expiry_time if o.expiry_time is not None else -1
                    for o in self.alternate_objects
                ]
            )

            alt_object_max_reward_delay = jnp.array(
                [o.max_reward_delay for o in self.alternate_objects]
            )
            alt_max_reward_delay = (
                int(jnp.max(alt_object_max_reward_delay)) + 1
                if len(self.alternate_objects) > 0
                else 0
            )
            # Use the maximum of both object sets for buffer size
            self.max_reward_delay = max(self.max_reward_delay, alt_max_reward_delay)

        self.biome_object_frequencies = jnp.array(
            [b.object_frequencies for b in biomes]
        )
        self.biome_starts = np.array(
            [b.start if b.start is not None else (-1, -1) for b in biomes]
        )
        self.biome_stops = np.array(
            [b.stop if b.stop is not None else (-1, -1) for b in biomes]
        )
        self.biome_sizes = np.prod(self.biome_stops - self.biome_starts, axis=1)
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
        for i in range(self.biome_object_frequencies.shape[0]):
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
            self.biome_masks.append(mask)

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

        # If we have alternate objects, also include their colors in the mapping
        if self.has_alternate_objects:
            alt_object_colors_no_empty = self.alt_object_colors[1:]
            alt_color_indices = jnp.zeros(len(alt_object_colors_no_empty), dtype=int)

            for i, color in enumerate(alt_object_colors_no_empty):
                color_tuple = tuple(color.tolist())
                if color_tuple not in color_map:
                    color_map[color_tuple] = next_channel
                    unique_colors.append(color)
                    next_channel += 1
                alt_color_indices = alt_color_indices.at[i].set(color_map[color_tuple])

            self.alt_object_to_color_map = alt_color_indices

        self.unique_colors = jnp.array(unique_colors)
        self.num_color_channels = len(unique_colors)
        # color_indices maps from object_id-1 to color_channel_index
        self.object_to_color_map = color_indices

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(
            max_steps_in_episode=None,
        )

    def _compute_fourier_reward(self, fourier_params: jax.Array, time: int) -> float:
        """Compute reward from Fourier series parameters using harmonic decomposition.

        Args:
            fourier_params: Array of shape (2*num_harmonics + 1,) where:
                - fourier_params[0] = period
                - fourier_params[1::2] = a_n (cosine coefficients)
                - fourier_params[2::2] = b_n (sine coefficients)
            time: Current timestep

        Returns:
            Reward value computed from normalized Fourier series sum
        """
        period = fourier_params[0]

        # Extract cosine and sine coefficients
        a_n = fourier_params[1::2]  # a_1, a_2, a_3, ...
        b_n = fourier_params[2::2]  # b_1, b_2, b_3, ...

        # Compute Fourier series: sum of a_n * cos(2πnx/period) + b_n * sin(2πnx/period)
        n_harmonics = len(a_n)
        harmonics = jnp.arange(1, n_harmonics + 1)

        # Compute all harmonic contributions
        cos_terms = a_n * jnp.cos(2 * jnp.pi * harmonics * time / period)
        sin_terms = b_n * jnp.sin(2 * jnp.pi * harmonics * time / period)

        # Sum all terms (already normalized to [-1, 1] during generation)
        reward = jnp.sum(cos_terms + sin_terms)

        return reward

    def _place_timer_at_position(
        self, grid: jax.Array, y: int, x: int, timer_val: int
    ) -> jax.Array:
        """Place a timer at a specific position."""
        return grid.at[y, x].set(timer_val)

    def _place_timer_at_random_position(
        self,
        grid: jax.Array,
        y: int,
        x: int,
        timer_val: int,
        biome_grid: jax.Array,
        rand_key: jax.Array,
    ) -> jax.Array:
        """Place a timer at a random valid position within the same biome."""
        # Set the original position to empty temporarily
        grid_temp = grid.at[y, x].set(0)

        # Find all valid spawn locations (empty cells within the same biome)
        biome_id = biome_grid[y, x]
        biome_mask = biome_grid == biome_id
        empty_mask = grid_temp == 0
        valid_spawn_mask = biome_mask & empty_mask

        num_valid_spawns = jnp.sum(valid_spawn_mask)

        # Get indices of valid spawn locations, padded to a static size
        y_indices, x_indices = jnp.nonzero(
            valid_spawn_mask, size=self.size[0] * self.size[1], fill_value=-1
        )
        valid_spawn_indices = jnp.stack([y_indices, x_indices], axis=1)

        # Select a random valid location
        random_idx = jax.random.randint(rand_key, (), 0, num_valid_spawns)
        new_spawn_pos = valid_spawn_indices[random_idx]

        # Place the timer at the new random position
        new_grid = grid_temp.at[new_spawn_pos[0], new_spawn_pos[1]].set(timer_val)
        return new_grid

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

        # Check for automatic teleport
        if self.teleport_interval is not None:
            should_teleport = jnp.mod(state.time + 1, self.teleport_interval) == 0
        else:
            should_teleport = False

        def teleport_fn():
            # Calculate squared distances from current position to each biome center
            diffs = self.biome_centers_jax - pos
            distances = jnp.sum(diffs**2, axis=1)
            # Find the index of the furthest biome center
            furthest_idx = jnp.argmax(distances)
            new_pos = self.biome_centers_jax[furthest_idx]
            return new_pos

        pos = jax.lax.cond(should_teleport, teleport_fn, lambda: pos)

        # 2. HANDLE COLLISIONS AND REWARDS
        obj_at_pos = current_objects[pos[1], pos[0]]
        is_collectable = self.object_collectable[obj_at_pos]
        should_collect = is_collectable & (obj_at_pos > 0)

        # Handle digestion: add reward to buffer if collected
        digestion_buffer = state.digestion_buffer
        key, reward_subkey = jax.random.split(key)

        # Check if this object has Fourier parameters set (sum of all params > 0)
        obj_fourier_params = state.fourier_params[obj_at_pos]
        has_fourier = jnp.sum(jnp.abs(obj_fourier_params)) > 0.0

        # Compute reward: use Fourier if available, otherwise use object's reward function
        fourier_reward = self._compute_fourier_reward(obj_fourier_params, state.time)
        default_reward = jax.lax.switch(
            obj_at_pos, self.reward_fns, state.time, reward_subkey
        )
        object_reward = jnp.where(has_fourier, fourier_reward, default_reward)

        key, digestion_subkey = jax.random.split(key)
        reward_delay = jax.lax.switch(
            obj_at_pos, self.reward_delay_fns, state.time, digestion_subkey
        )
        reward = jnp.where(should_collect & (reward_delay == 0), object_reward, 0.0)
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

        # 3. HANDLE OBJECT COLLECTION AND RESPAWNING
        key, regen_subkey, rand_key = jax.random.split(key, 3)

        # Decrement timers (stored as negative values)
        is_timer = state.object_grid < 0
        object_grid = jnp.where(
            is_timer, state.object_grid + num_obj_types, state.object_grid
        )

        # Track which objects just respawned (timer -> object transition)
        # An object respawned if it was negative and is now positive
        just_respawned = is_timer & (object_grid > 0)

        # Update spawn times for objects that just respawned
        object_spawn_time_grid = jnp.where(
            just_respawned, state.time, state.object_spawn_time_grid
        )

        # Collect object: set a timer
        regen_delay = jax.lax.switch(
            obj_at_pos, self.regen_delay_fns, state.time, regen_subkey
        )
        encoded_timer = obj_at_pos - ((regen_delay + 1) * num_obj_types)

        # If collected, replace object with timer; otherwise, keep it
        val_at_pos = object_grid[pos[1], pos[0]]
        should_collect = is_collectable & (val_at_pos > 0)

        # When not collecting, the value at the position remains unchanged.
        # When collecting, we either place the timer at the current position or a random one.
        def do_collection():
            return jax.lax.cond(
                self.object_random_respawn[obj_at_pos],
                lambda: self._place_timer_at_random_position(
                    object_grid,
                    pos[1],
                    pos[0],
                    encoded_timer,
                    state.biome_grid,
                    rand_key,
                ),
                lambda: self._place_timer_at_position(
                    object_grid, pos[1], pos[0], encoded_timer
                ),
            )

        def no_collection():
            return object_grid

        object_grid = jax.lax.cond(
            should_collect,
            do_collection,
            no_collection,
        )

        # 3.5. HANDLE OBJECT EXPIRY
        # Check each cell for objects that have exceeded their expiry time
        current_objects_for_expiry = jnp.maximum(0, object_grid)

        # Calculate age of each object (current_time - spawn_time)
        object_ages = state.time - object_spawn_time_grid

        # Get expiry time for each object type in the grid
        expiry_times = self.object_expiry_time[current_objects_for_expiry]

        # Check if object should expire (age >= expiry_time and expiry_time >= 0)
        should_expire = (
            (object_ages >= expiry_times)
            & (expiry_times >= 0)
            & (current_objects_for_expiry > 0)
        )

        # For expired objects, calculate expiry regen delay
        key, expiry_key = jax.random.split(key)

        # Process expiry for all positions that need it
        def process_expiry(y, x, grid, spawn_grid, key):
            obj_id = current_objects_for_expiry[y, x]
            should_exp = should_expire[y, x]

            def expire_object():
                # Get expiry regen delay for this object
                exp_key = jax.random.fold_in(key, y * self.size[0] + x)
                exp_delay = jax.lax.switch(
                    obj_id, self.expiry_regen_delay_fns, state.time, exp_key
                )
                encoded_exp_timer = obj_id - ((exp_delay + 1) * num_obj_types)

                # Check if this object should respawn randomly
                should_random_respawn = self.object_random_respawn[obj_id]

                # Use second split for randomness in random placement
                rand_key = jax.random.split(exp_key)[1]

                # Place timer either at current position or random position
                new_grid = jax.lax.cond(
                    should_random_respawn,
                    lambda: self._place_timer_at_random_position(
                        grid, y, x, encoded_exp_timer, state.biome_grid, rand_key
                    ),
                    lambda: self._place_timer_at_position(
                        grid, y, x, encoded_exp_timer
                    ),
                )

                return new_grid, spawn_grid

            def no_expire():
                return grid, spawn_grid

            return jax.lax.cond(should_exp, expire_object, no_expire)

        # Apply expiry to all cells (vectorized)
        def scan_expiry_row(carry, y):
            grid, spawn_grid, key = carry

            def scan_expiry_col(carry_col, x):
                grid_col, spawn_grid_col, key_col = carry_col
                grid_col, spawn_grid_col = process_expiry(
                    y, x, grid_col, spawn_grid_col, key_col
                )
                return (grid_col, spawn_grid_col, key_col), None

            (grid, spawn_grid, key), _ = jax.lax.scan(
                scan_expiry_col, (grid, spawn_grid, key), jnp.arange(self.size[0])
            )
            return (grid, spawn_grid, key), None

        (object_grid, object_spawn_time_grid, _), _ = jax.lax.scan(
            scan_expiry_row,
            (object_grid, object_spawn_time_grid, expiry_key),
            jnp.arange(self.size[1]),
        )

        info = {"discount": self.discount(state, params)}
        temperatures = jnp.zeros(len(self.objects))
        for obj_index, obj in enumerate(self.objects):
            if isinstance(obj, WeatherObject):
                temperatures = temperatures.at[obj_index].set(
                    get_temperature(obj.rewards, state.time, obj.repeat)
                )
        info["temperatures"] = temperatures
        info["biome_id"] = state.biome_grid[pos[1], pos[0]]
        info["object_collected_id"] = jax.lax.select(should_collect, obj_at_pos, -1)

        # 3.6. TRACK BIOME CONSUMPTION AND REGENERATION
        # Update collection counter if we collected an object
        biome_id = state.biome_grid[pos[1], pos[0]]
        biome_objects_collected = jax.lax.cond(
            should_collect,
            lambda: state.biome_objects_collected.at[biome_id].add(1),
            lambda: state.biome_objects_collected,
        )

        # Check if any biome should regenerate (>= threshold % collected)
        biome_consumption_ratio = jnp.where(
            state.biome_total_spawned > 0,
            biome_objects_collected / state.biome_total_spawned,
            0.0,
        )

        # Regenerate biomes that hit threshold and haven't been regenerated yet
        should_regenerate_biomes = (
            (biome_consumption_ratio >= self.biome_regen_threshold)
            & (self.biome_regen_threshold > 0)
            & ~state.biome_regenerated
        )

        # Apply biome regeneration if needed
        def regenerate_biome(biome_idx, grid, spawn_grid, colors, fourier, regen_key):
            """Regenerate a single biome with new colors and Fourier parameters."""
            mask = self.biome_masks[biome_idx]

            # Clear existing objects in this biome and regenerate
            grid = jnp.where(mask, 0, grid)
            spawn_grid = jnp.where(mask, state.time, spawn_grid)

            # Generate new objects using the same frequencies
            regen_key, biome_key, color_key, fourier_key = jax.random.split(
                regen_key, 4
            )
            if self.deterministic_spawn:
                biome_objects = self.generate_biome_new(biome_idx, biome_key)
                grid = grid.at[mask].set(biome_objects)
            else:
                biome_objects = self.generate_biome_old(biome_idx, biome_key)
                grid = jnp.where(mask, biome_objects, grid)

            # Generate new random colors for collectable objects in this biome
            # Simplest approach: Don't loop at all if biome has no objects
            # Just return unchanged colors and fourier params
            biome_freqs = self.biome_object_frequencies[biome_idx]

            # If biome has no objects, return early
            if len(biome_freqs) == 0:
                # No updates needed
                pass
            else:
                # Update colors and Fourier params for each object type
                def update_object_params(obj_idx, carry):
                    colors_updated, fourier_updated, key = carry
                    actual_obj_idx = obj_idx + 1  # +1 because index 0 is EMPTY

                    # Check if this object is in biome and collectable (using JAX operations)
                    in_biome = biome_freqs[obj_idx] > 0
                    is_collectable = self.object_collectable[actual_obj_idx]
                    should_update = in_biome & is_collectable

                    # Generate new color and Fourier params (even if not used)
                    key, color_subkey, period_key, coeff_key = jax.random.split(key, 4)
                    new_color = jax.random.randint(color_subkey, (3,), 0, 256)

                    # Generate Fourier series parameters following the provided approach
                    # 1. Random period between 10 and 1000
                    period = jax.random.randint(period_key, (), 10, 1001).astype(
                        jnp.float32
                    )

                    # 2. Generate random harmonic coefficients scaled by 1/n
                    coeff_key, a_key, b_key = jax.random.split(coeff_key, 3)
                    n_harmonics = self.num_fourier_harmonics
                    harmonic_indices = jnp.arange(1, n_harmonics + 1)

                    # Random coefficients scaled by 1/n (gives less weight to higher frequencies)
                    a_n = jax.random.normal(a_key, (n_harmonics,)) / harmonic_indices
                    b_n = jax.random.normal(b_key, (n_harmonics,)) / harmonic_indices

                    # 3. Compute a sample of the function to find normalization factor
                    # Sample at regular intervals over one period
                    n_samples = 100
                    sample_x = jnp.linspace(0, period, n_samples)
                    sample_y = jnp.zeros(n_samples)

                    # Compute function values at sample points
                    for n in range(1, n_harmonics + 1):
                        sample_y += a_n[n - 1] * jnp.cos(
                            2 * jnp.pi * n * sample_x / period
                        )
                        sample_y += b_n[n - 1] * jnp.sin(
                            2 * jnp.pi * n * sample_x / period
                        )

                    # 4. Normalize to [-1, 1]
                    max_val = jnp.max(jnp.abs(sample_y))
                    # Avoid division by zero
                    normalization_factor = jnp.where(max_val > 0, max_val, 1.0)
                    a_n_normalized = a_n / normalization_factor
                    b_n_normalized = b_n / normalization_factor

                    # Interleave a_n and b_n: [period, a_1, b_1, a_2, b_2, ...]
                    coeffs = jnp.empty(2 * n_harmonics)
                    coeffs = coeffs.at[::2].set(a_n_normalized)
                    coeffs = coeffs.at[1::2].set(b_n_normalized)
                    new_fourier_params = jnp.concatenate([jnp.array([period]), coeffs])

                    # Conditionally update (using jnp.where)
                    colors_updated = jnp.where(
                        should_update,
                        colors_updated.at[actual_obj_idx].set(new_color),
                        colors_updated,
                    )
                    fourier_updated = jnp.where(
                        should_update,
                        fourier_updated.at[actual_obj_idx].set(new_fourier_params),
                        fourier_updated,
                    )

                    return colors_updated, fourier_updated, key

                # Use lax.fori_loop
                colors, fourier, _ = jax.lax.fori_loop(
                    0,
                    len(biome_freqs),
                    update_object_params,
                    (colors, fourier, fourier_key),
                )

            return grid, spawn_grid, colors, fourier, regen_key

        # Regenerate each biome that should be regenerated
        key, regen_key = jax.random.split(key)
        object_colors = state.object_colors
        fourier_params = state.fourier_params

        for biome_idx in range(len(should_regenerate_biomes)):
            (
                object_grid,
                object_spawn_time_grid,
                object_colors,
                fourier_params,
                regen_key,
            ) = jax.lax.cond(
                should_regenerate_biomes[biome_idx],
                lambda og=object_grid,
                st=object_spawn_time_grid,
                oc=object_colors,
                fp=fourier_params,
                rk=regen_key: regenerate_biome(biome_idx, og, st, oc, fp, rk),
                lambda og=object_grid,
                st=object_spawn_time_grid,
                oc=object_colors,
                fp=fourier_params,
                rk=regen_key: (og, st, oc, fp, rk),
            )

        # Update biome tracking
        biome_regenerated = state.biome_regenerated | should_regenerate_biomes

        # Reset counters for regenerated biomes
        biome_objects_collected = jnp.where(
            should_regenerate_biomes, 0, biome_objects_collected
        )

        # Recalculate total spawned for regenerated biomes
        biome_total_spawned = state.biome_total_spawned
        for biome_idx in range(len(should_regenerate_biomes)):

            def update_total(idx):
                mask = self.biome_masks[idx]
                new_total = jnp.sum((object_grid > 0) & mask)
                return biome_total_spawned.at[idx].set(new_total)

            biome_total_spawned = jax.lax.cond(
                should_regenerate_biomes[biome_idx],
                lambda: update_total(biome_idx),
                lambda: biome_total_spawned,
            )

        # 4. UPDATE STATE
        state = EnvState(
            pos=pos,
            object_grid=object_grid,
            biome_grid=state.biome_grid,
            time=state.time + 1,
            digestion_buffer=digestion_buffer,
            object_spawn_time_grid=object_spawn_time_grid,
            biome_objects_collected=biome_objects_collected,
            biome_total_spawned=biome_total_spawned,
            biome_regenerated=biome_regenerated,
            object_colors=object_colors,
            fourier_params=fourier_params,
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
        object_grid = jnp.zeros((self.size[1], self.size[0]), dtype=int)
        biome_grid = jnp.full((self.size[1], self.size[0]), -1, dtype=int)
        key, iter_key = jax.random.split(key)
        for i in range(self.biome_object_frequencies.shape[0]):
            iter_key, biome_key = jax.random.split(iter_key)
            mask = self.biome_masks[i]
            biome_grid = jnp.where(mask, i, biome_grid)

            if self.deterministic_spawn:
                biome_objects = self.generate_biome_new(i, biome_key)
                object_grid = object_grid.at[mask].set(biome_objects)
            else:
                biome_objects = self.generate_biome_old(i, biome_key)
                object_grid = jnp.where(mask, biome_objects, object_grid)

        # Place agent in the center of the world
        agent_pos = jnp.array([self.size[0] // 2, self.size[1] // 2])

        # Initialize spawn times to 0 (all objects spawn at time 0)
        object_spawn_time_grid = jnp.zeros((self.size[1], self.size[0]), dtype=int)

        # Initialize biome consumption tracking
        num_biomes = self.biome_object_frequencies.shape[0]
        biome_objects_collected = jnp.zeros(num_biomes, dtype=int)

        # Count total spawned objects per biome
        biome_total_spawned = jnp.zeros(num_biomes, dtype=int)
        for i in range(num_biomes):
            mask = self.biome_masks[i]
            # Count non-zero (non-empty) objects in this biome
            biome_total_spawned = biome_total_spawned.at[i].set(
                jnp.sum((object_grid > 0) & mask)
            )

        biome_regenerated = jnp.zeros(num_biomes, dtype=bool)

        state = EnvState(
            pos=agent_pos,
            object_grid=object_grid,
            biome_grid=biome_grid,
            time=0,
            digestion_buffer=jnp.zeros((self.max_reward_delay,)),
            object_spawn_time_grid=object_spawn_time_grid,
            biome_objects_collected=biome_objects_collected,
            biome_total_spawned=biome_total_spawned,
            biome_regenerated=biome_regenerated,
            object_colors=self.initial_object_colors.copy(),
            fourier_params=self.initial_fourier_params.copy(),
        )

        return self.get_obs(state, params), state

    def generate_biome_old(self, i: int, biome_key: jax.Array):
        biome_freqs = self.biome_object_frequencies[i]
        grid_rand = jax.random.uniform(biome_key, (self.size[1], self.size[0]))
        empty_freq = 1.0 - jnp.sum(biome_freqs)
        all_freqs = jnp.concatenate([jnp.array([empty_freq]), biome_freqs])
        cumulative_freqs = jnp.cumsum(jnp.concatenate([jnp.array([0.0]), all_freqs]))
        biome_objects = jnp.searchsorted(cumulative_freqs, grid_rand, side="right") - 1
        return biome_objects

    def generate_biome_new(self, i: int, biome_key: jax.Array):
        biome_freqs = self.biome_object_frequencies[i]
        grid = jnp.linspace(0, 1, self.biome_sizes[i], endpoint=False)
        biome_objects = len(biome_freqs) - jnp.searchsorted(
            jnp.cumsum(biome_freqs[::-1]), grid, side="right"
        )
        flat_biome_objects = biome_objects.flatten()
        shuffled_objects = jax.random.permutation(biome_key, flat_biome_objects)
        return shuffled_objects

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
        return spaces.Dict(
            {
                "pos": spaces.Box(0, max(self.size), (2,), int),
                "object_grid": spaces.Box(
                    -1000 * len(self.object_ids),
                    len(self.object_ids),
                    (self.size[1], self.size[0]),
                    int,
                ),
                "biome_grid": spaces.Box(
                    0,
                    self.biome_object_frequencies.shape[0],
                    (self.size[1], self.size[0]),
                    int,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "digestion_buffer": spaces.Box(
                    -jnp.inf,
                    jnp.inf,
                    (self.max_reward_delay,),
                    float,
                ),
                "object_spawn_time_grid": spaces.Box(
                    0,
                    jnp.inf,
                    (self.size[1], self.size[0]),
                    int,
                ),
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

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        """Get observation based on observation_type and full_world."""
        # Decode grid for observation
        obs_grid = jnp.maximum(0, state.object_grid)

        if self.full_world:
            return self._get_world_obs(obs_grid, state)
        else:
            grid = self._get_aperture(obs_grid, state.pos)
            return self._get_aperture_obs(grid, state)

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
            obs = jax.nn.one_hot(obs_grid, num_obj_types)
            # Agent position
            obs = obs.at[state.pos[1], state.pos[0], :].set(0)
            obs = obs.at[state.pos[1], state.pos[0], -1].set(1)
            colors = state.object_colors / 255.0  # Use state colors
            obs = jnp.tensordot(obs, colors, axes=1)
            return obs
        elif self.observation_type == "color":
            # Handle case with no objects (only EMPTY)
            if self.num_color_channels == 0:
                return jnp.zeros(obs_grid.shape + (0,), dtype=jnp.float32)
            # Map object IDs to color channel indices
            color_channels = jnp.where(
                obs_grid == 0,
                -1,
                jnp.take(self.object_to_color_map, obs_grid - 1, axis=0),
            )
            obs = jax.nn.one_hot(color_channels, self.num_color_channels, axis=-1)
            return obs
        else:
            raise ValueError(f"Unknown observation_type: {self.observation_type}")

    def _get_aperture_obs(self, aperture: jax.Array, state: EnvState) -> jax.Array:
        """Get aperture observation."""
        if self.observation_type == "object":
            num_obj_types = len(self.object_ids)
            obs = jax.nn.one_hot(aperture, num_obj_types, axis=-1)
            return obs
        elif self.observation_type == "rgb":
            num_obj_types = len(self.object_ids)
            aperture_one_hot = jax.nn.one_hot(aperture, num_obj_types)
            colors = state.object_colors / 255.0  # Use state colors
            obs = jnp.tensordot(aperture_one_hot, colors, axes=1)
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

        return spaces.Box(0, 1, obs_shape, float)

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
                color = state.object_colors[i]  # Use state colors
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
            img = jnp.tensordot(
                aperture_one_hot, state.object_colors, axes=1
            )  # Use state colors

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
