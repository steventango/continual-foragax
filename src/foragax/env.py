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
    object_color_grid: jax.Array  # (H, W, 3) - RGB color per object
    object_generation_grid: jax.Array  # (H, W) - generation number for each object
    object_state_grid: jax.Array  # (H, W, N) - Fourier params per object
    biome_consumption_count: jax.Array  # (num_biomes,) - objects consumed per biome
    biome_total_objects: jax.Array  # (num_biomes,) - total objects spawned per biome
    biome_generation: jax.Array  # (num_biomes,) - current generation per biome


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
        dynamic_biomes: bool = False,
        biome_consumption_threshold: float = 0.9,
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
        self.dynamic_biomes = dynamic_biomes
        self.biome_consumption_threshold = biome_consumption_threshold

        objects = (EMPTY,) + objects
        if self.nowrap and not self.full_world:
            objects = objects + (PADDING,)
        self.objects = objects

        # Infer num_fourier_terms from objects
        self.num_fourier_terms = max(
            (
                obj.num_fourier_terms
                for obj in self.objects
                if isinstance(obj, FourierObject)
            ),
            default=0,
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
        self.biome_sizes = np.prod(self.biome_stops - self.biome_starts, axis=1)
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
            biome_masks_array.append(mask)

        # Convert to JAX array for indexing in JIT-compiled code
        self.biome_masks_array = jnp.array(biome_masks_array)

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

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(
            max_steps_in_episode=None,
        )

    def _place_timer(
        self,
        grid: jax.Array,
        y: int,
        x: int,
        timer_val: int,
        random_respawn: bool,
        biome_grid: jax.Array,
        rand_key: jax.Array,
    ) -> jax.Array:
        """Place a timer at position or randomly within the same biome.

        Args:
            grid: Object grid
            y, x: Original position
            timer_val: Timer value to place (negative for countdown, 0 for permanent removal)
            random_respawn: If True, place at random location in same biome
            biome_grid: Biome assignment grid
            rand_key: Random key for random placement

        Returns:
            Updated grid with timer placed
        """

        # Handle permanent removal (timer_val == 0)
        def place_empty():
            return grid.at[y, x].set(0)

        # Handle timer placement
        def place_timer():
            # Non-random: place at original position
            def place_at_position():
                return grid.at[y, x].set(timer_val)

            # Random: place at random location in same biome
            def place_randomly():
                grid_temp = grid.at[y, x].set(0)
                biome_id = biome_grid[y, x]
                biome_mask = biome_grid == biome_id
                empty_mask = grid_temp == 0
                valid_spawn_mask = biome_mask & empty_mask
                num_valid_spawns = jnp.sum(valid_spawn_mask)

                y_indices, x_indices = jnp.nonzero(
                    valid_spawn_mask, size=self.size[0] * self.size[1], fill_value=-1
                )
                valid_spawn_indices = jnp.stack([y_indices, x_indices], axis=1)
                random_idx = jax.random.randint(rand_key, (), 0, num_valid_spawns)
                new_spawn_pos = valid_spawn_indices[random_idx]

                return grid_temp.at[new_spawn_pos[0], new_spawn_pos[1]].set(timer_val)

            return jax.lax.cond(random_respawn, place_randomly, place_at_position)

        return jax.lax.cond(timer_val == 0, place_empty, place_timer)

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

        object_params = state.object_state_grid[pos[1], pos[0]]
        object_reward = jax.lax.switch(
            obj_at_pos, self.reward_fns, state.time, reward_subkey, object_params
        )

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
        encoded_timer = jax.lax.cond(
            regen_delay == jnp.iinfo(jnp.int32).max,
            lambda: 0,
            lambda: obj_at_pos - ((regen_delay + 1) * num_obj_types),
        )

        # If collected, replace object with timer; otherwise, keep it
        val_at_pos = object_grid[pos[1], pos[0]]
        # Use original should_collect for consumption tracking
        should_collect_now = is_collectable & (val_at_pos > 0)

        # Place timer on collection
        object_grid = jax.lax.cond(
            should_collect_now,
            lambda: self._place_timer(
                object_grid,
                pos[1],
                pos[0],
                encoded_timer,
                self.object_random_respawn[obj_at_pos],
                state.biome_grid,
                rand_key,
            ),
            lambda: object_grid,
        )

        # Clear color grid when object is collected
        object_color_grid = jax.lax.cond(
            should_collect_now,
            lambda: state.object_color_grid.at[pos[1], pos[0]].set(
                jnp.full((3,), 255, dtype=jnp.uint8)
            ),
            lambda: state.object_color_grid,
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
        def process_expiry(y, x, grid, color_grid, spawn_grid, key):
            obj_id = current_objects_for_expiry[y, x]
            should_exp = should_expire[y, x]

            def expire_object():
                exp_key = jax.random.fold_in(key, y * self.size[0] + x)
                exp_delay = jax.lax.switch(
                    obj_id, self.expiry_regen_delay_fns, state.time, exp_key
                )
                encoded_exp_timer = jax.lax.cond(
                    exp_delay == jnp.iinfo(jnp.int32).max,
                    lambda: 0,
                    lambda: obj_id - ((exp_delay + 1) * num_obj_types),
                )

                # Use unified timer placement method
                rand_key = jax.random.split(exp_key)[1]
                new_grid = self._place_timer(
                    grid,
                    y,
                    x,
                    encoded_exp_timer,
                    self.object_random_respawn[obj_id],
                    state.biome_grid,
                    rand_key,
                )

                # Clear color grid when object expires
                empty_color = jnp.full((3,), 255, dtype=jnp.uint8)
                new_color_grid = color_grid.at[y, x].set(empty_color)

                return new_grid, new_color_grid, spawn_grid

            def no_expire():
                return grid, color_grid, spawn_grid

            return jax.lax.cond(should_exp, expire_object, no_expire)

        # Apply expiry to all cells (vectorized)
        def scan_expiry_row(carry, y):
            grid, color_grid, spawn_grid, key = carry

            def scan_expiry_col(carry_col, x):
                grid_col, color_grid_col, spawn_grid_col, key_col = carry_col
                grid_col, color_grid_col, spawn_grid_col = process_expiry(
                    y, x, grid_col, color_grid_col, spawn_grid_col, key_col
                )
                return (grid_col, color_grid_col, spawn_grid_col, key_col), None

            (grid, color_grid, spawn_grid, key), _ = jax.lax.scan(
                scan_expiry_col,
                (grid, color_grid, spawn_grid, key),
                jnp.arange(self.size[0]),
            )
            return (grid, color_grid, spawn_grid, key), None

        (object_grid, object_color_grid, object_spawn_time_grid, _), _ = jax.lax.scan(
            scan_expiry_row,
            (object_grid, object_color_grid, object_spawn_time_grid, expiry_key),
            jnp.arange(self.size[1]),
        )

        # 3.6. HANDLE DYNAMIC BIOME CONSUMPTION AND RESPAWNING
        biome_consumption_count = state.biome_consumption_count
        biome_total_objects = state.biome_total_objects
        biome_generation = state.biome_generation
        object_generation_grid = state.object_generation_grid
        # object_color_grid is already updated above
        object_state_grid = state.object_state_grid

        if self.dynamic_biomes:
            # Update consumption count if an object was collected
            # Only count if the object belongs to the current generation of its biome
            collected_biome_id = state.biome_grid[pos[1], pos[0]]
            object_gen_at_pos = state.object_generation_grid[pos[1], pos[0]]
            current_biome_gen = state.biome_generation[collected_biome_id]
            is_current_generation = object_gen_at_pos == current_biome_gen

            biome_consumption_count = jax.lax.cond(
                should_collect & is_current_generation,
                lambda: biome_consumption_count.at[collected_biome_id].add(1),
                lambda: biome_consumption_count,
            )

            # Check each biome for threshold crossing and respawn if needed
            key, respawn_key = jax.random.split(key)
            (
                object_grid,
                object_color_grid,
                object_state_grid,
                biome_generation,
                biome_consumption_count,
                biome_total_objects,
                object_spawn_time_grid,
                object_generation_grid,
                respawn_key,
            ) = self._check_and_respawn_biomes(
                object_grid,
                state.biome_grid,
                object_color_grid,
                object_state_grid,
                biome_generation,
                biome_consumption_count,
                biome_total_objects,
                object_spawn_time_grid,
                state.object_generation_grid,
                state.time,
                respawn_key,
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

        # 4. UPDATE STATE
        state = EnvState(
            pos=pos,
            object_grid=object_grid,
            biome_grid=state.biome_grid,
            time=state.time + 1,
            digestion_buffer=digestion_buffer,
            object_spawn_time_grid=object_spawn_time_grid,
            object_color_grid=object_color_grid,
            object_state_grid=object_state_grid,
            biome_consumption_count=biome_consumption_count,
            biome_total_objects=biome_total_objects,
            biome_generation=biome_generation,
            object_generation_grid=object_generation_grid,
        )

        done = self.is_terminal(state, params)
        return (
            jax.lax.stop_gradient(self.get_obs(state, params)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def _check_and_respawn_biomes(
        self,
        object_grid: jax.Array,
        biome_grid: jax.Array,
        object_color_grid: jax.Array,
        object_state_grid: jax.Array,
        biome_generation: jax.Array,
        biome_consumption_count: jax.Array,
        biome_total_objects: jax.Array,
        object_spawn_time_grid: jax.Array,
        object_generation_grid: jax.Array,
        current_time: int,
        key: jax.Array,
    ) -> Tuple[jax.Array, ...]:
        """Check all biomes for consumption threshold and respawn if needed."""

        def check_biome(i, carry):
            (
                obj_grid,
                color_grid,
                params_grid,
                gen,
                consumption,
                total,
                spawn_grid,
                gen_grid,
                k,
            ) = carry

            consumption_rate = consumption[i] / jnp.maximum(1.0, total[i].astype(float))

            should_respawn = consumption_rate >= self.biome_consumption_threshold

            def respawn_biome_i():
                # Get biome mask and properties
                biome_mask = self.biome_masks_array[i]

                # Generate new objects for this biome using unified spawn method
                # Use deterministic spawn if configured, otherwise use random spawn
                # For deterministic spawn with fori_loop, we need to handle the traced index
                # by using switch to dispatch to concrete biome-specific spawn functions
                k_new, spawn_key = jax.random.split(k)

                if self.deterministic_spawn:
                    # Create spawn functions for each biome (using concrete indices)
                    def make_spawn_fn(biome_idx):
                        def spawn_fn(key):
                            return self._spawn_biome_objects(
                                biome_idx, key, deterministic=True
                            )

                        return spawn_fn

                    # Create list of spawn functions (one per biome)
                    spawn_fns = [make_spawn_fn(idx) for idx in range(len(self.biome_masks))]

                    # Use switch to select the correct spawn function based on traced index i
                    new_biome_objects_full, new_biome_colors_full, new_biome_params_full = (
                        jax.lax.switch(i, spawn_fns, spawn_key)
                    )
                else:
                    # Random spawn works fine with traced index
                    new_biome_objects_full, new_biome_colors_full, new_biome_params_full = (
                        self._spawn_biome_objects(i, spawn_key, deterministic=False)
                    )

                # CRITICAL FIX: Only replace where new spawn is NOT empty
                # Keep old objects where new spawn is empty (0)
                # Count new objects BEFORE merging
                new_object_count = jnp.sum((new_biome_objects_full > 0) & biome_mask)

                is_new_object = (new_biome_objects_full > 0) & biome_mask
                new_obj_grid = jnp.where(
                    is_new_object, new_biome_objects_full, obj_grid
                )

                # Update generation counter
                new_gen = gen.at[i].add(1)
                new_gen_value = new_gen[i]

                # Use precomputed colors and params for NEW objects only
                new_params_grid = jnp.where(
                    is_new_object[..., None], new_biome_params_full, params_grid
                )
                new_color_grid = jnp.where(
                    is_new_object[..., None], new_biome_colors_full, color_grid
                )

                # Tag new objects with the new generation number
                new_gen_grid = jnp.where(is_new_object, new_gen_value, gen_grid)

                # Reset consumption count and update total objects
                # IMPORTANT: Count only NEWLY spawned objects, not old preserved objects
                # This ensures consumption tracking is based on the current generation
                new_consumption = consumption.at[i].set(0)
                new_total = total.at[i].set(new_object_count)

                # Reset spawn times ONLY for newly placed objects
                new_spawn_grid = jnp.where(is_new_object, current_time, spawn_grid)

                return (
                    new_obj_grid,
                    new_color_grid,
                    new_params_grid,
                    new_gen,
                    new_consumption,
                    new_total,
                    new_spawn_grid,
                    new_gen_grid,
                    k_new,
                )

            def no_respawn():
                return (
                    obj_grid,
                    color_grid,
                    params_grid,
                    gen,
                    consumption,
                    total,
                    spawn_grid,
                    gen_grid,
                    k,
                )

            return jax.lax.cond(should_respawn, respawn_biome_i, no_respawn)

        # Process all biomes
        init_carry = (
            object_grid,
            object_color_grid,
            object_state_grid,
            biome_generation,
            biome_consumption_count,
            biome_total_objects,
            object_spawn_time_grid,
            object_generation_grid,
            key,
        )

        final_carry = jax.lax.fori_loop(
            0, self.biome_object_frequencies.shape[0], check_biome, init_carry
        )

        return final_carry

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        """Reset environment state."""
        object_grid = jnp.zeros((self.size[1], self.size[0]), dtype=int)
        object_color_grid = jnp.full(
            (self.size[1], self.size[0], 3), 255, dtype=jnp.uint8
        )
        num_object_params = 2 + 2 * self.num_fourier_terms
        object_state_grid = jnp.zeros(
            (self.size[1], self.size[0], num_object_params), dtype=jnp.float32
        )
        biome_grid = jnp.full((self.size[1], self.size[0]), -1, dtype=int)

        key, iter_key = jax.random.split(key)

        # Spawn objects in each biome using unified method
        for i in range(self.biome_object_frequencies.shape[0]):
            iter_key, biome_key = jax.random.split(iter_key)
            mask = self.biome_masks[i]
            biome_grid = jnp.where(mask, i, biome_grid)

            # Use unified spawn method
            biome_objects, biome_colors, biome_object_state = self._spawn_biome_objects(
                i, biome_key, self.deterministic_spawn
            )

            # Merge biome objects/colors/params into global grids
            object_grid = jnp.where(mask, biome_objects, object_grid)
            object_color_grid = jnp.where(
                mask[..., None], biome_colors, object_color_grid
            )
            object_state_grid = jnp.where(
                mask[..., None], biome_object_state, object_state_grid
            )

        # Place agent in the center of the world
        agent_pos = jnp.array([self.size[0] // 2, self.size[1] // 2])

        # Initialize spawn times to 0 (all objects spawn at time 0)
        object_spawn_time_grid = jnp.zeros((self.size[1], self.size[0]), dtype=int)

        # Initialize generation grid - all objects start in generation 0
        object_generation_grid = jnp.zeros((self.size[1], self.size[0]), dtype=int)

        # Initialize biome consumption tracking
        num_biomes = self.biome_object_frequencies.shape[0]
        biome_consumption_count = jnp.zeros(num_biomes, dtype=int)
        biome_total_objects = jnp.zeros(num_biomes, dtype=int)

        # Count objects in each biome
        for i in range(num_biomes):
            mask = self.biome_masks[i]
            # Count non-empty objects (object_id > 0)
            total = jnp.sum((object_grid > 0) & mask)
            biome_total_objects = biome_total_objects.at[i].set(total)

        biome_generation = jnp.zeros(num_biomes, dtype=int)

        state = EnvState(
            pos=agent_pos,
            object_grid=object_grid,
            biome_grid=biome_grid,
            time=0,
            digestion_buffer=jnp.zeros((self.max_reward_delay,)),
            object_spawn_time_grid=object_spawn_time_grid,
            object_color_grid=object_color_grid,
            object_state_grid=object_state_grid,
            biome_consumption_count=biome_consumption_count,
            biome_total_objects=biome_total_objects,
            biome_generation=biome_generation,
            object_generation_grid=object_generation_grid,
        )

        return self.get_obs(state, params), state

    def _spawn_biome_objects(
        self,
        biome_idx: int,
        key: jax.Array,
        deterministic: bool = False,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Spawn objects in a biome.

        Returns:
            object_grid: (H, W) array of object IDs
            color_grid: (H, W, 3) array of RGB colors
            state_grid: (H, W, num_state_params) array of object state parameters
        """
        biome_freqs = self.biome_object_frequencies[biome_idx]
        biome_mask = self.biome_masks_array[biome_idx]

        key, spawn_key, color_key, params_key = jax.random.split(key, 4)

        # Generate object IDs using deterministic or random spawn
        if deterministic:
            # Deterministic spawn: exact number of each object type
            # NOTE: Requires concrete biome_idx to compute size at trace time
            # Get static biome bounds
            biome_start = self.biome_starts[biome_idx]
            biome_stop = self.biome_stops[biome_idx]
            biome_height = biome_stop[1] - biome_start[1]
            biome_width = biome_stop[0] - biome_start[0]
            biome_size = int(self.biome_sizes[biome_idx])

            grid = jnp.linspace(0, 1, biome_size, endpoint=False)
            biome_objects_flat = len(biome_freqs) - jnp.searchsorted(
                jnp.cumsum(biome_freqs[::-1]), grid, side="right"
            )
            biome_objects_flat = jax.random.permutation(spawn_key, biome_objects_flat)

            # Reshape to match biome dimensions (use concrete dimensions)
            biome_objects = biome_objects_flat.reshape(biome_height, biome_width)

            # Place in full grid using slicing with static bounds
            object_grid = jnp.zeros((self.size[1], self.size[0]), dtype=int)
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
            )

        # Initialize color grid
        color_grid = jnp.full((self.size[1], self.size[0], 3), 255, dtype=jnp.uint8)

        # Sample ONE color per object type in this biome (not per instance)
        # This gives objects of the same type the same color within a biome generation
        # Skip index 0 (EMPTY object) - only sample colors for actual objects
        num_object_types = len(self.objects)
        num_actual_objects = num_object_types - 1  # Exclude EMPTY

        if num_actual_objects > 0:
            biome_object_colors = jax.random.randint(
                color_key,
                (num_actual_objects, 3),
                minval=0,
                maxval=256,
                dtype=jnp.uint8,
            )

            # Assign colors based on object type (starting from index 1)
            for obj_idx in range(1, num_object_types):
                obj_mask = (object_grid == obj_idx) & biome_mask
                obj_color = biome_object_colors[
                    obj_idx - 1
                ]  # Offset by 1 since we skip EMPTY
                color_grid = jnp.where(obj_mask[..., None], obj_color, color_grid)

        # Initialize parameters grid
        num_object_params = 2 + 2 * self.num_fourier_terms
        params_grid = jnp.zeros(
            (self.size[1], self.size[0], num_object_params), dtype=jnp.float32
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
            params_grid = jnp.where(obj_mask[..., None], obj_params, params_grid)

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
        num_object_params = 2 + 2 * self.num_fourier_terms
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
                "object_color_grid": spaces.Box(
                    0,
                    255,
                    (self.size[1], self.size[0], 3),
                    int,
                ),
                "object_generation_grid": spaces.Box(
                    0,
                    jnp.inf,
                    (self.size[1], self.size[0]),
                    int,
                ),
                "object_state_grid": spaces.Box(
                    -jnp.inf,
                    jnp.inf,
                    (self.size[1], self.size[0], num_object_params),
                    float,
                ),
                "biome_consumption_count": spaces.Box(
                    0,
                    jnp.inf,
                    (self.biome_object_frequencies.shape[0],),
                    int,
                ),
                "biome_total_objects": spaces.Box(
                    0,
                    jnp.inf,
                    (self.biome_object_frequencies.shape[0],),
                    int,
                ),
                "biome_generation": spaces.Box(
                    0,
                    jnp.inf,
                    (self.biome_object_frequencies.shape[0],),
                    int,
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

    def _get_aperture(self, object_grid: jax.Array, pos: jax.Array) -> jax.Array:
        """Extract the aperture view from the object grid."""
        y_coords, x_coords, y_coords_adj, x_coords_adj = (
            self._compute_aperture_coordinates(pos)
        )

        values = object_grid[y_coords_adj, x_coords_adj]

        if self.nowrap:
            # Mark out-of-bounds positions with padding
            y_out = (y_coords < 0) | (y_coords >= self.size[1])
            x_out = (x_coords < 0) | (x_coords >= self.size[0])
            out_of_bounds = y_out | x_out
            padding_index = self.object_ids[-1]
            aperture = jnp.where(out_of_bounds, padding_index, values)
        else:
            aperture = values

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
            colors = self.object_colors / 255.0
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
            colors = self.object_colors / 255.0
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
            # Use stateful object colors if dynamic_biomes is enabled, else use default colors
            if self.dynamic_biomes:
                # Use per-instance colors from state
                img = state.object_color_grid.copy()
            else:
                # Use default object colors
                img = jnp.zeros((self.size[1], self.size[0], 3))
                render_grid = jnp.maximum(0, state.object_grid)

                def update_image(i, img):
                    color = self.object_colors[i]
                    mask = render_grid == i
                    img = jnp.where(mask[..., None], color, img)
                    return img

                img = jax.lax.fori_loop(0, len(self.object_ids), update_image, img)

            # Tint the agent's aperture
            y_coords, x_coords, y_coords_adj, x_coords_adj = (
                self._compute_aperture_coordinates(state.pos)
            )

            alpha = 0.2
            agent_color = jnp.array(AGENT.color)

            if self.nowrap:
                # Create tint mask: any in-bounds original position maps to a cell makes it tinted
                tint_mask = jnp.zeros((self.size[1], self.size[0]), dtype=int)
                tint_mask = tint_mask.at[y_coords_adj, x_coords_adj].set(1)
                # Apply tint to masked positions
                original_colors = img
                tinted_colors = (1 - alpha) * original_colors + alpha * agent_color
                img = jnp.where(tint_mask[..., None], tinted_colors, img)
            else:
                original_colors = img[y_coords_adj, x_coords_adj]
                tinted_colors = (1 - alpha) * original_colors + alpha * agent_color
                img = img.at[y_coords_adj, x_coords_adj].set(tinted_colors)

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

            if self.dynamic_biomes:
                # Use per-instance colors from state - extract aperture view
                y_coords, x_coords, y_coords_adj, x_coords_adj = (
                    self._compute_aperture_coordinates(state.pos)
                )
                img = state.object_color_grid[y_coords_adj, x_coords_adj]

                if self.nowrap:
                    # For out-of-bounds, use padding object color
                    y_out = (y_coords < 0) | (y_coords >= self.size[1])
                    x_out = (x_coords < 0) | (x_coords >= self.size[0])
                    out_of_bounds = y_out | x_out
                    padding_color = jnp.array(self.objects[-1].color, dtype=jnp.float32)
                    img = jnp.where(out_of_bounds[..., None], padding_color, img)
            else:
                # Use default object colors
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
