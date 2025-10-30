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
            object_id=jnp.zeros((h, w), dtype=int),
            respawn_timer=jnp.zeros((h, w), dtype=int),
            respawn_object_id=jnp.zeros((h, w), dtype=int),
            spawn_time=jnp.zeros((h, w), dtype=int),
            color=jnp.full((h, w, 3), 255, dtype=jnp.uint8),
            generation=jnp.zeros((h, w), dtype=int),
            state_params=jnp.zeros((h, w, num_params), dtype=jnp.float32),
            biome_id=jnp.full((h, w), -1, dtype=int),
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

        # Handle permanent removal (timer_val == 0)
        def place_empty():
            return object_state.replace(
                object_id=object_state.object_id.at[y, x].set(0),
                respawn_timer=object_state.respawn_timer.at[y, x].set(0),
                respawn_object_id=object_state.respawn_object_id.at[y, x].set(0),
            )

        # Handle timer placement
        def place_timer():
            # Non-random: place at original position
            def place_at_position():
                return object_state.replace(
                    object_id=object_state.object_id.at[y, x].set(0),
                    respawn_timer=object_state.respawn_timer.at[y, x].set(timer_val),
                    respawn_object_id=object_state.respawn_object_id.at[y, x].set(
                        object_type
                    ),
                )

            # Random: place at random location in same biome
            def place_randomly():
                # Clear the collected object's position
                new_object_id = object_state.object_id.at[y, x].set(0)
                new_respawn_timer = object_state.respawn_timer.at[y, x].set(0)
                new_respawn_object_id = object_state.respawn_object_id.at[y, x].set(0)

                # Find valid spawn locations in the same biome
                biome_id = object_state.biome_id[y, x]
                biome_mask = object_state.biome_id == biome_id
                empty_mask = new_object_id == 0
                no_timer_mask = new_respawn_timer == 0
                valid_spawn_mask = biome_mask & empty_mask & no_timer_mask
                num_valid_spawns = jnp.sum(valid_spawn_mask)

                y_indices, x_indices = jnp.nonzero(
                    valid_spawn_mask, size=self.size[0] * self.size[1], fill_value=-1
                )
                valid_spawn_indices = jnp.stack([y_indices, x_indices], axis=1)
                random_idx = jax.random.randint(rand_key, (), 0, num_valid_spawns)
                new_spawn_pos = valid_spawn_indices[random_idx]

                # Place timer at the new random position
                new_respawn_timer = new_respawn_timer.at[
                    new_spawn_pos[0], new_spawn_pos[1]
                ].set(timer_val)
                new_respawn_object_id = new_respawn_object_id.at[
                    new_spawn_pos[0], new_spawn_pos[1]
                ].set(object_type)

                return object_state.replace(
                    object_id=new_object_id,
                    respawn_timer=new_respawn_timer,
                    respawn_object_id=new_respawn_object_id,
                )

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
        current_objects = state.object_state.object_id

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

        object_params = state.object_state.state_params[pos[1], pos[0]]
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

        # Decrement respawn timers
        has_timer = state.object_state.respawn_timer > 0
        new_respawn_timer = jnp.where(
            has_timer,
            state.object_state.respawn_timer - 1,
            state.object_state.respawn_timer,
        )

        # Track which cells have timers that just reached 0
        just_respawned = has_timer & (new_respawn_timer == 0)

        # Respawn objects where timer reached 0
        new_object_id = jnp.where(
            just_respawned,
            state.object_state.respawn_object_id,
            state.object_state.object_id,
        )

        # Clear respawn_object_id for cells that just respawned
        new_respawn_object_id = jnp.where(
            just_respawned,
            0,
            state.object_state.respawn_object_id,
        )

        # Update spawn times for objects that just respawned
        spawn_time = jnp.where(
            just_respawned, state.time, state.object_state.spawn_time
        )

        # Collect object: set a timer
        regen_delay = jax.lax.switch(
            obj_at_pos, self.regen_delay_fns, state.time, regen_subkey
        )
        timer_countdown = jax.lax.cond(
            regen_delay == jnp.iinfo(jnp.int32).max,
            lambda: 0,  # No timer (permanent removal)
            lambda: regen_delay + 1,  # Timer countdown
        )

        # If collected, replace object with timer; otherwise, keep it
        val_at_pos = current_objects[pos[1], pos[0]]
        # Use original should_collect for consumption tracking
        should_collect_now = is_collectable & (val_at_pos > 0)

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

        # 3.5. HANDLE OBJECT EXPIRY
        # Only process expiry if there are objects that can expire
        key, object_state = self.expire_objects(key, state, object_state)

        # 3.6. HANDLE DYNAMIC BIOME CONSUMPTION AND RESPAWNING
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
                lambda: biome_consumption_count.at[collected_biome_id].add(1),
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
        else:
            biome_state = state.biome_state

        info = {"discount": self.discount(state, params)}
        temperatures = jnp.zeros(len(self.objects))
        for obj_index, obj in enumerate(self.objects):
            if isinstance(obj, WeatherObject):
                temperatures = temperatures.at[obj_index].set(
                    get_temperature(obj.rewards, state.time, obj.repeat)
                )
        info["temperatures"] = temperatures
        info["biome_id"] = object_state.biome_id[pos[1], pos[0]]
        info["object_collected_id"] = jax.lax.select(should_collect, obj_at_pos, -1)

        # Compute reward at each grid position
        fixed_key = jax.random.key(0)  # Fixed key for deterministic reward computation

        def compute_reward(obj_id, params):
            return jax.lax.cond(
                obj_id > 0,
                lambda: jax.lax.switch(
                    obj_id, self.reward_fns, state.time, fixed_key, params
                ),
                lambda: 0.0,
            )

        reward_grid = jax.vmap(jax.vmap(compute_reward))(
            object_state.object_id, object_state.state_params
        )
        info["rewards"] = reward_grid

        # 4. UPDATE STATE
        state = EnvState(
            pos=pos,
            time=state.time + 1,
            digestion_buffer=digestion_buffer,
            object_state=object_state,
            biome_state=biome_state,
        )

        done = self.is_terminal(state, params)
        return (
            jax.lax.stop_gradient(self.get_obs(state, params)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            info,
        )

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
                & (expiry_times >= 0)
                & (current_objects_for_expiry > 0)
            )

            # Count how many objects actually need to expire
            num_expiring = jnp.sum(should_expire)

            # Only process expiry if there are actually objects to expire
            def process_expiries():
                # Get positions of objects that should expire
                # Use nonzero with fixed size to maintain JIT compatibility
                max_objects = self.size[0] * self.size[1]
                y_indices, x_indices = jnp.nonzero(
                    should_expire, size=max_objects, fill_value=-1
                )

                key_local, expiry_key = jax.random.split(key)

                def process_one_expiry(carry, i):
                    obj_state = carry
                    y = y_indices[i]
                    x = x_indices[i]

                    # Skip if this is a padding index (from fill_value)
                    is_valid = (y >= 0) & (x >= 0)

                    def expire_one():
                        obj_id = current_objects_for_expiry[y, x]
                        exp_key = jax.random.fold_in(expiry_key, y * self.size[0] + x)
                        exp_delay = jax.lax.switch(
                            obj_id, self.expiry_regen_delay_fns, state.time, exp_key
                        )
                        timer_countdown = jax.lax.cond(
                            exp_delay == jnp.iinfo(jnp.int32).max,
                            lambda: 0,  # No timer (permanent removal)
                            lambda: exp_delay + 1,  # Timer countdown
                        )

                        # Use unified timer placement method
                        rand_key = jax.random.split(exp_key)[1]
                        new_obj_state = self._place_timer(
                            obj_state,
                            y,
                            x,
                            obj_id,
                            timer_countdown,
                            self.object_random_respawn[obj_id],
                            rand_key,
                        )

                        return new_obj_state

                    def no_op():
                        return obj_state

                    return jax.lax.cond(is_valid, expire_one, no_op), None

                new_object_state, _ = jax.lax.scan(
                    process_one_expiry,
                    object_state,
                    jnp.arange(max_objects),
                )
                return key_local, new_object_state

            def no_expiries():
                return key, object_state

            key, object_state = jax.lax.cond(
                num_expiring > 0,
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

        # Compute consumption rates for all biomes
        consumption_rates = biome_state.consumption_count / jnp.maximum(
            1.0, biome_state.total_objects.astype(float)
        )
        should_respawn = consumption_rates >= self.biome_consumption_threshold

        # Split key for all biomes in parallel
        key, subkey = jax.random.split(key)
        biome_keys = jax.random.split(subkey, num_biomes)

        # Compute all new spawns in parallel using vmap for random, switch for deterministic
        if self.deterministic_spawn:
            # Use switch to dispatch to concrete biome spawns for deterministic
            def make_spawn_fn(biome_idx):
                def spawn_fn(key):
                    return self._spawn_biome_objects(biome_idx, key, deterministic=True)

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
                lambda i, k: self._spawn_biome_objects(i, k, deterministic=False)
            )(jnp.arange(num_biomes), biome_keys)

        # Initialize updated grids
        new_obj_id = object_state.object_id
        new_color = object_state.color
        new_params = object_state.state_params
        new_spawn = object_state.spawn_time
        new_gen = object_state.generation

        # Update biome state
        new_consumption_count = jnp.where(
            should_respawn, 0, biome_state.consumption_count
        )
        new_generation = biome_state.generation + should_respawn.astype(int)

        # Compute new total objects for respawning biomes
        def count_objects(i):
            return jnp.sum((all_new_objects[i] > 0) & self.biome_masks_array[i])

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

            # Only update where new spawn has objects and biome should respawn
            is_new_object = (
                (all_new_objects[i] > 0) & biome_mask & should_respawn[i][..., None]
            )

            new_obj_id = jnp.where(is_new_object, all_new_objects[i], new_obj_id)
            new_color = jnp.where(
                is_new_object[..., None], all_new_colors[i], new_color
            )
            new_params = jnp.where(
                is_new_object[..., None], all_new_params[i], new_params
            )
            new_gen = jnp.where(is_new_object, new_gen_value, new_gen)
            new_spawn = jnp.where(is_new_object, current_time, new_spawn)

        # Clear timers in respawning biomes
        new_respawn_timer = object_state.respawn_timer
        new_respawn_object_id = object_state.respawn_object_id
        for i in range(num_biomes):
            biome_mask = self.biome_masks_array[i]
            should_clear = biome_mask & should_respawn[i][..., None]
            new_respawn_timer = jnp.where(should_clear, 0, new_respawn_timer)
            new_respawn_object_id = jnp.where(should_clear, 0, new_respawn_object_id)

        object_state = object_state.replace(
            object_id=new_obj_id,
            respawn_timer=new_respawn_timer,
            respawn_object_id=new_respawn_object_id,
            color=new_color,
            state_params=new_params,
            generation=new_gen,
            spawn_time=new_spawn,
        )

        return object_state, new_biome_state, key

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        """Reset environment state."""
        num_object_params = 2 + 2 * self.num_fourier_terms
        object_state = ObjectState.create_empty(self.size, num_object_params)

        key, iter_key = jax.random.split(key)

        # Spawn objects in each biome using unified method
        for i in range(self.biome_object_frequencies.shape[0]):
            iter_key, biome_key = jax.random.split(iter_key)
            mask = self.biome_masks[i]

            # Set biome_id
            object_state = object_state.replace(
                biome_id=jnp.where(mask, i, object_state.biome_id)
            )

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

        # Place agent in the center of the world
        agent_pos = jnp.array([self.size[0] // 2, self.size[1] // 2])

        # Initialize biome consumption tracking
        num_biomes = self.biome_object_frequencies.shape[0]
        biome_consumption_count = jnp.zeros(num_biomes, dtype=int)
        biome_total_objects = jnp.zeros(num_biomes, dtype=int)

        # Count objects in each biome
        for i in range(num_biomes):
            mask = self.biome_masks[i]
            # Count non-empty objects (object_id > 0)
            total = jnp.sum((object_state.object_id > 0) & mask)
            biome_total_objects = biome_total_objects.at[i].set(total)

        biome_generation = jnp.zeros(num_biomes, dtype=int)

        state = EnvState(
            pos=agent_pos,
            time=0,
            digestion_buffer=jnp.zeros((self.max_reward_delay,)),
            object_state=object_state,
            biome_state=BiomeState(
                consumption_count=biome_consumption_count,
                total_objects=biome_total_objects,
                generation=biome_generation,
            ),
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
                "time": spaces.Discrete(params.max_steps_in_episode),
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

    def _get_aperture(self, object_id_grid: jax.Array, pos: jax.Array) -> jax.Array:
        """Extract the aperture view from the object id grid."""
        y_coords, x_coords, y_coords_adj, x_coords_adj = (
            self._compute_aperture_coordinates(pos)
        )

        values = object_id_grid[y_coords_adj, x_coords_adj]

        if self.nowrap:
            # Mark out-of-bounds positions with padding
            y_out = (y_coords < 0) | (y_coords >= self.size[1])
            x_out = (x_coords < 0) | (x_coords >= self.size[0])
            out_of_bounds = y_out | x_out

            # Handle both object_id grids (2D) and color grids (3D)
            if len(values.shape) == 3:
                # Color grid: use PADDING color (0, 0, 0)
                padding_value = jnp.array([0, 0, 0], dtype=values.dtype)
                aperture = jnp.where(out_of_bounds[..., None], padding_value, values)
            else:
                # Object ID grid: use PADDING index
                padding_index = self.object_ids[-1]
                aperture = jnp.where(out_of_bounds, padding_index, values)
        else:
            aperture = values

        return aperture

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        """Get observation based on observation_type and full_world."""
        obs_grid = state.object_state.object_id
        color_grid = state.object_state.color

        if self.full_world:
            return self._get_world_obs(obs_grid, state)
        else:
            grid = self._get_aperture(obs_grid, state.pos)
            color_grid = self._get_aperture(color_grid, state.pos)
            return self._get_aperture_obs(grid, color_grid, state)

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
                obs_grid == 0,
                -1,
                jnp.take(self.object_to_color_map, obs_grid - 1, axis=0),
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
            empty_mask = aperture == 0
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
                img = state.object_state.color.copy()
                # Mask empty cells (object_id == 0) to white
                empty_mask = state.object_state.object_id == 0
                white_color = jnp.array([255, 255, 255], dtype=jnp.uint8)
                img = jnp.where(empty_mask[..., None], white_color, img)
            else:
                # Use default object colors
                img = jnp.zeros((self.size[1], self.size[0], 3))
                render_grid = state.object_state.object_id

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
                render_grid = state.object_state.object_id
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
            obs_grid = state.object_state.object_id
            aperture = self._get_aperture(obs_grid, state.pos)

            if self.dynamic_biomes:
                # Use per-instance colors from state - extract aperture view
                y_coords, x_coords, y_coords_adj, x_coords_adj = (
                    self._compute_aperture_coordinates(state.pos)
                )
                img = state.object_state.color[y_coords_adj, x_coords_adj]

                # Mask empty cells (object_id == 0) to white
                aperture_object_ids = state.object_state.object_id[
                    y_coords_adj, x_coords_adj
                ]
                empty_mask = aperture_object_ids == 0
                white_color = jnp.array([255, 255, 255], dtype=jnp.uint8)
                img = jnp.where(empty_mask[..., None], white_color, img)

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
