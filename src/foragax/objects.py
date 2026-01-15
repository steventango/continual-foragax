import abc
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from foragax.weather import FILE_PATHS, get_temperature, load_data


class BaseForagaxObject:
    """Base class for objects in the Foragax environment."""

    def __init__(
        self,
        name: str = "empty",
        blocking: bool = False,
        collectable: bool = False,
        color: Tuple[int, int, int] = (0, 0, 0),
        random_respawn: bool = False,
        max_reward_delay: int = 0,
        expiry_time: Optional[int] = None,
    ):
        self.name = name
        self.blocking = blocking
        self.collectable = collectable
        self.color = color
        self.random_respawn = random_respawn
        self.max_reward_delay = max_reward_delay
        self.expiry_time = expiry_time

    def get_state(self, key: jax.Array) -> jax.Array:
        """Generate per-object reward parameters to store in the environment state.

        By default, objects don't use per-instance params. Override this method
        to provide per-instance parameters that will be stored in object_state_grid.

        Args:
            key: JAX random key for parameter generation

        Returns:
            Array of parameters (can be empty array for objects without params)
        """
        return jnp.array([], dtype=jnp.float32)

    @abc.abstractmethod
    def reward(
        self, clock: int, rng: jax.Array, params: Optional[jax.Array] = None
    ) -> float:
        """Reward function.

        Args:
            clock: Current time step
            rng: JAX random key
            params: Optional per-object parameters from object_state_grid
        """
        raise NotImplementedError

    @abc.abstractmethod
    def regen_delay(self, clock: int, rng: jax.Array) -> int:
        """Regeneration delay function."""
        raise NotImplementedError

    @abc.abstractmethod
    def reward_delay(self, clock: int, rng: jax.Array) -> int:
        """Reward delay function."""
        raise NotImplementedError

    @abc.abstractmethod
    def expiry_regen_delay(self, clock: int, rng: jax.Array) -> int:
        """Expiry regeneration delay function."""
        raise NotImplementedError


class DefaultForagaxObject(BaseForagaxObject):
    """Base class for default objects in the Foragax environment."""

    def __init__(
        self,
        name: str = "empty",
        reward: float = 0.0,
        blocking: bool = False,
        collectable: bool = False,
        regen_delay: Tuple[int, int] = (10, 100),
        color: Tuple[int, int, int] = (255, 255, 255),
        random_respawn: bool = False,
        reward_delay: int = 0,
        max_reward_delay: Optional[int] = None,
        expiry_time: Optional[int] = None,
        expiry_regen_delay: Tuple[int, int] = (10, 100),
    ):
        if max_reward_delay is None:
            max_reward_delay = reward_delay
        super().__init__(
            name,
            blocking,
            collectable,
            color,
            random_respawn,
            max_reward_delay,
            expiry_time,
        )
        self.reward_val = reward
        self.regen_delay_range = regen_delay
        self.reward_delay_val = reward_delay
        self.expiry_regen_delay_range = expiry_regen_delay

    def reward(
        self, clock: int, rng: jax.Array, params: Optional[jax.Array] = None
    ) -> float:
        """Default reward function."""
        return self.reward_val

    def regen_delay(self, clock: int, rng: jax.Array) -> int:
        """Default regeneration delay function."""
        min_delay, max_delay = self.regen_delay_range
        return jax.random.randint(rng, (), min_delay, max_delay)

    def reward_delay(self, clock: int, rng: jax.Array) -> int:
        """Default reward delay function."""
        return self.reward_delay_val

    def expiry_regen_delay(self, clock: int, rng: jax.Array) -> int:
        """Default expiry regeneration delay function."""
        min_delay, max_delay = self.expiry_regen_delay_range
        return jax.random.randint(rng, (), min_delay, max_delay)


class NormalRegenForagaxObject(DefaultForagaxObject):
    """Object with regeneration delay from a normal distribution."""

    def __init__(
        self,
        name: str = "empty",
        reward: float = 0.0,
        collectable: bool = False,
        mean_regen_delay: int = 10,
        std_regen_delay: int = 1,
        color: Tuple[int, int, int] = (0, 0, 0),
        random_respawn: bool = False,
        reward_delay: int = 0,
        max_reward_delay: Optional[int] = None,
        expiry_time: Optional[int] = None,
        mean_expiry_regen_delay: Optional[int] = None,
        std_expiry_regen_delay: Optional[int] = None,
    ):
        # If expiry regen delays not provided, use same as normal regen
        if mean_expiry_regen_delay is None:
            mean_expiry_regen_delay = mean_regen_delay
        if std_expiry_regen_delay is None:
            std_expiry_regen_delay = std_regen_delay

        super().__init__(
            name=name,
            reward=reward,
            collectable=collectable,
            regen_delay=(mean_regen_delay, mean_regen_delay),
            color=color,
            random_respawn=random_respawn,
            reward_delay=reward_delay,
            max_reward_delay=max_reward_delay,
            expiry_time=expiry_time,
            expiry_regen_delay=(mean_expiry_regen_delay, mean_expiry_regen_delay),
        )
        self.mean_regen_delay = mean_regen_delay
        self.std_regen_delay = std_regen_delay
        self.mean_expiry_regen_delay = mean_expiry_regen_delay
        self.std_expiry_regen_delay = std_expiry_regen_delay

    def regen_delay(self, clock: int, rng: jax.Array) -> int:
        """Regeneration delay from a normal distribution."""
        delay = self.mean_regen_delay + jax.random.normal(rng) * self.std_regen_delay
        return jnp.maximum(0, delay).astype(int)

    def expiry_regen_delay(self, clock: int, rng: jax.Array) -> int:
        """Expiry regeneration delay from a normal distribution."""
        delay = (
            self.mean_expiry_regen_delay
            + jax.random.normal(rng) * self.std_expiry_regen_delay
        )
        return jnp.maximum(0, delay).astype(int)


class WeatherObject(NormalRegenForagaxObject):
    """Object with reward based on temperature data."""

    def __init__(
        self,
        name: str,
        rewards: jnp.ndarray,
        repeat: int,
        multiplier: float = 1.0,
        mean_regen_delay: int = 10,
        std_regen_delay: int = 1,
        color: Tuple[int, int, int] = (0, 0, 0),
        random_respawn: bool = False,
        reward_delay: int = 0,
        max_reward_delay: Optional[int] = None,
        expiry_time: Optional[int] = None,
        mean_expiry_regen_delay: Optional[int] = None,
        std_expiry_regen_delay: Optional[int] = None,
    ):
        super().__init__(
            name=name,
            collectable=True,
            mean_regen_delay=mean_regen_delay,
            std_regen_delay=std_regen_delay,
            color=color,
            random_respawn=random_respawn,
            reward_delay=reward_delay,
            max_reward_delay=max_reward_delay,
            expiry_time=expiry_time,
            mean_expiry_regen_delay=mean_expiry_regen_delay,
            std_expiry_regen_delay=std_expiry_regen_delay,
        )
        self.rewards = rewards * multiplier
        self.repeat = repeat

    def reward(
        self, clock: int, rng: jax.Array, params: Optional[jax.Array] = None
    ) -> float:
        """Reward is based on temperature."""
        return get_temperature(self.rewards, clock, self.repeat)


class FourierObject(BaseForagaxObject):
    """Object with reward based on Fourier series with per-instance parameters.

    This object doesn't respawn on its own. Instead, objects are respawned
    biome-wide when consumption threshold is reached, with new random parameters.

    The reward function is a Fourier series with random period and harmonics,
    with coefficients scaled by 1/n and normalized to [-1, 1].
    """

    def __init__(
        self,
        name: str,
        num_fourier_terms: int = 10,
        base_magnitude: float = 1.0,
        color: Tuple[int, int, int] = (0, 0, 0),
        reward_delay: int = 0,
        max_reward_delay: Optional[int] = None,
        regen_delay: Optional[Tuple[int, int]] = None,
        reward_repeat: int = 1,
    ):
        if max_reward_delay is None:
            max_reward_delay = reward_delay
        super().__init__(
            name=name,
            blocking=False,
            collectable=True,
            color=color,
            random_respawn=True,
            max_reward_delay=max_reward_delay,
            expiry_time=None,
        )
        self.num_fourier_terms = num_fourier_terms
        self.base_magnitude = base_magnitude
        self.reward_delay_val = reward_delay
        self.regen_delay_range = regen_delay
        self.reward_repeat = reward_repeat

    def get_state(self, key: jax.Array) -> jax.Array:
        """Generate random Fourier series parameters.

        Returns array of shape (3 + 2*num_fourier_terms,) containing:
        [period, y_min, y_max, a1, b1, a2, b2, ...]
        """
        # Sample period uniformly from [10, 1000]
        key, period_key = jax.random.split(key)
        period = jax.random.randint(period_key, (), 10, 1001).astype(jnp.float32)

        # Generate coefficients with 1/n scaling
        key, a_key = jax.random.split(key)
        n_values = jnp.arange(1, self.num_fourier_terms + 1, dtype=jnp.float32)
        a_coeffs = jax.random.normal(a_key, (self.num_fourier_terms,)) / n_values

        key, b_key = jax.random.split(key)
        b_coeffs = jax.random.normal(b_key, (self.num_fourier_terms,)) / n_values

        # Compute min-max values for normalization
        num_samples = 1000
        t_samples = jnp.linspace(0, 2 * jnp.pi, num_samples)
        y_samples = jnp.zeros(num_samples)
        for n in range(1, self.num_fourier_terms + 1):
            y_samples += a_coeffs[n - 1] * jnp.cos(n * t_samples)
            y_samples += b_coeffs[n - 1] * jnp.sin(n * t_samples)

        # Store min and max for proper min-max normalization
        y_min = jnp.min(y_samples)
        y_max = jnp.max(y_samples)

        # Combine into parameter vector: [period, y_min, y_max, a1, b1, a2, b2, ...]
        ab_interleaved = jnp.empty(2 * self.num_fourier_terms, dtype=jnp.float32)
        ab_interleaved = ab_interleaved.at[::2].set(a_coeffs)
        ab_interleaved = ab_interleaved.at[1::2].set(b_coeffs)
        params_vec = jnp.concatenate(
            [jnp.array([period, y_min, y_max]), ab_interleaved]
        )

        return params_vec

    def reward(
        self, clock: int, rng: jax.Array, params: Optional[jax.Array] = None
    ) -> float:
        """Compute reward from Fourier series parameters.

        Args:
            clock: Current timestep
            rng: Random key (unused for Fourier objects)
            params: Array of shape (3 + 2*num_fourier_terms,) containing
                    [period, y_min, y_max, a1, b1, a2, b2, ...]

        Returns:
            Reward value computed from Fourier series, normalized to [-base_magnitude, base_magnitude]
        """
        if params is None or len(params) == 0:
            return 0.0

        # Extract period and min-max values
        period = params[0]
        y_min = params[1]
        y_max = params[2]

        # Normalize time to [0, 2π] using the object's period
        t = 2.0 * jnp.pi * ((clock // self.reward_repeat) % period) / period

        # Extract interleaved coefficients: [a1, b1, a2, b2, ...]
        ab_coeffs = params[3:]
        n_terms = len(ab_coeffs) // 2

        # Compute Fourier series: sum(a_n*cos(n*t) + b_n*sin(n*t))
        reward = 0.0
        for i in range(n_terms):
            freq = i + 1
            a_i = ab_coeffs[2 * i]  # a coefficient at index 2i
            b_i = ab_coeffs[2 * i + 1]  # b coefficient at index 2i+1
            reward += a_i * jnp.cos(freq * t) + b_i * jnp.sin(freq * t)

        # Apply min-max normalization to [-1, 1], then scale by base_magnitude
        # Formula: 2 * (x - min) / (max - min) - 1
        # If min == max (constant function), return 0
        range_val = jnp.maximum(y_max - y_min, 1e-8)  # Avoid division by zero
        # Check if this is a constant function (min == max)
        is_constant = jnp.abs(y_max - y_min) < 1e-8
        reward = jnp.where(
            is_constant,
            0.0,
            (2.0 * (reward - y_min) / range_val - 1.0) * self.base_magnitude,
        )

        return reward

    def reward_delay(self, clock: int, rng: jax.Array) -> int:
        """Reward delay function."""
        return self.reward_delay_val

    def regen_delay(self, clock: int, rng: jax.Array) -> int:
        """No individual regeneration - returns infinity."""
        if self.regen_delay_range is not None:
            min_delay, max_delay = self.regen_delay_range
            return jax.random.randint(rng, (), min_delay, max_delay)
        return jnp.iinfo(jnp.int32).max

    def expiry_regen_delay(self, clock: int, rng: jax.Array) -> int:
        """No expiry regeneration."""
        return jnp.iinfo(jnp.int32).max


EMPTY = DefaultForagaxObject()
WALL = DefaultForagaxObject(name="wall", blocking=True, color=(127, 127, 127))
FLOWER = DefaultForagaxObject(
    name="flower",
    reward=1.0,
    collectable=True,
    color=(0, 255, 0),
)

THORNS = DefaultForagaxObject(
    name="thorns",
    reward=-1.0,
    collectable=True,
    color=(255, 0, 0),
)

MOREL = DefaultForagaxObject(
    name="morel",
    reward=10.0,
    collectable=True,
    regen_delay=(100, 100),
    color=(63, 30, 25),
)
OYSTER = DefaultForagaxObject(
    name="oyster",
    reward=1.0,
    collectable=True,
    regen_delay=(10, 10),
    color=(124, 61, 81),
)
LARGE_MOREL = NormalRegenForagaxObject(
    name="large_morel",
    reward=30.0,
    collectable=True,
    mean_regen_delay=300,
    std_regen_delay=30,
    color=(63, 30, 25),
)
MEDIUM_MOREL = NormalRegenForagaxObject(
    name="medium_morel",
    reward=10.0,
    collectable=True,
    mean_regen_delay=100,
    std_regen_delay=10,
    color=(63, 30, 25),
)
LARGE_OYSTER = NormalRegenForagaxObject(
    name="large_oyster",
    reward=1.0,
    collectable=True,
    mean_regen_delay=10,
    std_regen_delay=1,
    color=(124, 61, 81),
)
DEATHCAP = DefaultForagaxObject(
    name="deathcap",
    reward=-1.0,
    collectable=True,
    regen_delay=(10, 10),
    color=(193, 178, 30),
)
AGENT = DefaultForagaxObject(name="agent", blocking=True, color=(0, 0, 255))

PADDING = DefaultForagaxObject(name="padding", blocking=True, color=(0, 0, 0))

BROWN_MOREL = NormalRegenForagaxObject(
    name="brown_morel",
    reward=30.0,
    collectable=True,
    color=(63, 30, 25),
    mean_regen_delay=300,
    std_regen_delay=30,
)
BROWN_MOREL_2 = NormalRegenForagaxObject(
    name="brown_morel",
    reward=10.0,
    collectable=True,
    color=(63, 30, 25),
    mean_regen_delay=100,
    std_regen_delay=10,
)
BROWN_OYSTER = NormalRegenForagaxObject(
    name="brown_oyster",
    reward=1.0,
    collectable=True,
    color=(63, 30, 25),
    mean_regen_delay=10,
    std_regen_delay=1,
)
GREEN_DEATHCAP = DefaultForagaxObject(
    name="green_deathcap",
    reward=-1.0,
    collectable=True,
    color=(0, 255, 0),
    regen_delay=(10, 10),
)
GREEN_DEATHCAP_2 = DefaultForagaxObject(
    name="green_deathcap",
    reward=-5.0,
    collectable=True,
    color=(0, 255, 0),
    regen_delay=(10, 10),
)
GREEN_DEATHCAP_3 = NormalRegenForagaxObject(
    name="green_deathcap",
    reward=-5.0,
    collectable=True,
    color=(0, 255, 0),
    mean_regen_delay=10,
    std_regen_delay=1,
)
GREEN_FAKE = DefaultForagaxObject(
    name="green_fake",
    reward=0.0,
    collectable=True,
    color=(0, 255, 0),
    regen_delay=(10, 10),
)
GREEN_FAKE_2 = NormalRegenForagaxObject(
    name="green_fake",
    reward=0.0,
    collectable=True,
    color=(0, 255, 0),
    mean_regen_delay=10,
    std_regen_delay=1,
)
BROWN_MOREL_UNIFORM = DefaultForagaxObject(
    name="brown_morel",
    reward=10.0,
    collectable=True,
    color=(63, 30, 25),
    regen_delay=(90, 110),
)
BROWN_OYSTER_UNIFORM = DefaultForagaxObject(
    name="brown_oyster",
    reward=1.0,
    collectable=True,
    color=(63, 30, 25),
    regen_delay=(9, 11),
)
GREEN_DEATHCAP_UNIFORM = DefaultForagaxObject(
    name="green_deathcap",
    reward=-5.0,
    collectable=True,
    color=(0, 255, 0),
    regen_delay=(9, 11),
)
GREEN_FAKE_UNIFORM = DefaultForagaxObject(
    name="green_fake",
    reward=0.0,
    collectable=True,
    color=(0, 255, 0),
    regen_delay=(9, 11),
)

# Random respawn variants
BROWN_MOREL_UNIFORM_RANDOM = DefaultForagaxObject(
    name="brown_morel",
    reward=10.0,
    collectable=True,
    color=(63, 30, 25),
    regen_delay=(90, 110),
    random_respawn=True,
)
BROWN_OYSTER_UNIFORM_RANDOM = DefaultForagaxObject(
    name="brown_oyster",
    reward=1.0,
    collectable=True,
    color=(63, 30, 25),
    regen_delay=(9, 11),
    random_respawn=True,
)
GREEN_DEATHCAP_UNIFORM_RANDOM = DefaultForagaxObject(
    name="green_deathcap",
    reward=-5.0,
    collectable=True,
    color=(0, 255, 0),
    regen_delay=(9, 11),
    random_respawn=True,
)
GREEN_FAKE_UNIFORM_RANDOM = DefaultForagaxObject(
    name="green_fake",
    reward=0.0,
    collectable=True,
    color=(0, 255, 0),
    regen_delay=(9, 11),
    random_respawn=True,
)

# Random respawn variants with expiry
BROWN_MOREL_UNIFORM_RANDOM_EXPIRY = DefaultForagaxObject(
    name="brown_morel",
    reward=10.0,
    collectable=True,
    color=(63, 30, 25),
    regen_delay=(90, 110),
    random_respawn=True,
    expiry_time=500,
)
BROWN_OYSTER_UNIFORM_RANDOM_EXPIRY = DefaultForagaxObject(
    name="brown_oyster",
    reward=1.0,
    collectable=True,
    color=(63, 30, 25),
    regen_delay=(9, 11),
    random_respawn=True,
    expiry_time=500,
)
GREEN_DEATHCAP_UNIFORM_RANDOM_EXPIRY = DefaultForagaxObject(
    name="green_deathcap",
    reward=-5.0,
    collectable=True,
    color=(0, 255, 0),
    regen_delay=(9, 11),
    random_respawn=True,
    expiry_time=500,
)
GREEN_FAKE_UNIFORM_RANDOM_EXPIRY = DefaultForagaxObject(
    name="green_fake",
    reward=0.0,
    collectable=True,
    color=(0, 255, 0),
    regen_delay=(9, 11),
    random_respawn=True,
    expiry_time=500,
)


def create_weather_objects(
    file_index: int = 0,
    repeat: int = 500,
    multiplier: float = 1.0,
    same_color: bool = False,
    random_respawn: bool = False,
    reward_delay: int = 0,
    expiry_time: Optional[int] = None,
    mean_expiry_regen_delay: Optional[int] = None,
    std_expiry_regen_delay: Optional[int] = None,
):
    """Create HOT and COLD WeatherObject instances using the specified file.

    Args:
        file_index: Index into `FILE_PATHS` to select the temperature file.
        repeat: How many steps each temperature value repeats for.
        multiplier: Base multiplier applied to HOT; COLD will use -multiplier.
        same_color: If True, both HOT and COLD use the same color.
        random_respawn: If True, objects respawn at random locations.
        reward_delay: Number of steps before reward is delivered.
        expiry_time: Time steps before object expires (None = no expiry).
        mean_expiry_regen_delay: Mean delay for expiry respawn.
        std_expiry_regen_delay: Standard deviation for expiry respawn delay.

    Returns:
        A tuple (HOT, COLD) of WeatherObject instances.
    """
    # Clamp file_index
    if file_index < 0 or file_index >= len(FILE_PATHS):
        raise IndexError(
            f"file_index {file_index} out of range (0..{len(FILE_PATHS) - 1})"
        )

    rewards = load_data(FILE_PATHS[file_index])

    hot_color = (63, 30, 25) if same_color else (255, 0, 255)

    hot = WeatherObject(
        name="hot",
        rewards=rewards,
        repeat=repeat,
        multiplier=multiplier,
        color=hot_color,
        random_respawn=random_respawn,
        reward_delay=reward_delay,
        expiry_time=expiry_time,
        mean_expiry_regen_delay=mean_expiry_regen_delay,
        std_expiry_regen_delay=std_expiry_regen_delay,
    )

    cold_color = hot_color if same_color else (0, 255, 255)
    cold = WeatherObject(
        name="cold",
        rewards=rewards,
        repeat=repeat,
        multiplier=-multiplier,
        color=cold_color,
        random_respawn=random_respawn,
        reward_delay=reward_delay,
        expiry_time=expiry_time,
        mean_expiry_regen_delay=mean_expiry_regen_delay,
        std_expiry_regen_delay=std_expiry_regen_delay,
    )

    return hot, cold


class SineObject(DefaultForagaxObject):
    """Object with reward based on sine wave with a base reward offset.

    The total reward is: base_reward + amplitude * sin(2*pi * clock / period)
    This allows for objects that have different base behaviors (positive/negative)
    with an underlying sine wave that drives continual learning.

    Uses uniform distribution for regeneration delays by default (from DefaultForagaxObject).
    """

    def __init__(
        self,
        name: str,
        base_reward: float = 0.0,
        amplitude: float = 1.0,
        period: int = 1000,
        phase: float = 0.0,
        regen_delay: Tuple[int, int] = (9, 11),
        color: Tuple[int, int, int] = (0, 0, 0),
        random_respawn: bool = False,
        reward_delay: int = 0,
        max_reward_delay: Optional[int] = None,
        expiry_time: Optional[int] = None,
        expiry_regen_delay: Tuple[int, int] = (9, 11),
    ):
        super().__init__(
            name=name,
            reward=base_reward,
            collectable=True,
            regen_delay=regen_delay,
            color=color,
            random_respawn=random_respawn,
            reward_delay=reward_delay,
            max_reward_delay=max_reward_delay,
            expiry_time=expiry_time,
            expiry_regen_delay=expiry_regen_delay,
        )
        self.base_reward = base_reward
        self.amplitude = amplitude
        self.period = period
        self.phase = phase

    def reward(
        self, clock: int, rng: jax.Array, params: Optional[jax.Array] = None
    ) -> float:
        """Reward is base_reward + amplitude * sin(2*pi * clock / period + phase)."""
        sine_value = jnp.sin(2.0 * jnp.pi * clock / self.period + self.phase)
        return self.base_reward + self.amplitude * sine_value


def create_fourier_objects(
    num_fourier_terms: int = 10,
    base_magnitude: float = 1.0,
    reward_delay: int = 0,
    regen_delay: Optional[Tuple[int, int]] = None,
    reward_repeat: int = 1,
):
    """Create HOT and COLD FourierObject instances.

    Args:
        num_fourier_terms: Number of Fourier terms in the reward function (default: 10).
        base_magnitude: Base magnitude for Fourier coefficients.
        reward_delay: Number of steps before reward is delivered.

    Returns:
        A tuple (HOT, COLD) of FourierObject instances.
    """
    hot = FourierObject(
        name="hot_fourier",
        num_fourier_terms=num_fourier_terms,
        base_magnitude=base_magnitude,
        color=(0, 0, 0),
        reward_delay=reward_delay,
        regen_delay=regen_delay,
        reward_repeat=reward_repeat,
    )

    cold = FourierObject(
        name="cold_fourier",
        num_fourier_terms=num_fourier_terms,
        base_magnitude=base_magnitude,
        color=(0, 0, 0),
        reward_delay=reward_delay,
        regen_delay=regen_delay,
        reward_repeat=reward_repeat,
    )

    return hot, cold


def create_sine_biome_objects(
    period: int = 1000,
    amplitude: float = 20.0,
    base_oyster_reward: float = 10.0,
    base_deathcap_reward: float = -10.0,
    regen_delay: Tuple[int, int] = (9, 11),
    reward_delay: int = 0,
    expiry_time: int = 500,
    expiry_regen_delay: Tuple[int, int] = (9, 11),
):
    """Create objects for the sine-based two-biome environment.

    Biome 1 (Left): Oyster (+base_reward), Death Cap (-base_reward)
    Biome 2 (Right): Oyster (-base_reward), Death Cap (+base_reward)

    Both biomes have an underlying sine curve with the specified amplitude.
    The sine curve of biome 2 is the negative of biome 1 (180 degree phase shift).

    Objects use uniform respawn and random expiry by default.

    Args:
        period: Period of the sine wave in timesteps
        amplitude: Amplitude of the sine wave
        base_oyster_reward: Base reward for oyster in biome 1 (will be negated in biome 2)
        base_deathcap_reward: Base reward for death cap in biome 1 (will be negated in biome 2)
        regen_delay: Tuple of (min, max) for uniform regeneration delay
        reward_delay: Number of steps before reward is delivered
        expiry_time: Time steps before object expires (None = no expiry)
        expiry_regen_delay: Tuple of (min, max) for uniform expiry regeneration delay

    Returns:
        A tuple of (biome1_oyster, biome1_deathcap, biome2_oyster, biome2_deathcap)
    """
    # Biome 1 objects (phase = 0)
    biome1_oyster = SineObject(
        name="oyster_sine_1",
        base_reward=base_oyster_reward,
        amplitude=amplitude,
        period=period,
        phase=0.0,
        regen_delay=regen_delay,
        color=(124, 61, 81),  # Oyster color
        random_respawn=True,
        reward_delay=reward_delay,
        expiry_time=expiry_time,
        expiry_regen_delay=expiry_regen_delay,
    )

    biome1_deathcap = SineObject(
        name="deathcap_sine_1",
        base_reward=base_deathcap_reward,
        amplitude=amplitude,
        period=period,
        phase=0.0,
        regen_delay=regen_delay,
        color=(0, 255, 0),  # Green color
        random_respawn=True,
        reward_delay=reward_delay,
        expiry_time=expiry_time,
        expiry_regen_delay=expiry_regen_delay,
    )

    # Biome 2 objects (phase = pi for 180 degree shift)
    biome2_oyster = SineObject(
        name="oyster_sine_2",
        base_reward=-base_oyster_reward,  # Negated
        amplitude=amplitude,
        period=period,
        phase=jnp.pi,  # 180 degree phase shift (negative of biome 1)
        regen_delay=regen_delay,
        color=(124, 61, 81),  # Same oyster color
        random_respawn=True,
        reward_delay=reward_delay,
        expiry_time=expiry_time,
        expiry_regen_delay=expiry_regen_delay,
    )

    biome2_deathcap = SineObject(
        name="deathcap_sine_2",
        base_reward=-base_deathcap_reward,  # Negated
        amplitude=amplitude,
        period=period,
        phase=jnp.pi,  # 180 degree phase shift (negative of biome 1)
        regen_delay=regen_delay,
        color=(0, 255, 0),  # Same green color
        random_respawn=True,
        reward_delay=reward_delay,
        expiry_time=expiry_time,
        expiry_regen_delay=expiry_regen_delay,
    )

    return biome1_oyster, biome1_deathcap, biome2_oyster, biome2_deathcap
