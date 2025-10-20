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

    @abc.abstractmethod
    def reward(self, clock: int, rng: jax.Array) -> float:
        """Reward function."""
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

    def reward(self, clock: int, rng: jax.Array) -> float:
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

    def reward(self, clock: int, rng: jax.Array) -> float:
        """Reward is based on temperature."""
        return get_temperature(self.rewards, clock, self.repeat)


class FourierRewardObject(NormalRegenForagaxObject):
    """Object with reward based on a randomly sampled Fourier series."""

    def __init__(
        self,
        name: str,
        fourier_coeffs: jnp.ndarray,
        period: int = 1000,
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
        """Initialize FourierRewardObject.

        Args:
            name: Object name
            fourier_coeffs: Array of shape (num_terms, 2) where each row contains [amplitude, phase]
                The reward is computed as: sum(amplitude * sin(2*pi*k*t/period + phase)) for k=1..num_terms
            period: Period of the Fourier series
            mean_regen_delay: Mean regeneration delay
            std_regen_delay: Standard deviation of regeneration delay
            color: RGB color tuple
            random_respawn: Whether to respawn at random locations
            reward_delay: Delay before reward is delivered
            max_reward_delay: Maximum reward delay
            expiry_time: Time before object expires
            mean_expiry_regen_delay: Mean expiry regeneration delay
            std_expiry_regen_delay: Standard deviation of expiry regen delay
        """
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
        self.fourier_coeffs = fourier_coeffs
        self.period = period

    def reward(self, clock: int, rng: jax.Array) -> float:
        """Reward based on Fourier series."""
        t = clock / self.period
        # Compute sum of amplitude * sin(2*pi*k*t + phase) for each term
        k = jnp.arange(1, len(self.fourier_coeffs) + 1)
        amplitudes = self.fourier_coeffs[:, 0]
        phases = self.fourier_coeffs[:, 1]
        terms = amplitudes * jnp.sin(2 * jnp.pi * k * t + phases)
        return jnp.sum(terms)


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


def create_fourier_objects(
    rng_key: jax.Array,
    num_terms: int = 5,
    period: int = 1000,
    amplitude_scale: float = 1.0,
    same_color: bool = True,
    random_respawn: bool = False,
    reward_delay: int = 0,
    expiry_time: Optional[int] = None,
    mean_expiry_regen_delay: Optional[int] = None,
    std_expiry_regen_delay: Optional[int] = None,
):
    """Create HOT and COLD FourierRewardObject instances with random coefficients.

    Args:
        rng_key: JAX random key for generating Fourier coefficients.
        num_terms: Number of Fourier terms (harmonics).
        period: Period of the Fourier series.
        amplitude_scale: Scale factor for amplitudes.
        same_color: If True, both HOT and COLD use the same color.
        random_respawn: If True, objects respawn at random locations.
        reward_delay: Number of steps before reward is delivered.
        expiry_time: Time steps before object expires (None = no expiry).
        mean_expiry_regen_delay: Mean delay for expiry respawn.
        std_expiry_regen_delay: Standard deviation for expiry respawn delay.

    Returns:
        A tuple (HOT, COLD) of FourierRewardObject instances.
    """
    # Generate random Fourier coefficients for hot object
    key_hot, key_cold = jax.random.split(rng_key)

    # Amplitudes: random from normal distribution scaled by amplitude_scale
    # Phases: random from uniform [0, 2π]
    key_amp_hot, key_phase_hot = jax.random.split(key_hot)
    hot_amplitudes = jax.random.normal(key_amp_hot, (num_terms,)) * amplitude_scale
    hot_phases = jax.random.uniform(key_phase_hot, (num_terms,)) * 2 * jnp.pi
    hot_coeffs = jnp.stack([hot_amplitudes, hot_phases], axis=1)

    # Cold object uses negated amplitudes (opposite rewards)
    key_amp_cold, key_phase_cold = jax.random.split(key_cold)
    cold_amplitudes = -hot_amplitudes  # Opposite sign
    cold_phases = jax.random.uniform(key_phase_cold, (num_terms,)) * 2 * jnp.pi
    cold_coeffs = jnp.stack([cold_amplitudes, cold_phases], axis=1)

    hot_color = (63, 30, 25) if same_color else (255, 128, 0)
    cold_color = hot_color if same_color else (0, 128, 255)

    hot = FourierRewardObject(
        name="hot_fourier",
        fourier_coeffs=hot_coeffs,
        period=period,
        color=hot_color,
        random_respawn=random_respawn,
        reward_delay=reward_delay,
        expiry_time=expiry_time,
        mean_expiry_regen_delay=mean_expiry_regen_delay,
        std_expiry_regen_delay=std_expiry_regen_delay,
    )

    cold = FourierRewardObject(
        name="cold_fourier",
        fourier_coeffs=cold_coeffs,
        period=period,
        color=cold_color,
        random_respawn=random_respawn,
        reward_delay=reward_delay,
        expiry_time=expiry_time,
        mean_expiry_regen_delay=mean_expiry_regen_delay,
        std_expiry_regen_delay=std_expiry_regen_delay,
    )

    return hot, cold
