# Foragax AI Coding Instructions

## Project Overview
Foragax is a JAX-first grid-world environment suite for continual reinforcement learning experiments. It provides multiple environment variants (weather, multi-biome, etc.) through a registry factory pattern, with support for different observation modalities and dynamic weather-driven rewards.

## Architecture & Key Components

### Core Structure
- **`foragax.registry.make()`**: Factory function for creating environments by ID (e.g., `"ForagaxWeather-v1"`)
- **Environment classes**: `ForagaxObjectEnv`, `ForagaxRGBEnv`, `ForagaxWorldEnv` - different observation modalities
- **Object system**: Extensible via `BaseForagaxObject` with rewards, regeneration, blocking/collectable properties
- **Biome system**: Defines regions with different object spawning frequencies and boundaries
- **Weather system**: Uses real ECA&D temperature data for dynamic rewards

### JAX-First Design Principles
- **Explicit PRNG keys**: Pass `jax.random.key()` instances to all stochastic operations
- **Immutable state**: Environment state is a flax.struct.dataclass, never mutated in-place
- **JIT-compatible**: All operations designed for `jax.jit` compilation
- **Functional API**: `env.reset(key, params)` → `(obs, state)`, `env.step(key, state, action, params)` → `(obs, state, reward, done, info)`

## Critical Developer Workflows

### Environment Creation & Usage
```python
from foragax.registry import make
import jax

# Create environment
env = make("ForagaxWeather-v1", aperture_size=5, observation_type="object")

# Initialize
key = jax.random.key(0)
key, key_reset = jax.random.split(key)
obs, env_state = env.reset(key_reset, env.default_params)

# Step through environment
key, key_act, key_step = jax.random.split(key, 3)
action = env.action_space(env.default_params).sample(key_act)
obs, next_state, reward, done, info = env.step(key_step, env_state, action, env.default_params)
```

### Testing Patterns
- Use `chex` for JAX array assertions: `chex.assert_trees_all_equal(actual, expected)`
- Test observation shapes: `chex.assert_shape(obs, (5, 5, 2))`
- Benchmark with `@pytest.mark.benchmark`: `benchmark(benchmark_fn)`

### Build & Development
- **Dependencies**: Managed with `uv` - use `uv pip install -e .[dev]` for development setup
- **Testing**: `uv run --frozen pytest tests` (CI runs across Python 3.8-3.13)
- **Linting**: `ruff check --fix` and `ruff format` (via pre-commit hooks)
- **Publishing**: Uses commitizen for version bumping and releases

## Project-Specific Conventions

### Object Definition
```python
from foragax.objects import DefaultForagaxObject

# Define custom objects
MY_OBJECT = DefaultForagaxObject(
    name="my_object",
    reward=5.0,
    collectable=True,
    regen_delay=(50, 100),  # Min/max respawn delay
    color=(255, 0, 0),      # RGB color for rendering
    blocking=False
)
```

### Environment Configuration
```python
from foragax.env import Biome

# Define biomes with object frequencies
biomes = (
    Biome(
        start=(0, 0), stop=(10, 10),  # Grid coordinates
        object_frequencies=(0.1, 0.2)  # Frequency per object type
    ),
)

# Create environment programmatically
env = ForagaxObjectEnv(
    size=(20, 20),
    aperture_size=(5, 5),
    objects=(MY_OBJECT,),
    biomes=biomes,
    nowrap=True  # Disable boundary wrapping
)
```

### Weather Integration
- Weather objects use `WeatherObject` class with temperature-based rewards
- Temperature data loaded from `foragax/data/ECA_non-blended_custom/` directory
- `create_weather_objects()` factory creates HOT/COLD object pairs
- Rewards scale with `multiplier` parameter and temperature normalization

### Observation Modes
- **Object mode**: Color-based partial observability (objects with same color = same observation channel)
- **RGB mode**: Full color rendering of aperture
- **World mode**: Complete grid observation with agent position channel

## Key Files & Patterns

### Essential Files
- `foragax/registry.py`: Environment factory and configuration registry
- `foragax/env.py`: Core environment implementations and biome logic
- `foragax/objects.py`: Object definitions and weather object creation
- `foragax/weather.py`: Temperature data loading and processing
- `tests/test_foragax.py`: Comprehensive test suite with JAX-specific patterns

### Common Patterns
- **Boundary handling**: `nowrap=True` prevents wrapping, uses padding objects
- **Aperture vision**: Agent sees `aperture_size` grid around current position
- **Timer encoding**: Respawning objects stored as negative values in grid
- **Color mapping**: Objects grouped by color for partial observability
- **Biome masking**: Objects only spawn within defined biome boundaries

## Integration Points

### External Dependencies
- **Gymnax**: Base environment framework - follow `gymnax.environments.environment.Environment` interface
- **JAX/Flax**: Core computation and struct definitions
- **ECA&D data**: Real climate datasets for weather environments
- **uv**: Modern Python packaging and dependency management

### CI/CD Integration
- **GitHub Actions**: Matrix testing across Python versions
- **Ruff**: Fast linting and formatting (replaces black/isort/flake8)
- **Commitizen**: Conventional commit versioning and changelog generation
- **pre-commit**: Automated code quality checks

## Debugging Tips

### Common Issues
- **JAX key management**: Always split keys with `jax.random.split()` before use
- **Shape mismatches**: Check observation shapes match `env.observation_space(params)`
- **JIT compilation**: Ensure all operations are JIT-compatible (no Python control flow)
- **Boundary conditions**: Test both `nowrap=True` and `nowrap=False` behaviors

### Visualization
- Use `examples/plot.py` for reward/temperature analysis
- Use `examples/visualize.py` for video generation with Gymnasium
- Render modes: `"world"`, `"aperture"`, `"world_true"`, `"aperture_true"`

Remember: Foragax is designed for continual RL research - focus on long-horizon, non-terminating environments with dynamic object respawning and weather-driven reward landscapes.