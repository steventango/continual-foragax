"""Tests for biome regeneration with dynamic object properties."""

import chex
import jax
import jax.numpy as jnp

from foragax.env import Biome, ForagaxEnv
from foragax.objects import DefaultForagaxObject
from foragax.registry import make


def test_object_colors_in_state():
    """Test that object colors are properly stored in state and can be accessed."""
    env = ForagaxEnv(
        size=(10, 10),
        aperture_size=(5, 5),
        objects=(
            DefaultForagaxObject(
                name="red", reward=1.0, collectable=True, color=(255, 0, 0)
            ),
            DefaultForagaxObject(
                name="blue", reward=2.0, collectable=True, color=(0, 0, 255)
            ),
        ),
        observation_type="rgb",
    )

    key = jax.random.key(42)
    obs, state = env.reset(key, env.default_params)

    # Check that state has object_colors field
    assert hasattr(state, "object_colors")
    assert state.object_colors.shape == (3, 3)  # EMPTY + 2 objects

    # Check initial colors match (EMPTY has default white color)
    chex.assert_trees_all_close(state.object_colors[0], jnp.array([255, 255, 255]))  # EMPTY (default)
    chex.assert_trees_all_close(state.object_colors[1], jnp.array([255, 0, 0]))  # red
    chex.assert_trees_all_close(state.object_colors[2], jnp.array([0, 0, 255]))  # blue


def test_fourier_params_in_state():
    """Test that Fourier parameters are properly initialized in state."""
    env = ForagaxEnv(
        size=(10, 10),
        aperture_size=(5, 5),
        objects=(
            DefaultForagaxObject(
                name="obj1", reward=1.0, collectable=True, color=(255, 0, 0)
            ),
        ),
        observation_type="rgb",
    )

    key = jax.random.key(42)
    obs, state = env.reset(key, env.default_params)

    # Check that state has fourier_params field
    assert hasattr(state, "fourier_params")
    # Shape should be (num_objects, 2*num_harmonics + 1)
    # 2 objects (EMPTY + 1), 50 harmonics by default, so 2*50+1 = 101 params per object
    assert state.fourier_params.shape == (2, 101)  # EMPTY + 1 object, 101 params each

    # Initially should be all zeros
    chex.assert_trees_all_close(state.fourier_params, jnp.zeros_like(state.fourier_params))


def test_biome_regeneration_triggers():
    """Test that biome regeneration triggers when threshold is reached."""
    # Create environment with low threshold for quick testing
    env = ForagaxEnv(
        size=(10, 10),
        aperture_size=(5, 5),
        objects=(
            DefaultForagaxObject(
                name="test_obj",
                reward=1.0,
                collectable=True,
                color=(255, 0, 0),
                regen_delay=(10, 10),
            ),
        ),
        biomes=(
            Biome(start=(2, 2), stop=(4, 4), object_frequencies=(1.0,)),  # 4 objects
        ),
        biome_regen_threshold=0.5,  # Regenerate when 50% consumed (2 objects)
        observation_type="rgb",
    )

    key = jax.random.key(42)
    obs, state = env.reset(key, env.default_params)

    # Store initial colors
    initial_colors = state.object_colors.copy()
    initial_fourier = state.fourier_params.copy()

    # Check initial regeneration state
    assert not state.biome_regenerated[0], "Biome should not be regenerated initially"

    # Count initial objects
    initial_count = jnp.sum(state.object_grid > 0)
    assert initial_count > 0, "Should have objects initially"

    # Collect objects by moving to them (we'll need to step multiple times)
    # This is a simplified test - in reality we'd need to navigate to objects
    # For now, let's just manually trigger regeneration by setting the counters

    # Manually set consumption to trigger regeneration
    state = state.replace(
        biome_objects_collected=state.biome_objects_collected.at[0].set(2),
        biome_total_spawned=state.biome_total_spawned.at[0].set(4),
    )

    # Step the environment - this should trigger regeneration
    key, step_key = jax.random.split(key)
    obs, state, reward, done, info = env.step(step_key, state, 0, env.default_params)

    # Check that regeneration happened
    assert state.biome_regenerated[0], "Biome should be regenerated"

    # Check that colors changed (at least for some objects)
    colors_changed = not jnp.allclose(state.object_colors, initial_colors)
    # Note: Colors might not change if random generation happens to produce same values
    # But Fourier params should definitely change from zeros
    fourier_changed = not jnp.allclose(state.fourier_params, initial_fourier)

    # At least one should have changed
    assert colors_changed or fourier_changed, "Either colors or Fourier params should change after regeneration"


def test_fourier_reward_computation():
    """Test that Fourier rewards are computed correctly."""
    env = ForagaxEnv(
        size=(10, 10),
        aperture_size=(5, 5),
        objects=(
            DefaultForagaxObject(
                name="test_obj", reward=1.0, collectable=True, color=(255, 0, 0)
            ),
        ),
        observation_type="rgb",
    )

    # Test the Fourier reward computation with known parameters
    # Simple case: period=100, single harmonic with a_1=1, b_1=0
    # Should give: cos(2π * 1 * t / 100) = cos(2π * t / 100)
    # Create params array: [period, a_1, b_1, a_2, b_2, ..., a_50, b_50]
    n_harmonics = 50
    fourier_params = jnp.zeros(2 * n_harmonics + 1)
    fourier_params = fourier_params.at[0].set(100.0)  # period
    fourier_params = fourier_params.at[1].set(1.0)    # a_1 = 1
    # All other coefficients are 0

    reward_t0 = env._compute_fourier_reward(fourier_params, 0)
    reward_t25 = env._compute_fourier_reward(fourier_params, 25)
    reward_t50 = env._compute_fourier_reward(fourier_params, 50)

    # At t=0: cos(0) = 1
    chex.assert_trees_all_close(reward_t0, 1.0, atol=1e-6)

    # At t=25: cos(2π * 25 / 100) = cos(π/2) = 0
    chex.assert_trees_all_close(reward_t25, 0.0, atol=1e-6)

    # At t=50: cos(2π * 50 / 100) = cos(π) = -1
    chex.assert_trees_all_close(reward_t50, -1.0, atol=1e-6)


def test_weather_v6_environment():
    """Test that ForagaxWeather-v6 can be created and run."""
    env = make("ForagaxWeather-v6", observation_type="rgb", aperture_size=5)

    key = jax.random.key(42)
    obs, state = env.reset(key, env.default_params)

    # Check that state has the required fields
    assert hasattr(state, "object_colors")
    assert hasattr(state, "fourier_params")
    assert hasattr(state, "biome_regenerated")

    # Check Fourier params shape (num_objects, 2*num_harmonics+1)
    # ForagaxWeather-v6 has EMPTY + HOT + COLD + PADDING = 4 objects (nowrap=True adds PADDING)
    n_harmonics = 50
    expected_shape = (4, 2 * n_harmonics + 1)
    chex.assert_shape(state.fourier_params, expected_shape)

    # Check that biome regeneration threshold is set
    assert env.biome_regen_threshold == 0.9

    # Step the environment a few times
    for _ in range(10):
        key, step_key, action_key = jax.random.split(key, 3)
        action = env.action_space(env.default_params).sample(action_key)
        obs, state, reward, done, info = env.step(step_key, state, action, env.default_params)

        # Check observation shape is correct
        assert obs.shape == env.observation_space(env.default_params).shape


def test_colors_persist_across_steps():
    """Test that object colors in state persist across steps without regeneration."""
    env = ForagaxEnv(
        size=(10, 10),
        aperture_size=(5, 5),
        objects=(
            DefaultForagaxObject(
                name="test_obj",
                reward=1.0,
                collectable=True,
                color=(255, 0, 0),
                regen_delay=(10, 10),
            ),
        ),
        biomes=(Biome(start=(2, 2), stop=(4, 4), object_frequencies=(0.5,)),),
        biome_regen_threshold=0.0,  # Disabled
        observation_type="rgb",
    )

    key = jax.random.key(42)
    obs, state = env.reset(key, env.default_params)

    initial_colors = state.object_colors.copy()
    initial_fourier = state.fourier_params.copy()

    # Step multiple times
    for _ in range(20):
        key, step_key = jax.random.split(key)
        obs, state, reward, done, info = env.step(step_key, state, 0, env.default_params)

    # Colors and Fourier params should remain the same (no regeneration)
    chex.assert_trees_all_equal(state.object_colors, initial_colors)
    chex.assert_trees_all_equal(state.fourier_params, initial_fourier)


def test_fourier_rewards_override_default():
    """Test that non-zero Fourier params override default reward functions."""
    env = ForagaxEnv(
        size=(10, 10),
        aperture_size=(5, 5),
        objects=(
            DefaultForagaxObject(
                name="test_obj",
                reward=10.0,  # Default reward
                collectable=True,
                color=(255, 0, 0),
                regen_delay=(1000, 1000),  # Long regen so we can collect
            ),
        ),
        biomes=(Biome(start=(4, 4), stop=(5, 5), object_frequencies=(1.0,)),),
        observation_type="rgb",
    )

    key = jax.random.key(42)
    obs, state = env.reset(key, env.default_params)

    # Find which object type is at position (4, 4)
    obj_at_pos = state.object_grid[4, 4]
    assert obj_at_pos > 0, "Should have an object at (4, 4)"

    # Set Fourier params for the object type that's actually at this position
    # Use a simple constant function: period=100, a_1=0.5, all others=0
    # This gives reward = 0.5 * cos(2π * t / 100)
    n_harmonics = 50
    new_params = jnp.zeros(2 * n_harmonics + 1)
    new_params = new_params.at[0].set(100.0)  # period
    new_params = new_params.at[1].set(0.5)     # a_1 = 0.5

    state = state.replace(
        fourier_params=state.fourier_params.at[obj_at_pos].set(new_params)
    )

    # Move agent to object position (4, 4)
    state = state.replace(pos=jnp.array([4, 4]))

    # Step with any action - should collect object and get Fourier reward
    # At t=0, reward should be 0.5 * cos(0) = 0.5
    key, step_key = jax.random.split(key)
    obs, next_state, reward, done, info = env.step(
        step_key, state, 0, env.default_params
    )

    # Should get Fourier reward (0.5) instead of default reward (10)
    chex.assert_trees_all_close(reward, 0.5, atol=1e-4)
