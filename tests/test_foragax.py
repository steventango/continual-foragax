import pickle

import chex
import gymnax
import jax
import jax.numpy as jnp
import pytest

from foragax import (
    AGENT,
    EMPTY,
    FLOWER,
    MOREL,
    OYSTER,
    THORNS,
    WALL,
    Actions,
    Biome,
    ForagerObject,
    ForagerWorld,
    ObjectType,
)


def test_observation_shape():
    """Test that the observation shape is correct."""
    env = ForagerObject(
        size=(500, 500),
        aperture_size=(9, 9),
        object_types=(WALL, FLOWER, THORNS),
    )
    params = env.default_params
    assert env.observation_space(params).shape == (9, 9, 3)


def test_gymnax_api():
    key = jax.random.PRNGKey(0)
    key, key_reset, key_act, key_step = jax.random.split(key, 4)

    env = ForagerObject(size=(5, 5))
    env_params = env.default_params

    obs, state = env.reset(key_reset, env_params)

    action = env.action_space(env_params).sample(key_act)

    # Perform the step transition.
    n_obs, n_state, reward, done, _ = env.step(key_step, state, action, env_params)


def test_sizes():
    # can specify sizes with integers
    env = ForagerObject(size=8, aperture_size=3)
    params = env.default_params
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)
    assert jnp.array_equal(state.pos, jnp.array([4, 4]))
    assert env.size == (8, 8)
    assert env.aperture_size == (3, 3)
    chex.assert_shape(obs, (3, 3, 0))


def test_uneven_sizes():
    # can specify sizes as uneven tuples
    env = ForagerObject(size=(10, 5), aperture_size=(5, 1))
    params = env.default_params
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)
    assert jnp.array_equal(state.pos, jnp.array([5, 2]))
    assert env.size == (10, 5)
    assert env.aperture_size == (5, 1)
    chex.assert_shape(obs, (5, 1, 0))


def test_add_objects():
    # can add objects
    env = ForagerObject(
        size=10,
        object_types=(
            EMPTY,
            FLOWER,
        ),
        biomes=(
            Biome((0, 0)),
            Biome(
                object_frequencies=(
                    0,
                    0.1,
                )
            ),
        ),
    )
    params = env.default_params
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)

    assert jnp.count_nonzero(state.object_grid) == int(100 * 0.1)
    chex.assert_shape(obs, (5, 1, 0))


def test_world_observation_mode():
    # can use world observation mode
    env = ForagerWorld(size=(10, 10))
    params = env.default_params
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)

    assert obs.shape == (10, 10, 1)


def test_basic_movement():
    """Test agent movement and collision with walls."""
    key = jax.random.PRNGKey(0)

    biome = Biome(
        object_frequencies=(
            0,
            1.0,
        ),
        start=(3, 4),
        stop=(4, 6),
    )
    env = ForagerObject(
        size=7,
        object_types=(
            EMPTY,
            WALL,
        ),
        biomes=(biome,),
    )
    params = env.default_params
    _, state = env.reset(key, params)

    assert jnp.array_equal(state.pos, jnp.array([3, 3]))

    # stays still when bumping into a wall
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([3, 3]))

    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)
    assert jnp.array_equal(state.pos, jnp.array([4, 3]))

    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)
    assert jnp.array_equal(state.pos, jnp.array([3, 3]))

    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([3, 2]))


def test_vision():
    """Test the agent's observation."""
    key = jax.random.PRNGKey(0)
    env = Forager(
        size=(7, 7), aperture_size=(3, 3), observation_mode=ObservationMode.OBJECTS
    )
    params = env.default_params
    _, state = env.reset_env(key, params)

    # Create a predictable environment
    grid = jnp.zeros(env.observation_space(params).shape[:2], dtype=jnp.int32)
    grid = grid.at[2, 3].set(WALL.id)
    state = state.replace(object_grid=grid, pos=jnp.array([3, 3]))

    obs = env.get_obs(state, params)

    # Agent is at the center of the aperture
    # Wall is at (2, 3), agent at (3, 3). Wall should be at top-center of aperture.
    expected_aperture = jnp.zeros((3, 3, len(env.object_ids)))
    expected_aperture = expected_aperture.at[0, 1, WALL.id].set(1)
    expected_aperture = expected_aperture.at[1, 1, AGENT.id].set(1)

    assert jnp.array_equal(obs, expected_aperture)


def test_respawn():
    """Test object respawning."""
    key = jax.random.PRNGKey(0)
    # predictable respawn
    flower_with_regen = FLOWER.replace(regen_delay=(5, 6))
    env = Forager(
        size=(7, 7),
        object_types=(EMPTY, WALL, flower_with_regen),
    )
    params = env.default_params
    _, state = env.reset_env(key, params)

    # Place a flower and move the agent to it
    grid = jnp.zeros((7, 7), dtype=jnp.int32)
    grid = grid.at[3, 4].set(flower_with_regen.id)
    state = state.replace(
        object_grid=grid, original_object_grid=grid, pos=jnp.array([3, 4])
    )

    # Collect the flower (action is irrelevant, agent is on the flower)
    key, step_key = jax.random.split(state.key)
    _, state, reward, _, _ = env.step_env(step_key, state, 0, params)
    assert reward == flower_with_regen.reward
    assert state.object_grid[3, 4] == EMPTY.id
    assert state.respawn_timers[3, 4] > 0

    # Step until it respawns
    for _ in range(4):
        key, step_key = jax.random.split(state.key)
        _, state, _, _, _ = env.step_env(step_key, state, 0, params)
        assert state.object_grid[3, 4] == EMPTY.id

    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step_env(step_key, state, 0, params)
    assert state.object_grid[3, 4] == flower_with_regen.id


def test_wrapping_dynamics():
    """Test that the agent wraps around the environment boundaries."""
    key = jax.random.PRNGKey(0)
    env = ForagerObject(size=(5, 5), object_types=(EMPTY,))
    params = env.default_params
    _, state = env.reset(key, params)

    # Go up
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 3]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 4]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 0]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 1]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))

    # Go down
    _, state = env.reset(key, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 1]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 0]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 4]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 3]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))

    # Go right
    _, state = env.reset(key, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)
    assert jnp.array_equal(state.pos, jnp.array([3, 2]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)
    assert jnp.array_equal(state.pos, jnp.array([4, 2]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)
    assert jnp.array_equal(state.pos, jnp.array([0, 2]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)
    assert jnp.array_equal(state.pos, jnp.array([1, 2]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))

    # Go left
    _, state = env.reset(key, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)
    assert jnp.array_equal(state.pos, jnp.array([1, 2]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)
    assert jnp.array_equal(state.pos, jnp.array([0, 2]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)
    assert jnp.array_equal(state.pos, jnp.array([4, 2]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)
    assert jnp.array_equal(state.pos, jnp.array([3, 2]))
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))


def test_wrapping_vision():
    """Test that the agent's vision wraps around the environment boundaries."""
    key = jax.random.PRNGKey(0)
    env = ForagerObject(size=(5, 5), aperture_size=(3, 3), object_types=(EMPTY, FLOWER))
    params = env.default_params
    obs, state = env.reset(key, params)

    # Create a predictable environment with a flower at (0, 0)
    grid = jnp.zeros((5, 5), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)
    state = state.replace(object_grid=grid)

    obs = env.get_obs(state, params)

    expected = jnp.zeros((3, 3, 1), dtype=jnp.int32)
    assert jnp.array_equal(obs, expected)

    # go left
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)

    # go down
    key, step_key = jax.random.split(state.key)
    obs, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)

    expected = jnp.zeros((3, 3, 1), dtype=jnp.int32)
    expected = expected.at[2, 0, 0].set(1)

    assert jnp.array_equal(state.pos, jnp.array([1, 1]))
    assert jnp.array_equal(obs, expected)

    # go left , go left
    key, step_key = jax.random.split(state.key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)
    key, step_key = jax.random.split(state.key)
    obs, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)

    expected = jnp.zeros((3, 3, 1), dtype=jnp.int32)
    expected = expected.at[2, 2, 0].set(1)

    assert jnp.array_equal(state.pos, jnp.array([4, 1]))
    assert jnp.array_equal(obs, expected)


def test_generate_objects_in_biome():
    """Test generating objects within a specific biome area."""
    env = Forager(
        size=(10, 10),
        object_types=(EMPTY, WALL, FLOWER, THORNS, MOREL, OYSTER),
        biomes=(
            Biome(
                object_frequencies=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                start=(2, 2),
                stop=(6, 6),
            ),
        ),
    )
    key = jax.random.PRNGKey(0)
    params = env.default_params

    _, state = env.reset_env(key, params)

    # Check that morels only appear within the biome
    morel_locations = jnp.argwhere(state.object_grid == MOREL.id)

    assert jnp.all(morel_locations >= 2)
    assert jnp.all(morel_locations < 6)

    # Check that no other objects were generated
    unique_objects = jnp.unique(state.object_grid)
    assert OYSTER.id not in unique_objects


def test_benchmark_big_env(benchmark):
    env = ForagerObject(
        size=10_000,
        aperture_size=61,
        object_types=(EMPTY, WALL, FLOWER),
        biomes=(Biome(object_frequencies=(0.9, 0.05, 0.05)),),
    )
    params = env.default_params
    key = jax.random.PRNGKey(0)

    # Reset is part of the setup, not benchmarked
    key, reset_key = jax.random.split(key)
    _, state = env.reset(reset_key, params)

    @jax.jit
    def _run(state, key):
        def f(carry, _):
            state, key = carry
            key, step_key = jax.random.split(key, 2)
            action = Actions.UP
            _, new_state, _, _, _ = env.step(step_key, state, action, params)
            return (new_state, key), None

        (final_state, _), _ = jax.lax.scan(f, (state, key), None, length=100)
        return final_state

    # warm-up compilation
    key, run_key = jax.random.split(key)
    _run(state, run_key).pos.block_until_ready()

    def benchmark_fn():
        # use a fixed key for benchmark consistency
        key, run_key = jax.random.split(jax.random.PRNGKey(1))
        _run(state, run_key).pos.block_until_ready()

    benchmark(benchmark_fn)
