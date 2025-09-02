import chex
import jax
import jax.numpy as jnp

from foragax.env import (
    Actions,
    Biome,
    ForagaxObjectEnv,
    ForagaxRGBEnv,
    ForagaxWorldEnv,
)
from foragax.objects import FLOWER, MOREL, OYSTER, THORNS, WALL


def test_observation_shape():
    """Test that the observation shape is correct."""
    env = ForagaxObjectEnv(
        size=(500, 500),
        aperture_size=(9, 9),
        objects=(WALL, FLOWER, THORNS),
    )
    params = env.default_params
    assert env.observation_space(params).shape == (9, 9, 3)


def test_gymnax_api():
    key = jax.random.key(0)
    key, key_reset, key_act, key_step = jax.random.split(key, 4)

    env = ForagaxObjectEnv(size=(5, 5))
    env_params = env.default_params

    obs, state = env.reset(key_reset, env_params)

    action = env.action_space(env_params).sample(key_act)

    # Perform the step transition.
    n_obs, n_state, reward, done, _ = env.step(key_step, state, action, env_params)


def test_sizes():
    # can specify sizes with integers
    env = ForagaxObjectEnv(size=8, aperture_size=3)
    params = env.default_params
    key = jax.random.key(0)
    obs, state = env.reset(key, params)
    assert jnp.array_equal(state.pos, jnp.array([4, 4]))
    assert env.size == (8, 8)
    assert env.aperture_size == (3, 3)
    chex.assert_shape(obs, (3, 3, 0))


def test_uneven_sizes():
    # can specify sizes as uneven tuples
    env = ForagaxObjectEnv(size=(10, 5), aperture_size=(5, 1))
    params = env.default_params
    key = jax.random.key(0)
    obs, state = env.reset(key, params)
    assert jnp.array_equal(state.pos, jnp.array([5, 2]))
    assert env.size == (10, 5)
    assert env.aperture_size == (5, 1)
    chex.assert_shape(obs, (5, 1, 0))


def test_add_objects():
    # can add objects
    size = 100
    freq = 0.1
    env = ForagaxObjectEnv(
        size=size,
        objects=(FLOWER,),
        biomes=(Biome(object_frequencies=(freq,)),),
    )
    params = env.default_params
    key = jax.random.key(0)
    obs, state = env.reset(key, params)

    empirical_freq = jnp.count_nonzero(state.object_grid) / size**2
    chex.assert_trees_all_close(empirical_freq, freq, rtol=0.1)
    chex.assert_shape(obs, (5, 5, 1))


def test_object_observation_mode():
    env = ForagaxObjectEnv(
        size=(10, 10),
        aperture_size=(5, 5),
        objects=(WALL, FLOWER),
    )
    params = env.default_params
    key = jax.random.key(0)
    obs, state = env.reset(key, params)

    assert obs.shape == (5, 5, 2)


def test_rgb_observation_mode():
    env = ForagaxRGBEnv(
        size=(10, 10),
        aperture_size=(5, 5),
        objects=(WALL, FLOWER),
    )
    params = env.default_params
    key = jax.random.key(0)
    obs, state = env.reset(key, params)

    assert obs.shape == (5, 5, 3)


def test_object_observation_mode_large_aperture():
    env = ForagaxObjectEnv(
        size=(10, 10),
        aperture_size=(20, 20),
        objects=(WALL, FLOWER),
    )
    params = env.default_params
    key = jax.random.key(0)
    obs, state = env.reset(key, params)

    assert obs.shape == (20, 20, 2)


def test_rgb_observation_mode_large_aperture():
    env = ForagaxRGBEnv(
        size=(10, 10),
        aperture_size=(20, 20),
        objects=(WALL, FLOWER),
    )
    params = env.default_params
    key = jax.random.key(0)
    obs, state = env.reset(key, params)

    assert obs.shape == (20, 20, 3)


def test_world_observation_mode():
    # can use world observation mode
    env = ForagaxWorldEnv(size=(10, 10))
    params = env.default_params
    key = jax.random.key(0)
    obs, state = env.reset(key, params)

    assert obs.shape == (10, 10, 1)


def test_basic_movement():
    """Test agent movement and collision with walls."""
    key = jax.random.key(0)

    biome = Biome(
        object_frequencies=(1.0,),
        start=(3, 4),
        stop=(4, 6),
    )
    env = ForagaxObjectEnv(
        size=7,
        objects=(WALL,),
        biomes=(biome,),
    )
    params = env.default_params
    _, state = env.reset(key, params)

    assert jnp.array_equal(state.pos, jnp.array([3, 3]))

    # stays still when bumping into a wall
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([3, 3]))

    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)
    assert jnp.array_equal(state.pos, jnp.array([4, 3]))

    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)
    assert jnp.array_equal(state.pos, jnp.array([3, 3]))

    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([3, 2]))


def test_vision():
    """Test the agent's observation."""
    key = jax.random.key(0)
    object_types = (WALL,)
    env = ForagaxObjectEnv(size=(7, 7), aperture_size=(3, 3), objects=object_types)
    params = env.default_params
    obs, state = env.reset(key, params)

    wall_id = 1  # 0 is EMPTY

    # Create a predictable environment
    grid = jnp.zeros((7, 7), dtype=jnp.int_)
    grid = grid.at[4, 3].set(wall_id)
    grid = grid.at[5, 3].set(wall_id)
    grid = grid.at[2, 0].set(wall_id)
    state = state.replace(object_grid=grid)

    chex.assert_trees_all_equal(state.pos, jnp.array([3, 3]))

    # No movement
    key, step_key = jax.random.split(key)
    obs, state, _, _, _ = env.step(step_key, state, Actions.UP, params)

    expected = jnp.zeros((3, 3, 1), dtype=jnp.int_)
    expected = expected.at[0, 1, 0].set(1)

    chex.assert_trees_all_equal(state.pos, jnp.array([3, 3]))
    chex.assert_trees_all_equal(obs, expected)

    # Move right
    key, step_key = jax.random.split(key)
    obs, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)
    key, step_key = jax.random.split(key)
    obs, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    expected = jnp.zeros((3, 3, 1), dtype=jnp.int_)
    expected = expected.at[0, 0, 0].set(1)
    expected = expected.at[1, 0, 0].set(1)

    chex.assert_trees_all_equal(state.pos, jnp.array([4, 4]))
    chex.assert_trees_all_equal(obs, expected)


def test_respawn():
    """Test object respawning."""
    key = jax.random.key(0)
    object_types = (FLOWER,)
    env = ForagaxObjectEnv(
        size=7,
        aperture_size=3,
        objects=object_types,
    )
    params = env.default_params
    _, state = env.reset_env(key, params)

    flower_id = 1  # 0 is EMPTY

    # Place a flower and move the agent to it
    grid = jnp.zeros((7, 7), dtype=jnp.int_)
    grid = grid.at[4, 3].set(flower_id)
    state = state.replace(object_grid=grid, pos=jnp.array([3, 3]))

    # Collect the flower
    key, step_key = jax.random.split(key)
    _, state, reward, _, _ = env.step_env(step_key, state, Actions.UP, params)
    assert reward == FLOWER.reward_val
    assert state.object_grid[4, 3] < 0

    # Step until it respawns
    for i in range(20):
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step_env(step_key, state, Actions.UP, params)
        assert state.object_grid[4, 3] < 0

    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert state.object_grid[4, 3] == flower_id


def test_wrapping_dynamics():
    """Test that the agent wraps around the environment boundaries."""
    key = jax.random.key(0)
    env = ForagaxObjectEnv(size=(5, 5), objects=())
    params = env.default_params
    _, state = env.reset(key, params)

    # Go up
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 3]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 4]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 0]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 1]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))

    # Go down
    _, state = env.reset(key, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 1]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 0]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 4]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 3]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))

    # Go right
    _, state = env.reset(key, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)
    assert jnp.array_equal(state.pos, jnp.array([3, 2]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)
    assert jnp.array_equal(state.pos, jnp.array([4, 2]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)
    assert jnp.array_equal(state.pos, jnp.array([0, 2]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)
    assert jnp.array_equal(state.pos, jnp.array([1, 2]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))

    # Go left
    _, state = env.reset(key, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)
    assert jnp.array_equal(state.pos, jnp.array([1, 2]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)
    assert jnp.array_equal(state.pos, jnp.array([0, 2]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)
    assert jnp.array_equal(state.pos, jnp.array([4, 2]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)
    assert jnp.array_equal(state.pos, jnp.array([3, 2]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))


def test_wrapping_vision():
    """Test that the agent's vision wraps around the environment boundaries."""
    key = jax.random.key(0)
    env = ForagaxObjectEnv(size=(5, 5), aperture_size=(3, 3), objects=(FLOWER,))
    params = env.default_params
    obs, state = env.reset(key, params)

    # Create a predictable environment with a flower at (0, 0)
    grid = jnp.zeros((5, 5), dtype=jnp.int_)
    grid = grid.at[0, 0].set(1)
    state = state.replace(object_grid=grid)

    obs = env.get_obs(state, params)

    expected = jnp.zeros((3, 3, 1), dtype=jnp.int_)
    assert jnp.array_equal(obs, expected)

    # go left
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)

    # go down
    key, step_key = jax.random.split(key)
    obs, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)

    expected = jnp.zeros((3, 3, 1), dtype=jnp.int_)
    expected = expected.at[2, 0, 0].set(1)

    assert jnp.array_equal(state.pos, jnp.array([1, 1]))
    assert jnp.array_equal(obs, expected)

    # go left , go left
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)
    key, step_key = jax.random.split(key)
    obs, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)

    expected = jnp.zeros((3, 3, 1), dtype=jnp.int_)
    expected = expected.at[2, 2, 0].set(1)

    assert jnp.array_equal(state.pos, jnp.array([4, 1]))
    assert jnp.array_equal(obs, expected)


def test_generate_objects_in_biome():
    """Test generating objects within a specific biome area."""
    object_types = (WALL, FLOWER, THORNS, MOREL, OYSTER)
    env = ForagaxObjectEnv(
        size=(10, 10),
        objects=object_types,
        biomes=(
            Biome(
                object_frequencies=(0.0, 0.0, 0.0, 0.1, 0.0),
                start=(2, 2),
                stop=(6, 6),
            ),
        ),
    )
    key = jax.random.key(0)
    params = env.default_params

    _, state = env.reset(key, params)

    morel_id = object_types.index(MOREL) + 1
    oyster_id = object_types.index(OYSTER) + 1
    wall_id = object_types.index(WALL) + 1
    thorns_id = object_types.index(THORNS) + 1

    # Check that morels only appear within the biome
    morel_locations = jnp.argwhere(state.object_grid == morel_id)

    assert jnp.all(morel_locations >= 2)
    assert jnp.all(morel_locations < 6)

    # Check that no other objects were generated
    unique_objects = jnp.unique(state.object_grid)
    assert oyster_id not in unique_objects
    assert wall_id not in unique_objects
    assert thorns_id not in unique_objects


def test_benchmark_vision(benchmark):
    env = ForagaxObjectEnv(size=7, aperture_size=3, objects=(WALL,))
    params = env.default_params
    key = jax.random.key(0)
    _, state = env.reset(key, params)

    grid = jnp.zeros((7, 7), dtype=jnp.int_)
    grid = grid.at[4, 3].set(1)
    grid = grid.at[5, 3].set(1)
    grid = grid.at[2, 0].set(1)
    state = state.replace(object_grid=grid)

    @jax.jit
    def _run(state, key):
        key, step_key = jax.random.split(key)
        obs, new_state, _, _, _ = env.step(step_key, state, Actions.UP, params)
        return obs, new_state

    # warm-up
    obs, new_state = _run(state, key)

    expected = jnp.zeros((3, 3, 1), dtype=jnp.int_)
    expected = expected.at[0, 1, 0].set(1)

    chex.assert_trees_all_equal(new_state.pos, jnp.array([3, 3]))
    chex.assert_trees_all_equal(obs, expected)

    def benchmark_fn():
        # use a fixed key for benchmark consistency
        _run(state, jax.random.key(1))[0].block_until_ready()

    benchmark(benchmark_fn)


def test_benchmark_creation(benchmark):
    env = ForagaxObjectEnv(
        size=1_000,
        aperture_size=31,
        objects=(WALL, FLOWER),
        biomes=(Biome(object_frequencies=(0.05, 0.05)),),
    )
    params = env.default_params

    @jax.jit
    def _build(key):
        _, state = env.reset(key, params)
        return state

    # no warm-up

    def benchmark_fn():
        _build(jax.random.key(1)).pos.block_until_ready()

    benchmark(benchmark_fn)


def test_benchmark_small_env(benchmark):
    env = ForagaxObjectEnv(
        size=1_000,
        aperture_size=11,
        objects=(WALL, FLOWER),
        biomes=(Biome(object_frequencies=(0.1, 0.1)),),
    )
    params = env.default_params
    key = jax.random.key(0)
    key, reset_key = jax.random.split(key)
    _, state = env.reset(reset_key, params)

    @jax.jit
    def _run(state, key):
        def f(carry, _):
            state, key = carry
            key, step_key = jax.random.split(key, 2)
            _, new_state, _, _, _ = env.step(step_key, state, Actions.UP, params)
            return (new_state, key), None

        (final_state, _), _ = jax.lax.scan(f, (state, key), None, length=1000)
        return final_state

    key, run_key = jax.random.split(key)
    _run(state, run_key).pos.block_until_ready()

    def benchmark_fn():
        key, run_key = jax.random.split(jax.random.key(1))
        _run(state, run_key).pos.block_until_ready()

    benchmark(benchmark_fn)


def test_benchmark_big_env(benchmark):
    env = ForagaxObjectEnv(
        size=10_000,
        aperture_size=61,
        objects=(WALL, FLOWER),
        biomes=(Biome(object_frequencies=(0.05, 0.05)),),
    )
    params = env.default_params
    key = jax.random.key(0)

    # Reset is part of the setup, not benchmarked
    key, reset_key = jax.random.split(key)
    _, state = env.reset(reset_key, params)

    @jax.jit
    def _run(state, key):
        def f(carry, _):
            state, key = carry
            key, step_key = jax.random.split(key, 2)
            _, new_state, _, _, _ = env.step(step_key, state, Actions.UP, params)
            return (new_state, key), None

        (final_state, _), _ = jax.lax.scan(f, (state, key), None, length=100)
        return final_state

    # warm-up compilation
    key, run_key = jax.random.split(key)
    _run(state, run_key).pos.block_until_ready()

    def benchmark_fn():
        # use a fixed key for benchmark consistency
        key, run_key = jax.random.split(jax.random.key(1))
        _run(state, run_key).pos.block_until_ready()

    benchmark(benchmark_fn)


def test_benchmark_vmap_env(benchmark):
    num_envs = 100
    env = ForagaxObjectEnv(
        size=1_000,
        aperture_size=11,
        objects=(WALL, FLOWER),
        biomes=(Biome(object_frequencies=(0.1, 0.1)),),
    )
    params = env.default_params
    key = jax.random.key(0)

    # Reset is part of the setup, not benchmarked
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, num_envs)
    states = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, params)[1]

    @jax.jit
    def _run(states, key):
        def f(carry, _):
            states, key = carry
            key, step_key = jax.random.split(key, 2)
            step_keys = jax.random.split(step_key, num_envs)
            _, new_states, _, _, _ = jax.vmap(env.step, in_axes=(0, 0, None, None))(
                step_keys, states, Actions.UP, params
            )
            return (new_states, key), None

        (final_states, _), _ = jax.lax.scan(f, (states, key), None, length=1000)
        return final_states

    # warm-up compilation
    key, run_key = jax.random.split(key)
    _run(states, run_key).pos.block_until_ready()

    def benchmark_fn():
        # use a fixed key for benchmark consistency
        key, run_key = jax.random.split(jax.random.key(1))
        _run(states, run_key).pos.block_until_ready()

    benchmark(benchmark_fn)


def test_benchmark_small_env_color(benchmark):
    env = ForagaxRGBEnv(
        size=1_000,
        aperture_size=15,
        objects=(WALL, FLOWER),
        biomes=(Biome(object_frequencies=(0.05, 0.05)),),
    )
    params = env.default_params
    key = jax.random.key(0)
    key, reset_key = jax.random.split(key)
    _, state = env.reset(reset_key, params)

    @jax.jit
    def _run(state, key):
        def f(carry, _):
            state, key = carry
            key, step_key = jax.random.split(key, 2)
            _, new_state, _, _, _ = env.step(step_key, state, Actions.UP, params)
            return (new_state, key), None

        (final_state, _), _ = jax.lax.scan(f, (state, key), None, length=100)
        return final_state

    key, run_key = jax.random.split(key)
    _run(state, run_key).pos.block_until_ready()

    def benchmark_fn():
        key, run_key = jax.random.split(jax.random.key(1))
        _run(state, run_key).pos.block_until_ready()

    benchmark(benchmark_fn)


def test_benchmark_small_env_world(benchmark):
    env = ForagaxWorldEnv(
        size=1_000,
        objects=(WALL, FLOWER),
        biomes=(Biome(object_frequencies=(0.05, 0.05)),),
    )
    params = env.default_params
    key = jax.random.key(0)
    key, reset_key = jax.random.split(key)
    _, state = env.reset(reset_key, params)

    @jax.jit
    def _run(state, key):
        def f(carry, _):
            state, key = carry
            key, step_key = jax.random.split(key, 2)
            _, new_state, _, _, _ = env.step(step_key, state, Actions.UP, params)
            return (new_state, key), None

        (final_state, _), _ = jax.lax.scan(f, (state, key), None, length=100)
        return final_state

    key, run_key = jax.random.split(key)
    _run(state, run_key).pos.block_until_ready()

    def benchmark_fn():
        key, run_key = jax.random.split(jax.random.key(1))
        _run(state, run_key).pos.block_until_ready()

    benchmark(benchmark_fn)
