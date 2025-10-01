import chex
import jax
import jax.numpy as jnp

from foragax.env import Actions, Biome, ForagaxObjectEnv, ForagaxRGBEnv, ForagaxWorldEnv
from foragax.objects import FLOWER, WALL


def test_benchmark_vision(benchmark):
    env = ForagaxObjectEnv(size=7, aperture_size=3, objects=(WALL,))
    params = env.default_params
    key = jax.random.key(0)
    _, state = env.reset(key, params)

    grid = jnp.zeros((7, 7), dtype=int)
    grid = grid.at[4, 3].set(1)
    grid = grid.at[5, 3].set(1)
    grid = grid.at[2, 0].set(1)
    state = state.replace(object_grid=grid)

    @jax.jit
    def _run(state, key):
        key, step_key = jax.random.split(key)
        obs, new_state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
        return obs, new_state

    # warm-up
    obs, new_state = _run(state, key)

    expected = jnp.zeros((3, 3, 1), dtype=int)
    expected = expected.at[2, 1, 0].set(1)

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
            _, new_state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
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
            _, new_state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
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
                step_keys, states, Actions.DOWN, params
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
            _, new_state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
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
            _, new_state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
            return (new_state, key), None

        (final_state, _), _ = jax.lax.scan(f, (state, key), None, length=100)
        return final_state

    key, run_key = jax.random.split(key)
    _run(state, run_key).pos.block_until_ready()

    def benchmark_fn():
        key, run_key = jax.random.split(jax.random.key(1))
        _run(state, run_key).pos.block_until_ready()

    benchmark(benchmark_fn)
