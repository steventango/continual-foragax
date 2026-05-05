import jax

from foragax import registry
from foragax.env import Actions


def test_benchmark_big_v5_vmap(benchmark):
    num_envs = 128
    env = registry.make("ForagaxBig-v5")
    params = env.default_params
    key = jax.random.key(0)
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

        (final_state, _), _ = jax.lax.scan(f, (states, key), None, length=1000)
        return final_state

    key, run_key = jax.random.split(key)
    _run(states, run_key).pos.block_until_ready()

    def benchmark_fn():
        key, run_key = jax.random.split(jax.random.key(1))
        _run(states, run_key).pos.block_until_ready()

    benchmark(benchmark_fn)


def test_benchmark_square_wave_two_biome_v11_vmap(benchmark):
    num_envs = 128
    env = registry.make("ForagaxSquareWaveTwoBiome-v11")
    params = env.default_params
    key = jax.random.key(0)
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

        (final_state, _), _ = jax.lax.scan(f, (states, key), None, length=1000)
        return final_state

    key, run_key = jax.random.split(key)
    _run(states, run_key).pos.block_until_ready()

    def benchmark_fn():
        key, run_key = jax.random.split(jax.random.key(1))
        _run(states, run_key).pos.block_until_ready()

    benchmark(benchmark_fn)
