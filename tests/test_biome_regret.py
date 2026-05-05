import jax
import jax.numpy as jnp
from foragax.registry import make


def test_no_negative_regret():
    """Verify that biome_regret is never negative, even if all regions have negative means."""
    print("Running negative regret safety test...")
    env = make("ForagaxBig-v5")
    params = env.default_params
    key = jax.random.key(0)

    # Reset
    obs, state = env.reset(key, params)

    # Manually check that regret >= 0 for many random positions
    def check_non_negative(k, s):
        _, _, _, _, info = env.step(k, s, 0, params)
        return info["biome_regret"]

    keys = jax.random.split(key, 100)
    for i in range(100):
        # Random position
        pos = jax.random.randint(keys[i], (2,), 0, jnp.array(env.size))
        state_random = state.replace(pos=pos)
        regret = check_non_negative(keys[i], state_random)
        assert regret >= -1e-7, f"Negative regret detected: {regret} at pos {pos}"

    print("Negative regret safety test PASSED!")
