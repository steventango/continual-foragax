# foragax

Foragax is a lightweight, JAX-first grid-world environment suite for continual / procedural
experiments. It provides a small collection of environment variants (weather, multi-biome,
etc.), a registry factory for easy construction, and simple example scripts for plotting and
visualization.

This version is a [Gymnax](https://github.com/RobertTLange/gymnax) environment implemented in JAX. The original implementation of Forager is implemented in Numba is available at [andnp/forager](https://github.com/andnp/forager). In addition to the original features, this implementation includes: biomes, visualization, and a weather environment.

Key ideas:

- Functional, JAX-friendly API (explicit PRNG keys, immutable env state objects).
- Multiple observation modalities: Object and RGB, as well as aperture based or
	full-world observations.
- Customizable biomes
- Customizable object placement, respawning, and rewards.
- Visualization via RGB rendering and plotting.

## Quickstart

We recommend installing with pip from https://pypi.org/project/continual-foragax/.

```bash
pip install continual-foragax
```

We support Python 3.8 through Python 3.13.

The codebase expects JAX and other numeric dependencies. If you don't have JAX installed, see
the JAX install instructions for your platform; the project `uv.lock` pins compatible versions.

## Minimal example (from examples)

Use the registry factory to create an environment and run it with JAX-style RNG keys and an
explicit environment state.

```python
from foragax.registry import make
import jax

# create env (observation_type is one of: 'object', 'rgb', 'world')
env = make(
		"ForagaxWeather-v1",
		aperture_size=5,
		observation_type="object",
)

# environment parameters and RNG
env_params = env.default_params
key = jax.random.key(0)
key, key_reset = jax.random.split(key)

# reset returns (obs, env_state)
_, env_state = env.reset(key_reset, env_params)

# sampling an action and stepping (functional-style)
key, key_act, key_step = jax.random.split(key, 3)
action = env.action_space(env_params).sample(key_act)
_, next_env_state, reward, done, info = env.step(key_step, env_state, action, env_params)

# rendering supports multiple modes: 'world' and 'aperture'
frame = env.render(env_state, env_params, render_mode="aperture")
```

See `examples/plot.py` and `examples/visualize.py` for runnable scripts that produce a sample
plot and saved videos using Gym/Gymnasium helpers.

## Registry and included environments

Use `foragax.registry.make` to construct environments by id. Example environment ids include:

- `ForagaxTwoBiomeSmall-v1` / `-v2` — hand-crafted small multi-biome layouts
- `ForagaxWeather-v1` — small weather-driven two-biome environment used by examples

The `make` factory accepts the following notable kwargs:

- `observation_type`: one of `"object"`, `"rgb"`, or `"world"`.
- `aperture_size`: integer or tuple controlling the agent's local observation aperture.
- `file_index`: used to pick weather locations.

## Custom objects and extensions

The codebase includes an object system for placing items into biomes and controlling
behaviour (rewards, respawn / regen behavior, blocking/collectable flags). See
`foragax.objects` for the canonical object definitions and helpers like
`create_weather_objects` used by the registry.

If you want to add new object classes, follow the examples in `foragax.objects` and add the
class into registry configs or construct environments programmatically.

## Design notes

- JAX-first: RNG keys and immutable env state are passed explicitly so environments can be
	stepped inside JIT/pmapped loops if desired.
- Small, composable environment variants are provided through the registry (easy to add more).

## Examples

- `examples/plot.py` — runs a short random policy in `ForagaxWeather-v1` and produces a
	temperature vs reward plot (saves to `plots/sample_plot.png`).
- `examples/visualize.py` — runs environments at multiple aperture sizes and saves short
	videos under `videos/` using `save_video`.

## Development

Run unit tests via pytest.

## Acknowledgments

We acknowledge the data providers in the ECA&D project. Klein Tank, A.M.G. and
Coauthors, 2002. Daily dataset of 20th-century surface air temperature and
precipitation series for the European Climate Assessment. Int. J. of Climatol.,
22, 1441-1453.

Data and metadata available at https://www.ecad.eu
