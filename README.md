# foragax

Foragax is a lightweight, JAX-first grid-world environment suite for continual / procedural
experiments. It provides a small collection of environment variants, a registry factory for
easy construction, and example scripts for visualization.

This version is a [Gymnax](https://github.com/RobertTLange/gymnax) environment implemented in JAX. The original implementation of Forager (in Numba) is available at [andnp/forager](https://github.com/andnp/forager). In addition to the original features, this implementation includes biomes and visualization.

Key ideas:

- Functional, JAX-friendly API (explicit PRNG keys, immutable env state objects).
- Multiple observation modalities: object, RGB, and color, as well as aperture-based or
	full-world observations.
- Customizable biomes.
- Customizable object placement, respawning, and rewards.
- Visualization via RGB rendering.

## Quickstart

We recommend installing with pip from https://pypi.org/project/continual-foragax/.

```bash
pip install continual-foragax
```

Requires Python 3.8 or newer.

The codebase expects JAX and other numeric dependencies. If you don't have JAX installed, see
the JAX install instructions for your platform; the project `uv.lock` pins compatible versions.

## Minimal example

Use the registry factory to create an environment and run it with JAX-style RNG keys and an
explicit environment state.

```python
from foragax.registry import make
import jax

env = make(
    "ForagaxSquareWaveTwoBiome-v11",
    aperture_size=9,
    observation_type="color",
)

env_params = env.default_params
key = jax.random.key(0)
key, key_reset = jax.random.split(key)
obs, env_state = env.reset(key_reset, env_params)

key, key_act, key_step = jax.random.split(key, 3)
action = env.action_space(env_params).sample(key_act)
obs, env_state, reward, done, info = env.step(key_step, env_state, action, env_params)

frame = env.render(env_state, env_params, render_mode="world")
```

See `examples/observation.py` and `examples/visualize.py` for runnable scripts that save
short videos under `videos/` using Gymnasium helpers.

## Registry and included environments

Use `foragax.registry.make` to construct environments by id. The registered ids are:

- `ForagaxBig-v5` — large multi-biome layout with Fourier-modulated rewards (used by
	`examples/visualize.py`).
- `ForagaxSquareWaveTwoBiome-v11` — two-biome layout with square-wave reward shifts (used
	by `examples/observation.py`).

The `make` factory accepts the following kwargs:

- `observation_type`: one of `"object"`, `"rgb"`, or `"color"` (default `"color"`).
- `aperture_size`: `int`, `(int, int)`, or `-1` for full-world observation. Defaults to
	`(5, 5)`; pass `None` to use the environment's own default.
- `reward_delay`: steps required to digest food items (default `0`).
- `random_shift_max_steps`: random initial offset on the underlying time signal
	(default `0`).
- Additional `**kwargs` are forwarded to the `ForagaxEnv` constructor and override config
	defaults.

## Custom objects and extensions

Object classes in `foragax.objects` define rewards, respawn / regen behavior, and
blocking/collectable flags. The registry presets above are built using two helpers from
this module:

- `create_fourier_objects` — Fourier-modulated reward objects (used by `ForagaxBig-v5`).
- `create_shift_square_wave_biome_objects` — square-wave biome objects (used by
	`ForagaxSquareWaveTwoBiome-v11`).

Weather-driven environments are also supported even though no weather preset is currently
registered: compose `WeatherObject` or `WeatherWaveObject` from `foragax.objects` with
`foragax.weather.get_temperature` (which reads ECA&D temperature data shipped under
`foragax/data/`) to build one programmatically.

To add new object classes, follow the patterns in `foragax.objects` and either register a
new entry in `foragax.registry.ENV_CONFIGS` or construct `ForagaxEnv` directly.

## Design notes

- JAX-first: RNG keys and immutable env state are passed explicitly so environments can be
	stepped inside JIT/pmapped loops if desired.
- Small, composable environment variants are provided through the registry (easy to add more).

## Examples

- `examples/observation.py` — runs a random policy in `ForagaxSquareWaveTwoBiome-v11` and
	saves a video of the color observations to `plots/`.
- `examples/visualize.py` — runs a random policy in `ForagaxBig-v5` and saves periodic
	`world_reward` render videos to `videos/`.

## Development

Run `uv run pytest` from the repo root. The project uses `uv` for package management; `ruff` for formatting and linting.

## Citation

If you use Foragax in your research, please cite:

```bibtex
@misc{tang2026forager,
    title={Forager: a lightweight testbed for continual learning with partial observability in RL},
    author={Steven Tang and Xinze Xiong and Anna Hakhverdyan and Andrew Patterson and Jacob Adkins and Jiamin He and Esraa Elelimy and Parham Mohammad Panahi and Martha White and Adam White},
    year={2026},
    eprint={2605.01131},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2605.01131},
}
```

## Acknowledgments

We acknowledge the data providers in the ECA&D project. Klein Tank, A.M.G. and
Coauthors, 2002. Daily dataset of 20th-century surface air temperature and
precipitation series for the European Climate Assessment. Int. J. of Climatol.,
22, 1441-1453.

Data and metadata available at https://www.ecad.eu
