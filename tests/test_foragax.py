import chex
import jax
import jax.numpy as jnp

from foragax.env import (
    Actions,
    Biome,
    ForagaxEnv,
)
from foragax.objects import (
    BROWN_MOREL_UNIFORM,
    BROWN_OYSTER_UNIFORM,
    FLOWER,
    GREEN_DEATHCAP_UNIFORM,
    GREEN_FAKE_UNIFORM,
    LARGE_MOREL,
    MEDIUM_MOREL,
    MOREL,
    OYSTER,
    THORNS,
    WALL,
    DefaultForagaxObject,
    NormalRegenForagaxObject,
    WeatherObject,
)


def test_observation_shape():
    """Test that the observation shape is correct."""
    env = ForagaxEnv(
        size=(500, 500),
        aperture_size=(9, 9),
        objects=(WALL, FLOWER, THORNS),
        observation_type="color",
    )
    params = env.default_params
    assert env.observation_space(params).shape == (9, 9, 3)


def test_gymnax_api():
    key = jax.random.key(0)
    key, key_reset, key_act, key_step = jax.random.split(key, 4)

    env = ForagaxEnv(size=(5, 5), observation_type="color")
    env_params = env.default_params

    obs, state = env.reset(key_reset, env_params)

    action = env.action_space(env_params).sample(key_act)

    # Perform the step transition.
    n_obs, n_state, reward, done, _ = env.step(key_step, state, action, env_params)


def test_sizes():
    # can specify sizes with integers
    env = ForagaxEnv(size=8, aperture_size=3, observation_type="color")
    params = env.default_params
    key = jax.random.key(0)
    obs, state = env.reset(key, params)
    assert jnp.array_equal(state.pos, jnp.array([4, 4]))
    assert env.size == (8, 8)
    assert env.aperture_size == (3, 3)
    chex.assert_shape(obs, (3, 3, 0))


def test_uneven_sizes():
    # can specify sizes as uneven tuples
    env = ForagaxEnv(size=(10, 5), aperture_size=(5, 1), observation_type="color")
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
    env = ForagaxEnv(
        size=size,
        objects=(FLOWER,),
        biomes=(Biome(object_frequencies=(freq,)),),
        observation_type="color",
    )
    params = env.default_params
    key = jax.random.key(0)
    obs, state = env.reset(key, params)

    empirical_freq = jnp.count_nonzero(state.object_grid) / size**2
    chex.assert_trees_all_close(empirical_freq, freq, rtol=0.1)
    chex.assert_shape(obs, (5, 5, 1))


def test_object_observation_mode():
    env = ForagaxEnv(
        size=(10, 10),
        aperture_size=(5, 5),
        objects=(WALL, FLOWER),
        observation_type="color",
    )
    params = env.default_params
    key = jax.random.key(0)
    obs, state = env.reset(key, params)

    assert obs.shape == (5, 5, 2)


def test_rgb_observation_mode():
    env = ForagaxEnv(
        size=(10, 10),
        aperture_size=(5, 5),
        objects=(WALL, FLOWER),
        observation_type="rgb",
    )
    params = env.default_params
    key = jax.random.key(0)
    obs, state = env.reset(key, params)

    assert obs.shape == (5, 5, 3)


def test_object_observation_mode_large_aperture():
    env = ForagaxEnv(
        size=(10, 10),
        aperture_size=(20, 20),
        objects=(WALL, FLOWER),
        observation_type="color",
    )
    params = env.default_params
    key = jax.random.key(0)
    obs, state = env.reset(key, params)

    assert obs.shape == (20, 20, 2)


def test_rgb_observation_mode_large_aperture():
    env = ForagaxEnv(
        size=(10, 10),
        aperture_size=(20, 20),
        objects=(WALL, FLOWER),
        observation_type="rgb",
    )
    params = env.default_params
    key = jax.random.key(0)
    obs, state = env.reset(key, params)

    assert obs.shape == (20, 20, 3)


def test_world_observation_mode():
    # can use world observation mode
    env = ForagaxEnv(
        size=(10, 10),
        aperture_size=-1,
        objects=(WALL, FLOWER),
        biomes=(Biome(object_frequencies=(0.1, 0.1)),),
        observation_type="object",
    )
    params = env.default_params
    key = jax.random.key(0)
    obs, state = env.reset(key, params)

    assert obs.shape == (10, 10, 3)
    # Check agent channel (last channel) is 0 except 1 at agent position
    agent_channel = obs[:, :, 2]
    assert jnp.sum(agent_channel) == 1
    assert agent_channel[state.pos[1], state.pos[0]] == 1

    # check non-zero values in other channels
    assert jnp.sum(obs[:, :, 0]) > 0
    assert jnp.sum(obs[:, :, 1]) > 0


def test_basic_movement():
    """Test agent movement and collision with walls."""
    key = jax.random.key(0)

    biome = Biome(
        object_frequencies=(1.0,),
        start=(3, 4),
        stop=(4, 6),
    )
    env = ForagaxEnv(
        size=7,
        objects=(WALL,),
        biomes=(biome,),
        observation_type="color",
    )
    params = env.default_params
    _, state = env.reset(key, params)

    assert jnp.array_equal(state.pos, jnp.array([3, 3]))

    # stays still when bumping into a wall
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([3, 3]))

    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)
    assert jnp.array_equal(state.pos, jnp.array([4, 3]))

    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)
    assert jnp.array_equal(state.pos, jnp.array([3, 3]))

    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([3, 2]))


def test_vision():
    """Test the agent's observation."""
    key = jax.random.key(0)
    object_types = (WALL,)
    env = ForagaxEnv(
        size=(7, 7),
        aperture_size=(3, 3),
        objects=object_types,
        observation_type="color",
    )
    params = env.default_params
    obs, state = env.reset(key, params)

    wall_id = 1  # 0 is EMPTY

    # Create a predictable environment
    grid = jnp.zeros((7, 7), dtype=int)
    grid = grid.at[4, 3].set(wall_id)
    grid = grid.at[5, 3].set(wall_id)
    grid = grid.at[2, 0].set(wall_id)
    state = state.replace(object_grid=grid)

    chex.assert_trees_all_equal(state.pos, jnp.array([3, 3]))

    # No movement
    key, step_key = jax.random.split(key)
    obs, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)

    expected = jnp.zeros((3, 3, 1), dtype=int)
    expected = expected.at[2, 1, 0].set(1)

    chex.assert_trees_all_equal(state.pos, jnp.array([3, 3]))
    chex.assert_trees_all_equal(obs, expected)

    # Move right
    key, step_key = jax.random.split(key)
    obs, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)
    key, step_key = jax.random.split(key)
    obs, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    expected = jnp.zeros((3, 3, 1), dtype=int)
    expected = expected.at[1, 0, 0].set(1)
    expected = expected.at[2, 0, 0].set(1)

    chex.assert_trees_all_equal(state.pos, jnp.array([4, 4]))
    chex.assert_trees_all_equal(obs, expected)


def test_respawn():
    """Test object respawning."""
    key = jax.random.key(0)
    object_types = (FLOWER,)
    env = ForagaxEnv(
        size=7,
        aperture_size=3,
        objects=object_types,
        observation_type="color",
    )
    params = env.default_params
    _, state = env.reset(key, params)

    flower_id = 1  # 0 is EMPTY

    # Place a flower and move the agent to it
    grid = jnp.zeros((7, 7), dtype=int)
    grid = grid.at[4, 3].set(flower_id)
    state = state.replace(object_grid=grid, pos=jnp.array([3, 3]))

    # Collect the flower
    key, step_key = jax.random.split(key)
    _, state, reward, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert reward == FLOWER.reward_val
    assert state.object_grid[4, 3] < 0

    steps_until_respawn = -state.object_grid[4, 3] // 2

    # Step until it respawns
    for i in range(steps_until_respawn):
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
        assert state.object_grid[4, 3] < 0

    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert state.object_grid[4, 3] == flower_id


def test_random_respawn():
    """Test that an object respawns at a random empty location within its biome."""
    key = jax.random.key(0)

    flower_random = DefaultForagaxObject(
        name="flower",
        reward=1.0,
        collectable=True,
        color=(0, 255, 0),
        random_respawn=True,
    )

    object_types = (flower_random, WALL)
    biome = Biome(start=(2, 2), stop=(5, 5), object_frequencies=(0.0, 0.0))
    env = ForagaxEnv(
        size=7,
        aperture_size=3,
        objects=object_types,
        biomes=(biome,),
        observation_type="color",
    )
    params = env.default_params
    _, state = env.reset(key, params)

    flower_id = 1  # 0 is EMPTY
    original_pos = jnp.array([3, 3])

    # Place a flower and move the agent to it
    grid = jnp.zeros((7, 7), dtype=int)
    grid = grid.at[original_pos[1], original_pos[0]].set(flower_id)
    # Add a wall to make sure it doesn't spawn there
    grid = grid.at[4, 4].set(2)  # Use a fixed ID for the wall
    state = state.replace(object_grid=grid, pos=jnp.array([2, 3]))

    # Collect the flower
    key, step_key = jax.random.split(key)
    _, new_state, reward, _, _ = env.step(step_key, state, Actions.RIGHT, params)

    assert reward == flower_random.reward_val
    # Original position should be empty
    assert new_state.object_grid[original_pos[1], original_pos[0]] == 0

    # A timer should be placed somewhere
    assert jnp.sum(new_state.object_grid < 0) == 1
    timer_pos_flat = jnp.argmin(new_state.object_grid)
    timer_pos = jnp.array(jnp.unravel_index(timer_pos_flat, (7, 7)))
    # New position should not be the original position
    assert not jnp.array_equal(timer_pos, original_pos)

    # New position should be within the biome
    assert jnp.all(timer_pos >= jnp.array(biome.start))
    assert jnp.all(timer_pos < jnp.array(biome.stop))

    # New position should be on an empty cell (not the wall)
    assert not jnp.array_equal(timer_pos, jnp.array([4, 4]))


def test_random_respawn_no_empty_space():
    """Test that an object respawns at the same spot if no empty space is available."""
    key = jax.random.key(0)

    flower_random = DefaultForagaxObject(
        name="flower",
        reward=1.0,
        collectable=True,
        color=(0, 255, 0),
        random_respawn=True,
    )

    object_types = (WALL, flower_random)
    # A 1x1 biome
    biome = Biome(start=(3, 3), stop=(4, 4), object_frequencies=(0.0, 0.0))
    env = ForagaxEnv(
        size=7,
        objects=object_types,
        biomes=(biome,),
        observation_type="color",
    )
    params = env.default_params
    _, state = env.reset(key, params)

    flower_id = 2  # WALL is 1
    original_pos = jnp.array([3, 3])

    # Place a flower in the 1x1 biome
    grid = jnp.zeros((7, 7), dtype=int)
    grid = grid.at[original_pos[1], original_pos[0]].set(flower_id)
    state = state.replace(object_grid=grid, pos=jnp.array([2, 3]))

    # Collect the flower
    key, step_key = jax.random.split(key)
    _, new_state, reward, _, _ = env.step(step_key, state, Actions.RIGHT, params)

    assert reward == flower_random.reward_val
    # The timer should be placed back at the original position
    assert new_state.object_grid[original_pos[1], original_pos[0]] < 0


def test_wrapping_dynamics():
    """Test that the agent wraps around the environment boundaries."""
    key = jax.random.key(0)
    env = ForagaxEnv(size=(5, 5), objects=(), observation_type="color")
    params = env.default_params
    _, state = env.reset(key, params)

    # Go up
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 3]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 4]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 0]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 1]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))

    # Go down
    _, state = env.reset(key, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 2]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 1]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 0]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 4]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 3]))
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
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


def test_no_wrapping_dynamics():
    """Test that the agent does not wrap around the environment boundaries when nowrap=True."""
    key = jax.random.key(0)
    env = ForagaxEnv(size=(5, 5), objects=(), nowrap=True, observation_type="color")
    params = env.default_params
    _, state = env.reset(key, params)

    # Agent starts at center (2,2)

    # Move to top edge
    for _ in range(2):
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 0]))  # at top

    # Try to move up again, should stay put
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.UP, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 0]))  # still at top

    # Move to bottom edge
    _, state = env.reset(key, params)
    for _ in range(2):
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 4]))  # at bottom

    # Try to move down again, should stay put
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert jnp.array_equal(state.pos, jnp.array([2, 4]))  # still at bottom

    # Move to left edge
    _, state = env.reset(key, params)
    for _ in range(2):
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)
    assert jnp.array_equal(state.pos, jnp.array([0, 2]))  # at left

    # Try to move left again, should stay put
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)
    assert jnp.array_equal(state.pos, jnp.array([0, 2]))  # still at left

    # Move to right edge
    _, state = env.reset(key, params)
    for _ in range(2):
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)
    assert jnp.array_equal(state.pos, jnp.array([4, 2]))  # at right

    # Try to move right again, should stay put
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)
    assert jnp.array_equal(state.pos, jnp.array([4, 2]))  # still at right


def test_wrapping_vision():
    """Test that the agent's vision wraps around the environment boundaries."""
    key = jax.random.key(0)
    env = ForagaxEnv(
        size=(5, 5), aperture_size=(3, 3), objects=(FLOWER,), observation_type="color"
    )
    params = env.default_params
    obs, state = env.reset(key, params)

    # Create a predictable environment with a flower at (0, 0)
    grid = jnp.zeros((5, 5), dtype=int)
    grid = grid.at[0, 0].set(1)
    state = state.replace(object_grid=grid)

    obs = env.get_obs(state, params)

    expected = jnp.zeros((3, 3, 1), dtype=int)
    assert jnp.array_equal(obs, expected)

    # go left
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)

    # go down
    key, step_key = jax.random.split(key)
    obs, state, _, _, _ = env.step(step_key, state, Actions.UP, params)

    expected = jnp.zeros((3, 3, 1), dtype=int)
    expected = expected.at[0, 0, 0].set(1)

    assert jnp.array_equal(state.pos, jnp.array([1, 1]))
    assert jnp.array_equal(obs, expected)

    # go left , go left
    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)
    key, step_key = jax.random.split(key)
    obs, state, _, _, _ = env.step(step_key, state, Actions.LEFT, params)

    expected = jnp.zeros((3, 3, 1), dtype=int)
    expected = expected.at[0, 2, 0].set(1)

    assert jnp.array_equal(state.pos, jnp.array([4, 1]))
    assert jnp.array_equal(obs, expected)


def test_no_wrapping_vision():
    """Test that the agent's vision does not wrap around boundaries when nowrap=True."""
    key = jax.random.key(0)
    object_types = (FLOWER,)
    env_no_wrap = ForagaxEnv(
        size=(7, 7),
        aperture_size=(3, 3),
        objects=object_types,
        nowrap=True,
        observation_type="color",
    )
    params = env_no_wrap.default_params

    # Place a flower at the opposite corner (6,6)
    grid = jnp.zeros((7, 7), dtype=int)
    grid = grid.at[6, 6].set(1)  # FLOWER

    # Agent at (0,0)
    state = env_no_wrap.reset(key, params)[1]
    state = state.replace(object_grid=grid, pos=jnp.array([0, 0]))

    # With no wrapping, should not see the flower, see padding
    obs_no_wrap = env_no_wrap.get_obs(state, params)
    assert env_no_wrap.num_color_channels == 2  # Flower + padding
    # Check that padding channel is activated for out of bound positions
    padding_mask = jnp.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    )
    assert jnp.all(obs_no_wrap[padding_mask, 1] == 1)
    # And flower not visible
    assert jnp.all(obs_no_wrap[:, :, 0] == 0)


def test_generate_objects_in_biome():
    """Test generating objects within a specific biome area."""
    object_types = (WALL, FLOWER, THORNS, MOREL, OYSTER)
    env = ForagaxEnv(
        size=(10, 10),
        objects=object_types,
        biomes=(
            Biome(
                object_frequencies=(0.0, 0.0, 0.0, 0.1, 0.0),
                start=(2, 2),
                stop=(6, 6),
            ),
        ),
        observation_type="color",
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


def test_deterministic_object_spawning():
    """Test deterministic object spawning with fixed counts and shuffled positions."""
    object_types = (WALL, FLOWER)
    env = ForagaxEnv(
        size=(10, 10),
        objects=object_types,
        biomes=(
            Biome(
                object_frequencies=(0.1, 0.1),
                start=(2, 2),
                stop=(6, 6),  # 4x4 = 16 cells
            ),
        ),
        deterministic_spawn=True,
        observation_type="color",
    )
    key = jax.random.key(0)
    params = env.default_params

    _, state = env.reset_env(key, params)

    wall_id = object_types.index(WALL) + 1
    flower_id = object_types.index(FLOWER) + 1

    # Check exact counts: 16 * 0.1 = 1.6, rounded to 2 each
    wall_count = jnp.sum(state.object_grid == wall_id)
    flower_count = jnp.sum(state.object_grid == flower_id)
    assert wall_count == 2
    assert flower_count == 2

    # Check that objects only appear within the biome
    wall_locations = jnp.argwhere(state.object_grid == wall_id)
    flower_locations = jnp.argwhere(state.object_grid == flower_id)

    assert jnp.all(wall_locations >= 2)
    assert jnp.all(wall_locations < 6)
    assert jnp.all(flower_locations >= 2)
    assert jnp.all(flower_locations < 6)

    # Test different positions: different key should produce different positions but same counts
    key_1 = jax.random.key(1)
    _, state_1 = env.reset(key_1, params)
    assert jnp.sum(state_1.object_grid == wall_id) == 2
    assert jnp.sum(state_1.object_grid == flower_id) == 2
    # Positions may be the same due to implementation (not shuffled)
    assert not jnp.array_equal(state.object_grid, state_1.object_grid)


def test_complex_deterministic_object_spawning():
    """Test a more complex deterministic object spawning case with two biomes."""
    aperture_size = (5, 5)
    objects = (
        BROWN_MOREL_UNIFORM,
        BROWN_OYSTER_UNIFORM,
        GREEN_DEATHCAP_UNIFORM,
        GREEN_FAKE_UNIFORM,
    )
    margin = aperture_size[1] // 2 + 1
    width = 2 * margin + 9
    config = {
        "size": (width, 15),
        "objects": objects,
        "biomes": (
            # Morel biome
            Biome(
                start=(margin, 0),
                stop=(margin + 2, 15),
                object_frequencies=(0.25, 0.0, 0.5, 0.0),
            ),
            # Oyster biome
            Biome(
                start=(margin + 7, 0),
                stop=(margin + 9, 15),
                object_frequencies=(0.0, 0.25, 0.0, 0.5),
            ),
        ),
        "deterministic_spawn": True,
        "aperture_size": aperture_size,
        "observation_type": "color",
    }

    env = ForagaxEnv(**config)
    params = env.default_params
    key = jax.random.key(0)

    _, state = env.reset_env(key, params)

    morel_id = objects.index(BROWN_MOREL_UNIFORM) + 1
    oyster_id = objects.index(BROWN_OYSTER_UNIFORM) + 1
    deathcap_id = objects.index(GREEN_DEATHCAP_UNIFORM) + 1
    fake_id = objects.index(GREEN_FAKE_UNIFORM) + 1

    # Biome 1 area: (2 * 15) = 30 cells. 30 * 0.25 = 7.5 -> 8 morels. 30 * 0.5 = 15 deathcaps.
    # Biome 2 area: (2 * 15) = 30 cells. 30 * 0.25 = 7.5 -> 8 oysters. 30 * 0.5 = 15 fakes.
    assert jnp.sum(state.object_grid == morel_id) == 8
    assert jnp.sum(state.object_grid == oyster_id) == 8
    assert jnp.sum(state.object_grid == deathcap_id) == 15
    assert jnp.sum(state.object_grid == fake_id) == 15

    # Check that objects are within their biomes
    morel_locs = jnp.argwhere(state.object_grid == morel_id)
    oyster_locs = jnp.argwhere(state.object_grid == oyster_id)
    deathcap_locs = jnp.argwhere(state.object_grid == deathcap_id)
    fake_locs = jnp.argwhere(state.object_grid == fake_id)

    # Morel biome checks
    assert jnp.all(morel_locs[:, 1] >= margin)
    assert jnp.all(morel_locs[:, 1] < margin + 2)
    assert jnp.all(deathcap_locs[:, 1] >= margin)
    assert jnp.all(deathcap_locs[:, 1] < margin + 2)

    # Oyster biome checks
    assert jnp.all(oyster_locs[:, 1] >= margin + 7)
    assert jnp.all(oyster_locs[:, 1] < margin + 9)
    assert jnp.all(fake_locs[:, 1] >= margin + 7)
    assert jnp.all(fake_locs[:, 1] < margin + 9)


def test_color_based_partial_observability():
    """Test that objects with the same color are grouped into the same observation channel."""
    env = ForagaxEnv(
        size=(10, 10),
        aperture_size=(5, 5),
        objects=(MOREL, LARGE_MOREL, MEDIUM_MOREL, FLOWER),
        observation_type="color",
    )
    params = env.default_params
    key = jax.random.key(0)

    # Check that morels (all same color) use 1 channel, plus flower = 2 total
    assert env.num_color_channels == 2

    # Create a state with different morels and a flower
    state = env.reset(key, params)[1]

    # Manually place objects
    grid = jnp.zeros((10, 10), dtype=int)
    grid = grid.at[5, 5].set(1)  # MOREL
    grid = grid.at[5, 6].set(2)  # LARGE_MOREL
    grid = grid.at[5, 7].set(3)  # MEDIUM_MOREL
    grid = grid.at[6, 5].set(4)  # FLOWER
    state = state.replace(object_grid=grid)

    obs = env.get_obs(state, params)

    # All morels should activate the same channel (channel 0 for brown)
    # Flower should activate channel 1 (green)
    center_obs = obs[2, 2, :]  # MOREL at center
    morel_obs = obs[2, 3, :]  # LARGE_MOREL
    med_morel_obs = obs[2, 4, :]  # MEDIUM_MOREL
    flower_obs = obs[3, 2, :]  # FLOWER (flipped coordinates)

    # All morels should have the same observation (channel 0 activated)
    chex.assert_trees_all_equal(center_obs, jnp.array([1.0, 0.0]))
    chex.assert_trees_all_equal(center_obs, morel_obs)
    chex.assert_trees_all_equal(center_obs, med_morel_obs)

    # Flower should have different observation (channel 1 activated)
    chex.assert_trees_all_equal(flower_obs, jnp.array([0.0, 1.0]))


def test_color_channel_mapping():
    """Test that the color channel mapping is correct."""
    env = ForagaxEnv(
        size=(10, 10),
        aperture_size=(3, 3),
        objects=(WALL, FLOWER, THORNS),
        observation_type="color",
    )

    # WALL (gray), FLOWER (green), THORNS (red) - all different colors
    assert env.num_color_channels == 3

    # Check that unique colors are correctly identified in order of first appearance
    expected_colors = jnp.array(
        [
            [127, 127, 127],  # gray (WALL) - appears first
            [0, 255, 0],  # green (FLOWER) - appears second
            [255, 0, 0],  # red (THORNS) - appears third
        ]
    )
    chex.assert_trees_all_equal(env.unique_colors, expected_colors)


def test_same_color_objects_same_channel():
    """Test that objects with identical colors produce identical observations."""
    obj1 = DefaultForagaxObject(name="obj1", color=(100, 50, 25))
    obj2 = DefaultForagaxObject(name="obj2", color=(100, 50, 25))  # Same color
    obj3 = DefaultForagaxObject(name="obj3", color=(200, 100, 50))  # Different color

    env = ForagaxEnv(
        size=(7, 7),
        aperture_size=(3, 3),
        objects=(obj1, obj2, obj3),
        observation_type="color",
    )
    params = env.default_params

    # Should have 2 color channels
    assert env.num_color_channels == 2

    # Create test state
    key = jax.random.key(0)
    state = env.reset(key, params)[1]

    # Place objects
    grid = jnp.zeros((7, 7), dtype=int)
    grid = grid.at[3, 3].set(1)  # obj1
    grid = grid.at[3, 4].set(2)  # obj2 (same color as obj1)
    grid = grid.at[4, 3].set(3)  # obj3 (different color)
    state = state.replace(object_grid=grid)

    obs = env.get_obs(state, params)

    # obj1 and obj2 should have identical observations
    obj1_obs = obs[1, 1, :]  # Center
    obj2_obs = obs[1, 2, :]  # Right of center
    obj3_obs = obs[2, 1, :]  # Below center

    chex.assert_trees_all_equal(obj1_obs, obj2_obs)
    assert not jnp.allclose(obj1_obs, obj3_obs)


def test_empty_environment_observation():
    """Test observation shape when no objects are present."""
    env = ForagaxEnv(
        size=(5, 5),
        aperture_size=(3, 3),
        objects=(),  # No objects
        observation_type="color",
    )
    params = env.default_params
    key = jax.random.key(0)
    obs, state = env.reset(key, params)

    # Should have 0 color channels
    assert env.num_color_channels == 0
    assert obs.shape == (3, 3, 0)


def test_single_color_all_objects():
    """Test when all objects have the same color."""
    # Create multiple objects with same color
    from foragax.objects import DefaultForagaxObject

    obj1 = DefaultForagaxObject(name="obj1", color=(50, 100, 150))
    obj2 = DefaultForagaxObject(name="obj2", color=(50, 100, 150))
    obj3 = DefaultForagaxObject(name="obj3", color=(50, 100, 150))

    env = ForagaxEnv(
        size=(7, 7),
        aperture_size=(3, 3),
        objects=(obj1, obj2, obj3),
        observation_type="color",
    )
    params = env.default_params

    # Should have 1 color channel
    assert env.num_color_channels == 1

    # Create test state
    key = jax.random.key(0)
    state = env.reset(key, params)[1]

    # Place all objects
    grid = jnp.zeros((7, 7), dtype=int)
    grid = grid.at[3, 2].set(1)  # obj1
    grid = grid.at[3, 3].set(2)  # obj2
    grid = grid.at[3, 4].set(3)  # obj3
    state = state.replace(object_grid=grid)

    obs = env.get_obs(state, params)

    # All objects should activate the same channel
    for i in range(3):
        obj_obs = obs[1, i, :]  # Row 1, columns 0-2
        chex.assert_trees_all_equal(obj_obs, jnp.array([1.0]))


def test_teleporting():
    """Test automatic teleporting to the furthest biome center from current position."""
    key = jax.random.key(0)

    # Create environment with two biomes and teleport every 5 steps
    env = ForagaxEnv(
        size=(10, 10),
        aperture_size=(3, 3),
        objects=(WALL,),
        biomes=(
            Biome(
                start=(1, 1), stop=(5, 5), object_frequencies=(0.0,)
            ),  # Biome 0 center at (2,2)
            Biome(
                start=(6, 6), stop=(10, 10), object_frequencies=(0.0,)
            ),  # Biome 1 center at (7,7)
        ),
        teleport_interval=5,
        nowrap=True,
        observation_type="color",
    )
    params = env.default_params

    # Reset and get initial state
    obs, state = env.reset_env(key, params)

    # Agent should start at center (5, 5), which is not in either biome initially
    # But let's manually place it in biome 0 for testing
    state = state.replace(pos=jnp.array([2, 2]))  # In biome 0

    # Step 4 times (time will be 0,1,2,3,4 after these steps)
    for i in range(4):
        key, step_key = jax.random.split(key)
        obs, state, _, _, _ = env.step_env(step_key, state, Actions.LEFT, params)

    # After 4 steps, time=4, next step should teleport (4+1) % 5 == 0
    key, step_key = jax.random.split(key)
    obs, state, _, _, _ = env.step_env(step_key, state, Actions.LEFT, params)

    # Should have teleported to the furthest biome center (biome 1 center)
    # From (2,2), (7,7) is further than (2,2)
    expected_pos = jnp.array([7, 7])
    chex.assert_trees_all_equal(state.pos, expected_pos)

    # Step another 4 times to reach time=9, then teleport back
    # From [7,7], move right to stay in biome: [8,7], [9,7], then stay
    for i in range(4):
        key, step_key = jax.random.split(key)
        obs, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)

    # After another 4 steps, time=9, next step should teleport (9+1) % 5 == 0
    key, step_key = jax.random.split(key)
    obs, state, _, _, _ = env.step(step_key, state, Actions.RIGHT, params)

    # Should teleport to the furthest biome center (biome 0 center)
    # From (7,7), (2,2) is further than (7,7)
    expected_pos = jnp.array([2, 2])
    chex.assert_trees_all_equal(state.pos, expected_pos)


def test_info_discount():
    """Test that info contains discount."""
    key = jax.random.key(0)
    env = ForagaxEnv(size=(5, 5), objects=(), observation_type="color")
    params = env.default_params
    obs, state = env.reset(key, params)

    key, step_key = jax.random.split(key)
    _, _, _, _, info = env.step(step_key, state, Actions.UP, params)

    assert "discount" in info
    assert info["discount"] == 1.0


def test_info_temperature():
    """Test that info contains temperature when weather object is present."""
    key = jax.random.key(0)
    # Create a simple weather object
    weather_obj = WeatherObject(
        name="hot",
        rewards=jnp.array([10.0, 20.0]),
        repeat=1,
        multiplier=1.0,
    )

    env = ForagaxEnv(
        size=(5, 5),
        objects=(weather_obj,),
        biomes=(Biome(object_frequencies=(0.1,)),),
        observation_type="color",
    )
    params = env.default_params
    obs, state = env.reset(key, params)

    key, step_key = jax.random.split(key)
    _, state, _, _, info = env.step(step_key, state, Actions.UP, params)

    assert "temperatures" in info
    # Temperatures should be an array with temperature at index 1 (object ID 1 for the weather object)
    assert len(info["temperatures"]) == 2  # EMPTY + 1 weather object
    assert info["temperatures"][0] == 0.0  # EMPTY
    assert info["temperatures"][1] == 10.0  # weather object at index 1

    key, step_key = jax.random.split(key)
    _, state, _, _, info = env.step(step_key, state, Actions.UP, params)
    assert info["temperatures"][1] == 20.0  # Next temperature value


def test_info_multiple_weather_objects():
    """Test that info contains temperatures for multiple weather objects."""
    key = jax.random.key(0)
    # Create two weather objects
    hot_obj = WeatherObject(
        name="hot",
        rewards=jnp.array([10.0, 20.0]),
        repeat=1,
        multiplier=1.0,
    )
    cold_obj = WeatherObject(
        name="cold",
        rewards=jnp.array([5.0, 15.0]),
        repeat=1,
        multiplier=-1.0,
    )

    env = ForagaxEnv(
        size=(5, 5),
        objects=(hot_obj, cold_obj),
        biomes=(Biome(object_frequencies=(0.1, 0.1)),),
        observation_type="color",
    )
    params = env.default_params
    obs, state = env.reset(key, params)

    key, step_key = jax.random.split(key)
    _, state, _, _, info = env.step(step_key, state, Actions.UP, params)

    # Should have temperatures array
    assert "temperatures" in info
    assert len(info["temperatures"]) == 3  # EMPTY + 2 objects
    assert info["temperatures"][0] == 0.0  # EMPTY
    assert info["temperatures"][1] == 10.0  # hot object at index 1
    assert info["temperatures"][2] == 5.0  # cold object at index 2 (raw temperature)

    key, step_key = jax.random.split(key)
    _, state, _, _, info = env.step(step_key, state, Actions.UP, params)
    assert info["temperatures"][1] == 20.0  # hot next
    assert info["temperatures"][2] == 15.0  # cold next


def test_info_biome_id():
    """Test that info contains biome_id at agent's position."""
    key = jax.random.key(0)

    # Create environment with two biomes: one at (3,y) and one at (4,y)
    biomes = (
        Biome(start=(3, 0), stop=(4, 5), object_frequencies=(0.0,)),
        Biome(start=(4, 0), stop=(5, 5), object_frequencies=(0.0,)),
    )

    env = ForagaxEnv(size=(5, 5), objects=(), biomes=biomes, observation_type="color")
    params = env.default_params
    obs, state = env.reset(key, params)

    # Agent starts at center (2, 2), which should be in neither biome (biome_id = -1)
    assert state.biome_grid[2, 2] == -1

    # Move right to biome 0: from (2,2) to (3,2)
    key, step_key = jax.random.split(key)
    _, state, _, _, info = env.step(step_key, state, Actions.RIGHT, params)

    assert "biome_id" in info
    assert state.biome_grid[2, 3] == 0  # Position (3,2) is in biome 0
    assert info["biome_id"] == 0

    # Move right again to biome 1: from (3,2) to (4,2)
    key, step_key = jax.random.split(key)
    _, state, _, _, info = env.step(step_key, state, Actions.RIGHT, params)

    assert state.biome_grid[2, 4] == 1  # Position (4,2) is in biome 1
    assert info["biome_id"] == 1


def test_info_object_collected_id():
    """Test that info contains object_eaten_id when collecting objects."""
    key = jax.random.key(0)
    env = ForagaxEnv(
        size=(7, 7),
        objects=(FLOWER,),
        observation_type="color",
    )
    params = env.default_params
    _, state = env.reset(key, params)

    flower_id = 1  # 0 is EMPTY

    # Place a flower and move the agent to it
    grid = jnp.zeros((7, 7), dtype=int)
    grid = grid.at[4, 3].set(flower_id)
    state = state.replace(object_grid=grid, pos=jnp.array([3, 3]))

    # Collect the flower by moving down
    key, step_key = jax.random.split(key)
    _, _, reward, _, info = env.step(step_key, state, Actions.DOWN, params)

    assert "object_collected_id" in info
    assert info["object_collected_id"] == 1  # FLOWER id
    assert reward == FLOWER.reward_val

    # Next step should not collect anything
    key, step_key = jax.random.split(key)
    _, _, reward, _, info = env.step(step_key, state, Actions.UP, params)

    assert info["object_collected_id"] == -1  # No object collected
    assert reward == 0.0


def test_reward_delay():
    """Test that rewards are delayed by reward_delay for objects with k > 0."""
    key = jax.random.key(0)

    # Create an object with digestion delay of 2 steps
    delayed_flower = DefaultForagaxObject(
        name="delayed_flower",
        reward=5.0,
        collectable=True,
        color=(0, 255, 0),
        reward_delay=2,
    )

    env = ForagaxEnv(
        size=(7, 7),
        objects=(delayed_flower,),
        observation_type="color",
    )
    params = env.default_params
    _, state = env.reset(key, params)

    delayed_flower_id = 1  # 0 is EMPTY

    # Place the delayed flower and move the agent to it
    grid = jnp.zeros((7, 7), dtype=int)
    grid = grid.at[4, 3].set(delayed_flower_id)
    state = state.replace(object_grid=grid, pos=jnp.array([3, 3]))

    # Collect the flower by moving down - should get no immediate reward
    key, step_key = jax.random.split(key)
    _, state, reward, _, info = env.step(step_key, state, Actions.DOWN, params)

    assert info["object_collected_id"] == delayed_flower_id
    assert reward == 0.0  # No immediate reward due to digestion delay

    # Step 1: Still no reward (time=1, reward should arrive at time=2)
    key, step_key = jax.random.split(key)
    _, state, reward, _, _ = env.step(step_key, state, Actions.UP, params)
    assert reward == 0.0

    # Step 2: Reward should arrive now (time=2)
    key, step_key = jax.random.split(key)
    _, state, reward, _, _ = env.step(step_key, state, Actions.UP, params)
    assert reward == 5.0

    # Step 3: No more rewards
    key, step_key = jax.random.split(key)
    _, state, reward, _, _ = env.step(step_key, state, Actions.UP, params)
    assert reward == 0.0


def test_basic_object_expiry():
    """Test that objects expire and respawn after expiry_time steps."""
    # Create an object that expires after 5 steps with expiry_regen_delay=(2, 2)
    # Note: Due to timer encoding, delay=2 actually takes 3 steps (consistent with regen_delay)
    expiring_object = DefaultForagaxObject(
        name="expiring",
        reward=1.0,
        collectable=False,  # Not collectable, so it won't be removed by collection
        color=(255, 0, 0),
        expiry_time=5,
        expiry_regen_delay=(2, 2),  # Takes 3 steps due to +1 in timer encoding
    )

    env = ForagaxEnv(
        size=(3, 3),
        aperture_size=(3, 3),
        objects=(expiring_object,),
        biomes=(Biome(object_frequencies=(1.0,)),),
        observation_type="object",
    )

    key = jax.random.key(0)
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset, env.default_params)

    # Initial state - all objects present
    assert jnp.all(state.object_grid == 1)
    assert jnp.all(state.object_spawn_time_grid == 0)

    # Step through 5 times - objects should still be present
    for _ in range(5):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.DOWN, env.default_params
        )

    assert jnp.all(state.object_grid == 1)
    assert state.time == 5

    # Step once more - objects should expire and become timers
    key, key_step = jax.random.split(key)
    obs, state, _, _, _ = env.step(key_step, state, Actions.DOWN, env.default_params)

    assert jnp.all(state.object_grid < 0)  # All are now timers
    assert jnp.all(
        state.object_spawn_time_grid == 0
    )  # Spawn time NOT updated yet (still at reset value)

    # Step through regen delay (3 more steps due to +1 in encoding)
    for _ in range(3):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.DOWN, env.default_params
        )

    assert jnp.all(state.object_grid == 1)  # Objects respawned
    assert jnp.all(
        state.object_spawn_time_grid == 8
    )  # Spawn time = time at beginning of step when respawn occurred
    assert state.time == 9  # Current time after step completes


def test_no_expiry_backwards_compatibility():
    """Test that objects without expiry_time don't expire."""
    non_expiring_object = DefaultForagaxObject(
        name="non_expiring",
        reward=1.0,
        collectable=False,
        color=(0, 255, 0),
    )

    env = ForagaxEnv(
        size=(3, 3),
        aperture_size=(3, 3),
        objects=(non_expiring_object,),
        biomes=(Biome(object_frequencies=(1.0,)),),
        observation_type="object",
    )

    key = jax.random.key(42)
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset, env.default_params)

    initial_grid = state.object_grid.copy()

    # Step through many timesteps - grid should remain unchanged
    for _ in range(20):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.DOWN, env.default_params
        )

    assert jnp.all(state.object_grid == initial_grid)
    assert jnp.all(state.object_grid > 0)  # No timers


def test_expiry_with_collection():
    """Test that expiry works alongside collection."""
    collectable_expiring = DefaultForagaxObject(
        name="collectable_expiring",
        reward=5.0,
        collectable=True,
        color=(255, 255, 0),
        regen_delay=(3, 3),
        expiry_time=10,
        expiry_regen_delay=(5, 5),
    )

    env = ForagaxEnv(
        size=(5, 5),
        aperture_size=(5, 5),
        objects=(collectable_expiring,),
        biomes=(Biome(object_frequencies=(0.3,)),),
        observation_type="object",
    )

    key = jax.random.key(123)
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset, env.default_params)

    # Count initial objects
    initial_object_count = jnp.sum(state.object_grid > 0)
    assert initial_object_count > 0

    # Step through and collect/expire objects
    collected_count = 0
    for _ in range(20):
        key, key_step = jax.random.split(key)
        obs, state, reward, _, _ = env.step(
            key_step, state, Actions.UP, env.default_params
        )
        if reward > 0:
            collected_count += 1

    # Should have collected at least some objects
    assert collected_count > 0


def test_expiry_with_normal_regen():
    """Test expiry with NormalRegenForagaxObject."""
    normal_expiring = NormalRegenForagaxObject(
        name="normal_expiring",
        reward=3.0,
        collectable=False,
        mean_regen_delay=10,
        std_regen_delay=1,
        color=(128, 128, 255),
        expiry_time=8,
        mean_expiry_regen_delay=4,
        std_expiry_regen_delay=1,
    )

    env = ForagaxEnv(
        size=(4, 4),
        aperture_size=(4, 4),
        objects=(normal_expiring,),
        biomes=(Biome(object_frequencies=(1.0,)),),
        observation_type="object",
    )

    key = jax.random.key(456)
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset, env.default_params)

    # All objects start present
    assert jnp.all(state.object_grid == 1)

    # Step until expiry
    for _ in range(8):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.LEFT, env.default_params
        )

    # Objects should expire
    key, key_step = jax.random.split(key)
    obs, state, _, _, _ = env.step(key_step, state, Actions.LEFT, env.default_params)

    assert jnp.all(state.object_grid < 0)  # All expired


def test_mixed_expiry_times():
    """Test objects with different expiry times."""
    fast_expiry = DefaultForagaxObject(
        name="fast",
        reward=1.0,
        collectable=False,
        color=(255, 0, 0),
        expiry_time=3,
        expiry_regen_delay=(1, 1),
    )

    slow_expiry = DefaultForagaxObject(
        name="slow",
        reward=2.0,
        collectable=False,
        color=(0, 0, 255),
        expiry_time=6,
        expiry_regen_delay=(2, 2),
    )

    env = ForagaxEnv(
        size=(5, 5),
        aperture_size=(5, 5),
        objects=(fast_expiry, slow_expiry),
        biomes=(Biome(object_frequencies=(0.3, 0.3)),),
        observation_type="object",
    )

    key = jax.random.key(789)
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset, env.default_params)

    # Count initial objects
    fast_count = jnp.sum(state.object_grid == 1)
    slow_count = jnp.sum(state.object_grid == 2)

    # Step 4 times - fast should expire, slow should not
    for _ in range(4):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.RIGHT, env.default_params
        )

    # Fast objects should be timers, slow objects should still be present
    fast_after = jnp.sum(state.object_grid == 1)
    slow_after = jnp.sum(state.object_grid == 2)

    # Fast objects should have become timers (decreased count)
    assert fast_after < fast_count
    # Slow objects should still be present
    assert slow_after == slow_count


def test_expiry_spawn_time_tracking():
    """Test that spawn times are correctly tracked for expiry."""
    expiring_obj = DefaultForagaxObject(
        name="expiring",
        reward=1.0,
        collectable=False,
        color=(200, 100, 50),
        expiry_time=5,
        expiry_regen_delay=(2, 2),
    )

    env = ForagaxEnv(
        size=(2, 2),
        aperture_size=(2, 2),
        objects=(expiring_obj,),
        biomes=(Biome(object_frequencies=(1.0,)),),
        observation_type="object",
    )

    key = jax.random.key(111)
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset, env.default_params)

    # Initial spawn times should be 0
    assert jnp.all(state.object_spawn_time_grid == 0)

    # Step until expiry
    for step in range(6):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.DOWN, env.default_params
        )

    # After expiry (but before respawn), spawn times should still be 0
    assert jnp.all(state.object_spawn_time_grid == 0)
    assert jnp.all(state.object_grid < 0)  # Objects are timers

    # Step through regen delay (3 steps)
    for step in range(3):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.DOWN, env.default_params
        )

    # After respawn, spawn times should be updated to when respawn occurred
    assert jnp.all(
        state.object_spawn_time_grid == 8
    )  # Time at beginning of step when respawn occurred
    assert jnp.all(state.object_grid > 0)  # Objects respawned
    assert state.time == 9  # Current time after step completes

    # Ages should be 1 (respawned 1 step ago: current_time=9 - spawn_time=8)
    ages = state.time - state.object_spawn_time_grid
    assert jnp.all(ages == 1)


def test_expiry_with_collectable_spawn_time_update():
    """Test that spawn times update correctly when objects are collected."""
    obj = DefaultForagaxObject(
        name="test",
        reward=1.0,
        collectable=True,
        color=(100, 200, 100),
        regen_delay=(5, 5),
        expiry_time=10,
        expiry_regen_delay=(3, 3),
    )

    env = ForagaxEnv(
        size=(3, 3),
        aperture_size=(3, 3),
        objects=(obj,),
        biomes=(Biome(object_frequencies=(1.0,)),),
        observation_type="object",
    )

    key = jax.random.key(222)
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset, env.default_params)

    # Initial spawn times should be 0
    assert jnp.all(state.object_spawn_time_grid == 0)

    # Move and collect object
    total_reward = 0
    collected_time = None
    for step in range(10):
        key, key_step = jax.random.split(key)
        action = step % 4
        obs, state, reward, _, _ = env.step(key_step, state, action, env.default_params)
        total_reward += reward

        if reward > 0 and collected_time is None:
            collected_time = state.time
            # When collected, spawn time should NOT change (still 0)
            assert jnp.all(state.object_spawn_time_grid == 0)
            break

    # Should have collected at least one object
    assert total_reward > 0
    assert collected_time is not None

    # Step through regen delay (6 more steps due to +1 in encoding)
    for step in range(6):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.DOWN, env.default_params
        )

    # After respawn, spawn time should be updated to respawn time
    # Respawn happens at beginning of step (collected_time + 6 - 1)
    expected_respawn_time = collected_time + 5
    # At least one object should have respawned and had its spawn time set
    assert jnp.any(state.object_spawn_time_grid == expected_respawn_time)
