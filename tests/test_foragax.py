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
from foragax.objects import (
    FLOWER,
    LARGE_MOREL,
    MEDIUM_MOREL,
    MOREL,
    OYSTER,
    THORNS,
    WALL,
    DefaultForagaxObject,
)


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
    env = ForagaxWorldEnv(
        size=(10, 10),
        aperture_size=5,
        objects=(WALL, FLOWER),
        biomes=(Biome(object_frequencies=(0.1, 0.1)),),
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
    env = ForagaxObjectEnv(size=(7, 7), aperture_size=(3, 3), objects=object_types)
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
    env = ForagaxObjectEnv(
        size=7,
        aperture_size=3,
        objects=object_types,
    )
    params = env.default_params
    _, state = env.reset_env(key, params)

    flower_id = 1  # 0 is EMPTY

    # Place a flower and move the agent to it
    grid = jnp.zeros((7, 7), dtype=int)
    grid = grid.at[4, 3].set(flower_id)
    state = state.replace(object_grid=grid, pos=jnp.array([3, 3]))

    # Collect the flower
    key, step_key = jax.random.split(key)
    _, state, reward, _, _ = env.step_env(step_key, state, Actions.DOWN, params)
    assert reward == FLOWER.reward_val
    assert state.object_grid[4, 3] < 0

    steps_until_respawn = -state.object_grid[4, 3] // 2

    # Step until it respawns
    for i in range(steps_until_respawn):
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step_env(step_key, state, Actions.DOWN, params)
        assert state.object_grid[4, 3] < 0

    key, step_key = jax.random.split(key)
    _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
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
    env = ForagaxObjectEnv(size=(5, 5), objects=(), nowrap=True)
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
    env = ForagaxObjectEnv(size=(5, 5), aperture_size=(3, 3), objects=(FLOWER,))
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
    env_no_wrap = ForagaxObjectEnv(
        size=(7, 7), aperture_size=(3, 3), objects=object_types, nowrap=True
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


def test_color_based_partial_observability():
    """Test that objects with the same color are grouped into the same observation channel."""
    env = ForagaxObjectEnv(
        size=(10, 10),
        aperture_size=(5, 5),
        objects=(MOREL, LARGE_MOREL, MEDIUM_MOREL, FLOWER),
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
    env = ForagaxObjectEnv(
        size=(10, 10),
        aperture_size=(3, 3),
        objects=(WALL, FLOWER, THORNS),
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

    env = ForagaxObjectEnv(
        size=(7, 7),
        aperture_size=(3, 3),
        objects=(obj1, obj2, obj3),
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
    env = ForagaxObjectEnv(
        size=(5, 5),
        aperture_size=(3, 3),
        objects=(),  # No objects
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

    env = ForagaxObjectEnv(
        size=(7, 7),
        aperture_size=(3, 3),
        objects=(obj1, obj2, obj3),
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
