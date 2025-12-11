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
    FourierObject,
    NormalRegenForagaxObject,
    SineObject,
    WeatherObject,
    create_fourier_objects,
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

    empirical_freq = jnp.count_nonzero(state.object_state.object_id) / size**2
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
    state = state.replace(object_state=state.object_state.replace(object_id=grid))

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
    state = state.replace(
        object_state=state.object_state.replace(object_id=grid), pos=jnp.array([3, 3])
    )

    # Collect the flower
    key, step_key = jax.random.split(key)
    _, state, reward, _, _ = env.step(step_key, state, Actions.DOWN, params)
    assert reward == FLOWER.reward_val
    assert state.object_state.object_id[4, 3] == 0  # Object removed
    assert state.object_state.respawn_timer[4, 3] > 0  # Timer set

    steps_until_respawn = int(state.object_state.respawn_timer[4, 3])

    # Step until it respawns
    for i in range(steps_until_respawn):
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step(step_key, state, Actions.DOWN, params)
        if i < steps_until_respawn - 1:
            assert state.object_state.object_id[4, 3] == 0  # Still empty
            assert state.object_state.respawn_timer[4, 3] > 0  # Timer still counting

    # After timer reaches 0, object should respawn
    assert state.object_state.object_id[4, 3] == flower_id
    assert state.object_state.respawn_timer[4, 3] == 0


def test_random_respawn():
    """Test that an object can respawn at random empty locations within its biome."""
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

    flower_id = 1  # 0 is EMPTY
    original_pos = jnp.array([3, 3])

    # Test multiple times to verify randomness
    timer_positions = []
    for i in range(20):
        key, reset_key = jax.random.split(key)
        _, state = env.reset(reset_key, params)

        # Place a flower and move the agent to it
        grid = jnp.zeros((7, 7), dtype=int)
        grid = grid.at[original_pos[1], original_pos[0]].set(flower_id)
        # Add a wall to make sure it doesn't spawn there
        grid = grid.at[4, 4].set(2)  # Use a fixed ID for the wall
        state = state.replace(
            object_state=state.object_state.replace(object_id=grid),
            pos=jnp.array([2, 3]),
        )

        # Collect the flower
        key, step_key = jax.random.split(key)
        _, new_state, reward, _, _ = env.step(step_key, state, Actions.RIGHT, params)

        assert reward == flower_random.reward_val
        # Original position should be empty
        assert new_state.object_state.object_id[original_pos[1], original_pos[0]] == 0

        # A timer should be placed somewhere (check respawn_timer instead of object_id)
        assert jnp.sum(new_state.object_state.respawn_timer > 0) == 1
        timer_pos_flat = jnp.argmax(new_state.object_state.respawn_timer)
        timer_pos_array = jnp.unravel_index(timer_pos_flat, (7, 7))
        timer_pos = (int(timer_pos_array[0]), int(timer_pos_array[1]))

        # New position should be within the biome
        assert timer_pos[0] >= biome.start[0] and timer_pos[0] < biome.stop[0]
        assert timer_pos[1] >= biome.start[1] and timer_pos[1] < biome.stop[1]

        # New position should be on an empty cell (not the wall)
        assert timer_pos != (4, 4)

        # Verify respawn_object_id is set correctly
        assert (
            new_state.object_state.respawn_object_id[timer_pos[0], timer_pos[1]]
            == flower_id
        )

        timer_positions.append(timer_pos)

    # Verify that we get multiple different positions (randomness works)
    # With 20 trials and multiple valid positions, we should see at least 2 different locations
    unique_positions = set(timer_positions)
    assert len(unique_positions) >= 2, (
        f"Expected at least 2 unique positions, got {len(unique_positions)}: {unique_positions}"
    )


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
    state = state.replace(
        object_state=state.object_state.replace(object_id=grid), pos=jnp.array([2, 3])
    )

    # Collect the flower
    key, step_key = jax.random.split(key)
    _, new_state, reward, _, _ = env.step(step_key, state, Actions.RIGHT, params)

    assert reward == flower_random.reward_val
    # The timer should be placed back at the original position
    assert new_state.object_state.object_id[original_pos[1], original_pos[0]] == 0
    assert new_state.object_state.respawn_timer[original_pos[1], original_pos[0]] > 0
    assert (
        new_state.object_state.respawn_object_id[original_pos[1], original_pos[0]]
        == flower_id
    )


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
    state = state.replace(object_state=state.object_state.replace(object_id=grid))

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
    state = state.replace(
        object_state=state.object_state.replace(object_id=grid), pos=jnp.array([0, 0])
    )

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
    morel_locations = jnp.argwhere(state.object_state.object_id == morel_id)

    assert jnp.all(morel_locations >= 2)
    assert jnp.all(morel_locations < 6)

    # Check that no other objects were generated
    unique_objects = jnp.unique(state.object_state.object_id)
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
    wall_count = jnp.sum(state.object_state.object_id == wall_id)
    flower_count = jnp.sum(state.object_state.object_id == flower_id)
    assert wall_count == 2
    assert flower_count == 2

    # Check that objects only appear within the biome
    wall_locations = jnp.argwhere(state.object_state.object_id == wall_id)
    flower_locations = jnp.argwhere(state.object_state.object_id == flower_id)

    assert jnp.all(wall_locations >= 2)
    assert jnp.all(wall_locations < 6)
    assert jnp.all(flower_locations >= 2)
    assert jnp.all(flower_locations < 6)

    # Test different positions: different key should produce different positions but same counts
    key_1 = jax.random.key(1)
    _, state_1 = env.reset(key_1, params)
    assert jnp.sum(state_1.object_state.object_id == wall_id) == 2
    assert jnp.sum(state_1.object_state.object_id == flower_id) == 2
    # Positions may be the same due to implementation (not shuffled)
    assert not jnp.array_equal(
        state.object_state.object_id, state_1.object_state.object_id
    )


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
    assert jnp.sum(state.object_state.object_id == morel_id) == 8
    assert jnp.sum(state.object_state.object_id == oyster_id) == 8
    assert jnp.sum(state.object_state.object_id == deathcap_id) == 15
    assert jnp.sum(state.object_state.object_id == fake_id) == 15

    # Check that objects are within their biomes
    morel_locs = jnp.argwhere(state.object_state.object_id == morel_id)
    oyster_locs = jnp.argwhere(state.object_state.object_id == oyster_id)
    deathcap_locs = jnp.argwhere(state.object_state.object_id == deathcap_id)
    fake_locs = jnp.argwhere(state.object_state.object_id == fake_id)

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
    state = state.replace(object_state=state.object_state.replace(object_id=grid))

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
    state = state.replace(object_state=state.object_state.replace(object_id=grid))

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
    state = state.replace(object_state=state.object_state.replace(object_id=grid))

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
    assert info["temperatures"][2] == -5.0  # cold object at index 2 (raw temperature)

    key, step_key = jax.random.split(key)
    _, state, _, _, info = env.step(step_key, state, Actions.UP, params)
    assert info["temperatures"][1] == 20.0  # hot next
    assert info["temperatures"][2] == -15.0  # cold next


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
    assert state.object_state.biome_id[2, 2] == -1

    # Move right to biome 0: from (2,2) to (3,2)
    key, step_key = jax.random.split(key)
    _, state, _, _, info = env.step(step_key, state, Actions.RIGHT, params)

    assert "biome_id" in info
    assert state.object_state.biome_id[2, 3] == 0  # Position (3,2) is in biome 0
    assert info["biome_id"] == 0

    # Move right again to biome 1: from (3,2) to (4,2)
    key, step_key = jax.random.split(key)
    _, state, _, _, info = env.step(step_key, state, Actions.RIGHT, params)

    assert state.object_state.biome_id[2, 4] == 1  # Position (4,2) is in biome 1
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
    state = state.replace(
        object_state=state.object_state.replace(object_id=grid), pos=jnp.array([3, 3])
    )

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
    state = state.replace(
        object_state=state.object_state.replace(object_id=grid), pos=jnp.array([3, 3])
    )

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
        max_expiries_per_step=10,  # Ensure all expiries processed each step
    )

    key = jax.random.key(0)
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset, env.default_params)

    # Initial state - all objects present
    assert jnp.all(state.object_state.object_id == 1)
    assert jnp.all(state.object_state.spawn_time == 0)

    # Step through 5 times - objects should still be present
    for _ in range(5):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.DOWN, env.default_params
        )

    assert jnp.all(state.object_state.object_id == 1)
    assert state.time == 5

    # Step once more - objects should expire and become timers
    key, key_step = jax.random.split(key)
    obs, state, _, _, _ = env.step(key_step, state, Actions.DOWN, env.default_params)

    assert jnp.all(state.object_state.object_id == 0)  # All objects removed
    assert jnp.all(state.object_state.respawn_timer > 0)  # All have timers
    assert jnp.all(
        state.object_state.respawn_object_id == 1
    )  # Will respawn as object 1
    assert jnp.all(
        state.object_state.spawn_time == 0
    )  # Spawn time NOT updated yet (still at reset value)

    # Step through regen delay (3 more steps due to +1 in encoding)
    for _ in range(3):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.DOWN, env.default_params
        )

    assert jnp.all(state.object_state.object_id == 1)  # Objects respawned
    assert jnp.all(state.object_state.respawn_timer == 0)  # Timers cleared
    assert jnp.all(
        state.object_state.spawn_time == 8
    )  # Spawn time = time at beginning of step when respawn occurred
    assert state.time == 9  # Current time after step completes


def test_basic_object_expiry_one():
    """Test that objects expire and respawn after expiry_time steps."""
    # Create an object that expires after 5 steps with expiry_regen_delay=(2, 2)
    # Note: Due to timer encoding, delay=2 actually takes 3 steps (consistent with regen_delay)
    expiring_object = DefaultForagaxObject(
        name="expiring",
        reward=1.0,
        collectable=False,  # Not collectable, so it won't be removed by collection
        color=(255, 0, 0),
        expiry_time=10,
        expiry_regen_delay=(8, 8),  # Takes 9 steps due to +1 in timer encoding
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
    assert jnp.all(state.object_state.object_id == 1)
    assert jnp.all(state.object_state.spawn_time == 0)

    # Step through 10 times - objects should still be present
    for _ in range(10):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.DOWN, env.default_params
        )

    assert jnp.all(state.object_state.object_id == 1)
    assert state.time == 10

    # Step once more - objects should expire and become timers
    key, key_step = jax.random.split(key)
    obs, state, _, _, _ = env.step(key_step, state, Actions.DOWN, env.default_params)

    assert (
        jnp.count_nonzero(state.object_state.object_id == 0) == 1
    )  # One objects removed
    assert (
        jnp.count_nonzero(state.object_state.respawn_timer > 0) == 1
    )  # One have timers
    assert (
        jnp.count_nonzero(state.object_state.respawn_object_id == 1) == 1
    )  # Will respawn as object 1
    assert jnp.all(
        state.object_state.spawn_time == 0
    )  # Spawn time NOT updated yet (still at reset value)

    # Step through all removals and they all become timers (8 more steps)
    for _ in range(8):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.DOWN, env.default_params
        )

    assert jnp.all(state.object_state.object_id == 0)  # All objects removed
    assert jnp.all(state.object_state.respawn_timer > 0)  # All have timers
    assert jnp.all(
        state.object_state.respawn_object_id == 1
    )  # Will respawn as object 1
    assert jnp.all(
        state.object_state.spawn_time == 0
    )  # Spawn time NOT updated yet (still at reset value)

    # Step once more - one object should respawn
    key, key_step = jax.random.split(key)
    obs, state, _, _, _ = env.step(key_step, state, Actions.DOWN, env.default_params)

    assert (
        jnp.count_nonzero(state.object_state.object_id == 0) == 8
    )  # One objects present
    assert (
        jnp.count_nonzero(state.object_state.respawn_timer > 0) == 8
    )  # One don't have timers
    assert (
        jnp.count_nonzero(state.object_state.respawn_object_id == 1) == 8
    )  # Remaining will respawn as object 1
    assert (
        jnp.count_nonzero(state.object_state.spawn_time == 19) == 1
    )  # One with updated spawn time

    # Step through regen delay for remaining (8 more objects)
    for _ in range(8):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.DOWN, env.default_params
        )

    assert jnp.all(state.object_state.object_id == 1)  # Objects respawned
    assert jnp.all(state.object_state.respawn_timer == 0)  # Timers cleared
    print(state.object_state.spawn_time)
    assert jnp.all(
        jnp.sort(state.object_state.spawn_time.flatten())
        == jnp.sort(jnp.arange(19, 28))
    )  # Spawn time = time at beginning of step when respawn occurred
    assert state.time == 28  # Current time after step completes


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

    initial_grid = state.object_state.object_id.copy()

    # Step through many timesteps - grid should remain unchanged
    for _ in range(20):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.DOWN, env.default_params
        )

    assert jnp.all(state.object_state.object_id == initial_grid)
    assert jnp.all(state.object_state.object_id > 0)  # No timers


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
    initial_object_count = jnp.sum(state.object_state.object_id > 0)
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
        max_expiries_per_step=16,  # Ensure all expiries processed each step
    )

    key = jax.random.key(456)
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset, env.default_params)

    # All objects start present
    assert jnp.all(state.object_state.object_id == 1)

    # Step until expiry
    for _ in range(8):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.LEFT, env.default_params
        )

    # Objects should expire
    key, key_step = jax.random.split(key)
    obs, state, _, _, _ = env.step(key_step, state, Actions.LEFT, env.default_params)

    assert jnp.all(state.object_state.object_id == 0)  # All objects removed
    assert jnp.all(state.object_state.respawn_timer > 0)  # All have timers


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
    fast_count = jnp.sum(state.object_state.object_id == 1)
    slow_count = jnp.sum(state.object_state.object_id == 2)

    # Step 4 times - fast should expire, slow should not
    for _ in range(4):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.RIGHT, env.default_params
        )

    # Fast objects should be timers, slow objects should still be present
    fast_after = jnp.sum(state.object_state.object_id == 1)
    slow_after = jnp.sum(state.object_state.object_id == 2)

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
        max_expiries_per_step=4,  # Ensure all expiries processed each step
    )

    key = jax.random.key(111)
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset, env.default_params)

    # Initial spawn times should be 0
    assert jnp.all(state.object_state.spawn_time == 0)

    # Step until expiry
    for step in range(6):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.DOWN, env.default_params
        )

    # After expiry (but before respawn), spawn times should still be 0
    assert jnp.all(state.object_state.spawn_time == 0)
    assert jnp.all(state.object_state.object_id == 0)  # Objects removed
    assert jnp.all(state.object_state.respawn_timer > 0)  # Timers set

    # Step through regen delay (3 steps)
    for step in range(3):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.DOWN, env.default_params
        )

    # After respawn, spawn times should be updated to when respawn occurred
    assert jnp.all(
        state.object_state.spawn_time == 8
    )  # Time at beginning of step when respawn occurred
    assert jnp.all(state.object_state.object_id > 0)  # Objects respawned
    assert jnp.all(state.object_state.respawn_timer == 0)  # Timers cleared
    assert state.time == 9  # Current time after step completes

    # Ages should be 1 (respawned 1 step ago: current_time=9 - spawn_time=8)
    ages = state.time - state.object_state.spawn_time
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
    assert jnp.all(state.object_state.spawn_time == 0)

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
            assert jnp.all(state.object_state.spawn_time == 0)
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
    assert jnp.any(state.object_state.spawn_time == expected_respawn_time)


def test_expiry_with_random_respawn():
    """Test that expired objects with random_respawn=True respawn at random locations within the same biome."""
    expiring_random = DefaultForagaxObject(
        name="expiring_random",
        reward=1.0,
        collectable=False,
        random_respawn=True,  # Enable random respawn
        color=(100, 200, 100),
        expiry_time=3,
        expiry_regen_delay=(1, 1),  # Short delay for quick test
    )

    # Create environment with lower frequency to allow empty cells for random respawn
    env = ForagaxEnv(
        size=(6, 6),
        aperture_size=(6, 6),
        objects=(expiring_random,),
        biomes=(
            Biome(
                start=(0, 0), stop=(6, 6), object_frequencies=(0.5,)
            ),  # Single biome with 50% occupancy
        ),
        observation_type="object",
    )

    key = jax.random.key(999)
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset, env.default_params)

    # Record initial object positions
    initial_object_positions = state.object_state.object_id == 1

    # Step until expiry (4 steps - expiry happens when age >= expiry_time)
    for _ in range(4):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.DOWN, env.default_params
        )

    # Some objects should have expired and have timers
    has_timers = jnp.any(state.object_state.respawn_timer > 0)
    assert has_timers, "At least some objects should have expired"
    # With random respawn, timers may be placed at different locations than where objects expired
    # So we just check that timers exist and respawn_object_id is set correctly
    assert jnp.all(
        (state.object_state.respawn_timer > 0)
        == (state.object_state.respawn_object_id > 0)
    )

    # Step through regen delay (2 steps due to +1 encoding)
    for _ in range(2):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.DOWN, env.default_params
        )

    # Objects should have respawned
    final_object_positions = state.object_state.object_id == 1

    # With random respawn, objects should not all be in their original positions
    # (This is a probabilistic test - with true randomness, it's very unlikely all objects
    # would end up in exactly the same positions)
    positions_unchanged = jnp.all(initial_object_positions == final_object_positions)
    assert not positions_unchanged, (
        "Objects should respawn at random locations, not all in original positions"
    )

    # But they should still be within the biome
    biome_mask = state.object_state.biome_id == 0
    assert jnp.all(
        (state.object_state.object_id == 0)
        | ((state.object_state.object_id == 1) & biome_mask)
    ), "All objects should be within the biome"


def test_dynamic_biome_respawn_threshold():
    """Test that biomes respawn when consumption threshold is reached."""
    hot = create_fourier_objects(num_fourier_terms=2, base_magnitude=1.0)[0]

    env = ForagaxEnv(
        name="TestDynamicBiome",
        size=(4, 4),
        aperture_size=(3, 3),
        objects=(hot,),
        biomes=(Biome(start=(0, 0), stop=(4, 4), object_frequencies=(1.0,)),),
        nowrap=True,
        deterministic_spawn=True,
        observation_type="object",
        dynamic_biomes=True,
        biome_consumption_threshold=0.7,
    )

    key = jax.random.key(123)
    obs, state = env.reset(key, env.default_params)

    # Record initial state
    initial_generation = state.biome_state.generation[0]
    initial_total = state.biome_state.total_objects[0]
    initial_params = state.object_state.state_params.copy()

    # Calculate how many objects need to be consumed to trigger respawn (70% of 16 = 11.2, so 12)
    threshold_count = int(jnp.ceil(initial_total * 0.7))

    # Collect objects by taking steps - agent will collect when moving onto objects
    current_state = state
    steps_taken = 0
    max_steps = 50  # Safety limit

    while (
        current_state.biome_state.consumption_count[0] < threshold_count
        and current_state.biome_state.generation[0] == initial_generation
        and steps_taken < max_steps
    ):
        key, key_step = jax.random.split(key)
        # Take a random action
        action = jax.random.randint(key_step, (), 0, 4)
        obs, current_state, reward, done, info = env.step(
            key_step, current_state, action, env.default_params
        )
        steps_taken += 1

    # Check that respawn occurred
    assert current_state.biome_state.generation[0] > initial_generation, (
        f"Generation should increase. Consumed {current_state.biome_state.consumption_count[0]}/{threshold_count} in {steps_taken} steps"
    )
    assert current_state.biome_state.consumption_count[0] == 0, (
        "Consumption should reset after respawn"
    )
    assert current_state.biome_state.total_objects[0] > 0, "Objects should be respawned"

    # Reward parameters should have changed
    current_params = current_state.object_state.state_params
    params_changed = jnp.any(jnp.abs(current_params - initial_params) > 0.01)
    assert params_changed, "Reward parameters should change after respawn"


def test_object_no_individual_respawn():
    """Test that objects with max regen_delay do not have a timer placed."""
    # Create an object with infinite regen delay (no individual respawn)
    no_respawn_obj = DefaultForagaxObject(
        name="no_respawn",
        reward=1.0,
        collectable=True,
        regen_delay=(jnp.iinfo(jnp.int32).max, jnp.iinfo(jnp.int32).max),
        color=(255, 0, 0),
    )

    env = ForagaxEnv(
        size=(3, 3),
        aperture_size=(3, 3),
        objects=(no_respawn_obj,),
        biomes=(Biome(start=(0, 0), stop=(3, 3), object_frequencies=(1.0,)),),
        nowrap=True,
        observation_type="object",
    )

    key = jax.random.key(42)
    obs, state = env.reset(key, env.default_params)

    # Find an object position
    obj_pos = jnp.argwhere(state.object_state.object_id > 0)[0]
    y, x = obj_pos

    # Position agent above the object so moving down collects it
    state = state.replace(pos=jnp.array([x, y - 1]))
    key, key_step = jax.random.split(key)
    obs, state, reward, done, info = env.step(
        key_step, state, Actions.DOWN, env.default_params
    )

    # Verify object was collected
    assert info["object_collected_id"] == 1, "Object should have been collected"

    # Check that no timer was placed (position should be 0, not negative)
    assert state.object_state.object_id[y, x] == 0, (
        f"No timer should be placed for objects with max regen_delay, "
        f"but got {state.object_state.object_id[y, x]}"
    )


def test_object_color_grid_cleared_on_collection():
    """Test that object colors are preserved in state but masked in rendering when collected."""
    key = jax.random.key(0)
    env = ForagaxEnv(
        size=(7, 7),
        objects=(FLOWER,),
        biomes=(Biome(object_frequencies=(1.0,)),),
        observation_type="color",
        dynamic_biomes=True,  # Enable dynamic biomes to test color grid
    )
    params = env.default_params
    _, state = env.reset(key, params)

    flower_id = 1  # 0 is EMPTY

    # Find a flower position
    flower_positions = jnp.argwhere(state.object_state.object_id == flower_id)
    flower_pos = flower_positions[0]
    y, x = flower_pos

    # Store original color
    original_color = state.object_state.color[y, x].copy()

    # Move agent to flower and collect it
    # Position agent above the flower so moving down collects it
    state = state.replace(pos=jnp.array([x, y - 1]))
    key, step_key = jax.random.split(key)
    _, state, reward, _, _ = env.step(step_key, state, Actions.DOWN, params)

    assert reward == FLOWER.reward_val

    # NEW BEHAVIOR: Color is preserved in state (not cleared to white)
    collected_color = state.object_state.color[y, x]
    chex.assert_trees_all_equal(collected_color, original_color)

    # But object_id should be 0 (or timer should be set)
    assert (
        state.object_state.object_id[y, x] == 0
        or state.object_state.respawn_timer[y, x] > 0
    )

    # Check that other positions with objects still have their colors
    other_flower_positions = jnp.argwhere(
        (state.object_state.object_id == flower_id)
        & (state.object_state.object_id != 0)
    )
    other_pos = other_flower_positions[0]
    other_color = state.object_state.color[other_pos[0], other_pos[1]]
    # Should not be the empty color
    expected_empty_color = jnp.array([255, 255, 255], dtype=jnp.uint8)
    assert not jnp.allclose(other_color, expected_empty_color)

    # Test RGB observation: empty cells should appear white
    # Create an RGB environment with the same setup
    env_rgb = ForagaxEnv(
        size=(7, 7),
        objects=(FLOWER,),
        biomes=(Biome(object_frequencies=(1.0,)),),
        observation_type="rgb",
        dynamic_biomes=True,
        aperture_size=-1,  # Use world view to get full grid observation
    )
    # Use the same state to get observation
    obs_rgb = env_rgb.get_obs(state, params)

    # Check that the collected position shows white in RGB observation
    # RGB observations are normalized to [0, 1], so white is [1.0, 1.0, 1.0]
    collected_rgb = obs_rgb[y, x]
    expected_white_rgb = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
    chex.assert_trees_all_close(collected_rgb, expected_white_rgb, rtol=1e-5)


def test_object_color_grid_cleared_on_expiry():
    """Test that object colors are preserved in state but masked in rendering when expired."""
    # Create an object that expires quickly
    expiring_obj = DefaultForagaxObject(
        name="expiring",
        reward=1.0,
        collectable=False,
        color=(255, 0, 0),
        expiry_time=3,
        expiry_regen_delay=(1, 1),
    )

    env = ForagaxEnv(
        size=(5, 5),
        aperture_size=(5, 5),
        objects=(expiring_obj,),
        biomes=(Biome(object_frequencies=(0.5,)),),
        observation_type="color",
        dynamic_biomes=True,  # Enable dynamic biomes to test color grid
        max_expiries_per_step=25,  # Ensure all expiries processed each step
    )

    key = jax.random.key(42)
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset, env.default_params)

    # Find an object position
    obj_positions = jnp.argwhere(state.object_state.object_id == 1)
    obj_pos = obj_positions[0]
    y, x = obj_pos

    # Store original color
    original_color = state.object_state.color[y, x].copy()

    # Step until expiry (4 steps: expiry happens when age >= expiry_time)
    for _ in range(4):
        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(
            key_step, state, Actions.DOWN, env.default_params
        )

    # NEW BEHAVIOR: Color is preserved in state (not cleared to white)
    expired_color = state.object_state.color[y, x]
    chex.assert_trees_all_equal(expired_color, original_color)

    # But object_id should be 0 (or timer should be set)
    assert (
        state.object_state.object_id[y, x] == 0
        or state.object_state.respawn_timer[y, x] > 0
    )

    # Test RGB observation: empty/expired cells should appear white
    # Create an RGB environment with the same setup
    env_rgb = ForagaxEnv(
        size=(5, 5),
        aperture_size=-1,  # Use world view to get full grid observation
        objects=(expiring_obj,),
        biomes=(Biome(object_frequencies=(0.5,)),),
        observation_type="rgb",
        dynamic_biomes=True,
    )
    # Use the same state to get observation
    obs_rgb = env_rgb.get_obs(state, env.default_params)

    # Check that the expired position shows white in RGB observation
    # RGB observations are normalized to [0, 1], so white is [1.0, 1.0, 1.0]
    expired_rgb = obs_rgb[y, x]
    expected_white_rgb = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
    chex.assert_trees_all_close(expired_rgb, expected_white_rgb, rtol=1e-5)


def test_biome_regeneration_preserves_old_objects():
    """Test that biome regeneration only replaces intersecting objects."""
    # Create a simple environment with one biome
    OBJ_A = FourierObject(
        name="OBJ_A",
        base_magnitude=1.0,
        color=(255, 0, 0),
    )
    OBJ_B = FourierObject(
        name="OBJ_B",
        base_magnitude=1.0,
        color=(0, 255, 0),
    )

    # Small grid for easier testing
    size = (10, 10)
    biomes = (
        Biome(
            start=(0, 0),
            stop=(10, 10),
            object_frequencies=(0.5, 0.3),  # 80% occupancy
        ),
    )

    env = ForagaxEnv(
        size=size,
        aperture_size=5,
        objects=(OBJ_A, OBJ_B),
        biomes=biomes,
        dynamic_biomes=True,
        biome_consumption_threshold=0.1,  # 10% threshold for easier testing
        deterministic_spawn=False,  # Use random spawn for realistic test
    )

    key = jax.random.key(42)
    key_reset, key_step = jax.random.split(key)

    # Reset environment
    obs, state = env.reset(key_reset, env.default_params)

    # Store initial object grid and object state
    initial_object_grid = state.object_state.object_id.copy()
    initial_object_state_grid = state.object_state.state_params.copy()

    # Consume objects until biome regeneration triggers
    # Directly manipulate state to collect objects and trigger respawn
    current_state = state

    # Collect all objects to trigger regeneration
    for y in range(size[1]):
        for x in range(size[0]):
            if current_state.object_state.object_id[y, x] > 0:
                obj_id = int(current_state.object_state.object_id[y, x])
                if (
                    obj_id < len(env.object_collectable)
                    and env.object_collectable[obj_id]
                ):
                    # Teleport agent to object and collect it
                    current_state = current_state.replace(pos=jnp.array([x, y]))
                    key_step, step_key = jax.random.split(key_step)
                    obs, current_state, reward, done, info = env.step(
                        step_key, current_state, 0, env.default_params
                    )

                    # Stop once regeneration happens
                    if current_state.biome_state.generation[0] > 0:
                        break
        if current_state.biome_state.generation[0] > 0:
            break

    assert current_state.biome_state.generation[0] > 0, "Biome should have regenerated"

    # Verify regeneration behavior:
    # 1. Object grid has changed
    assert not jnp.array_equal(
        initial_object_grid, current_state.object_state.object_id
    ), "Object grid should have changed after respawn"

    # 2. At least one old object should persist (wasn't replaced by new spawn)
    biome_0_mask = current_state.object_state.biome_id == 0
    old_objects_remaining = 0
    for y in range(size[1]):
        for x in range(size[0]):
            if biome_0_mask[y, x]:
                # Check if this position had an object initially and still has the same object
                if (
                    initial_object_grid[y, x] > 0
                    and current_state.object_state.object_id[y, x] > 0
                ):
                    # For FourierObjects, check if the state parameters are identical
                    # If they are, it's the same object instance (not a new spawn)
                    if jnp.array_equal(
                        initial_object_state_grid[y, x],
                        current_state.object_state.state_params[y, x],
                    ):
                        old_objects_remaining += 1

    assert old_objects_remaining > 0, (
        "At least one old object should persist after regeneration"
    )

    # 3. Consumption counter should be reset
    assert current_state.biome_state.consumption_count[0] == 0, (
        "Consumption count should reset"
    )

    # 4. New total objects should be tracked correctly
    actual_objects = jnp.sum((current_state.object_state.object_id > 0) & biome_0_mask)
    new_total = current_state.biome_state.total_objects[0]
    assert new_total <= actual_objects, (
        f"New total {new_total} should be <= actual {actual_objects} "
        "(actual may include preserved old objects)"
    )


def test_biome_regeneration_updates_only_new_objects():
    """Test that object properties are only updated for newly spawned objects."""
    # Create environment with Fourier objects
    OBJ = FourierObject(
        name="OBJ",
        base_magnitude=1.0,
        color=(255, 0, 0),
    )

    size = (10, 10)
    biomes = (
        Biome(
            start=(0, 0),
            stop=(10, 10),
            object_frequencies=(0.8,),  # High occupancy
        ),
    )

    env = ForagaxEnv(
        size=size,
        aperture_size=5,
        objects=(OBJ,),
        biomes=biomes,
        dynamic_biomes=True,
        biome_consumption_threshold=0.4,  # Lower threshold for easier testing
        deterministic_spawn=False,
    )

    key = jax.random.key(123)
    key_reset, key_step = jax.random.split(key)

    obs, state = env.reset(key_reset, env.default_params)

    # Store initial Fourier parameters for all object positions
    initial_params = {}
    for y in range(size[1]):
        for x in range(size[0]):
            if state.object_state.object_id[y, x] > 0:
                initial_params[(x, y)] = state.object_state.state_params[y, x].copy()

    print(f"Initial objects: {len(initial_params)}")

    # Force respawn by consuming objects
    current_state = state
    biome_mask = state.object_state.biome_id == 0
    objects_in_biome = jnp.sum((state.object_state.object_id > 0) & biome_mask)
    consumption_needed = int(jnp.ceil(objects_in_biome * 0.5))

    collections = 0
    for y in range(size[1]):
        for x in range(size[0]):
            if (
                collections >= consumption_needed + 10
            ):  # Collect extra to ensure threshold is hit
                break
            if biome_mask[y, x] and current_state.object_state.object_id[y, x] > 0:
                current_state = current_state.replace(pos=jnp.array([x, y]))
                key_step, step_key = jax.random.split(key_step)
                obs, current_state, reward, done, info = env.step(
                    step_key, current_state, 0, env.default_params
                )
                if info["object_collected_id"] >= 0:
                    collections += 1

        if current_state.biome_state.generation[0] > 0:
            break

    assert current_state.biome_state.generation[0] > 0, (
        f"Biome should have respawned after {collections} collections"
    )

    # After respawn, check that:
    # 1. NEW object positions have DIFFERENT parameters
    # 2. OLD object positions (not replaced) should have SAME parameters OR
    #    be replaced with new objects

    params_changed_count = 0
    params_same_count = 0

    for y in range(size[1]):
        for x in range(size[0]):
            if current_state.object_state.object_id[y, x] > 0:
                new_params = current_state.object_state.state_params[y, x]

                # Check if this position had an object initially
                if (x, y) in initial_params:
                    old_params = initial_params[(x, y)]
                    if jnp.allclose(old_params, new_params):
                        params_same_count += 1
                    else:
                        params_changed_count += 1
                else:
                    # This is a truly new object
                    params_changed_count += 1

    print(f"Parameters changed: {params_changed_count}")
    print(f"Parameters same: {params_same_count}")

    # The respawn should generate new objects with new parameters
    # Some positions might be replaced, others might not be
    # But we should see SOME new parameters
    assert params_changed_count > 0, "Should have some objects with new parameters"


def test_consumption_threshold_per_generation():
    """Test that consumption tracking resets per generation.

    With deterministic spawn and threshold 0.5:
    - Initial spawn: 5 objects (generation 0)
    - After consuming 3 objects: spawn 5 new objects (generation 1)
    - Consuming remaining gen0 objects should NOT count toward gen1 consumption

    The key is that consumption count resets to 0 after each regeneration,
    and only objects from the current generation count toward consumption.
    """
    OBJ_A = FourierObject(
        name="OBJ_A",
        base_magnitude=1.0,
        color=(255, 0, 0),
    )

    size = (3, 3)  # 9 cells total
    biomes = (
        Biome(
            start=(0, 0),
            stop=(3, 3),
            object_frequencies=(0.5,),  # All cells have objects
        ),
    )

    env = ForagaxEnv(
        size=size,
        aperture_size=3,
        objects=(OBJ_A,),
        biomes=biomes,
        dynamic_biomes=True,
        biome_consumption_threshold=0.5,  # 50% threshold
        deterministic_spawn=True,  # Deterministic spawn for predictable behavior
    )

    key = jax.random.key(42)
    key_reset, key_step = jax.random.split(key)

    # Reset environment
    obs, state = env.reset(key_reset, env.default_params)

    # Verify initial state
    biome_mask = state.object_state.biome_id == 0
    initial_objects = jnp.sum((state.object_state.object_id > 0) & biome_mask)
    assert initial_objects == 5, f"Should start with 5 objects, got {initial_objects}"
    assert state.biome_state.generation[0] == 0, "Should start at generation 0"
    assert state.biome_state.total_objects[0] == 5, "Should track 5 total objects"
    assert state.biome_state.consumption_count[0] == 0, (
        "Should start with 0 consumption"
    )

    # Store generation 0 properties and positions
    gen0_params = {}
    for y in range(size[1]):
        for x in range(size[0]):
            if state.object_state.object_id[y, x] > 0:
                gen0_params[(x, y)] = state.object_state.state_params[y, x].copy()

    # Consume 3 objects (>= 50% of 5 = 2.5, so 3 triggers respawn)
    # We need to collect objects by moving onto them, not by teleporting
    current_state = state
    collections = 0
    target_collections = 3
    gen0_collected_positions = set()

    # Collect objects by actually moving onto them
    max_steps = 100  # Safety limit
    steps = 0
    while collections < target_collections and steps < max_steps:
        # Find the nearest object
        found = False
        for y in range(size[1]):
            for x in range(size[0]):
                if current_state.object_state.object_id[y, x] > 0:
                    # Position agent next to object (above it)
                    if y > 0:
                        current_state = current_state.replace(pos=jnp.array([x, y - 1]))
                        action = 0  # DOWN
                    elif x > 0:
                        current_state = current_state.replace(pos=jnp.array([x - 1, y]))
                        action = 1  # RIGHT
                    elif y < size[1] - 1:
                        current_state = current_state.replace(pos=jnp.array([x, y + 1]))
                        action = 2  # UP
                    else:
                        current_state = current_state.replace(pos=jnp.array([x + 1, y]))
                        action = 3  # LEFT

                    key_step, step_key = jax.random.split(key_step)
                    obs, current_state, reward, done, info = env.step(
                        step_key, current_state, action, env.default_params
                    )

                    if info["object_collected_id"] >= 0:
                        gen0_collected_positions.add((x, y))
                        collections += 1
                        found = True
                        break
            if found:
                break
        steps += 1

    # Verify generation 1 spawned
    assert current_state.biome_state.generation[0] == 1, (
        "Should be at generation 1 after consuming 3 objects"
    )
    assert current_state.biome_state.consumption_count[0] == 0, (
        "Consumption should reset to 0 after respawn"
    )

    # Verify we have objects from generation 1
    # Note: total visible objects includes both gen 1 (new) and gen 0 (preserved)
    # biome_total_objects tracks only the NEW gen 1 objects spawned
    gen1_objects_total = jnp.sum(
        (current_state.object_state.object_id > 0) & biome_mask
    )
    assert gen1_objects_total >= 5, (
        f"Should have at least 5 objects (new gen1 + preserved gen0), got {gen1_objects_total}"
    )
    assert current_state.biome_state.total_objects[0] == 5, (
        "Should track 5 NEW objects spawned in gen 1"
    )

    # Store generation 1 properties
    gen1_params = {}
    for y in range(size[1]):
        for x in range(size[0]):
            if current_state.object_state.object_id[y, x] > 0:
                gen1_params[(x, y)] = current_state.object_state.state_params[
                    y, x
                ].copy()

    # Verify that generation 1 has DIFFERENT parameters than generation 0
    params_different = False
    for pos in gen1_params:
        if pos in gen0_params:
            if not jnp.allclose(gen0_params[pos], gen1_params[pos]):
                params_different = True
                break
    assert params_different, (
        "Generation 1 should have different Fourier parameters than generation 0"
    )

    # Find any remaining gen0 objects (preserved because new spawn didn't place object there)
    gen0_remaining_positions = []
    for pos in gen0_params:
        if pos not in gen0_collected_positions:  # Not collected
            # Check if it's still the same object by comparing params
            x, y = pos  # pos is (x, y) tuple
            if current_state.object_state.object_id[y, x] > 0:
                if jnp.allclose(
                    gen0_params[pos], current_state.object_state.state_params[y, x]
                ):
                    gen0_remaining_positions.append(pos)

    # Consume a gen0 object and verify they DON'T count toward gen1 consumption
    consumption_before = current_state.biome_state.consumption_count[0]

    # Just consume one gen0 object - position agent adjacent and move onto it
    assert len(gen0_remaining_positions) > 0, (
        "Should have at least one gen0 object remaining"
    )
    x, y = gen0_remaining_positions[0]

    # Position agent next to the object and move onto it
    if y > 0:
        current_state = current_state.replace(pos=jnp.array([x, y - 1]))
        action = 0  # DOWN
    elif x > 0:
        current_state = current_state.replace(pos=jnp.array([x - 1, y]))
        action = 1  # RIGHT
    elif y < size[1] - 1:
        current_state = current_state.replace(pos=jnp.array([x, y + 1]))
        action = 2  # UP
    else:
        current_state = current_state.replace(pos=jnp.array([x + 1, y]))
        action = 3  # LEFT

    key_step, step_key = jax.random.split(key_step)
    obs, current_state, reward, done, info = env.step(
        step_key, current_state, action, env.default_params
    )

    # Assert that we collected an object
    assert info["object_collected_id"] >= 0, "Should have collected a gen0 object"

    # Consumption count should NOT increase when collecting old gen0 objects
    assert current_state.biome_state.consumption_count[0] == consumption_before, (
        "Consuming gen0 objects should NOT count toward gen1 consumption"
    )


def test_biome_respawn_maintains_total_object_count_nondeterministic():
    """Test that biome respawn maintains the same total object count.

    When a biome respawns, it should generate a NEW set of objects based on
    the biome's object frequencies, replacing ALL objects in the biome.
    The total object count should remain approximately the same (within
    statistical variance for random spawn).
    """
    OBJ = FourierObject(
        name="OBJ",
        base_magnitude=1.0,
        color=(255, 0, 0),
    )

    size = (5, 5)  # 25 cells
    biomes = (
        Biome(
            start=(0, 0),
            stop=(5, 5),
            object_frequencies=(0.6,),  # ~60% occupancy = ~15 objects
        ),
    )

    env = ForagaxEnv(
        size=size,
        aperture_size=5,
        objects=(OBJ,),
        biomes=biomes,
        dynamic_biomes=True,
        biome_consumption_threshold=0.5,  # 50% threshold
        deterministic_spawn=False,  # Random spawn to test statistical behavior
    )

    key = jax.random.key(42)
    key_reset, key_step = jax.random.split(key)

    obs, state = env.reset(key_reset, env.default_params)

    # Record initial object count
    biome_mask = state.object_state.biome_id == 0
    initial_total = state.biome_state.total_objects[0]
    initial_count = jnp.sum((state.object_state.object_id > 0) & biome_mask)

    assert initial_total == initial_count, "Initial total should match actual count"

    # Consume objects to trigger respawn
    # Collect enough objects by stepping through the grid systematically
    current_state = state
    target_collections = (initial_total // 2) + 1  # Need > 50%

    # Collect objects by moving agent directly to each object position
    collected = 0
    for y in range(size[1]):
        for x in range(size[0]):
            if collected >= target_collections:
                break
            # Check if there's a collectable object at this position in ORIGINAL state
            if state.object_state.object_id[y, x] > 0:
                # Move agent to this position
                current_state = current_state.replace(pos=jnp.array([x, y]))
                # Step to collect
                key_step, step_key = jax.random.split(key_step)
                obs, current_state, reward, done, info = env.step(
                    step_key, current_state, 0, env.default_params
                )
                if info["object_collected_id"] >= 0:
                    collected += 1
        if collected >= target_collections:
            break

    # Verify respawn happened
    assert current_state.biome_state.generation[0] == 1, (
        f"Should have respawned after collecting {collected}/{initial_total} "
        f"(threshold: {env.biome_consumption_threshold})"
    )

    # Check new object count
    new_total = current_state.biome_state.total_objects[0]
    new_count_all = jnp.sum((current_state.object_state.object_id > 0) & biome_mask)

    # The new_total should reflect only NEWLY spawned objects (from this generation)
    # NOT the total including old preserved objects
    # So new_total should be similar to initial_total (both based on same frequency)
    variance_threshold = 0.4
    lower_bound = initial_total * (1 - variance_threshold)
    upper_bound = initial_total * (1 + variance_threshold)

    assert lower_bound <= new_total <= upper_bound, (
        f"New total {new_total} should be similar to initial total {initial_total} "
        f"(expected range: {lower_bound:.1f} - {upper_bound:.1f})"
    )

    # The total_in_biome might be higher than new_total because it includes
    # old unconsumed objects that were preserved
    assert new_count_all >= new_total, (
        f"Total in biome {new_count_all} should be >= new_total {new_total} "
        "(may include preserved old objects)"
    )


def test_biome_respawn_maintains_total_object_count_deterministic():
    """Test that biome respawn maintains the same total object count.

    When a biome respawns, it should generate a NEW set of objects based on
    the biome's object frequencies, replacing ALL objects in the biome.
    The total object count should remain the same.
    """
    OBJ = FourierObject(
        name="OBJ",
        base_magnitude=1.0,
        color=(255, 0, 0),
    )

    size = (5, 5)  # 25 cells
    biomes = (
        Biome(
            start=(0, 0),
            stop=(5, 5),
            object_frequencies=(0.6,),  # ~60% occupancy = ~15 objects
        ),
    )

    env = ForagaxEnv(
        size=size,
        aperture_size=5,
        objects=(OBJ,),
        biomes=biomes,
        dynamic_biomes=True,
        biome_consumption_threshold=0.5,  # 50% threshold
        deterministic_spawn=True,
    )

    key = jax.random.key(42)
    key_reset, key_step = jax.random.split(key)

    obs, state = env.reset(key_reset, env.default_params)

    # Record initial object count
    biome_mask = state.object_state.biome_id == 0
    initial_total = state.biome_state.total_objects[0]
    initial_count = jnp.sum((state.object_state.object_id > 0) & biome_mask)

    assert initial_total == initial_count, "Initial total should match actual count"

    # Consume objects to trigger respawn
    # Collect enough objects by stepping through the grid systematically
    current_state = state
    target_collections = (initial_total // 2) + 1  # Need > 50%

    # Collect objects by moving agent directly to each object position
    collected = 0
    for y in range(size[1]):
        for x in range(size[0]):
            if collected >= target_collections:
                break
            # Check if there's a collectable object at this position in ORIGINAL state
            if state.object_state.object_id[y, x] > 0:
                # Move agent to this position
                current_state = current_state.replace(pos=jnp.array([x, y]))
                # Take NOOP action to collect
                key_step, step_key = jax.random.split(key_step)
                obs, current_state, reward, done, info = env.step(
                    step_key, current_state, 0, env.default_params
                )
                if info["object_collected_id"] >= 0:
                    collected += 1
        if collected >= target_collections:
            break

    # Verify respawn happened
    assert current_state.biome_state.generation[0] == 1, (
        f"Should have respawned after collecting {collected}/{initial_total} "
        f"(threshold: {env.biome_consumption_threshold})"
    )

    # Check new object count
    new_total = current_state.biome_state.total_objects[0]
    new_count_all = jnp.sum((current_state.object_state.object_id > 0) & biome_mask)

    assert new_total == initial_total, (
        f"New total {new_total} should equal initial total {initial_total} "
    )

    # The total_in_biome might be higher than new_total because it includes
    # old unconsumed objects that were preserved
    assert new_count_all >= new_total, (
        f"Total in biome {new_count_all} should be >= new_total {new_total} "
        "(may include preserved old objects)"
    )


def test_empty_object_has_no_sampled_color():
    """Test that EMPTY object (index 0) does not get a sampled color during spawn."""
    OBJ_A = FourierObject(
        name="OBJ_A",
        base_magnitude=1.0,
        color=(255, 0, 0),
    )
    OBJ_B = FourierObject(
        name="OBJ_B",
        base_magnitude=1.0,
        color=(0, 255, 0),
    )

    size = (10, 10)
    biomes = (
        Biome(
            start=(0, 0),
            stop=(10, 10),
            object_frequencies=(0.3, 0.3),  # 60% occupancy, 40% empty
        ),
    )

    env = ForagaxEnv(
        size=size,
        aperture_size=5,
        objects=(OBJ_A, OBJ_B),
        biomes=biomes,
        dynamic_biomes=True,
        biome_consumption_threshold=0.9,
        deterministic_spawn=False,
    )

    key = jax.random.key(999)
    obs, state = env.reset(key, env.default_params)

    # Check that all EMPTY positions (object_grid == 0) have color [255, 255, 255]
    empty_mask = state.object_state.object_id == 0
    empty_colors = state.object_state.color[empty_mask]

    # All empty positions should have [255, 255, 255] color (default empty color)
    expected_empty_color = jnp.full(3, 255, dtype=jnp.uint8)

    # Check each empty position
    for color in empty_colors:
        chex.assert_trees_all_equal(color, expected_empty_color)

    print(
        f"Found {jnp.sum(empty_mask)} empty positions, all with [255, 255, 255] color"
    )

    # Additionally check that non-empty objects DO have colors
    non_empty_mask = state.object_state.object_id > 0
    non_empty_colors = state.object_state.color[non_empty_mask]

    # At least some non-empty objects should have non-zero colors
    has_color = jnp.any(non_empty_colors != 0, axis=1)
    assert jnp.sum(has_color) > 0, "Non-empty objects should have colors"

    print(f"Found {jnp.sum(non_empty_mask)} non-empty positions with colors")


def test_fourier_reward_zero_parameters():
    """Test that Fourier objects with zero parameters return zero reward."""
    fourier_obj = FourierObject(
        name="test_fourier",
        num_fourier_terms=5,
        base_magnitude=2.0,
        color=(255, 0, 0),
    )

    key = jax.random.key(42)
    zero_params = jnp.zeros(3 + 2 * fourier_obj.num_fourier_terms, dtype=jnp.float32)
    reward_zero = fourier_obj.reward(100, key, zero_params)
    assert reward_zero == 0.0, "Zero parameters should give zero reward"


def test_fourier_reward_parameter_diversity():
    """Test that different Fourier objects have different parameters and basic properties."""
    fourier_obj = FourierObject(
        name="test_fourier",
        num_fourier_terms=5,
        base_magnitude=2.0,
        color=(255, 0, 0),
    )

    key = jax.random.key(42)
    num_test_objects = 10
    test_timesteps = jnp.array([0, 50, 100, 200, 500, 1000])

    rewards_over_time = []
    all_params = []

    for i in range(num_test_objects):
        key, param_key = jax.random.split(key)
        params = fourier_obj.get_state(param_key)
        all_params.append(params)

        # Compute rewards at different timesteps
        object_rewards = []
        for t in test_timesteps:
            reward = fourier_obj.reward(t, jax.random.key(i), params)
            object_rewards.append(reward)
            # All rewards should be finite
            assert jnp.isfinite(reward), f"Reward should be finite at timestep {t}"
            # Rewards should be within expected bounds [-base_magnitude, base_magnitude]
            assert (
                -fourier_obj.base_magnitude <= reward <= fourier_obj.base_magnitude
            ), (
                f"Reward {reward} should be within [{-fourier_obj.base_magnitude}, {fourier_obj.base_magnitude}]"
            )
        rewards_over_time.append(jnp.array(object_rewards))

    # Different objects should have different parameters (with high probability)
    params_array = jnp.stack(all_params)
    params_differ = False
    for i in range(num_test_objects):
        for j in range(i + 1, num_test_objects):
            if not jnp.allclose(params_array[i], params_array[j], atol=1e-6):
                params_differ = True
                break
        if params_differ:
            break
    assert params_differ, "Different objects should have different parameters"

    # Rewards should vary over time for most objects
    rewards_array = jnp.stack(rewards_over_time)  # Shape: (num_objects, num_timesteps)
    for i in range(num_test_objects):
        obj_rewards = rewards_array[i]
        reward_variation = jnp.max(obj_rewards) - jnp.min(obj_rewards)
        assert reward_variation > 1e-6, f"Object {i} rewards should vary over time"


def test_fourier_reward_periodicity():
    """Test that Fourier rewards exhibit proper periodicity."""
    fourier_obj = FourierObject(
        name="test_fourier",
        num_fourier_terms=5,
        base_magnitude=2.0,
        color=(255, 0, 0),
    )

    key = jax.random.key(42)
    test_period = 100  # Use a reasonable period for testing
    period_params = jnp.array(
        [test_period, -1.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=jnp.float32,
    )  # Simple sine wave

    reward_t0 = fourier_obj.reward(0, key, period_params)
    reward_t_period = fourier_obj.reward(test_period, key, period_params)
    reward_t_2period = fourier_obj.reward(2 * test_period, key, period_params)

    # Should be approximately periodic
    chex.assert_trees_all_close(reward_t0, reward_t_period, rtol=1e-5)
    chex.assert_trees_all_close(reward_t0, reward_t_2period, rtol=1e-5)


def test_fourier_reward_mathematical_properties():
    """Test mathematical correctness of Fourier series computation."""
    fourier_obj = FourierObject(
        name="test_fourier",
        num_fourier_terms=5,
        base_magnitude=2.0,
        color=(255, 0, 0),
    )

    key = jax.random.key(42)

    # Create a simple single-term Fourier series: a1*cos(t) + b1*sin(t)
    # We use period=8.0 so that clock=1 corresponds to phase 2π/8 = π/4
    simple_params = jnp.array(
        [
            8.0,  # period
            -1.0,
            1.0,  # min=-1, max=1
            0.5,
            0.5,  # a1=0.5, b1=0.5
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=jnp.float32,
    )

    # At t=0 (clock=0): phase=0
    # 0.5*(cos(0) + sin(0)) = 0.5*(1 + 0) = 0.5
    reward_t0_simple = fourier_obj.reward(0, key, simple_params)
    # The actual reward is normalized: 2*(raw - min)/(max - min) - 1, then * base_magnitude
    # raw = 0.5, min=-1, max=1, so: 2*(0.5 - (-1))/(1 - (-1)) - 1 = 2*(1.5)/2 - 1 = 1.5 - 1 = 0.5
    # Then: 0.5 * base_magnitude = 0.5 * 2.0 = 1.0
    assert jnp.allclose(reward_t0_simple, 1.0, atol=1e-5), (
        f"Expected ~1.0, got {reward_t0_simple}"
    )

    # At t=1 (clock=1): phase = 2π * 1/8 = π/4
    # 0.5*(cos(π/4) + sin(π/4)) = 0.5*(0.707 + 0.707) = 0.5*1.414 ≈ 0.707
    reward_t_quarter = fourier_obj.reward(1, key, simple_params)
    t_quarter = jnp.pi / 4
    raw_quarter = 0.5 * (jnp.cos(t_quarter) + jnp.sin(t_quarter))
    expected_quarter = (
        2 * (raw_quarter - (-1)) / (1 - (-1)) - 1
    ) * fourier_obj.base_magnitude
    chex.assert_trees_all_close(reward_t_quarter, expected_quarter, rtol=1e-4)


def test_sine_object():
    """Test the SineObject class directly."""
    key = jax.random.key(0)

    # Create a sine object with base_reward=10, amplitude=20, period=1000
    sine_obj = SineObject(
        name="test_sine",
        base_reward=10.0,
        amplitude=20.0,
        period=1000,
        phase=0.0,
        regen_delay=(9, 11),
        color=(255, 0, 0),
    )

    # Test reward at different time points
    # At t=0: sin(0) = 0, reward = 10 + 20*0 = 10
    r0 = sine_obj.reward(0, key, None)
    assert jnp.allclose(r0, 10.0, atol=0.01)

    # At t=250 (quarter period): sin(2π*250/1000) = sin(π/2) = 1, reward = 10 + 20*1 = 30
    r250 = sine_obj.reward(250, key, None)
    assert jnp.allclose(r250, 30.0, atol=0.01)

    # At t=500 (half period): sin(π) = 0, reward = 10 + 20*0 = 10
    r500 = sine_obj.reward(500, key, None)
    assert jnp.allclose(r500, 10.0, atol=0.01)

    # At t=750 (three-quarter period): sin(3π/2) = -1, reward = 10 + 20*(-1) = -10
    r750 = sine_obj.reward(750, key, None)
    assert jnp.allclose(r750, -10.0, atol=0.01)

    # At t=1000 (full period): sin(2π) = 0, reward = 10 + 20*0 = 10
    r1000 = sine_obj.reward(1000, key, None)
    assert jnp.allclose(r1000, 10.0, atol=0.01)

    # Test with phase shift (π radians = 180 degrees)
    sine_obj_shifted = SineObject(
        name="test_sine_shifted",
        base_reward=-10.0,
        amplitude=20.0,
        period=1000,
        phase=jnp.pi,
        regen_delay=(9, 11),
        color=(0, 255, 0),
    )

    # At t=0 with phase=π: sin(0 + π) = sin(π) = 0, reward = -10 + 20*0 = -10
    r0_shifted = sine_obj_shifted.reward(0, key, None)
    assert jnp.allclose(r0_shifted, -10.0, atol=0.01)

    # At t=250 with phase=π: sin(π/2 + π) = sin(3π/2) = -1, reward = -10 + 20*(-1) = -30
    r250_shifted = sine_obj_shifted.reward(250, key, None)
    assert jnp.allclose(r250_shifted, -30.0, atol=0.01)

    # Verify phase inversion: original + shifted should sum to 0
    for t in [0, 250, 500, 750]:
        r_orig = sine_obj.reward(t, key, None)
        r_shift = sine_obj_shifted.reward(t, key, None)
        assert jnp.allclose(r_orig + r_shift, 0.0, atol=0.01)


def test_info_rewards():
    """Test that info contains rewards with next reward values for each grid position."""
    key = jax.random.key(0)
    env = ForagaxEnv(
        size=(5, 5),
        objects=(FLOWER,),
        biomes=(Biome(object_frequencies=(0.5,)),),
        observation_type="object",
    )
    params = env.default_params
    obs, state = env.reset(key, params)

    key, step_key = jax.random.split(key)
    _, state, _, _, info = env.step(step_key, state, Actions.UP, params)

    # Check that rewards is in info
    assert "rewards" in info

    # Check shape matches environment size
    assert info["rewards"].shape == (5, 5)

    # Check that positions with objects have non-zero rewards
    object_mask = state.object_state.object_id > 0
    rewards_at_objects = info["rewards"][object_mask]
    assert jnp.all(rewards_at_objects == FLOWER.reward_val)

    # Check that empty positions have zero rewards
    empty_mask = state.object_state.object_id == 0
    rewards_at_empty = info["rewards"][empty_mask]
    assert jnp.all(rewards_at_empty == 0.0)
