import chex
import jax
import jax.numpy as jnp

from foragax.registry import make


def test_foragax_weather_v4_registry():
    """Test that ForagaxWeather-v4 can be created via registry and has correct config."""
    env = make("ForagaxWeather-v4", aperture_size=(5, 5))

    # Check basic configuration
    assert env.name == "ForagaxWeather-v4"
    assert env.deterministic_spawn is True
    assert env.nowrap is True

    # Check that weather objects have random_respawn=True
    hot, cold = env.objects[1], env.objects[2]  # Skip EMPTY (index 0)
    assert hot.name == "hot"
    assert cold.name == "cold"
    assert hot.random_respawn is True
    assert cold.random_respawn is True

    # Test basic functionality
    key = jax.random.key(0)
    obs, state = env.reset(key, env.default_params)
    assert obs.shape == (5, 5, 2)  # 2 color channels (hot/cold, padding)

    # Test stepping
    key, step_key = jax.random.split(key)
    action = env.action_space(env.default_params).sample(step_key)
    obs2, state2, reward, done, info = env.step(
        step_key, state, action, env.default_params
    )
    assert obs2.shape == (5, 5, 2)
    assert not done


def test_foragax_weather_v4_deterministic_spawn():
    """Test that ForagaxWeather-v4 uses deterministic spawning."""
    env = make("ForagaxWeather-v4", aperture_size=(5, 5))
    params = env.default_params

    # Test that multiple resets with same key produce same object placement
    key = jax.random.key(42)
    _, state1 = env.reset(key, params)

    key = jax.random.key(42)  # Same key
    _, state2 = env.reset(key, params)

    # Object grids should be identical (deterministic spawn)
    chex.assert_trees_all_equal(state1.object_grid, state2.object_grid)

    # But different keys should produce different placements
    key1 = jax.random.key(42)
    key2 = jax.random.key(43)
    _, state1 = env.reset(key1, params)
    _, state2 = env.reset(key2, params)

    # Should be different (shuffled deterministically)
    assert not jnp.array_equal(state1.object_grid, state2.object_grid)

    # Test that number of objects is the same
    num_hot_1 = jnp.sum(state1.object_grid == 1)
    num_cold_1 = jnp.sum(state1.object_grid == 2)
    num_hot_2 = jnp.sum(state2.object_grid == 1)
    num_cold_2 = jnp.sum(state2.object_grid == 2)
    assert num_hot_1 == num_hot_2
    assert num_cold_1 == num_cold_2


def test_foragax_weather_v4_random_respawn():
    """Test that ForagaxWeather-v4 weather objects have random_respawn=True."""
    env = make("ForagaxWeather-v4", aperture_size=(5, 5))

    # Check that weather objects have random_respawn=True
    hot, cold = env.objects[1], env.objects[2]  # Skip EMPTY
    assert hot.random_respawn is True, "Hot objects should have random_respawn=True"
    assert cold.random_respawn is True, "Cold objects should have random_respawn=True"

    # Compare with v3
    env_v3 = make("ForagaxWeather-v3", aperture_size=(5, 5))
    hot_v3, cold_v3 = env_v3.objects[1], env_v3.objects[2]
    assert hot_v3.random_respawn is False, (
        "v3 hot objects should have random_respawn=False"
    )
    assert cold_v3.random_respawn is False, (
        "v3 cold objects should have random_respawn=False"
    )


def test_foragax_weather_v4_color_configuration():
    """Test that ForagaxWeather-v4 weather objects use the same color."""
    env = make("ForagaxWeather-v4", aperture_size=(5, 5))
    hot_v4, cold_v4 = env.objects[1], env.objects[2]

    # Same color configuration: v4 uses same color
    assert hot_v4.color == cold_v4.color, "v4 should use same color for hot and cold"


def test_foragax_weather_v5_registry():
    """Test that ForagaxWeather-v5 can be created via registry and has correct config."""
    env = make("ForagaxWeather-v5", aperture_size=(5, 5))

    # Check basic configuration
    assert env.name == "ForagaxWeather-v5"
    assert env.deterministic_spawn is True
    assert env.nowrap is False  # v5 enables wrapping

    # Check that weather objects have random_respawn=True
    hot, cold = env.objects[1], env.objects[2]  # Skip EMPTY (index 0)
    assert hot.name == "hot"
    assert cold.name == "cold"
    assert hot.random_respawn is True
    assert cold.random_respawn is True

    # Test basic functionality
    key = jax.random.key(0)
    obs, state = env.reset(key, env.default_params)
    assert obs.shape == (5, 5, 1)  # 1 color channel (hot/cold same color, no padding)

    # Test stepping
    key, step_key = jax.random.split(key)
    action = env.action_space(env.default_params).sample(step_key)
    obs2, state2, reward, done, info = env.step(
        step_key, state, action, env.default_params
    )
    assert obs2.shape == (5, 5, 1)
    assert not done


def test_foragax_weather_v5_deterministic_spawn():
    """Test that ForagaxWeather-v5 uses deterministic spawning."""
    env = make("ForagaxWeather-v5", aperture_size=(5, 5))
    params = env.default_params

    # Test that multiple resets with same key produce same object placement
    key = jax.random.key(42)
    _, state1 = env.reset(key, params)

    key = jax.random.key(42)  # Same key
    _, state2 = env.reset(key, params)

    # Object grids should be identical (deterministic spawn)
    chex.assert_trees_all_equal(state1.object_grid, state2.object_grid)

    # But different keys should produce different placements
    key1 = jax.random.key(42)
    key2 = jax.random.key(43)
    _, state1 = env.reset(key1, params)
    _, state2 = env.reset(key2, params)

    # Should be different (shuffled deterministically)
    assert not jnp.array_equal(state1.object_grid, state2.object_grid)

    # Test that number of objects is the same
    num_hot_1 = jnp.sum(state1.object_grid == 1)
    num_cold_1 = jnp.sum(state1.object_grid == 2)
    num_hot_2 = jnp.sum(state2.object_grid == 1)
    num_cold_2 = jnp.sum(state2.object_grid == 2)
    assert num_hot_1 == num_hot_2
    assert num_cold_1 == num_cold_2


def test_foragax_weather_v5_random_respawn():
    """Test that ForagaxWeather-v5 weather objects have random_respawn=True."""
    env = make("ForagaxWeather-v5", aperture_size=(5, 5))

    # Check that weather objects have random_respawn=True
    hot, cold = env.objects[1], env.objects[2]  # Skip EMPTY
    assert hot.random_respawn is True, "Hot objects should have random_respawn=True"
    assert cold.random_respawn is True, "Cold objects should have random_respawn=True"


def test_foragax_weather_v5_color_configuration():
    """Test that ForagaxWeather-v5 weather objects use the same color."""
    env = make("ForagaxWeather-v5", aperture_size=(5, 5))
    hot_v5, cold_v5 = env.objects[1], env.objects[2]

    # Same color configuration: v5 uses same color
    assert hot_v5.color == cold_v5.color, "v5 should use same color for hot and cold"


def test_foragax_twobiome_v10_registry():
    """Test that ForagaxTwoBiome-v10 can be created via registry and has correct config."""
    env = make("ForagaxTwoBiome-v10", aperture_size=(5, 5))

    # Check basic configuration
    assert env.name == "ForagaxTwoBiome-v10"
    assert env.deterministic_spawn is True
    assert env.nowrap is True

    # Check that objects have random_respawn=True
    morel, oyster, deathcap, fake = (
        env.objects[1],
        env.objects[2],
        env.objects[3],
        env.objects[4],
    )  # Skip EMPTY
    assert morel.name == "brown_morel"
    assert oyster.name == "brown_oyster"
    assert deathcap.name == "green_deathcap"
    assert fake.name == "green_fake"
    assert morel.random_respawn is True
    assert oyster.random_respawn is True
    assert deathcap.random_respawn is True
    assert fake.random_respawn is True

    # Test basic functionality
    key = jax.random.key(0)
    obs, state = env.reset(key, env.default_params)
    assert obs.shape == (5, 5, 3)  # 3 color channels (brown, green, black padding)

    # Test stepping
    key, step_key = jax.random.split(key)
    action = env.action_space(env.default_params).sample(step_key)
    obs2, state2, reward, done, info = env.step(
        step_key, state, action, env.default_params
    )
    assert obs2.shape == (5, 5, 3)
    assert not done


def test_foragax_twobiome_v10_deterministic_spawn():
    """Test that ForagaxTwoBiome-v10 uses deterministic spawning."""
    env = make("ForagaxTwoBiome-v10", aperture_size=(5, 5))
    params = env.default_params

    # Test that multiple resets with same key produce same object placement
    key = jax.random.key(42)
    _, state1 = env.reset(key, params)

    key = jax.random.key(42)  # Same key
    _, state2 = env.reset(key, params)

    # Object grids should be identical (deterministic spawn)
    chex.assert_trees_all_equal(state1.object_grid, state2.object_grid)

    # But different keys should produce different placements
    key1 = jax.random.key(42)
    key2 = jax.random.key(43)
    _, state1 = env.reset(key1, params)
    _, state2 = env.reset(key2, params)

    # Should be different (shuffled deterministically)
    assert not jnp.array_equal(state1.object_grid, state2.object_grid)

    # Test that number of objects is the same
    num_morel_1 = jnp.sum(state1.object_grid == 1)
    num_oyster_1 = jnp.sum(state1.object_grid == 2)
    num_deathcap_1 = jnp.sum(state1.object_grid == 3)
    num_fake_1 = jnp.sum(state1.object_grid == 4)
    num_morel_2 = jnp.sum(state2.object_grid == 1)
    num_oyster_2 = jnp.sum(state2.object_grid == 2)
    num_deathcap_2 = jnp.sum(state2.object_grid == 3)
    num_fake_2 = jnp.sum(state2.object_grid == 4)
    assert num_morel_1 == num_morel_2
    assert num_oyster_1 == num_oyster_2
    assert num_deathcap_1 == num_deathcap_2
    assert num_fake_1 == num_fake_2


def test_foragax_twobiome_v10_random_respawn():
    """Test that ForagaxTwoBiome-v10 objects have random_respawn=True."""
    env = make("ForagaxTwoBiome-v10", aperture_size=(5, 5))

    # Check that all objects have random_respawn=True
    morel, oyster, deathcap, fake = (
        env.objects[1],
        env.objects[2],
        env.objects[3],
        env.objects[4],
    )
    assert morel.random_respawn is True, "Morel objects should have random_respawn=True"
    assert oyster.random_respawn is True, (
        "Oyster objects should have random_respawn=True"
    )
    assert deathcap.random_respawn is True, (
        "Deathcap objects should have random_respawn=True"
    )
    assert fake.random_respawn is True, "Fake objects should have random_respawn=True"


def test_foragax_twobiome_v13_registry():
    """Test that ForagaxTwoBiome-v13 can be created via registry and has correct config."""
    env = make("ForagaxTwoBiome-v13", aperture_size=(5, 5))

    # Check basic configuration
    assert env.name == "ForagaxTwoBiome-v13"
    assert env.deterministic_spawn is True
    assert env.nowrap is False

    # Check that objects have random_respawn=True
    morel, oyster, deathcap, fake = (
        env.objects[1],
        env.objects[2],
        env.objects[3],
        env.objects[4],
    )  # Skip EMPTY
    assert morel.name == "brown_morel"
    assert oyster.name == "brown_oyster"
    assert deathcap.name == "green_deathcap"
    assert fake.name == "green_fake"
    assert morel.random_respawn is True
    assert oyster.random_respawn is True
    assert deathcap.random_respawn is True
    assert fake.random_respawn is True

    # Test basic functionality
    key = jax.random.key(0)
    obs, state = env.reset(key, env.default_params)
    assert obs.shape == (5, 5, 2)  # 2 color channels (brown, green)

    # Test stepping
    key, step_key = jax.random.split(key)
    action = env.action_space(env.default_params).sample(step_key)
    obs2, state2, reward, done, info = env.step(
        step_key, state, action, env.default_params
    )
    assert obs2.shape == (5, 5, 2)
    assert not done


def test_foragax_twobiome_v13_deterministic_spawn():
    """Test that ForagaxTwoBiome-v13 uses deterministic spawning."""
    env = make("ForagaxTwoBiome-v13", aperture_size=(5, 5))
    params = env.default_params

    # Test that multiple resets with same key produce same object placement
    key = jax.random.key(42)
    _, state1 = env.reset(key, params)

    key = jax.random.key(42)  # Same key
    _, state2 = env.reset(key, params)

    # Object grids should be identical (deterministic spawn)
    chex.assert_trees_all_equal(state1.object_grid, state2.object_grid)

    # But different keys should produce different placements
    key1 = jax.random.key(42)
    key2 = jax.random.key(43)
    _, state1 = env.reset(key1, params)
    _, state2 = env.reset(key2, params)

    # Should be different (shuffled deterministically)
    assert not jnp.array_equal(state1.object_grid, state2.object_grid)

    # Test that number of objects is the same
    num_morel_1 = jnp.sum(state1.object_grid == 1)
    num_oyster_1 = jnp.sum(state1.object_grid == 2)
    num_deathcap_1 = jnp.sum(state1.object_grid == 3)
    num_fake_1 = jnp.sum(state1.object_grid == 4)
    num_morel_2 = jnp.sum(state2.object_grid == 1)
    num_oyster_2 = jnp.sum(state2.object_grid == 2)
    num_deathcap_2 = jnp.sum(state2.object_grid == 3)
    num_fake_2 = jnp.sum(state2.object_grid == 4)
    assert num_morel_1 == num_morel_2
    assert num_oyster_1 == num_oyster_2
    assert num_deathcap_1 == num_deathcap_2
    assert num_fake_1 == num_fake_2


def test_foragax_twobiome_v13_random_respawn():
    """Test that ForagaxTwoBiome-v13 objects have random_respawn=True."""
    env = make("ForagaxTwoBiome-v13", aperture_size=(5, 5))

    # Check that all objects have random_respawn=True
    morel, oyster, deathcap, fake = (
        env.objects[1],
        env.objects[2],
        env.objects[3],
        env.objects[4],
    )
    assert morel.random_respawn is True, "Morel objects should have random_respawn=True"
    assert oyster.random_respawn is True, (
        "Oyster objects should have random_respawn=True"
    )
    assert deathcap.random_respawn is True, (
        "Deathcap objects should have random_respawn=True"
    )
    assert fake.random_respawn is True, "Fake objects should have random_respawn=True"


def test_repeat_parameter_weather_environments():
    """Test that the repeat parameter controls temperature cycling speed."""
    # Create environments with different repeat values
    env_repeat_100 = make("ForagaxWeather-v1", repeat=100, aperture_size=(5, 5))
    env_repeat_200 = make("ForagaxWeather-v1", repeat=200, aperture_size=(5, 5))

    # Get the weather objects
    hot_100, cold_100 = env_repeat_100.objects[1], env_repeat_100.objects[2]
    hot_200, cold_200 = env_repeat_200.objects[1], env_repeat_200.objects[2]

    # Check that repeat values are set correctly
    assert hot_100.repeat == 100
    assert cold_100.repeat == 100
    assert hot_200.repeat == 200
    assert cold_200.repeat == 200

    # Test temperature cycling by checking rewards at different clock times
    key = jax.random.key(0)

    # At clock=0, both should give the same temperature (first value)
    temp_100_t0 = hot_100.reward(0, key)
    temp_200_t0 = hot_200.reward(0, key)
    assert temp_100_t0 == temp_200_t0, "Temperatures should be identical at clock=0"

    # At clock=100, repeat_100 should move to next temperature (index 1), repeat_200 should stay at first (index 0)
    temp_100_t100 = hot_100.reward(100, key)
    temp_200_t100 = hot_200.reward(100, key)
    assert temp_100_t0 != temp_100_t100, "repeat_100 should cycle at step 100"
    assert temp_200_t0 == temp_200_t100, "repeat_200 should not cycle at step 100"

    # At clock=200, repeat_100 should be at index 2, repeat_200 should be at index 1
    temp_100_t200 = hot_100.reward(200, key)
    temp_200_t200 = hot_200.reward(200, key)
    assert temp_100_t200 != temp_200_t200, (
        "Different repeat values should give different temperatures at step 200"
    )

    # At clock=400, repeat_100 should be at index 4, repeat_200 should be at index 2
    temp_100_t400 = hot_100.reward(400, key)
    temp_200_t400 = hot_200.reward(400, key)
    assert temp_100_t400 != temp_100_t200, "repeat_100 should cycle again at step 400"
    assert temp_200_t400 != temp_200_t200, "repeat_200 should cycle again at step 400"
    assert temp_100_t400 != temp_200_t400, (
        "Different repeat values should give different temperatures at step 400"
    )


def test_reward_delay_parameter_weather_environments():
    """Test that the reward_delay parameter controls digestion delay."""
    # Create environments with different reward_delay values
    env_delays_0 = make("ForagaxWeather-v1", reward_delay=0, aperture_size=(5, 5))
    env_delays_5 = make("ForagaxWeather-v1", reward_delay=5, aperture_size=(5, 5))

    # Get the weather objects
    hot_0, cold_0 = env_delays_0.objects[1], env_delays_0.objects[2]
    hot_5, cold_5 = env_delays_5.objects[1], env_delays_5.objects[2]

    # Check that reward_delay values are set correctly
    assert hot_0.reward_delay_val == 0
    assert cold_0.reward_delay_val == 0
    assert hot_5.reward_delay_val == 5
    assert cold_5.reward_delay_val == 5

    # Test reward_delay function returns the correct values
    key = jax.random.key(0)
    delays_0 = hot_0.reward_delay(0, key)
    delays_5 = hot_5.reward_delay(0, key)

    assert delays_0 == 0, "reward_delay should return 0 for delays=0"
    assert delays_5 == 5, "reward_delay should return 5 for delays=5"
