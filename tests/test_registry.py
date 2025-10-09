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


def test_foragax_weather_v6_registry():
    """Test that ForagaxWeather-v6 can be created via registry and has correct config."""
    env = make("ForagaxWeather-v6", aperture_size=(5, 5))

    # Check basic configuration
    assert env.name == "ForagaxWeather-v6"
    assert env.deterministic_spawn is True
    assert env.nowrap is False  # v6 enables wrapping

    # Check that weather objects have random_respawn=True
    hot, cold = env.objects[1], env.objects[2]  # Skip EMPTY (index 0)
    assert hot.name == "hot"
    assert cold.name == "cold"
    assert hot.random_respawn is True
    assert cold.random_respawn is True

    # Check digestion steps
    assert hot.max_digestion_steps == 10  # digestion_steps=10
    assert cold.max_digestion_steps == 10  # digestion_steps=10

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


def test_foragax_weather_v6_deterministic_spawn():
    """Test that ForagaxWeather-v6 uses deterministic spawning."""
    env = make("ForagaxWeather-v6", aperture_size=(5, 5))
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


def test_foragax_weather_v6_random_respawn():
    """Test that ForagaxWeather-v6 weather objects have random_respawn=True."""
    env = make("ForagaxWeather-v6", aperture_size=(5, 5))

    # Check that weather objects have random_respawn=True
    hot, cold = env.objects[1], env.objects[2]  # Skip EMPTY
    assert hot.random_respawn is True, "Hot objects should have random_respawn=True"
    assert cold.random_respawn is True, "Cold objects should have random_respawn=True"


def test_foragax_weather_v6_color_configuration():
    """Test that ForagaxWeather-v6 weather objects use the same color."""
    env = make("ForagaxWeather-v6", aperture_size=(5, 5))
    hot_v6, cold_v6 = env.objects[1], env.objects[2]

    # Same color configuration: v6 uses same color
    assert hot_v6.color == cold_v6.color, "v6 should use same color for hot and cold"


def test_foragax_weather_v6_digestion_steps():
    """Test that ForagaxWeather-v6 weather objects have digestion_steps=10."""
    env = make("ForagaxWeather-v6", aperture_size=(5, 5))

    # Check that weather objects have digestion_steps=10
    hot, cold = env.objects[1], env.objects[2]  # Skip EMPTY
    assert hot.digestion_steps(0, jax.random.key(0)) == 10, (
        "Hot objects should have digestion_steps=10"
    )
    assert cold.digestion_steps(0, jax.random.key(0)) == 10, (
        "Cold objects should have digestion_steps=10"
    )


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
