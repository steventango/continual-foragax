import chex
import jax
import jax.numpy as jnp

from foragax.objects import FourierObject
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
    chex.assert_trees_all_equal(
        state1.object_state.object_id, state2.object_state.object_id
    )

    # But different keys should produce different placements
    key1 = jax.random.key(42)
    key2 = jax.random.key(43)
    _, state1 = env.reset(key1, params)
    _, state2 = env.reset(key2, params)

    # Should be different (shuffled deterministically)
    assert not jnp.array_equal(
        state1.object_state.object_id, state2.object_state.object_id
    )

    # Test that number of objects is the same
    num_hot_1 = jnp.sum(state1.object_state.object_id == 1)
    num_cold_1 = jnp.sum(state1.object_state.object_id == 2)
    num_hot_2 = jnp.sum(state2.object_state.object_id == 1)
    num_cold_2 = jnp.sum(state2.object_state.object_id == 2)
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
    chex.assert_trees_all_equal(
        state1.object_state.object_id, state2.object_state.object_id
    )

    # But different keys should produce different placements
    key1 = jax.random.key(42)
    key2 = jax.random.key(43)
    _, state1 = env.reset(key1, params)
    _, state2 = env.reset(key2, params)

    # Should be different (shuffled deterministically)
    assert not jnp.array_equal(
        state1.object_state.object_id, state2.object_state.object_id
    )

    # Test that number of objects is the same
    num_hot_1 = jnp.sum(state1.object_state.object_id == 1)
    num_cold_1 = jnp.sum(state1.object_state.object_id == 2)
    num_hot_2 = jnp.sum(state2.object_state.object_id == 1)
    num_cold_2 = jnp.sum(state2.object_state.object_id == 2)
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
    chex.assert_trees_all_equal(
        state1.object_state.object_id, state2.object_state.object_id
    )

    # But different keys should produce different placements
    key1 = jax.random.key(42)
    key2 = jax.random.key(43)
    _, state1 = env.reset(key1, params)
    _, state2 = env.reset(key2, params)

    # Should be different (shuffled deterministically)
    assert not jnp.array_equal(
        state1.object_state.object_id, state2.object_state.object_id
    )

    # Test that number of objects is the same
    num_morel_1 = jnp.sum(state1.object_state.object_id == 1)
    num_oyster_1 = jnp.sum(state1.object_state.object_id == 2)
    num_deathcap_1 = jnp.sum(state1.object_state.object_id == 3)
    num_fake_1 = jnp.sum(state1.object_state.object_id == 4)
    num_morel_2 = jnp.sum(state2.object_state.object_id == 1)
    num_oyster_2 = jnp.sum(state2.object_state.object_id == 2)
    num_deathcap_2 = jnp.sum(state2.object_state.object_id == 3)
    num_fake_2 = jnp.sum(state2.object_state.object_id == 4)
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
    chex.assert_trees_all_equal(
        state1.object_state.object_id, state2.object_state.object_id
    )

    # But different keys should produce different placements
    key1 = jax.random.key(42)
    key2 = jax.random.key(43)
    _, state1 = env.reset(key1, params)
    _, state2 = env.reset(key2, params)

    # Should be different (shuffled deterministically)
    assert not jnp.array_equal(
        state1.object_state.object_id, state2.object_state.object_id
    )

    # Test that number of objects is the same
    num_morel_1 = jnp.sum(state1.object_state.object_id == 1)
    num_oyster_1 = jnp.sum(state1.object_state.object_id == 2)
    num_deathcap_1 = jnp.sum(state1.object_state.object_id == 3)
    num_fake_1 = jnp.sum(state1.object_state.object_id == 4)
    num_morel_2 = jnp.sum(state2.object_state.object_id == 1)
    num_oyster_2 = jnp.sum(state2.object_state.object_id == 2)
    num_deathcap_2 = jnp.sum(state2.object_state.object_id == 3)
    num_fake_2 = jnp.sum(state2.object_state.object_id == 4)
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


def test_foragax_twobiome_v17_registry():
    """Test that ForagaxTwoBiome-v17 can be created via registry and has correct config."""
    env = make("ForagaxTwoBiome-v17", aperture_size=(5, 5))

    # Check basic configuration
    assert env.name == "ForagaxTwoBiome-v17"
    assert env.deterministic_spawn is True
    assert env.nowrap is False

    # Check that objects have random_respawn=True and expiry_time=500
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
    assert morel.expiry_time == 500
    assert oyster.expiry_time == 500
    assert deathcap.expiry_time == 500
    assert fake.expiry_time == 500

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


def test_foragax_twobiome_v17_expiry():
    """Test that ForagaxTwoBiome-v17 objects have expiry_time=500."""
    env = make("ForagaxTwoBiome-v17", aperture_size=(5, 5))

    # Check that objects have expiry_time=500
    morel, oyster, deathcap, fake = (
        env.objects[1],
        env.objects[2],
        env.objects[3],
        env.objects[4],
    )
    assert morel.expiry_time == 500
    assert oyster.expiry_time == 500
    assert deathcap.expiry_time == 500
    assert fake.expiry_time == 500

    # Check that expiry_time is properly stored in the environment
    assert env.object_expiry_time[1] == 500  # morel
    assert env.object_expiry_time[2] == 500  # oyster
    assert env.object_expiry_time[3] == 500  # deathcap
    assert env.object_expiry_time[4] == 500  # fake


def test_foragax_diwali_v1_creation():
    """Test that ForagaxDiwali-v1 can be created and initialized."""
    env = make("ForagaxDiwali-v1", observation_type="object", aperture_size=(5, 5))

    # Check that dynamic biomes are enabled
    assert env.dynamic_biomes, "Dynamic biomes should be enabled for ForagaxDiwali-v1"
    assert env.biome_consumption_threshold == 0.9, "Threshold should be 0.9"
    assert env.num_fourier_terms == 10, "Should have 10 Fourier terms"

    # Check that objects are FourierObjects
    assert isinstance(env.objects[1], FourierObject), (
        "First object should be FourierObject"
    )
    assert isinstance(env.objects[2], FourierObject), (
        "Second object should be FourierObject"
    )

    # Initialize environment
    key = jax.random.key(0)
    obs, state = env.reset(key, env.default_params)

    # Check state has new fields
    assert hasattr(state, "object_state"), "State should have object_state"
    assert hasattr(state.object_state, "color"), "ObjectState should have color"
    assert hasattr(state.object_state, "state_params"), (
        "ObjectState should have state_params"
    )
    assert hasattr(state, "biome_state"), "State should have biome_state"
    assert hasattr(state.biome_state, "consumption_count"), (
        "BiomeState should have consumption_count"
    )
    assert hasattr(state.biome_state, "total_objects"), (
        "BiomeState should have total_objects"
    )
    assert hasattr(state.biome_state, "generation"), "BiomeState should have generation"

    # Check shapes
    chex.assert_shape(state.object_state.color, (15, 15, 3))
    chex.assert_shape(
        state.object_state.state_params, (15, 15, 22)
    )  # 2 + 2*10 = 22 params
    chex.assert_shape(state.biome_state.consumption_count, (2,))  # 2 biomes
    chex.assert_shape(state.biome_state.total_objects, (2,))
    chex.assert_shape(state.biome_state.generation, (2,))

    # Check initial generation is 0
    assert jnp.all(state.biome_state.generation == 0), "Initial generation should be 0"


def test_foragax_diwali_v2_creation():
    """Test that ForagaxDiwali-v2 can be created and initialized."""
    env = make("ForagaxDiwali-v2", observation_type="object", aperture_size=(5, 5))

    # Check that dynamic biomes are enabled
    assert env.dynamic_biomes, "Dynamic biomes should be enabled for ForagaxDiwali-v2"
    assert env.biome_consumption_threshold == 200, "Threshold should be 200"
    assert env.num_fourier_terms == 10, "Should have 10 Fourier terms"

    # Check that objects are FourierObjects
    assert isinstance(env.objects[1], FourierObject), (
        "First object should be FourierObject"
    )
    assert isinstance(env.objects[2], FourierObject), (
        "Second object should be FourierObject"
    )

    # Check regen_delay for objects
    assert env.objects[1].regen_delay_range == (9, 11), (
        "Object 1 should have regen_delay_range (9, 11)"
    )
    assert env.objects[2].regen_delay_range == (9, 11), (
        "Object 2 should have regen_delay_range (9, 11)"
    )

    # Initialize environment
    key = jax.random.key(0)
    obs, state = env.reset(key, env.default_params)

    # Check state has new fields
    assert hasattr(state, "object_state"), "State should have object_state"
    assert hasattr(state.object_state, "color"), "ObjectState should have color"
    assert hasattr(state.object_state, "state_params"), (
        "ObjectState should have state_params"
    )
    assert hasattr(state, "biome_state"), "State should have biome_state"
    assert hasattr(state.biome_state, "consumption_count"), (
        "BiomeState should have consumption_count"
    )
    assert hasattr(state.biome_state, "total_objects"), (
        "BiomeState should have total_objects"
    )
    assert hasattr(state.biome_state, "generation"), "BiomeState should have generation"

    # Check shapes
    chex.assert_shape(state.object_state.color, (15, 15, 3))
    chex.assert_shape(
        state.object_state.state_params, (15, 15, 22)
    )  # 2 + 2*10 = 22 params
    chex.assert_shape(state.biome_state.consumption_count, (2,))  # 2 biomes
    chex.assert_shape(state.biome_state.total_objects, (2,))
    chex.assert_shape(state.biome_state.generation, (2,))

    # Check initial generation is 0
    assert jnp.all(state.biome_state.generation == 0), "Initial generation should be 0"


def test_foragax_sine_twobiome_v1_creation():
    """Test that ForagaxSineTwoBiome-v1 can be created and initialized."""
    env = make("ForagaxSineTwoBiome-v1", observation_type="color", aperture_size=(5, 5))

    # Check basic configuration
    assert env.name == "ForagaxSineTwoBiome-v1"
    assert env.size == (15, 15)
    assert env.nowrap is False, "ForagaxSineTwoBiome-v1 should not have nowrap"
    assert len(env.objects) == 5, "Should have 5 objects (EMPTY + 4 sine objects)"
    assert len(env.biome_masks) == 2, "Should have 2 biomes"

    # Check object names
    object_names = [obj.name for obj in env.objects]
    assert object_names == [
        "empty",
        "oyster_sine_1",
        "deathcap_sine_1",
        "oyster_sine_2",
        "deathcap_sine_2",
    ], "Should have correct object names"

    # Check that all sine objects have correct base properties
    from foragax.objects import SineObject

    assert isinstance(env.objects[1], SineObject), "Object 1 should be SineObject"
    assert isinstance(env.objects[2], SineObject), "Object 2 should be SineObject"
    assert isinstance(env.objects[3], SineObject), "Object 3 should be SineObject"
    assert isinstance(env.objects[4], SineObject), "Object 4 should be SineObject"

    # Check sine object parameters
    oyster1 = env.objects[1]
    deathcap1 = env.objects[2]
    oyster2 = env.objects[3]
    deathcap2 = env.objects[4]

    # Check base rewards
    assert oyster1.base_reward == 10.0, "Biome 1 Oyster should have base reward +10"
    assert deathcap1.base_reward == -10.0, (
        "Biome 1 DeathCap should have base reward -10"
    )
    assert oyster2.base_reward == -10.0, "Biome 2 Oyster should have base reward -10"
    assert deathcap2.base_reward == 10.0, "Biome 2 DeathCap should have base reward +10"

    # Check amplitude and period
    for obj in [oyster1, deathcap1, oyster2, deathcap2]:
        assert obj.amplitude == 20.0, "All objects should have amplitude 20"
        assert obj.period == 1000, "All objects should have period 1000"

    # Check phase shift (Biome 2 should be inverted)
    assert oyster1.phase == 0.0, "Biome 1 Oyster should have phase 0"
    assert deathcap1.phase == 0.0, "Biome 1 DeathCap should have phase 0"
    assert oyster2.phase == jnp.pi, "Biome 2 Oyster should have phase π"
    assert deathcap2.phase == jnp.pi, "Biome 2 DeathCap should have phase π"

    # Initialize environment
    key = jax.random.key(0)
    obs, state = env.reset(key, env.default_params)

    # Check observation shape (2 color channels: oyster and deathcap)
    assert obs.shape == (5, 5, 2), "Observation should have shape (5, 5, 2)"

    # Check complementary rewards
    key = jax.random.key(42)
    for t in [0, 250, 500, 750, 1000]:
        r1 = env.objects[1].reward(t, key, None)
        r2 = env.objects[3].reward(t, key, None)
        sum_reward = r1 + r2
        assert jnp.abs(sum_reward) < 0.01, (
            f"Complementary rewards should sum to 0 at t={t}, got {sum_reward}"
        )

    # Test stepping
    key, step_key = jax.random.split(key)
    action = env.action_space(env.default_params).sample(step_key)
    obs2, state2, reward, done, info = env.step(
        step_key, state, action, env.default_params
    )
    assert obs2.shape == (5, 5, 2), "Observation should maintain shape after step"
    assert not done, "Environment should not terminate"


def test_sine_twobiome_environment():
    """Test the SineTwoBiome environment with dynamic sine rewards."""
    env = make("ForagaxSineTwoBiome-v1", aperture_size=(5, 5), observation_type="color")

    # Check environment configuration
    assert env.name == "ForagaxSineTwoBiome-v1"
    assert env.size == (15, 15)
    assert len(env.objects) == 5  # EMPTY + 4 sine objects
    assert len(env.biome_masks) == 2  # Two biomes

    # Check object names
    object_names = [obj.name for obj in env.objects]
    assert object_names == [
        "empty",
        "oyster_sine_1",
        "deathcap_sine_1",
        "oyster_sine_2",
        "deathcap_sine_2",
    ]

    # Test sine reward behavior
    key = jax.random.key(42)

    # At t=0: sine = 0
    # Biome 1: Oyster = 10 + 20*0 = 10, DeathCap = -10 + 20*0 = -10
    # Biome 2: Oyster = -10 + 20*0 = -10, DeathCap = 10 + 20*0 = 10
    t0_rewards = [env.objects[i].reward(0, key, None) for i in range(1, 5)]
    assert jnp.allclose(t0_rewards[0], 10.0, atol=0.01)  # Biome 1 Oyster
    assert jnp.allclose(t0_rewards[1], -10.0, atol=0.01)  # Biome 1 DeathCap
    assert jnp.allclose(t0_rewards[2], -10.0, atol=0.01)  # Biome 2 Oyster
    assert jnp.allclose(t0_rewards[3], 10.0, atol=0.01)  # Biome 2 DeathCap

    # At t=250 (quarter period): sine = 1 for biome 1, sine = -1 for biome 2
    # Biome 1: Oyster = 10 + 20*1 = 30, DeathCap = -10 + 20*1 = 10
    # Biome 2: Oyster = -10 + 20*(-1) = -30, DeathCap = 10 + 20*(-1) = -10
    t250_rewards = [env.objects[i].reward(250, key, None) for i in range(1, 5)]
    assert jnp.allclose(t250_rewards[0], 30.0, atol=0.01)  # Biome 1 Oyster
    assert jnp.allclose(t250_rewards[1], 10.0, atol=0.01)  # Biome 1 DeathCap
    assert jnp.allclose(t250_rewards[2], -30.0, atol=0.01)  # Biome 2 Oyster
    assert jnp.allclose(t250_rewards[3], -10.0, atol=0.01)  # Biome 2 DeathCap

    # At t=500 (half period): sine = 0 again
    t500_rewards = [env.objects[i].reward(500, key, None) for i in range(1, 5)]
    assert jnp.allclose(t500_rewards[0], 10.0, atol=0.01)  # Biome 1 Oyster
    assert jnp.allclose(t500_rewards[1], -10.0, atol=0.01)  # Biome 1 DeathCap
    assert jnp.allclose(t500_rewards[2], -10.0, atol=0.01)  # Biome 2 Oyster
    assert jnp.allclose(t500_rewards[3], 10.0, atol=0.01)  # Biome 2 DeathCap

    # At t=750 (three-quarter period): sine = -1 for biome 1, sine = 1 for biome 2
    # Biome 1: Oyster = 10 + 20*(-1) = -10, DeathCap = -10 + 20*(-1) = -30
    # Biome 2: Oyster = -10 + 20*1 = 10, DeathCap = 10 + 20*1 = 30
    t750_rewards = [env.objects[i].reward(750, key, None) for i in range(1, 5)]
    assert jnp.allclose(t750_rewards[0], -10.0, atol=0.01)  # Biome 1 Oyster
    assert jnp.allclose(t750_rewards[1], -30.0, atol=0.01)  # Biome 1 DeathCap
    assert jnp.allclose(t750_rewards[2], 10.0, atol=0.01)  # Biome 2 Oyster
    assert jnp.allclose(t750_rewards[3], 30.0, atol=0.01)  # Biome 2 DeathCap

    # Test complementary behavior: Biome 1 + Biome 2 = 0
    for t in [0, 250, 500, 750, 1000]:
        b1_oyster = env.objects[1].reward(t, key, None)
        b2_oyster = env.objects[3].reward(t, key, None)
        b1_deathcap = env.objects[2].reward(t, key, None)
        b2_deathcap = env.objects[4].reward(t, key, None)

        # Sum of complementary objects should be 0
        assert jnp.allclose(b1_oyster + b2_oyster, 0.0, atol=0.01)
        assert jnp.allclose(b1_deathcap + b2_deathcap, 0.0, atol=0.01)

    # Test environment can be reset and stepped
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset, env.default_params)
    assert obs.shape == (5, 5, 2)  # Two color channels (oyster and deathcap)

    # Take a few steps
    for _ in range(10):
        key, key_act, key_step = jax.random.split(key, 3)
        action = env.action_space(env.default_params).sample(key_act)
        obs, state, reward, done, info = env.step(
            key_step, state, action, env.default_params
        )
        assert not done  # Continuing environment
        assert obs.shape == (5, 5, 2)
