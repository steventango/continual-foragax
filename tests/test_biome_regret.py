import jax
import jax.numpy as jnp
from foragax.registry import make


def test_biome_regret_metrics():
    print("Running FINAL consolidated biome regret metrics test with ForagaxBig-v2...")
    env = make("ForagaxBig-v2")
    params = env.default_params
    key = jax.random.key(0)

    print("Resetting environment...")
    obs, state = env.reset(key, params)
    print(f"Environment reset. Grid size: {env.size}")

    # Inspect the means and counts directly from the env helper
    means, counts = env._get_biome_mean_rewards(state)
    print("\nFull Metrics Region Data:")
    num_food = env.num_food_biomes
    for i in range(len(means)):
        name = (
            f"Food Biome {env.food_biome_indices[i]}" if i < num_food else "Void (-1)"
        )
        print(f"  {name}: Mean={means[i]:.4f}, Count={counts[i]}")

    def check_at_pos(y, x, name):
        # NOTE: y=row, x=col
        state_at = state.replace(pos=jnp.array([x, y], dtype=jnp.int32))
        _, _, _, _, info = env.step(key, state_at, 0, params)
        print(f"\n{name} at x={x}, y={y}:")
        print(f"  Reported Biome ID: {info['biome_id']}")
        print(f"  Mean: {info['current_biome_mean']:.4f}, Rank: {info['biome_rank']}")
        return info

    # Food biomes 0-3 (Boxes)
    # B0: Top-Left, B1: Bottom-Left, B2: Top-Right, B3: Bottom-Right
    info0 = check_at_pos(6, 6, "Box 0 (TL)")
    info1 = check_at_pos(24, 6, "Box 1 (BL)")
    info2 = check_at_pos(6, 24, "Box 2 (TR)")
    info3 = check_at_pos(24, 24, "Box 3 (BR)")

    # Wall regions (inside food biomes)
    # Wall 4 in B0, Wall 5 in B1, Wall 6 in B2, Wall 7 in B3
    infoW4 = check_at_pos(9, 9, "Wall 4 (Inside Box 0)")
    infoW5 = check_at_pos(27, 9, "Wall 5 (Inside Box 1)")
    infoW6 = check_at_pos(9, 27, "Wall 6 (Inside Box 2)")

    # Void region
    info_minus_1 = check_at_pos(0, 0, "Biome -1 (Void)")

    # Assertions
    # 1. Reported IDs should be the original ones
    assert info0["biome_id"] == 0
    assert info1["biome_id"] == 1
    assert infoW4["biome_id"] == 4
    assert infoW5["biome_id"] == 5
    assert infoW6["biome_id"] == 6
    assert info_minus_1["biome_id"] == -1

    # 3. Total Ranks should be 1-5 (4 food biomes + 1 void)
    all_ranks = [
        info["biome_rank"].item() for info in [info0, info1, info2, info3, info_minus_1]
    ]
    unique_ranks = sorted(list(set(all_ranks)))
    assert len(unique_ranks) == 5
    assert unique_ranks == [1, 2, 3, 4, 5]

    # 3. Consolidation: Wall 4 inherits from Box 0, Wall 5 from Box 1, etc.
    assert jnp.abs(infoW4["current_biome_mean"] - info0["current_biome_mean"]) < 1e-6
    assert infoW4["biome_rank"] == info0["biome_rank"]

    assert jnp.abs(infoW5["current_biome_mean"] - info1["current_biome_mean"]) < 1e-6
    assert infoW5["biome_rank"] == info1["biome_rank"]

    assert jnp.abs(infoW6["current_biome_mean"] - info2["current_biome_mean"]) < 1e-6
    assert infoW6["biome_rank"] == info2["biome_rank"]

    print("\nForagaxBig-v2 consolidated benchmarks PASSED!")
