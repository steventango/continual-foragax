from foragax.registry import make


def test_make_big_v5():
    env = make("ForagaxBig-v5")
    assert env.name == "ForagaxBig-v5"
    assert env.size == (28, 28)
    assert env.dynamic_biomes
    assert env.return_hint


def test_make_square_wave_two_biome_v11():
    env = make("ForagaxSquareWaveTwoBiome-v11")
    assert env.name == "ForagaxSquareWaveTwoBiome-v11"
    assert env.size == (24, 15)
    assert env.deterministic_spawn
