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
    assert env.objects[1].period == 500000

    # custom period test
    env = make("ForagaxSquareWaveTwoBiome-v11", period=250)
    assert env.objects[1].period == 250


def test_make_two_biome_large_v1():
    env = make("ForagaxTwoBiomeLarge-v1")
    assert env.name == "ForagaxTwoBiomeLarge-v1"
    assert env.size == (15, 15)


def test_make_unending_tasks_v1_alias():
    env = make("ForagaxUnendingTasks-v1")
    assert env.name == "ForagaxUnendingTasks-v1"
    assert env.size == (28, 28)
    assert env.dynamic_biomes
    assert env.return_hint


def test_make_never_ending_relearning_v1_alias():
    env = make("ForagaxNeverEndingRelearning-v1")
    assert env.name == "ForagaxNeverEndingRelearning-v1"
    assert env.size == (24, 15)
    assert env.deterministic_spawn
