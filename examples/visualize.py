import os
from collections import defaultdict

import jax
from gymnasium.utils.save_video import save_video
from tqdm import tqdm

from foragax.registry import make


def main():
    """Generate a visualization of a Foragax environment under random behavior."""
    key = jax.random.key(0)
    env = make("ForagaxTwoBiomeSmall", aperture_size=11, observation_type="object")
    env_params = env.default_params

    frames = defaultdict(list)
    key, key_reset = jax.random.split(key)
    _, env_state = env.reset(key_reset, env_params)
    for _ in tqdm(range(1000)):
        for render_mode in ("world", "aperture"):
            frames[render_mode].append(
                env.render(env_state, env_params, render_mode=render_mode)
            )
        key, key_act, key_step = jax.random.split(key, 3)
        action = env.action_space(env_params).sample(key_act)
        _, next_env_state, reward, done, info = env.step(
            key_step, env_state, action, env_params
        )
        if done:
            break
        else:
            env_state = next_env_state

    video_folder = "videos"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    for render_mode in ("world", "aperture"):
        save_video(
            frames[render_mode],
            video_folder,
            name_prefix=f"foragax_{render_mode}_test",
            fps=8,
        )


if __name__ == "__main__":
    main()
