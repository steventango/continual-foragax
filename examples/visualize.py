import os
from collections import defaultdict

import jax
from gymnasium.utils.save_video import save_video
from tqdm import tqdm

from foragax.registry import make


def main():
    """Generate a visualization of a Foragax environment under random behavior."""
    key = jax.random.key(0)
    aperture_sizes = [5, 9]
    render_modes = ["world", "world_true", "aperture", "aperture_true"]
    video_folder = "videos"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    for aperture_size in aperture_sizes:
        env = make(
            "ForagaxWeather-v1",
            aperture_size=aperture_size,
            observation_type="object",
        )
        env_params = env.default_params

        frames = defaultdict(list)
        key, key_reset = jax.random.split(key)
        _, env_state = env.reset(key_reset, env_params)
        for _ in tqdm(range(100), desc=f"Aperture {aperture_size}"):
            for render_mode in render_modes:
                frames[render_mode].append(
                    env.render(env_state, env_params, render_mode=render_mode)
                )
            key, key_act, key_step = jax.random.split(key, 3)
            action = env.action_space(env_params).sample(key_act)
            _, next_env_state, reward, done, info = env.step(
                key_step, env_state, action, env_params
            )
            env_state = next_env_state

        for render_mode in render_modes:
            save_video(
                frames[render_mode],
                video_folder,
                name_prefix=f"foragax_{render_mode}_test-aperture-{aperture_size}",
                fps=8,
            )


if __name__ == "__main__":
    main()
