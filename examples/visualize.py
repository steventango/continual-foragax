import os
from collections import defaultdict

import jax
from gymnasium.utils.save_video import save_video
from tqdm import tqdm

from foragax.registry import make


def main():
    """Generate a visualization of a Foragax environment under random behavior."""
    video_length = 1000
    video_every = 100_000
    for seed in range(1):
        key = jax.random.key(seed)
        aperture_sizes = [9]
        render_modes = ["world_reward"]
        video_folder = "videos"
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)

        for aperture_size in aperture_sizes:
            env = make(
                "ForagaxDiwali-v5",
                aperture_size=aperture_size,
                observation_type="color",
            )
            env_params = env.default_params

            frames = defaultdict(list)
            key, key_reset = jax.random.split(key)
            _, env_state = env.reset_env(key_reset, env_params)
            for frame in tqdm(range(1_000_000), desc=f"Aperture {aperture_size}"):
                if frame % video_every < video_length:
                    for render_mode in render_modes:
                        frames[render_mode].append(
                            env.render(env_state, env_params, render_mode=render_mode)
                        )
                if len(frames[render_mode]) >= video_length:
                    for render_mode in render_modes:
                        save_video(
                            frames[render_mode],
                            video_folder,
                            name_prefix=f"foragax_{seed}_{render_mode}_aperture-{aperture_size}_{frame - video_length + 1}_{frame}",
                            fps=8,
                        )
                        frames[render_mode] = []
                key, key_act, key_step = jax.random.split(key, 3)
                action = env.action_space(env_params).sample(key_act)
                _, env_state, reward, done, info = env.step(
                    key_step, env_state, action, env_params
                )

if __name__ == "__main__":
    main()
