import os
from collections import defaultdict

import jax
import jax.numpy as jnp
from gymnasium.utils.save_video import save_video
from tqdm import tqdm

from foragax.registry import make


def obs_to_frame(obs: jax.Array) -> jax.Array:
    obs *= 255
    obs = obs.astype(jnp.uint8)
    frame = jnp.concatenate([obs, jnp.ones((*obs.shape[:2], 1))], axis=-1)

    frame = jax.image.resize(
        frame,
        (frame.shape[1] * 24, frame.shape[0] * 24, 3),
        jax.image.ResizeMethod.NEAREST,
    )
    return frame


def main():
    """Generate a visualization of a Foragax environment under random behavior."""
    for seed in range(5):
        key = jax.random.key(seed)
        aperture_sizes = [1]
        render_modes = ["world"]
        video_folder = "videos"
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)

        for aperture_size in aperture_sizes:
            env = make(
                "ForagaxTwoBiome-v10",
                aperture_size=aperture_size,
                observation_type="world",
            )
            env_params = env.default_params

            frames = defaultdict(list)
            key, key_reset = jax.random.split(key)
            obs, env_state = env.reset_env(key_reset, env_params)
            print(obs.shape)
            for _ in tqdm(range(100), desc=f"Aperture {aperture_size}"):
                for render_mode in render_modes:
                    frame = obs_to_frame(obs)
                    frames[render_mode].append(frame)
                key, key_act, key_step = jax.random.split(key, 3)
                action = env.action_space(env_params).sample(key_act)
                obs, next_env_state, reward, done, info = env.step(
                    key_step, env_state, action, env_params
                )
                env_state = next_env_state

            for render_mode in render_modes:
                save_video(
                    frames[render_mode],
                    video_folder,
                    name_prefix=f"foragax_{seed}_{render_mode}_test-aperture-{aperture_size}",
                    fps=8,
                )


if __name__ == "__main__":
    main()
