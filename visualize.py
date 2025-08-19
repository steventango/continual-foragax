import os

import jax
from gymnasium.utils.save_video import save_video

from registry import make
from tqdm import tqdm

def main():
    """Generate a visualization of a Foragax environment under random behavior."""
    key = jax.random.PRNGKey(0)
    env = make("ForagaxTwoBiomeSmall", observation_type="object")
    env_params = env.default_params

    frames = []
    key, key_reset = jax.random.split(key)
    _, env_state = env.reset(key_reset, env_params)
    for _ in tqdm(range(1000)):
        frames.append(env.render(env_state, env_params))
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
    save_video(frames, video_folder, name_prefix="foragax_test", fps=8)


if __name__ == "__main__":
    main()
