import os

import jax
import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foragax.registry import make


def main():
    """Check weather environment and plot temperature over the run."""
    out_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(out_dir, exist_ok=True)

    key = jax.random.key(0)
    file_index = 0
    env = make(
        "ForagaxWeather-v5",
        aperture_size=5,
        file_index=file_index,
        observation_type="object",
    )
    env_params = env.default_params

    key, key_reset = jax.random.split(key)
    _, env_state = env.reset(key_reset, env_params)

    max_steps = 10000
    temps = np.zeros(max_steps)
    rewards = np.zeros(max_steps)

    for step in tqdm(range(max_steps)):
        key, key_act, key_step = jax.random.split(key, 3)
        action = env.action_space(env_params).sample(key_act)
        _, next_env_state, reward, done, info = env.step(
            key_step, env_state, action, env_params
        )
        temp = info["temperatures"][1]
        assert -abs(temp) <= reward <= abs(temp), (
            f"Reward {reward} out of bounds for temperature {temp}"
        )
        temps[step] = temp
        rewards[step] = reward
        env_state = next_env_state

    sample_path = os.path.abspath(os.path.join(out_dir, f"plot_{file_index}.png"))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.plot(-temps, color="gray", linewidth=1)
    ax.plot(temps, color="tab:orange", linewidth=1, label="Temperature")
    ax.plot(rewards, color="tab:blue", linewidth=1, label="Reward")
    ax.legend(loc="upper right")
    plt.title("Sample: Temperature and Reward over run")
    fig.tight_layout()
    fig.savefig(sample_path, dpi=150)
    plt.close(fig)
    print(f"Saved combined sample plot to: {sample_path}")


if __name__ == "__main__":
    main()
