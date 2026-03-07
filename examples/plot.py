import os

import jax
import matplotlib
import numpy as np
from matplotlib import ticker
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foragax.registry import make

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "plots", "big_v5_data.npz")
PLOT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "plots", "ForagaxBig-v5-rewards.pdf"
)
MAX_STEPS = 10_000_000
CHUNK_SIZE = 1_000


def generate(data_path: str = DATA_PATH) -> None:
    """Run ForagaxBig-v5 for MAX_STEPS steps and save per-biome reward data."""
    os.makedirs(os.path.dirname(os.path.abspath(data_path)), exist_ok=True)

    key = jax.random.key(0)
    env = make("ForagaxBig-v5", aperture_size=9, observation_type="color")
    env_params = env.default_params

    num_food_biomes = env.num_food_biomes
    food_biome_indices = env.food_biome_indices.tolist()

    key, key_reset = jax.random.split(key)
    _, env_state = env.reset(key_reset, env_params)

    # Retrieve each biome's color from the first step's object_state.color.
    # FourierObject colors are randomly sampled per biome at reset time.
    color_grid = np.array(env_state.object_state.color)  # (H, W, 3)
    object_id_grid = np.array(env_state.object_state.object_id)  # (H, W)
    metrics_grid = np.array(env.cell_to_metrics_idx_grid)  # (H, W)
    real_start = int(env.real_object_start)
    real_end = int(env.real_object_end)

    biome_colors = []
    object_blocking = np.array(env.object_blocking)  # (num_objects,)
    is_not_blocking = ~object_blocking[object_id_grid]  # (H, W)
    for i in range(num_food_biomes):
        biome_mask = metrics_grid == i
        active_mask = (
            biome_mask
            & (object_id_grid >= real_start)
            & (object_id_grid < real_end)
            & is_not_blocking
        )
        ys, xs = np.where(active_mask)
        if len(ys) > 0:
            r, g, b = color_grid[ys[0], xs[0]]
        else:
            # Fallback: cycle through tab10 colors
            cycle = plt.get_cmap("tab10")
            r, g, b = (np.array(cycle(i)[:3]) * 255).astype(int)
        biome_colors.append((r / 255.0, g / 255.0, b / 255.0))

    n_chunks = MAX_STEPS // CHUNK_SIZE
    biome_rewards = np.zeros((num_food_biomes, n_chunks))

    @jax.jit
    def run_chunk(key, env_state):
        """Run CHUNK_SIZE steps and return per-food-biome mean rewards at the end."""

        def step_fn(carry, _):
            key, state = carry
            key, key_act, key_step = jax.random.split(key, 3)
            action = env.action_space(env_params).sample(key_act)
            _, state, _, _, _ = env.step(key_step, state, action, env_params)
            return (key, state), None

        (key, env_state), _ = jax.lax.scan(
            step_fn, (key, env_state), None, length=CHUNK_SIZE
        )
        means, _ = env._get_biome_mean_rewards(env_state)
        global_mean = env._get_global_mean_reward(env_state)
        centered_means = means - global_mean
        return key, env_state, centered_means[:num_food_biomes]

    for chunk in tqdm(range(n_chunks), desc="Simulating ForagaxBig-v5"):
        key, env_state, means = run_chunk(key, env_state)
        biome_rewards[:, chunk] = np.array(means)

    np.savez(
        data_path,
        biome_rewards=biome_rewards,
        biome_colors=np.array(biome_colors),
        food_biome_indices=np.array(food_biome_indices),
        chunk_size=CHUNK_SIZE,
    )
    print(f"Saved data to: {os.path.abspath(data_path)}")


def plot(data_path: str = DATA_PATH, plot_path: str = PLOT_PATH) -> None:
    """Load saved data and produce the biome-rewards plot."""
    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)

    data = np.load(data_path)
    biome_rewards = data["biome_rewards"]  # (num_food_biomes, n_chunks)
    biome_colors = data["biome_colors"]  # (num_food_biomes, 3)
    food_biome_indices = data["food_biome_indices"].tolist()
    chunk_size = int(data["chunk_size"])

    max_plot_steps = 1_000_000
    n_chunks = min(biome_rewards.shape[1], max_plot_steps // chunk_size)
    biome_rewards = biome_rewards[:, :n_chunks]
    num_food_biomes = biome_rewards.shape[0]
    x = np.arange(n_chunks) * chunk_size

    # Set font

    plt.rcParams.update(
        {
            "text.usetex": False,  # Don't use LaTeX
            "font.family": "serif",  # Use a serif font
            "font.serif": ["Linux Libertine O"],  # Specifically, use Linux Libertine
            "font.sans-serif": ["Linux Biolinum O"],  # Use Linux Biolinum for sans-serif
            "mathtext.fontset": "custom",  # Use custom math fonts
            "mathtext.rm": "Linux Libertine O",  # Roman text in math
            "mathtext.it": "Linux Libertine O:italic",  # Italic text in math
            "mathtext.bf": "Linux Libertine O:bold",  # Bold text in math
            "axes.unicode_minus": False,  # Ensure minus signs are rendered correctly
        }
    )

    FONTSIZE = 16
    plt.rcParams["axes.labelsize"] = FONTSIZE  # Axis labels
    plt.rcParams["xtick.labelsize"] = FONTSIZE  # X-tick labels
    plt.rcParams["ytick.labelsize"] = FONTSIZE  # Y-tick labels

    fig, ax = plt.subplots(figsize=(3, 3))
    for i in range(num_food_biomes):
        ax.plot(
            x,
            biome_rewards[i],
            color=tuple(biome_colors[i]),
            linewidth=1,
            label=f"Biome {food_biome_indices[i]}",
            alpha=0.8,
        )
    ax.set_xlabel(r"Time steps $(\times 10^6)$")
    ax.set_ylabel("Reward")
    ax.set_xticks([0, 1e6])
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x / 1000000:g}")
    )
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    # despine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # plt.title(f"ForagaxBig-v5: Mean Biome Rewards over {max_plot_steps:,} Steps")
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight", pad_inches=0)

    plt.close(fig)
    print(f"Saved plot to: {os.path.abspath(plot_path)}")


def main():
    plot()


if __name__ == "__main__":
    main()
