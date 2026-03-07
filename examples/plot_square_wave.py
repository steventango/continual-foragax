import os

import matplotlib
import matplotlib.ticker as ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "plots", "square_wave.png"
)
PERIOD = 500_000
MAX_STEPS = 1_000_000


def square_wave(t: np.ndarray, period: int) -> np.ndarray:
    """Square wave oscillating between -1 and +1 with the given period."""
    return np.where((t % period) < (period / 2), 1.0, -1.0)


def plot(plot_path: str = PLOT_PATH) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)

    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Linux Libertine O"],
            "font.sans-serif": ["Linux Biolinum O"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Linux Libertine O",
            "mathtext.it": "Linux Libertine O:italic",
            "mathtext.bf": "Linux Libertine O:bold",
            "axes.unicode_minus": False,
        }
    )

    FONTSIZE = 16
    plt.rcParams["axes.labelsize"] = FONTSIZE
    plt.rcParams["xtick.labelsize"] = FONTSIZE
    plt.rcParams["ytick.labelsize"] = FONTSIZE

    t = np.arange(MAX_STEPS)
    y = square_wave(t, PERIOD)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(t, y, color="tab:blue", linewidth=1, alpha=0.8)
    ax.set_xlabel(r"Time steps $(\times 10^6)$")
    ax.set_ylabel("Reward")
    ax.set_xticks([0, 1e6])
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x / 1_000_000:g}")
    )
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved plot to: {os.path.abspath(plot_path)}")


def main():
    plot()


if __name__ == "__main__":
    main()
