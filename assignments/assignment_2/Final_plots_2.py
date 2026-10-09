import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

CWD = Path.cwd()
DATA = CWD / "__data__"
OUTPUT_FILE = DATA / "final_convergence_plot_2.png"


def load(path: Path) -> list[list[float]]:
    with open(path) as f:
        return json.load(f)


histories_variant1 = load(DATA / "EA_1" / "histories_variant1.json")
histories_variant2 = load(DATA / "EA_2" / "histories_variant2.json")
histories_variant3 = load(DATA / "EA_3" / "histories_variant3.json")
histories_random = load(DATA / "A2_2026_random_search" / "histories_random_search.json")

# The first random-search checkpoint corresponds to the initial population
# (generation 0), which the EA histories do not contain, so drop it.
histories_random = [h[1:] for h in histories_random]


def plotting(
    histories_variant1: list[list[float]],
    histories_variant2: list[list[float]],
    histories_variant3: list[list[float]],
    histories_random: list[list[float]],
) -> None:
    """
    Plots the mean +- one standard deviation (across seeds, ddof=1) of the
    best fitness per generation for the three EA variants and random search.
    Runs that stopped early are extended at their final value.
    """
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(7, 4))

    series = [
        (histories_variant1, "EA1 (1 mutation)", "tab:blue"),
        (histories_variant2, "EA2 (3 mutations)", "tab:red"),
        (histories_variant3, "EA3 (5 mutations)", "tab:green"),
        (histories_random, "Random search", "black"),
    ]

    max_generations = max(len(h) for histories, _, _ in series for h in histories)
    lowest = np.inf
    highest = -np.inf

    for histories, label, color in series:
        max_length = max(len(history) for history in histories)
        padded = [h + [h[-1]] * (max_length - len(h)) for h in histories]
        history_array = np.array(padded)
        mean_per_gen = history_array.mean(axis=0)
        std_per_gen = history_array.std(axis=0, ddof=1)
        generations = np.arange(len(mean_per_gen))

        ax.plot(generations, mean_per_gen, label=label, color=color, linewidth=1.5)
        ax.fill_between(
            generations,
            mean_per_gen - std_per_gen,
            mean_per_gen + std_per_gen,
            color=color,
            alpha=0.12,
        )
        lowest = min(lowest, float((mean_per_gen - std_per_gen).min()))
        highest = max(highest, float((mean_per_gen + std_per_gen).max()))

    ax.set_xlim(0, max_generations - 1)
    ax.set_ylim(max(0.0, lowest - 0.05), highest + 0.05)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (distance to target, m)")
    ax.set_title("Convergence of the EA variants and random search")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {OUTPUT_FILE}")


plotting(histories_variant1, histories_variant2, histories_variant3, histories_random)