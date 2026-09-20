print("SCRIPT STARTED")
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

with open(HERE / "__data__" / "A1_2026_EA1" / "histories_variant1.json") as f:
    histories_variant1 = json.load(f)

with open(HERE / "__data__" / "A1_2026_EA2" / "histories_variant2.json") as f:
    histories_variant2 = json.load(f)

with open(HERE / "__data__" / "A1_2026_random_search" / "histories_random_search.json") as f:
    histories_random = json.load(f)


def plotting(
    histories_variant1: list[list[float]],
    histories_variant2: list[list[float]],
    histories_random: list[list[float]],
) -> None:
    """
    Plots mean + std of best fitness per generation, across independent runs,
    for both EA variants and the random search baseline.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for histories, label, color in [
        (histories_variant1, "EA Variant 1 (crossover + mutation)", "blue"),
        (histories_variant2, "EA Variant 2 (mutation only)", "red"),
        (histories_random, "Random search", "black"),
    ]:
        history_array = np.array(histories)
        mean_per_gen = history_array.mean(axis=0)
        std_per_gen = history_array.std(axis=0)
        generations = np.arange(len(mean_per_gen))

        ax.plot(generations, mean_per_gen, label=label, color=color)
        ax.fill_between(
            generations,
            mean_per_gen - std_per_gen,
            mean_per_gen + std_per_gen,
            color=color,
            alpha=0.2,
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (tree edit distance)")
    ax.set_title("Convergence: EA Variant 1 vs Variant 2 vs Random Search")
    ax.legend()
    plt.tight_layout()
    plt.savefig(HERE / "final_convergence_plot.png", dpi=300)
    print(f"Saved plot to {HERE / 'final_convergence_plot.png'}")
    plt.show()

plotting(histories_variant1, histories_variant2, histories_random)