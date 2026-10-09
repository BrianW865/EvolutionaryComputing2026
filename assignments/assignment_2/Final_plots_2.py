print("SCRIPT STARTED")
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()

with open(CWD / "__data__" / "EA_1" / "histories_variant1.json") as f:
    histories_variant1 = json.load(f)

with open(CWD / "__data__" / "EA_2" / "histories_variant2.json") as f:
    histories_variant2 = json.load(f)

with open(CWD / "__data__" / "EA_3" / "histories_variant3.json") as f:
    histories_variant3 = json.load(f)

with open(CWD / "__data__" / "A2_2026_random_search" / "histories_random_search.json") as f:
    histories_random = json.load(f)


def plotting(
    histories_variant1: list[list[float]],
    histories_variant2: list[list[float]],
    histories_variant3: list[list[float]],
    histories_random: list[list[float]],
) -> None:
    """
    Plots mean + std of best fitness per generation, across independent runs,
    for both EA variants and the random search baseline.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for histories, label, color in [
        (histories_variant1, "EA Variant 1 (1 mutation)", "blue"),
        (histories_variant2, "EA Variant 2 (3 mutations)", "red"),
        (histories_variant3, "EA Variant 3 (5 mutations)", "green"),
        (histories_random, "Random search", "black"),
    ]:
        max_length = max(len(history) for history in histories)
        padded_histories = [
            history + [history[-1]] * (max_length - len(history))
            for history in histories
        ]
        history_array = np.array(padded_histories)
        mean_per_gen = history_array.mean(axis=0)
        std_per_gen = history_array.std(axis=0)
        generations = np.arange(len(mean_per_gen))

        ax.plot(generations, mean_per_gen, label=label, color=color)
        ax.fill_between(
            generations,
            mean_per_gen - std_per_gen,
            mean_per_gen + std_per_gen,
            color=color,
            alpha=0.1,
        )

    ax.set_ylim(0.8, 2.1)
    max_generations = max(
        len(history)
        for histories in (histories_variant1, histories_variant2, histories_variant3, histories_random)
        for history in histories
    )
    ax.set_xlim(0, max_generations - 1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (distance till endpoint)")
    ax.set_title("Convergence: EA Variant 1 vs Variant 2 vs variant 3 vs Random Search")
    ax.legend()
    plt.tight_layout()
    plt.savefig(CWD / "__data__" / "final_convergence_plot_2.png", dpi=300)
    print(f"Saved plot to {CWD / '__data__' / 'final_convergence_plot_3.png'}")
    plt.show()

plotting(histories_variant1, histories_variant2, histories_variant3, histories_random)