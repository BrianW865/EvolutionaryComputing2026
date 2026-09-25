import random
from pathlib import Path
from typing import Literal
import statistics
import json
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from rich.console import Console
from rich.traceback import install

from ariel import console
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)

from ariel.ec.genotypes.tree.operators import random_tree
from tree_edit_distance import mean_plus_std_tree_edit_distance
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from ariel.ec import (
    Individual
)

install()
console = Console()

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
HERE = Path(__file__).parent
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
TARGET_DIR: Path = HERE / "target_bodies"  # the bodies you must approach
NUM_OF_MODULES: int = 20  # module budget per evolved body
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]

NUM_OF_EVALUATIONS = 120

# --- Load targets --- #
def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    """..."""
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]

TARGETS: list[nx.DiGraph] = load_targets()

# --- Individual making --- #
# Using the tree genotype representation for this assignment (as opposed to NDE).
def make_individual() -> Individual:
    ind = Individual()
    # Individual.genotype is stored as a JSON database column, so we must store
    # the genome as a dict (.to_dict()), not as a raw TreeGenome object -
    # otherwise Individual persistence fails with a JSON serialization error.
    ind.genotype = random_tree(NUM_OF_MODULES).to_dict()
    return ind

def plotting(
    histories_variant1: list[list[float]],
) -> None:
    """
    Plots mean ± std of best fitness per generation, across independent runs.
    """
    history_array = np.array(histories_variant1)
    mean_per_gen = history_array.mean(axis=0)
    std_per_gen = history_array.std(axis=0)

    generations = np.arange(len(mean_per_gen))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(generations, mean_per_gen, label="Random search", color="blue")
    ax.fill_between(
        generations,
        mean_per_gen - std_per_gen,
        mean_per_gen + std_per_gen,
        color="blue",
        alpha=0.2,
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (tree edit distance)")
    ax.set_title("Convergence: Random search")
    ax.legend()
    plt.savefig(DATA / "random_search_convergence.png", dpi=300)
    plt.close()

def random_search():
    ind = make_individual()
    graph = TreeGenome.from_dict(ind.genotype).to_networkx()
    ind.fitness = mean_plus_std_tree_edit_distance(graph, TARGETS)

    return ind.fitness

def run_random_search(seed: int) -> list[float]:
    random.seed(seed)
    POPULATION_SIZE = 50
    INITIAL_POPULATION = 50
    AMOUNT_OF_GENERATIONS = 100
    REPETITIONS = INITIAL_POPULATION + (POPULATION_SIZE * AMOUNT_OF_GENERATIONS)

    best_fitness: float = float("inf")
    history: list[float] = []
    fitnesses: list[float] = []

    console.log(f"--- Starting run for seed {seed} ---")

    for i in range(REPETITIONS):
        fitness = random_search()
        fitnesses.append(fitness)

        if fitness < best_fitness:
            best_fitness = fitness

        if (i+1) % 50 == 0:
            history.append(best_fitness)

    
    console.log(f"best fitness = {best_fitness}")
    console.log(f"median = {statistics.median(fitnesses)}")
    console.log(f"mean = {statistics.mean(fitnesses)}")
    console.log("--- Run ended ---")

    return history

def main()-> None:
    seeds = [42, 43, 44, 45, 46]
    all_histories: list[list[float]] = []

    for seed in seeds:
        history = run_random_search(seed)
        number = 0
        for _ in history:
            number += 1
        console.log(f"Amount of numbers stored: {number}")
        all_histories.append(history)

    with open(DATA / "histories_random_search.json", "w") as f:
        json.dump(all_histories, f)

    plotting(all_histories)

if __name__ == "__main__":
    main()