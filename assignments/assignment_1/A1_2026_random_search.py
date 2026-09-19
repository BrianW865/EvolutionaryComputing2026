import random
from pathlib import Path
from typing import Literal
import statistics

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
NUM_OF_MODULES: int = 10  # module budget per evolved body
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

def random_search():
    console.log("hello")
    ind = make_individual()
    graph = TreeGenome.from_dict(ind.genotype).to_networkx()
    ind.fitness = mean_plus_std_tree_edit_distance(graph, TARGETS)

    return ind.fitness

def main() -> None:
    POPULATION_SIZE = 20
    INITIAL_POPULATION = 20
    AMOUNT_OF_GENERATIONS = 10
    REPETITIONS = POPULATION_SIZE + (INITIAL_POPULATION * AMOUNT_OF_GENERATIONS)

    best_fitness = float("inf")
    fitnesses = []

    console.log("--- Starting run ---")

    for _ in range(REPETITIONS):
        fitness = random_search()
        fitnesses.append(fitness)

        if fitness < best_fitness:
            best_fitness = fitness
        
        console.log(f"best = {best_fitness}")
        console.log(f"mean = {statistics.mean(fitnesses)}")
    
    console.log("--- Run ended ---")

if __name__ == "__main__":
    main()