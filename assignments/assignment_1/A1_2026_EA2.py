# ──────────────────────────────────────────────── EA2 - ONLY Mutation operator ──────────────────────────────────────────────────────────────────────
# Standard library
import random
from pathlib import Path
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
import json

import mujoco as mj
import networkx as nx
from mujoco import viewer
from rich.console import Console
from rich.traceback import install

from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)

from ariel.ec.genotypes.tree.operators import random_tree
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.video_recorder import VideoRecorder
from tree_edit_distance import mean_plus_std_tree_edit_distance
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.operators import random_tree, mutate_replace_node, crossover_subtree

# Type aliases
type GenotypeTypes = Literal["nde", "tree"]
type ViewerTypes = Literal["launcher", "video", "frame", "none"]

from ariel.body_phenotypes.robogen_lite.decoders._blueprint import load_graph_from_json

from ariel.ec import (
    EA,
    EAOperation,
    Individual,
    Population,
    config,
)

install()
console = Console()
# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
HERE = Path(__file__).parent
CWD = Path.cwd()
DATA = HERE / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
TARGET_DIR: Path = HERE / "target_bodies"  # the bodies you must approach
NUM_OF_MODULES: int = 20  # module budget per evolved body
GENOTYPE: GenotypeTypes = "tree"  # "nde" | "tree" 
MODE: ViewerTypes = "frame"  # see show_body() for the options
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]



# --- Individual making --- #
# Using the tree genotype representation for this assignment (as opposed to NDE).
def make_individual() -> Individual:
    ind = Individual()
    # Individual.genotype is stored as a JSON database column, so we must store
    # the genome as a dict (.to_dict()), not as a raw TreeGenome object -
    # otherwise Individual persistence fails with a JSON serialization error.
    ind.genotype = random_tree(NUM_OF_MODULES).to_dict()
    return ind


# --- Load targets --- #
def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    """..."""
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]

TARGETS: list[nx.DiGraph] = load_targets()

# --- Evaluation --- #
def evaluate(population: Population) -> Population:
    #Only score individuals that don't already have a fitness
    for ind in population.unevaluated:
        graph = TreeGenome.from_dict(ind.genotype).to_networkx()
        # Mean + std tree edit distance across all targets
        ind.fitness = mean_plus_std_tree_edit_distance(graph,TARGETS)
    return population


# --- Mutation Selection --- #

def mutation_selection(population: Population) -> Population:


    population = population.sort(sort="min")
    cutoff = len(population) // 2
    for i, ind in enumerate(population):
        ind.tags["mutate"] = i < cutoff

    selected_count = sum(1 for ind in population if ind.tags.get("mutate", False))
    console.log(
        f"[cyan]Mutation Selection: {selected_count}/{len(population)} marked for mutation[/cyan]",
    )
    return population

"""

# --- Crossover --- #

def crossover(population: Population) -> Population:
    parents = population.where(lambda ind: bool(ind.tags.get("selected", False)))
    for idx in range(0, len(parents) - 1, 2):
        p_a = parents[idx]
        p_b = parents[idx + 1]
        ## Genotypes are stored as dicts so we have to reconstruct real
        # TreeGenome objects before crossing, then convert children back to dicts.
        g_a, g_b = crossover_subtree(TreeGenome.from_dict(p_a.genotype), TreeGenome.from_dict(p_b.genotype))

        child_a = Individual()
        child_a.genotype = g_a.to_dict()
        #Take tagged genotypes only to avoid mutating parents as well
        child_a.tags = {"mutate": True}

        child_b = Individual()
        child_b.genotype = g_b.to_dict()
        child_b.tags = {"mutate": True}

        # Children are added on top of the existing population (which now grows
        # more).Thats why we use survivor_selection to trim it back down later.
        population.extend([child_a, child_b])
    return population


"""

# --- Mutation --- #
# EA VARIANT 1: uses mutate_replace_node (point mutation: reassigns one
# nodes type/rotation. Compare against EA variant 2, which uses mutate_subtree_replacement
# This is the only deliberate difference between the two EA files!!
# Everything else is held identical to isolate the effects of mutation operator.

MUTATION_RATE: float = 0.3

def mutate(population: Population) -> Population:
    to_mutate = population.where(lambda ind: bool(ind.tags.get("mutate", False)))

    for ind in to_mutate:
        if random.random() < MUTATION_RATE:
            genome = TreeGenome.from_dict(ind.genotype)
            mutate_replace_node(genome)
            #Mutate a copy of the original individual and add it to the population rather than mutating the original individual in place.
            ind_mutated = Individual()
            ind_mutated.genotype = genome.to_dict()
            population.extend([ind_mutated])

    return population

# --- Survivor Selection --- #
def survivor_selection(population: Population) -> Population:
     # We keep the target_population_size individuals with the lowest fitness.
    survivors = population.best(sort="min", n=config.target_population_size)
    survivor_ids = {ind.id for ind in survivors}
    # Mark everyone's alive status explicitly, not just the survivors. This matters because
    # EA._commit() only persists changes for individuals present in whatever
    # this function returns. If we returned just the survivors, the 
    # individuals' alive=False would never reach the database, and they'd
    # keep getting re-fetched every generation, causing unbounded population
    # growth (this was a real bug we hit and fixed).
    for ind in population:
        ind.alive = ind.id in survivor_ids
    return population

def show_body(
    body: nx.DiGraph,
    mode: ViewerTypes = MODE,
    file_name: str = "body",
) -> None:
    """Build a body graph in MuJoCo and look at it.

    There is no controller and no physics worth speaking of - this exists so
    you can SEE what your fitness function is actually rewarding. Do this
    early and often. A number going down is not evidence that the bodies look
    anything like the targets.
    """
    if mode == "none":
        return

    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    world = SimpleFlatWorld()
    robot = construct_mjspec_from_graph(body)
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    match mode:
        case "launcher":
            # Interactive window. Drag the modules around; nothing drives them.
            viewer.launch(model=model, data=data)
        case "frame":
            # A still image - the cheapest way to eyeball a body.
            save_path = str(DATA / f"{file_name}.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
            console.log(f"saved {save_path}")
        case "video":
            # Mostly useful for showing a body slumping under gravity.
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(model, data, duration=5.0, video_recorder=recorder)


# --- PLOTTING the results --- #
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

    ax.plot(generations, mean_per_gen, label="EA Variant 2", color="red")
    ax.fill_between(
        generations,
        mean_per_gen - std_per_gen,
        mean_per_gen + std_per_gen,
        color="red",
        alpha=0.2,
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (tree edit distance)")
    ax.set_title("Convergence: EA Variant 2")
    ax.legend()
    plt.tight_layout()
    plt.show()


# --- Run the EA for 5 different seeds --- #
NUM_GENERATIONS: int = 100

def run_ea(seed: int) -> list[float]:
    random.seed(seed)
    initial = Population([make_individual() for _ in range(config.target_population_size)])
    initial = evaluate(initial)

    ops: list[EAOperation] = [
        EAOperation(mutation_selection),
        EAOperation(mutate),
        EAOperation(evaluate),
        EAOperation(survivor_selection),
    ]

    ea = EA(initial, ops, num_steps=NUM_GENERATIONS, is_maximisation=False)

    history: list[float] = []
    for gen in range(NUM_GENERATIONS):
        ea.step()
        best = ea.get_solution('best', only_alive=False)
        history.append(best.fitness)

    console.log(f"--- Results (seed={seed}) ---")
    console.log(f"best = {ea.get_solution('best', only_alive=False)}")
    console.log(f"median = {ea.get_solution('median', only_alive=False)}")
    console.log(f"worst = {ea.get_solution('worst', only_alive=False)}")

    best = ea.get_solution('best', only_alive=False)
    graph = TreeGenome.from_dict(best.genotype).to_networkx()
    show_body(graph, MODE, file_name=f"best_individual_seed{seed}")

    ea.engine.dispose()  # release the DB connection so the next seed's EA can delete/recreate the file

    return history

# ─────────────────────────────────────────────── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    config.target_population_size = 50

    seeds = [42, 43, 44, 45, 46]
    all_histories: list[list[float]] = []
    for seed in seeds:
        history = run_ea(seed)
        all_histories.append(history)

    # Save results so they can be combined with other variants later
    with open(DATA / "histories_variant2.json", "w") as f:
        json.dump(all_histories, f)

    plotting(all_histories)

if __name__ == "__main__":
    main()