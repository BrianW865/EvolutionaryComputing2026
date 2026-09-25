# Standard library
from pathlib import Path
from typing import Literal

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
import random
import json

from ariel.ec import (
    EA,
    EAOperation,
    Individual,
    Population,
    config,
)

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko   #
from ariel.ec import set_seed
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

# Type aliases
type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]

SEED = 42                                   # can be changed, and should be tested using multiple seeds
RNG = np.random.default_rng(SEED)
set_seed(SEED)                              # needs to be reseeded when the seed is changed!

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]                # where the robot starts
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]          # where it should end up
SIM_DURATION: float = 15.0                              # seconds of simulated time per evaluation
MODE: ViewerTypes = "launcher"                          # see run_experiment() for the options


#------------constants that can be changed-------------------
target_population_size: int = 20
INITIAL_POPULATION: int = 20
NUM_GENERATIONS: int = 5
HIDDEN_SIZE: int = 6    #can be changed is own preference (explain!) / the hidden layer of the NN
MUTATION_RATE: float = 0.4

def build_world() -> SimpleFlatWorld:                   # the world that the robot moves in, is constant and can be changed!
    return SimpleFlatWorld()

def build_robot() -> CoreModule:
    return gecko()                      # the body can be changed, but also update the OUPUT size (hinges) and INPUT (amount of qpos)

def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
    weights: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    #Map robot state to hinge commands: in -> hidden -> actions.

    w1, w2 = weights   # the layer weight matrices 
    inputs = data.qpos   # input is bare qpos (simplest), but can be changed for better performance (qpos, qvel, time, target info...etc)
    layer1 = np.tanh(inputs @ w1)    #the hidden layer of the NN
    outputs = np.tanh(layer1 @ w2)   # in [-1, 1]

    return outputs * (np.pi / 2)  # in [-pi/2, pi/2]   -> rescales the hinges!

def make_random_weights(
    input_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:
    """Draw a random parameter set for `nn_controller`.

    THIS IS THE FUNCTION YOUR EA REPLACES. Instead of sampling weights from a
    normal distribution, your EA will search for them.

    Note the total parameter count printed by main(): that is the length of the
    flat vector an individual's genotype has to encode. Reshaping a flat
    genotype back into these matrices is on you.
    """
    return [
        RNG.normal(scale=0.5, size=(input_size, HIDDEN_SIZE)),
        RNG.normal(scale=0.5, size=(HIDDEN_SIZE, output_size)),
    ]

def make_individual() -> Individual:
    world = build_world()
    robot = build_robot()

    world.spawn(
        robot.spec,
        position = SPAWN_POS,
        correct_collision_with_floor = True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)
    input_size = len(data.qpos)
    output_size = model.nu

    weights = make_random_weights(input_size, output_size)
    genotype = np.concatenate([weights[0].flatten(), weights[1].flatten()])

    ind = Individual()
    ind.genotype = genotype.tolist()

    return ind

def evaluate(population: Population) -> Population:
    world = build_world()
    robot = build_robot()
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)

    input_size = len(data.qpos)
    output_size = model.nu

    w1_size = input_size * HIDDEN_SIZE
    w2_size = HIDDEN_SIZE * output_size

    for ind in population.unevaluated:
        genotype = np.asarray(ind.genotype)
        w1 = genotype[:w1_size].reshape(input_size, HIDDEN_SIZE,)
        w2 = genotype[w1_size:w1_size + w2_size].reshape(HIDDEN_SIZE, output_size,)
        weights = [w1, w2]

        ind.fitness = run_experiment(weights, mode = "simple")
    
    return population

def parent_selection(population: Population) -> Population:
    amount_of_parents: int = len(population)
    i: int = 0

    for ind in population:
        ind.tags["selected"] = False

    while i < amount_of_parents:
        tournament_selections = random.sample(list(population), 5)
        best_one = min(tournament_selections, key=lambda individual: individual.fitness)
        best_one.tags["selected"] = True
        i += 1

    selected_count = sum(1 for ind in population if ind.tags.get("selected", False))
    console.log(
        f"[cyan]Parent Selection: {selected_count}/{len(population)} marked for reproduction[/cyan]",
    )

    return population

def crossover(population: Population) -> Population:
    parents = population.where(lambda ind: bool(ind.tags.get("selected", False)))

    for idx in range(0, len(parents) - 1, 2):
        parent_1 = parents[idx]
        parent_2 = parents[idx + 1]
        crossover_point = np.random.randint(1, len(parent_1.genotype))

        child_1 = Individual()
        child_1.genotype = np.concatenate([parent_1.genotype[:crossover_point], parent_2.genotype[crossover_point:]]).tolist()
        child_1.tags = {"mutate": True}

        child_2 = Individual()
        child_2.genotype = np.concatenate([parent_2.genotype[:crossover_point], parent_1.genotype[crossover_point:]]).tolist()
        child_2.tags = {"mutate": True}

        population.extend([child_1, child_2])

    return population

def mutate(population: Population) -> Population:
    to_mutate = population.where(lambda ind: bool(ind.tags.get("mutate", False)))

    for ind in to_mutate:
        if random.random() < MUTATION_RATE:
            index = random.randint(0, len(ind.genotype) - 1)
            console.log(f"mutation number before: {ind.genotype[index]}")
            mutation_addition = np.random.normal(0, 0.03)  #this is gaussian mutation!
            console.log(f"Adding mutation_addition: {mutation_addition}")
            ind.genotype[index] += mutation_addition
            console.log(f"mutation number after: {ind.genotype[index]}")

    return population

def survivor_selection(population: Population) -> Population:
    survivors = population.best(sort = "min", n = config.target_population_size)
    survior_ids = {ind.id for ind in survivors}

    for ind in population:
        ind.alive = ind.id in survior_ids
    
    return population

def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    return np.asarray(data.qpos[0:3]).copy()    # Return the robot core's current (x, y, z) world position, read before and after stepping (data.geom("robot1_core").xpos)

def fitness_function(
    initial_position: npt.NDArray[np.float64],
    final_position: npt.NDArray[np.float64],
) -> float:
    """Score one evaluation. LOWER IS BETTER.

    The plain version: how far is the robot from the target when time runs out?

    `initial_position` is unused here on purpose - it is passed in because the
    moment you want a less naive fitness you will need it. Some things worth
    thinking about (and, ideally, comparing in your report):
      * Distance *reduced* rather than distance remaining, so a robot that
        starts closer is not rewarded for standing still.
      * Penalising a robot that falls over or leaves the arena.
      * Whether the z-axis should count at all - a robot that jumps is not
        closer to the target in any way you care about.
    See `ariel.simulation.tasks.targeted_locomotion` for some worked variants.
    """
    target = np.asarray(TARGET_POSITION)
    return float(np.linalg.norm(final_position[:2] - target[:2]))

def run_experiment(weights: list[npt.NDArray[np.float64]], mode: ViewerTypes = MODE) -> float:
    world = build_world()
    robot = build_robot()

    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)

    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    #input_size = len(data.qpos)
    #output_size = model.nu

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        actions = nn_controller(m, d, weights)
        d.ctrl[:] = actions

    initial_position = get_core_position(data)

    if mode != "no_control":
        mj.set_mjcb_control(control_callback)

    match mode:
        case "launcher":
            viewer.launch(model=model, data=data)
        case "simple":
            simple_runner(model, data, duration=SIM_DURATION)
        case "video":
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(
                model,
                data,
                duration=SIM_DURATION,
                video_recorder=recorder,
            )
        case "frame":
            single_frame_renderer(model, data, steps=1, show=True)
        case "no_control":
            viewer.launch(model=model, data=data)

    mj.set_mjcb_control(None)
    final_position = get_core_position(data)
    fitness = fitness_function(initial_position, final_position)

    #console.log(f"start  : {np.round(initial_position, 3)}")
    #console.log(f"end    : {np.round(final_position, 3)}")
    #console.log(f"target : {np.round(TARGET_POSITION, 3)}")
    #console.log(f"fitness: {fitness:.4f}   (lower is better)")

    return fitness

def run_ea(seed: int) -> list[float]:
    global RNG

    random.seed(seed)
    RNG = np.random.default_rng(seed)
    set_seed(seed)

    initial = Population([make_individual() for _ in range (config.target_population_size)])
    initial = evaluate(initial)

    ops: list[EAOperation] = [
        EAOperation(parent_selection),
        EAOperation(crossover),
        EAOperation(mutate),
        EAOperation(evaluate),
        EAOperation(survivor_selection),
    ]

    ea = EA(initial, ops, num_steps=NUM_GENERATIONS, is_maximisation = False)
    history: list[float] = []

    for gen in range(NUM_GENERATIONS):
        ea.step()
        best = ea.get_solution('best', only_alive = False)
        history.append(best.fitness)

    console.log(f"--- Results (seed={seed}) ---")
    console.log(f"best = {ea.get_solution('best', only_alive=False)}")
    console.log(f"median = {ea.get_solution('median', only_alive=False)}")
    console.log(f"worst = {ea.get_solution('worst', only_alive=False)}")

    best = ea.get_solution('best', only_alive=False)
    ea.engine.dispose()
    return history

def main() -> None:
    config.target_population_size = 20

    seeds = [42]
    all_histories: list[list[float]] = []

    for seed in seeds:
        history = run_ea(seed)
        all_histories.append(history)

    with open(DATA / "histories_variant1.json", "w") as f:
        json.dump(all_histories, f)
    
if __name__ == "__main__":
    main()