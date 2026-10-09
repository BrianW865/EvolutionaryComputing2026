# Standard library
from pathlib import Path
from typing import Literal
import json
import matplotlib.pyplot as plt

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider import spider
from ariel.ec import set_seed
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

# Type aliases
type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]                # where the robot starts
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]          # where it should end up
SIM_DURATION: float = 15.0                              # seconds of simulated time per evaluation
MODE: ViewerTypes = "simple"                          # see run_experiment() for the options


#------------constants that can be changed-------------------
POPULATION_SIZE: int = 100
MAX_GENERATIONS: int = 150
MIN_GENERATIONS: int = 50
HIDDEN_SIZE: int = 6    #can be changed is own preference (explain!) / the hidden layer of the NN
MUTATION_RATE: float = 0.4
TOLERANCE: float = 0.01
WINDOW_SIZE: int = 20   #number of generations without significant improvement before stopping the EA

def build_world() -> SimpleFlatWorld:
    world = SimpleFlatWorld()
    target_body = world.spec.worldbody.add_body(
        name="target_marker",
        pos=TARGET_POSITION,
    )
    target_body.add_geom(
        name="target_marker_geom",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=[0.1, 0.1, 0.1],
        rgba=[1.0, 0.0, 0.0, 0.7],
        contype=0,
        conaffinity=0,
    )
    return world

def build_robot() -> CoreModule:
    return spider() 

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
 
    return [
        RNG.normal(scale=0.5, size=(input_size, HIDDEN_SIZE)),
        RNG.normal(scale=0.5, size=(HIDDEN_SIZE, output_size)),
    ]

def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    return np.asarray(data.qpos[0:3]).copy()

def fitness_function(
    initial_position: npt.NDArray[np.float64],
    final_position: npt.NDArray[np.float64],
) -> float:
 
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

    return fitness

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

    ax.plot(generations, mean_per_gen, label="Random search", color="black")
    ax.fill_between(
        generations,
        mean_per_gen - std_per_gen,
        mean_per_gen + std_per_gen,
        color="black",
        alpha=0.2,
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (distance to target)")
    ax.set_title("Convergence: Random search")
    ax.legend()
    plt.savefig(DATA / "random_search_convergence.png", dpi=300)
    plt.close()

def random_search(seed: int, budget: int, checkpoint_every: int) -> list[float]:
    global RNG
    RNG = np.random.default_rng(seed)
    set_seed(seed)

    world = build_world()
    robot = build_robot()
    world.spawn(robot.spec, position=SPAWN_POS, correct_collision_with_floor=True)

    model = world.spec.compile()
    data = mj.MjData(model)
    input_size = len(data.qpos)
    output_size = model.nu

    best_fitness = float("inf")
    history = []

    for i in range(1, budget + 1):
        weights = make_random_weights(input_size, output_size)
        fitness = run_experiment(weights, mode="simple")
        if fitness < best_fitness:
            best_fitness = fitness
        if i % checkpoint_every == 0:
            history.append(best_fitness)
            console.log(f"seed progress: {i}/{budget}")

    return history

#Budget based on all three EAs for the random search
#Requires EA_1.py (and EA_2/EA_3, if pooling) to have been run first,
#So their histories_*.json files already exist in __data__/.
ea_histories = []
for variant_folder, variant_file in [
    ("EA_1", "histories_variant1.json"),
    ("EA_2", "histories_variant2.json"),
    ("EA_3", "histories_variant3.json"),
]:
    with open(CWD / "__data__" / variant_folder / variant_file) as f:
        ea_histories.extend(json.load(f))

avg_generations = sum(len(h) for h in ea_histories) / len(ea_histories)
NUM_GENERATIONS = round(avg_generations)

budget= POPULATION_SIZE * NUM_GENERATIONS
checkpoint_every = POPULATION_SIZE

def main()-> None:
    seeds = [42, 43, 44, 45, 46]
    all_histories: list[list[float]] = []

    for seed in seeds:
        history = random_search(seed, budget, checkpoint_every)
        number = 0
        for _ in history:
            number += 1
        console.log(f"Amount of numbers stored: {number}")
        all_histories.append(history)

    with open(DATA / "histories_random_search_3.json", "w") as f:
        json.dump(all_histories, f)

    plotting(all_histories)

if __name__ == "__main__":
    main()