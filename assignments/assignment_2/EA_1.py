# Standard library
from pathlib import Path
from typing import Literal

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

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

def build_world() -> SimpleFlatWorld:                   # the world that the robot moves in, is constant and can be changed!
    return SimpleFlatWorld()


def build_robot() -> CoreModule:
    return gecko()                      # the body can be changed, but also update the OUPUT size (hinges) and INPUT (amount of qpos)


# ============================================================================ #
#  2. THE CONTROLLER CONTRACT
# ============================================================================ #
#
# MuJoCo calls the controller every physics step with (model, data); its job
# is to write into data.ctrl.
#
#   INPUTS   : whatever you read from `data` (qpos, qvel, time, ...), plus any
#              task info you already know, e.g. the vector to TARGET_POSITION.
#              INPUT SIZE is your choice, but must stay CONSTANT.
#   OUTPUTS  : exactly `model.nu` values, one per actuated hinge.
#   RANGE    : hinges accept [-pi/2, +pi/2] radians. A tanh output gives
#              [-1, 1] - rescale: actions * (np.pi / 2).
#   WRITING  : DIRECT (data.ctrl[:] = actions) commands the angle straight -
#              fast, but can destabilise the sim on large jumps. DELTA
#              (data.ctrl[:] += actions * alpha, alpha ~ 0.05, then clip) is
#              smoother but accumulates, so clipping is required. Pick one,
#              justify it, use it everywhere.
#   NaN      : blown-up weights silently write NaN into data.ctrl. Assert
#              against it while developing.
#
# ============================================================================ #

HIDDEN_SIZE: int = 6    #can be changed is own preference (explain!) / the hidden layer of the NN

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
    ind = Individual
    return ind

def evaluate(population: Population) -> Population:
    # use the fitness function
    return 

def select_parents(population: Population) -> List[Individual]:
    #Tournament selection?
    parents = []
    return parents

def crossover(parents: List[Individual], population: Population) -> Population:
    """"
        parent 1 -> [0.2, 0.7, 0.1, 0.4, 0.3]
        paretn 2 -> [0.5, 0.1, 0.9, 0.6, 0.7]
          crossover -> [0.2, 0.7, | 0.1, 0.4, 0.3]
               child 1 -> [0.2, 0.7, 0.9, 0.6, 0.7]
               child 2 -> [0.5, 0.1, 0.1, 0.4, 0.3]
    """"
    return population

def mutate(population: Population) -> Population:
    #add small certain number to certain weights of individuals
    #ones selected with mutate = True 
    return population

def survivor_selection(population: Population) -> Population:
    #select the 50% best fitnesses
    return population

def EA_1(input_size: int, output_size: int) -> list(npt.NDArray[np.float64]):
    return [None, None]

def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    return np.asarray(data.qpos[0:3]).copy()    # Return the robot core's current (x, y, z) world position, read before and after stepping (data.geom("robot1_core").xpos)

#!!!TO DO!!!
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

def run_experiment(mode: ViewerTypes = MODE) -> float:
    # This is the function your EA calls once per individual, with `mode` set to "simple" (headless).
    # Returns the fitness of this run. Lower is better.

    world = build_world()
    robot = build_robot()

    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    # Compile the world into a model. USE AS IS.
    model = world.spec.compile()
    data = mj.MjData(model)

    # Put the simulation in a clean, known state before reading anything.
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    # --- Wire up the controller -------------------------------------------- #
    # Sizes are read from the compiled model, never hardcoded - they depend on
    # the body you chose in build_robot().
    input_size = len(data.qpos)
    output_size = model.nu

    #change to the EA version
    weights = make_random_weights(input_size, output_size)


    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        """Compute and apply actions; MuJoCo calls this every physics step."""
        actions = nn_controller(m, d, weights)

        # DIRECT application (see the controller contract above).
        d.ctrl[:] = actions

        # DELTA application - comment out the line above and use these instead:
        # delta = 0.05
        # d.ctrl[:] += actions * delta
        # d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

    # --- Record the starting point ----------------------------------------- #
    initial_position = get_core_position(data)

    # --- Run ---------------------------------------------------------------- #
    if mode != "no_control":
        mj.set_mjcb_control(control_callback)

    match mode:
        case "launcher":
            # Interactive window. Great for seeing what your robot does,
            # useless inside an evolutionary loop.
            viewer.launch(model=model, data=data)
        case "simple":
            # Headless. THIS is the one your EA uses.
            simple_runner(model, data, duration=SIM_DURATION)
        case "video":
            # Render to an mp4 - for the figures in your report.
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(
                model,
                data,
                duration=SIM_DURATION,
                video_recorder=recorder,
            )
        case "frame":
            # A single image of the scene. Useful to check your spawn position
            # and that the robot is not clipping through the floor.
            single_frame_renderer(model, data, steps=1, show=True)
        case "no_control":
            # No controller attached: drag the hinges around by hand.
            viewer.launch(model=model, data=data)

    # Detach the callback again so the next run starts clean.
    mj.set_mjcb_control(None)

    # --- Score -------------------------------------------------------------- #
    final_position = get_core_position(data)
    fitness = fitness_function(initial_position, final_position)

    console.log(f"start  : {np.round(initial_position, 3)}")
    console.log(f"end    : {np.round(final_position, 3)}")
    console.log(f"target : {np.round(TARGET_POSITION, 3)}")
    console.log(f"fitness: {fitness:.4f}   (lower is better)")

    return fitness


def main() -> None:
    """Run a single demo evaluation with a randomly-weighted controller."""
    # A quick look at the size of the problem you are about to search.
    mj.set_mjcb_control(None)
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
    num_weights = (
        input_size * HIDDEN_SIZE
        + HIDDEN_SIZE * output_size
    )
    console.log(f"controller inputs (len(data.qpos)) : {input_size}")
    console.log(f"controller outputs (model.nu)      : {output_size}")
    console.log(f"genotype length (total weights)    : {num_weights}")

    run_experiment(MODE)


if __name__ == "__main__":
    main()