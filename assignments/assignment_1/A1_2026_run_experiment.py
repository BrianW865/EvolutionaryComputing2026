from A1_2026_EA1 import run_EA1
from A1_2026_EA2 import run_EA2
from A1_2026_random_search import random_search

import numpy as np
import matplotlib.pyplot as plt
from rich.traceback import install
from rich.console import Console
from rich.table import Table

install()
console = Console()

SEEDS = [0,1,2,3,4]
NUM_OF_GENERATIONS = 10
NUM_OF_CHILDREN_PER_GEN = 10
POPULATION_SIZE = 20
EVALUATION_BUDGET = POPULATION_SIZE + NUM_OF_CHILDREN_PER_GEN * NUM_OF_GENERATIONS

def main() -> None:
    for seed in SEEDS:
        console.log(f"Running for Seed: {seed}")
        run_EA1 = run_EA1(seed, NUM_OF_GENERATIONS, POPULATION_SIZE)
        run_EA2 = run_EA2(seed, NUM_OF_GENERATIONS, POPULATION_SIZE)
        run_random = random_search(seed, EVALUATION_BUDGET)

        #log the results
        #make a graph for each seed including EA1, EA2 and random_search
        #after executing report on the mean and spread
