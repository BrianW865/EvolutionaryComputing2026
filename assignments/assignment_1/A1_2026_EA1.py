"""Example: one-max EA using the refactored prototype of ariel.ec module."""

NUM_OF_MODULES: int = 20

from ariel.ec import (
    EA,
    EAOperation,
    Individual,
    Population,
    config,
)
from ariel.ec.genotypes.tree import TreeGenome
from ariel.ec.genotypes.tree.operators import (
    random_tree,
    crossover_subtree,
    mutate_replace_node
) 

from A1_template_2026 import (
    fitness_function,
    load_targets
)

targets = load_targets()

def make_individual() -> Individual:
    ind = Individual()
    treeGenome = random_tree(max_modules = NUM_OF_MODULES)
    ind.genotype = treeGenome.to_dict()

    return ind

    #make a random tree generation

def evaluate(population: Population) -> Population:
    for ind in population.unevaluated:
        genome = TreeGenome.from_dict(ind.genotype)
        body = genome.to_networkx()
        ind.fitness = fitness_function(body, targets)

    return population

    #evaluate the population based on the tree_edit_distance 


def parent_selection(population: Population) -> Population:

    return population

    #select parents for offspring

def crossover(population: Population) -> Population:
    parents = population.where(lambda ind: bool(ind.tags.get("selected", False)))

    return parents

    #make offspring based on the selected parents 

def mutate(population: Population) -> Population:

    #use mutate_replace_node
    return population

    #mutate the offspring based on mutate possibility rate and mutate operator (different per EA version)


def survivor_selection(population: Population) -> Population:
    
    return population

    #select survivors 

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    config.target_population_size = 20
    initial = Population([make_individual() for _ in range(20)])   #make 
    initial = evaluate(initial)

    ops: list[EAOperation] = [
        EAOperation(parent_selection),
        EAOperation(crossover),
        EAOperation(mutate),
        EAOperation(evaluate),
        EAOperation(survivor_selection),
    ]

    ea = EA(initial, ops, num_steps=50)
    ea.run()

    console.log("--- Results ---")
    console.log(f"best = {ea.get_solution('best', only_alive=False)}")
    console.log(f"median = {ea.get_solution('median', only_alive=False)}")
    console.log(f"worst = {ea.get_solution('worst', only_alive=False)}")

    #make 20 (or different amount) of random trees
    #select the a certain amount (best 10 or tournament selection) -> def parent_selection()
    #evaluate these chosen ones  --> def evaluate()
    #do crossover on parents --> def crossover()
    #do the mutation --> def mutate()
    #select survivors --> def survivor_selection()

    #sudo code
    # population = make_random_population(20)
    # evaluate(population)
    #
    # for i in range (NUM_GENERATIONS)
    #   parents = parent_selection(population)
    #   offspring = crossover(parents)
    #   offspring = mutation(offspring)
    #   evaluate(offspring)
    #   population = survivor_selection(population, offspring)
    # 
    # best = get_best(population)

if __name__ == "__main__":
    main()
