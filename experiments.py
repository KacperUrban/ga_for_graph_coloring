import numpy as np
import random
from deap import base, creator, tools, algorithms
import json
import optuna
import mlflow


def load_matrices(file_name):
    with open(file_name, 'r') as f:
        matrices = json.load(f)
    return [np.array(matrix) for matrix in matrices]



def train(adj_matrix, num_colors = 2, pop_size=100, num_generations=100, cxpb=0.7, mutpb=0.2, cxtype="cxTwoPoint", seltype="selTournament"):
    def random_color():
        return random.randint(0, num_colors - 1)


    def evaluate(individual):
        conflicts = 0
        for i in range(len(individual)):
            for j in range(i + 1, len(individual)):
                if adj_matrix[i, j] == 1 and individual[i] == individual[j]:
                    conflicts += 1
        return (conflicts,)
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("attr_color", random_color)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_color,
        n=len(adj_matrix),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    if cxtype == "cxOnePoint":
        toolbox.register("mate", tools.cxOnePoint)
    elif cxtype == "cxTwoPoint":
        toolbox.register("mate", tools.cxTwoPoint)
    elif cxtype == "cxUniform":
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=num_colors - 1, indpb=0.1)

    if seltype == "selTournament":
        toolbox.register("select", tools.selTournament, tournsize=3)
    elif seltype == "selRoulette":
        toolbox.register("select", tools.selRoulette)
    elif seltype == "selBest":
        toolbox.register("select", tools.selBest)


    population = toolbox.population(n=pop_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", min)
    stats.register("avg", np.mean)

    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=num_generations,
        stats=stats,
        verbose=False,
    )

    best_ind = tools.selBest(population, 1)[0]

    return best_ind.fitness.values[0]


def objective(trail):
    with mlflow.start_run():
        params = {
            "population_size" : trail.suggest_int("population_size", low=10, high=500),
            "num_of_generations" : trail.suggest_int("num_of_generations", low=10, high=500),
            "crossover_rate" : trail.suggest_float("crossover_rate", low=0.05, high=1.0),
            "mutation_rate" : trail.suggest_float("mutation_rate", low=0.05, high=1.0),
            "crossover_type" : trail.suggest_categorical("crossover_type", ["cxOnePoint", "cxTwoPoint", "cxUniform"]),
            "selection_type" : trail.suggest_categorical("selection_type", ["selTournament", "selRoulette", "selBest"]), 
        }
        mlflow.log_params(params)

        matrices = load_matrices("matrices.json")

        loss = 0

        for matrix in matrices:
            loss += train(matrix, 7, params["population_size"], params["num_of_generations"], params["crossover_rate"], 
                            params["mutation_rate"], params["crossover_type"], params["selection_type"])
        mlflow.log_metric("Error", loss)
    return loss



if __name__ == "__main__":
    mlflow.set_experiment("Graph-coloring")

    study = optuna.create_study(direction="minimize")

    study.optimize(objective, n_trials=30)

    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)