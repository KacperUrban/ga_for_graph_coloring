import numpy as np
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import networkx as nx
import json

def load_matrices(file_name):
    with open(file_name, 'r') as f:
        matrices = json.load(f)
    return [np.array(matrix) for matrix in matrices]

def generate_random_adjacency_matrix(low=2, high=8):
    n = random.randint(low, high)
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i, n):
            if i != j:
                value = random.randint(0, 1)
                matrix[i][j] = value
                matrix[j][i] = value
    return matrix



def visualize_graph(adjacency_matrix, colors=None):
    # Tworzenie grafu z macierzy sąsiedztwa
    G = nx.from_numpy_array(adjacency_matrix)

    # Tworzenie układu grafu
    pos = nx.spring_layout(G)

    plt.figure(figsize=(5, 5))

    if colors is not None:
        # Jeśli podano kolory, użyj ich do pokolorowania wierzchołków
        color_map = ["C" + str(color) for color in colors]
        nx.draw(
            G,
            pos,
            node_color=color_map,
            with_labels=True,
            node_size=500,
            font_size=16,
            font_weight="bold",
        )
    else:
        # Jeśli nie podano kolorów, użyj domyślnego koloru
        nx.draw(
            G, pos, with_labels=True, node_size=500, font_size=16, font_weight="bold"
        )

    plt.title("Graf z macierzy sąsiedztwa")
    plt.show()


def main(adj_mat, num_colors, params):
    # Funkcja generująca losowy kolor
    def random_color():
        return random.randint(0, num_colors - 1)

    # Funkcja oceny (fitness)
    def evaluate(individual):
        conflicts = 0
        for i in range(len(individual)):
            for j in range(i + 1, len(individual)):
                if adj_mat[i, j] == 1 and individual[i] == individual[j]:
                    conflicts += 1
        return (conflicts,)
    # Tworzenie typów dla DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Inicjalizacja narzędzi
    toolbox = base.Toolbox()

    # Rejestracja funkcji w toolbox
    toolbox.register("attr_color", random_color)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_color,
        n=len(adj_mat),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    # Funkcje genetyczne
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", params["crossover_type"])
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=num_colors - 1, indpb=0.1)
    toolbox.register("select", params["selection_type"], tournsize=3)

    # Tworzenie początkowej populacji
    population = toolbox.population(n=params["population_size"])

    # Statystyki
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", min)
    stats.register("avg", np.mean)

    # Algorytm
    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=params["crossover_rate"],  # prawdopodobieństwo krzyżowania
        mutpb=params["mutation_rate"],  # prawdopodobieństwo mutacji
        ngen=params["num_of_generations"],
        stats=stats,
        verbose=True,
    )

    # Najlepsze znalezione rozwiązanie
    best_ind = tools.selBest(population, 1)[0]
    print("\nNajlepsze znalezione rozwiązanie:", best_ind)
    print("Liczba konfliktów:", best_ind.fitness.values[0])

    # Sprawdzenie poprawności kolorowania
    if best_ind.fitness.values[0] == 0:
        print("Znaleziono poprawne kolorowanie grafu!")
    else:
        print("Nie znaleziono poprawnego kolorowania grafu.")

    return best_ind


if __name__ == "__main__":
    # best params
    params = {
        "crossover_rate" : 0.9140770311656784,
        "crossover_type" : tools.cxTwoPoint,
        "mutation_rate" : 0.4165544386688508,
        "num_of_generations" : 424,
        "population_size" : 306,
        "selection_type" : tools.selTournament,
    }
    matrices = load_matrices("matrices.json")
    for matrix in matrices:
        best_solution = main(matrix, 7, params)
        visualize_graph(matrix, best_solution)
