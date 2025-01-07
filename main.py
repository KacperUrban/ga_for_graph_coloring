import numpy as np
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import networkx as nx



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


# Funkcja generująca losowy kolor
def random_color():
    return random.randint(0, NUM_COLORS - 1)



# Funkcja oceny (fitness)
def evaluate(individual):
    conflicts = 0
    for i in range(len(individual)):
        for j in range(i + 1, len(individual)):
            if adjacency_matrix[i, j] == 1 and individual[i] == individual[j]:
                conflicts += 1
    return (conflicts,)


def main():
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
        n=len(adjacency_matrix),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    # Funkcje genetyczne
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=NUM_COLORS - 1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    # Parametry algorytmu
    pop_size = 100
    num_generations = 100

    # Tworzenie początkowej populacji
    population = toolbox.population(n=pop_size)

    # Statystyki
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", min)
    stats.register("avg", np.mean)

    # Algorytm
    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.7,  # prawdopodobieństwo krzyżowania
        mutpb=0.2,  # prawdopodobieństwo mutacji
        ngen=num_generations,
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

    # Definicja grafu za pomocą macierzy sąsiedztwa
    # adjacency_matrix = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])

    adjacency_matrix = generate_random_adjacency_matrix(low=3, high=6)

    # Liczba dostępnych kolorów
    NUM_COLORS = 2
    # Uruchomienie algorytmu
    best_solution = main()
    visualize_graph(adjacency_matrix, best_solution)
