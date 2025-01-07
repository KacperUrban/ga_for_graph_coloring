import numpy as np
import random
import json

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


def generate_and_save_matrices(file_name, num_matrices, low=2, high=8):
    matrices = []
    for _ in range(num_matrices):
        matrix = generate_random_adjacency_matrix(low, high)
        matrices.append(matrix.tolist())
    
    with open(file_name, 'w') as f:
        json.dump(matrices, f)

if __name__ == '__main__':
    generate_and_save_matrices("matrices.json", 5, low=50, high=100)
    