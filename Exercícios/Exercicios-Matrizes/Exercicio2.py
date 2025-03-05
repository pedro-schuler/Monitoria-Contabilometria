import numpy as np
from numpy.linalg import inv

matriz_original = np.array(
    [
        [3.0, 1.0, 3.0],
        [1.0, 4.0, 2.0],
        [1.0, 0.0, 2.0],
    ]
)
matriz_inversa = inv(matriz_original)

matriz_calculada = np.array(
    [
        [2 / 3, -1 / 6, -5 / 6],
        [0, 1 / 4, -1 / 4],
        [-1 / 3, 1 / 12, 11 / 12],
    ],
)

print(matriz_inversa)
print("------------")
print(matriz_calculada)
