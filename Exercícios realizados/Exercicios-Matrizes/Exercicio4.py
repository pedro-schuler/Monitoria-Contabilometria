import numpy as np
from numpy.linalg import inv

matriz_original = np.array(
    [
        [1.0, 0.0, 7.0],
        [0.0, 6.0, 0.0],
        [5.0, 0.0, 8.0],
    ]
)
matriz_inversa = inv(matriz_original)

matriz_calculada = np.array(
    [
        [-8 / 27, 0, 7 / 27],
        [0, 1 / 6, 0],
        [5 / 27, 0, -1 / 27],
    ],
)

print(matriz_inversa)
print("------------")
print(matriz_calculada)
