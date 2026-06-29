import numpy as np
from numpy.linalg import inv

matriz_original = np.array(
    [
        [1.0, 2.0, 2.0],
        [6.0, 5.0, 4.0],
        [2.0, 8.0, 2.0],
    ]
)
matriz_inversa = inv(matriz_original)

matriz_calculada = np.array(
    [
        [-11 / 23, 6 / 23, -1 / 23],
        [-2 / 23, -1 / 23, 4 / 23],
        [19 / 23, -2 / 23, -7 / 46],
    ],
)

print(matriz_inversa)
print("------------")
print(matriz_calculada)
