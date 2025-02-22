import numpy as np
from numpy.linalg import inv

matriz_original = np.array(
    [
        [3.0, 2.0, 4.0],
        [2.0, 5.0, 1.0],
        [4.0, 1.0, 2.0],
    ]
)
matriz_inversa = inv(matriz_original)

matriz_calculada = np.array(
    [
        [-9 / 45, 0, 18 / 45],
        [0, 10 / 45, -5 / 45],
        [18 / 45, -5 / 45, -11 / 45],
    ],
)

print(matriz_inversa)
print("------------")
print(matriz_calculada)
