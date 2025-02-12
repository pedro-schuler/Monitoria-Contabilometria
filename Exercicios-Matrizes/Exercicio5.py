import numpy as np
from numpy.linalg import inv

matriz_original = np.array(
    [
        [-1.0, -2.0, 3.0],
        [1.0, -4.0, 2.0],
        [2.0, 5.0, 6.0],
    ]
)
matriz_inversa = inv(matriz_original)

matriz_calculada = np.array(
    [
        [-34 / 77, 27 / 77, 8 / 77],
        [-2 / 77, -12 / 77, 5 / 77],
        [13 / 77, 1 / 77, 6 / 77],
    ],
)

print(matriz_inversa)
print("------------")
print(matriz_calculada)
