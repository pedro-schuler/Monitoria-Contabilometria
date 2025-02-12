import numpy as np
from numpy.linalg import inv

matriz_original = np.array(
    [
        [2.0, 0.0, 6.0],
        [2.0, 3.0, 1.0],
        [4.0, 1.0, 5.0],
    ]
)
matriz_inversa = inv(matriz_original)

matriz_calculada = np.array(
    [
        [-7 / 16, -3 / 16, 9 / 16],
        [3 / 16, 7 / 16, -5 / 16],
        [5 / 16, 1 / 16, 3 / 16],
    ],
)

print(matriz_inversa)
print("------------")
print(matriz_calculada)
