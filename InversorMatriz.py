import numpy as np
from numpy.linalg import inv

matriz = np.array(
    [
        [3.0, 2.0, 4.0],
        [2.0, 5.0, 1.0],
        [4.0, 1.0, 2.0],
    ]
)
matriz_inversa = inv(matriz)

print(matriz_inversa)
