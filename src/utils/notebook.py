import numpy as np


def saddle_mesh(n_vertices):
    n = int(np.round(np.sqrt(n_vertices)))
    if n % 2 == 0:
        n += 1

    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)

    Z = X**2 - Y**2

    V = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = v0 + 1
            v2 = (i + 1) * n + j
            v3 = v2 + 1

            F.append([v0, v1, v2])
            F.append([v1, v3, v2])

    F = np.array(F, dtype=np.int32)

    saddle_vertex = (n // 2) * n + (n // 2)

    return V, F, saddle_vertex


def get_closest_vertex(vertices, point):
    distances_sq = np.sum((vertices - point) ** 2, axis=1)

    return np.argmin(distances_sq)
