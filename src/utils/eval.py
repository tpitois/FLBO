import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra


def compute_surface_area(V, F):
    v1, v2, v3 = V[F].transpose(1, 0, 2)

    return 0.5 * np.sum(np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1))


def compute_adjacency_matrix(V, F):
    N = V.shape[0]
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])

    distances = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)

    W = sp.coo_matrix((distances, (edges[:, 0], edges[:, 1])), shape=(N, N))
    W = W + W.T
    return W


def evaluate_predictions(preds, labels, V, F, num_samples=1000):
    area = compute_surface_area(V, F)
    norm_factor = np.sqrt(area)

    W = compute_adjacency_matrix(V, F)

    eval_indices = np.random.choice(
        len(labels), min(num_samples, len(labels)), replace=False
    )

    errors = []

    for idx in eval_indices:
        pred_vertex = preds[idx]
        true_vertex = labels[idx]

        if pred_vertex == true_vertex:
            errors.append(0.0)
            continue

        dist_map = dijkstra(W, indices=true_vertex, directed=False, unweighted=False)

        geo_dist = dist_map[pred_vertex]
        normalized_error = geo_dist / norm_factor
        errors.append(normalized_error)

    return np.array(errors)


def plot_pck_curve(errors, max_threshold=0.25, num_bins=100, save_path="pck_curve.png"):
    thresholds = np.linspace(0, max_threshold, num_bins)
    pck = []

    for t in thresholds:
        correct_percentage = np.mean(errors <= t) * 100
        pck.append(correct_percentage)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, pck, linewidth=2, color="b")
    plt.title("Correspondences Evaluation (PCK Curve)")
    plt.xlabel("Geodesic Error Threshold (D)")
    plt.ylabel("Percentage of Matches (%)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlim([0, max_threshold])
    plt.ylim([0, 105])

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
