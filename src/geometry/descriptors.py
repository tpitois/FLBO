import numpy as np
import scipy.sparse.linalg as sla


def compute_eigen_decomposition(W, S, k=100):
    """
    Calcule les k premières valeurs et vecteurs propres du Laplacien.
    Résout le problème généralisé : W * phi = lambda * S * phi
    """
    evals, evecs = sla.eigsh(-W, k=k, M=S, sigma=-1e-3, which="LM")

    idx = evals.argsort()
    evals = evals[idx]
    evecs = evecs[:, idx]

    evals = np.maximum(evals, 1e-10)

    return evals, evecs


def compute_wks(evals, evecs, num_energies=100):
    """
    Calcule le Wave Kernel Signature (WKS).
    """
    N = evecs.shape[0]
    WKS = np.zeros((N, num_energies))

    log_E = np.log(np.maximum(evals, 1e-10))
    e_min, e_max = log_E[1], log_E[-1]

    sigma = 7.0 * (e_max - e_min) / num_energies

    e_vals = np.linspace(e_min, e_max, num_energies)

    for i, e in enumerate(e_vals):
        weights = np.exp(-((e - log_E) ** 2) / (2 * sigma**2))

        WKS[:, i] = np.sum((evecs**2) * weights, axis=1)

        sum_weights = np.sum(weights)
        if sum_weights > 1e-10:
            WKS[:, i] /= sum_weights

    return WKS
