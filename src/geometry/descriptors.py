import numpy as np
import scipy.sparse.linalg as sla


def compute_eigen_decomposition(W, S,k=100):
    """
    Calcule les k premières valeurs et vecteurs propres du Laplacien.
    Résout le problème généralisé : W * phi = lambda * S * phi
    """
    # On cherche les k plus petites valeurs propres (SM = Smallest Magnitude)
    # sigma=-1e-3 aide le solveur à trouver la valeur propre 0 sans instabilité numérique
    evals, evecs = sla.eigsh(-W, k=k, M=S, sigma=-1e-3, which='LM')

    # Tri par ordre croissant (eigsh ne garantit pas toujours l'ordre parfait)
    idx = evals.argsort()
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Nettoyage des petites erreurs numériques (la première valeur propre doit être 0)
    evals = np.maximum(evals, 1e-10)

    return evals, evecs


def compute_wks(evals, evecs, num_energies=100):
    """
    Calcule le Wave Kernel Signature (WKS) de manière sécurisée.
    """
    N = evecs.shape[0]
    WKS = np.zeros((N, num_energies))

    # Sécurité 1 : Nettoyer les éventuels NaN dans evals
    if np.any(np.isnan(evals)):
        print("⚠️ ATTENTION : Le solveur a renvoyé des NaN dans les valeurs propres !")
        evals = np.nan_to_num(evals, nan=1e-10)

    log_E = np.log(np.maximum(evals, 1e-10))
    e_min, e_max = log_E[1], log_E[-1]

    sigma = 7.0 * (e_max - e_min) / num_energies

    # Sécurité 2 : Empêcher sigma de valoir 0
    if sigma < 1e-6 or np.isnan(sigma):
        print(f"⚠️ ATTENTION : sigma invalide ({sigma}). Les valeurs propres sont anormales.")
        print(f"  -> e_min = {e_min}, e_max = {e_max}")
        sigma = 1e-4  # Valeur de secours stricte pour éviter le crash

    e_vals = np.linspace(e_min, e_max, num_energies)

    for i, e in enumerate(e_vals):
        # La division par sigma**2 est maintenant 100% sécurisée
        weights = np.exp(-((e - log_E) ** 2) / (2 * sigma ** 2))

        WKS[:, i] = np.sum((evecs ** 2) * weights, axis=1)

        # Sécurité 3 : Éviter une division par 0 lors de la normalisation
        sum_weights = np.sum(weights)
        if sum_weights > 1e-10:
            WKS[:, i] /= sum_weights

    return WKS