import numpy as np
from dataclasses import dataclass

@dataclass
class Options:
    alpha: float
    angle: float
    tau: float

def compute_D_finsler(U: np.ndarray, options: Options) -> np.ndarray:
    """
    Calcule le tenseur de métrique de Finsler (D_finsler) de façon hyper-optimisée.
    """
    # 1. Matrice de rotation dans le plan tangent
    c = np.cos(options.angle)
    s = np.sin(options.angle)
    R = np.array([
        [ s,  c, 0],  # U1
        [ c, -s, 0],  # U2
        [ 0,  0, 1]   # N
    ])

    # Application de la rotation à tous les repères d'un coup
    U_rot = R @ U  # (3, 3) @ (N_faces, 3, 3) -> U_rot a (U1, U2, N) en lignes

    # 2. Construction de la matrice de diffusion de base M = U^T * D * U
    D_mat = np.diag([1.0 / (1.0 + options.alpha), 1.0, 1.0])

    # U_rot.transpose(0, 2, 1) est U^T pour chaque face
    M = U_rot.transpose(0, 2, 1) @ D_mat @ U_rot

    # 3. Métrique de Randers et dualité
    # On sait mathématiquement que <w, Mw> = tau^2 puisque w = tau * U2 et D2 = 1
    eta = 1.0 - (options.tau ** 2)

    # Extraction de U2 (qui est sur la 2ème ligne de U_rot, index 1)
    U2 = U_rot[:, 1, :]
    w = options.tau * U2

    # Calcul de w* (vecteur dual)
    # np.einsum('mij,mj->mi', M, w) est le produit matrice-vecteur M @ w
    Mw = np.einsum('mij,mj->mi', M, w)
    wstar = -Mw / eta

    # Produit externe w* @ w*^T
    wstar_outer = np.einsum('mi,mj->mij', wstar, wstar)

    # 4. Tenseur de Finsler
    # M_star = (Mw @ Mw^T + eta * M) / eta^2
    Mw_outer = np.einsum('mi,mj->mij', Mw, Mw)
    Mstar = (Mw_outer + eta * M) / (eta ** 2)

    D_finsler = Mstar - wstar_outer

    # Nettoyage des petites erreurs numériques
    D_finsler[np.abs(D_finsler) < 1e-10] = 0.0
    D_finsler[np.abs(D_finsler - 1.0) < 1e-10] = 1.0

    return D_finsler