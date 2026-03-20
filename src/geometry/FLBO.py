import numpy as np
import scipy.sparse as sp
import torch

from scipy.sparse.linalg import eigs


def compute_stiffness_matrix(V, F, D_F):
    """
    Assemble la matrice de rigidité creuse W selon la logique du code MATLAB original.
    Fidèle aux équations du papier FLBO.
    """
    n_vertices = V.shape[0]
    m_faces = F.shape[0]

    # --- 1. Calcul des angles de chaque triangle ---
    angles = np.zeros_like(F, dtype=float)
    for i in range(3):
        i1, i2, i3 = i, (i + 1) % 3, (i + 2) % 3

        pp = V[F[:, i2]] - V[F[:, i1]]
        qq = V[F[:, i3]] - V[F[:, i1]]

        # Normalisation sécurisée
        norm_pp = np.maximum(np.linalg.norm(pp, axis=1, keepdims=True), 1e-12)
        norm_qq = np.maximum(np.linalg.norm(qq, axis=1, keepdims=True), 1e-12)
        pp /= norm_pp
        qq /= norm_qq

        # Produit scalaire et angle
        dot_prod = np.sum(pp * qq, axis=1)
        dot_prod = np.clip(dot_prod, -1.0, 1.0) # Évite les NaN dus aux imprécisions flottantes
        angles[:, i1] = np.arccos(dot_prod)

    # --- 2. Construction vectorisée de la Matrice de Rigidité (W) ---
    I_W, J_W, V_W = [], [], []

    for i in range(3):
        i1, i2, i3 = i, (i + 1) % 3, (i + 2) % 3

        # Vecteurs arêtes opposés aux sommets i1 et i2
        e1 = V[F[:, i3]] - V[F[:, i2]]
        e2 = V[F[:, i1]] - V[F[:, i3]]

        norm_e1 = np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
        norm_e2 = np.maximum(np.linalg.norm(e2, axis=1, keepdims=True), 1e-12)
        e1 /= norm_e1
        e2 /= norm_e2

        # Calcul vectoriel des formes bilinéaires: e1^T * D_finsler * e2 et e1^T * D_finsler * e1
        # np.einsum('mij,mj->mi') fait le produit matrice-vecteur pour chaque face
        D_e2 = np.einsum('mij,mj->mi', D_F, e2)
        D_e1 = np.einsum('mij,mj->mi', D_F, e1)

        e1_D_e2 = np.sum(e1 * D_e2, axis=1)
        e1_D_e1 = np.sum(e1 * D_e1, axis=1)

        # Termes trigonométriques
        sin_i3 = np.sin(angles[:, i3])
        cot_i2 = 1.0 / np.tan(angles[:, i2])
        cot_i3 = 1.0 / np.tan(angles[:, i3])

        # Poids de l'arête (factore) et de la diagonale (factord)
        factore = -0.5 * e1_D_e2 / sin_i3
        factord = -0.5 * e1_D_e1 * (cot_i2 + cot_i3)

        # Accumulation dans les listes au format COO
        # W[i1, i2] += factore
        I_W.append(F[:, i1])
        J_W.append(F[:, i2])
        V_W.append(factore)

        # W[i2, i1] += factore
        I_W.append(F[:, i2])
        J_W.append(F[:, i1])
        V_W.append(factore)

        # W[i1, i1] += factord
        I_W.append(F[:, i1])
        J_W.append(F[:, i1])
        V_W.append(factord)

    I_W = np.concatenate(I_W)
    J_W = np.concatenate(J_W)
    V_W = np.concatenate(V_W)

    # scipy.sparse.coo_matrix somme automatiquement les doublons (i, j)
    W = sp.coo_matrix((V_W, (I_W, J_W)), shape=(n_vertices, n_vertices)).tocsr()


    return W.tocsr()


def compute_mass_matrix(V: np.ndarray, F: np.ndarray) -> sp.csr_matrix:
    n_vertices = V.shape[0]

    # 1. Extraction vectorisée en une seule passe
    v1, v2, v3 = V[F].transpose(1, 0, 2)

    # 2. Calcul des aires
    triangle_areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)

    # 3. Distribution aux sommets
    vertex_masses = np.bincount(
        F.flatten(),
        weights=np.repeat(triangle_areas / 3.0, 3),
        minlength=n_vertices
    )

    # 4. Matrice creuse
    return sp.diags(vertex_masses, format='csr')

def build_normalized_block_flbo(L_list):
    """
    Normalise une liste d'opérateurs FLBO (scipy.sparse) pour Chebyshev
    et les assemble en une grande matrice bloc-diagonale au format PyTorch.

    Arguments:
      L_list : Liste de matrices scipy.sparse (ex: [L_0, L_1, ..., L_7])
               où chaque L = S_inv @ W.

    Retourne:
      L_torch : torch.sparse.FloatTensor contenant la matrice globale.
    """
    L_norm_list = []

    for L in L_list:
        # 1. Calcul de la plus grande valeur propre (en module)
        # L_FLBO étant asymétrique, on utilise eigs (et non eigsh)

        vals, _ = eigs(L, k=1, which='LM')
        lambda_max = np.real(vals[0])

        # 2. Normalisation de Chebyshev (Shift and Scale)
        # Formule : L_norm = (2 / lambda_max) * L - I
        I = sp.eye(L.shape[0], format='csr')
        L_norm = (2.0 / lambda_max) * L - I

        L_norm_list.append(L_norm)

    # 3. Assemblage en une seule grande matrice bloc-diagonale
    # Si on a 8 matrices de taille (N, N), on obtient une (8N, 8N)
    L_block = sp.block_diag(L_norm_list, format='coo')

    # 4. Conversion ultra-rapide vers PyTorch Sparse Tensor
    # On évite les boucles en passant directement par les tableaux internes du format COO
    indices = np.vstack((L_block.row, L_block.col))
    indices_torch = torch.from_numpy(indices).long()
    values_torch = torch.from_numpy(L_block.data).float()
    shape_torch = torch.Size(L_block.shape)

    L_torch = torch.sparse_coo_tensor(indices_torch, values_torch, shape_torch)

    # coalesce() fusionne les doublons et optimise l'agencement mémoire pour le GPU
    return L_torch.coalesce()