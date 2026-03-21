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


def build_normalized_block_flbo(W_list, S):
    """
    Traduction exacte du MATLAB des auteurs d'ACSCNN.

    Arguments:
      W_list : Liste des matrices de raideur W pour chaque angle (scipy.sparse)
      S : Matrice de masse/aire diagonale (scipy.sparse)
    """
    L_norm_list = []
    n = S.shape[0]

    # 1. Calcul de l'inverse de l'aire (équivalent de 1./area)
    inv_area = 1.0 / S.diagonal()
    inv_S = sp.diags(inv_area)

    for W in W_list:
        # --- ETAPE 1 : shift_norm_laplacian (MATLAB) ---
        # L = inv_S @ W
        L_raw = inv_S @ W

        # M = full(diag(L)); M1 = sparse(1./sqrt(M))
        diag_L = L_raw.diagonal()
        M1_diag = 1.0 / np.sqrt(np.abs(diag_L) + 1e-12)
        M1 = sp.diags(M1_diag)

        # W_norm = M1 * W * M1
        W_norm = M1 @ W @ M1

        # W_norm = (W_norm + W_norm') / 2.0; % keep symmetric
        W_norm = (W_norm + W_norm.T) / 2.0

        # --- ETAPE 2 : shift_laplacian (MATLAB) ---
        L_temp = inv_S @ W_norm

        # lmax = eigs(W_norm, A, 1) -> équivalent à eigs(inv_S @ W_norm, k=1)
        vals, _ = eigs(L_temp, k=1, which='LM')
        lmax = np.real(vals[0])

        # L = (2 / lmax) * L_temp - I
        I = sp.eye(n, format='csr')
        L_norm = (2.0 / lmax) * L_temp - I

        L_norm_list.append(L_norm)

    # --- ETAPE 3 : Assemblage du bloc diagonal (PyTorch) ---
    L_block = sp.block_diag(L_norm_list, format='coo')

    indices = np.vstack((L_block.row, L_block.col))
    indices_torch = torch.from_numpy(indices).long()
    values_torch = torch.from_numpy(L_block.data).float()
    shape_torch = torch.Size(L_block.shape)

    L_torch = torch.sparse_coo_tensor(indices_torch, values_torch, shape_torch)

    return L_torch.coalesce()