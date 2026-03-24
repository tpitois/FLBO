import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import eigs


def compute_stiffness_matrix(V, F, D_F):
    n_vertices = V.shape[0]

    angles = np.zeros_like(F, dtype=float)
    for i in range(3):
        i1, i2, i3 = i, (i + 1) % 3, (i + 2) % 3

        # (N_VERTICES, 3)
        pp = V[F[:, i2]] - V[F[:, i1]]
        qq = V[F[:, i3]] - V[F[:, i1]]

        norm_pp = np.maximum(np.linalg.norm(pp, axis=1, keepdims=True), 1e-12)
        norm_qq = np.maximum(np.linalg.norm(qq, axis=1, keepdims=True), 1e-12)
        pp /= norm_pp
        qq /= norm_qq

        # (N_VERTICES)
        dot_prod = np.sum(pp * qq, axis=1)
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        angles[:, i1] = np.arccos(dot_prod)

    I_W, J_W, V_W = [], [], []
    for i in range(3):
        i1, i2, i3 = i, (i + 1) % 3, (i + 2) % 3

        # (N_VERTICES, 3)
        e1 = V[F[:, i3]] - V[F[:, i2]]
        e2 = V[F[:, i1]] - V[F[:, i3]]

        norm_e1 = np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
        norm_e2 = np.maximum(np.linalg.norm(e2, axis=1, keepdims=True), 1e-12)
        e1 /= norm_e1
        e2 /= norm_e2

        # (N_VERTICES)
        e1_D_e2 = np.einsum("mi,mij,mj->m", e1, D_F, e2)
        e1_D_e1 = np.einsum("mi,mij,mj->m", e1, D_F, e1)

        # (N_VERTICES)
        sin_i3 = np.sin(angles[:, i3])
        cot_i2 = 1.0 / np.tan(angles[:, i2])
        cot_i3 = 1.0 / np.tan(angles[:, i3])

        # (N_VERTICES)
        factore = -0.5 * e1_D_e2 / sin_i3
        factord = -0.5 * e1_D_e1 * (cot_i2 + cot_i3)

        I_W.append(F[:, i1])
        J_W.append(F[:, i2])
        V_W.append(factore)

        I_W.append(F[:, i2])
        J_W.append(F[:, i1])
        V_W.append(factore)

        I_W.append(F[:, i1])
        J_W.append(F[:, i1])
        V_W.append(factord)

    I_W = np.concatenate(I_W)
    J_W = np.concatenate(J_W)
    V_W = np.concatenate(V_W)

    W = sp.coo_matrix((V_W, (I_W, J_W)), shape=(n_vertices, n_vertices)).tocsr()

    return W


def compute_mass_matrix(V: np.ndarray, F: np.ndarray) -> sp.csr_matrix:
    n_vertices = V.shape[0]

    v1, v2, v3 = V[F].transpose(1, 0, 2)

    triangle_areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)

    vertex_masses = np.bincount(
        F.flatten(), weights=np.repeat(triangle_areas / 3.0, 3), minlength=n_vertices
    )

    return sp.diags(vertex_masses, format="csr")


def build_normalized_block_flbo(W_list, S):
    L_norm_list = []
    n = S.shape[0]

    inv_area = 1.0 / S.diagonal()
    inv_S = sp.diags(inv_area)

    for W in W_list:
        L_raw = inv_S @ W

        diag_L = L_raw.diagonal()
        M1_diag = 1.0 / np.sqrt(np.abs(diag_L) + 1e-12)
        M1 = sp.diags(M1_diag)

        W_norm = M1 @ W @ M1

        W_norm = (W_norm + W_norm.T) / 2.0

        L_temp = inv_S @ W_norm

        vals, _ = eigs(L_temp, k=1, which="LM")
        lmax = np.real(vals[0])

        I = sp.eye(n, format="csr")
        L_norm = (2.0 / lmax) * L_temp - I

        L_norm_list.append(L_norm)

    L_block = sp.block_diag(L_norm_list, format="coo")

    indices = np.vstack((L_block.row, L_block.col))
    indices_torch = torch.from_numpy(indices).long()
    values_torch = torch.from_numpy(L_block.data).float()
    shape_torch = torch.Size(L_block.shape)

    L_torch = torch.sparse_coo_tensor(indices_torch, values_torch, shape_torch)

    return L_torch.coalesce()
