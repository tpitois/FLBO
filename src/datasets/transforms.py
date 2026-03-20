import torch
import numpy as np
import scipy.sparse as sp

from src.geometry.frames import compute_local_frames
from src.geometry.finsler import compute_D_finsler, Options
from src.geometry.FLBO import (
    compute_mass_matrix, compute_stiffness_matrix, build_normalized_block_flbo
)
from src.geometry.descriptors import compute_eigen_decomposition, compute_wks


class FLBOTransform(object):
    def __init__(self, n_angles=8, alpha=10.0, tau=0.5, num_wks=100, k_eigen=100):
        self.n_angles = n_angles
        self.alpha = alpha
        self.tau = tau
        self.k_eigen = k_eigen
        self.num_wks = num_wks

    def __call__(self, data):
        # Extraction des sommets et des faces depuis PyG
        V = data.pos.numpy().astype(np.float64)
        V = V / np.max(np.linalg.norm(V - np.mean(V, axis=0), axis=1))

        F = data.face.t().numpy().astype(np.int64)  # PyG stocke les faces en (3, num_faces)

        # 1. Calcul du repère local (à l'aide de ta fonction)
        U = compute_local_frames(V, F)

        S = compute_mass_matrix(V, F)
        S_inv = sp.diags(1.0 / S.diagonal())

        opt_iso = Options(alpha=0.0, angle=0.0, tau=0.0)
        D_iso = compute_D_finsler(U, opt_iso)
        W_iso = compute_stiffness_matrix(V, F, D_iso)

        # Extraction du spectre et calcul du WKS
        evals, evecs = compute_eigen_decomposition(W_iso, S, k=self.k_eigen)
        wks_descriptors = compute_wks(evals, evecs, num_energies=self.num_wks)

        # On remplace data.x (qui était pos) par nos descripteurs WKS
        data.x = torch.from_numpy(wks_descriptors).float()

        # 2. Boucle sur les angles pour construire les Laplaciens
        angles = np.linspace(0, np.pi, self.n_angles, endpoint=False)
        L_list = []


        for angle in angles:
            D = compute_D_finsler(
                U,
                Options(alpha=self.alpha, angle=angle, tau=self.tau)
            )

            W = compute_stiffness_matrix(V, F, D)

            L = S_inv @ W

            L_list.append(L)

        # 3. Assemblage du super-tenseur PyTorch (fonction précédente)
        data.L = build_normalized_block_flbo(L_list)

        # 5. Labels de correspondance (y)
        # Dans TOSCA, le sommet i du Chat 1 correspond au sommet i du Chat 2
        data.y = torch.arange(V.shape[0], dtype=torch.long)

        return data