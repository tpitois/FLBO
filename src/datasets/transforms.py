import igl
import torch
import numpy as np

from src.geometry.frames import compute_local_frames
from src.geometry.finsler import compute_D_finsler, Options
from src.geometry.FLBO import (
    compute_mass_matrix, compute_stiffness_matrix, build_normalized_block_flbo
)
from src.geometry.descriptors import compute_eigen_decomposition, compute_wks


class FLBOTransform(object):
    def __init__(self, n_angles=8, alpha=10.0, tau=0.5, num_wks=100, k_eigen=100, decimate=None):
        self.n_angles = n_angles
        self.alpha = alpha
        self.tau = tau
        self.k_eigen = k_eigen
        self.num_wks = num_wks
        self.decimate = decimate

    def __call__(self, data):
        V = data.pos.numpy().astype(np.float64)
        V = V / np.max(np.linalg.norm(V - np.mean(V, axis=0), axis=1))

        F = data.face.t().numpy().astype(np.int64)

        if self.decimate is not None:
            V, F, _, _ = igl.decimate(V, F, self.decimate)

            data.pos = torch.from_numpy(V).float()
            data.face = torch.from_numpy(F).long().t().contiguous()

        U = compute_local_frames(V, F)

        S = compute_mass_matrix(V, F)

        opt_iso = Options(alpha=0.0, angle=0.0, tau=0.0)
        D_iso = compute_D_finsler(U, opt_iso)
        W_iso = compute_stiffness_matrix(V, F, D_iso)

        evals, evecs = compute_eigen_decomposition(W_iso, S, k=self.k_eigen)
        wks_descriptors = compute_wks(evals, evecs, num_energies=self.num_wks)

        data.x = torch.from_numpy(wks_descriptors).float()

        angles = np.linspace(0, np.pi, self.n_angles, endpoint=False)
        W_list = []

        for angle in angles:
            D = compute_D_finsler(
                U,
                Options(alpha=self.alpha, angle=angle, tau=self.tau)
            )

            W_list.append(compute_stiffness_matrix(V, F, D))

        data.L =  build_normalized_block_flbo(W_list, S)

        data.y = torch.arange(V.shape[0], dtype=torch.long)

        return data
