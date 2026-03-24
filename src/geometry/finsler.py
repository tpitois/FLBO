from dataclasses import dataclass

import numpy as np


@dataclass
class Options:
    alpha: float
    angle: float
    tau: float


def compute_D_finsler(U: np.ndarray, options: Options) -> np.ndarray:
    c = np.cos(options.angle)
    s = np.sin(options.angle)
    R = np.array([[s, c, 0], [c, -s, 0], [0, 0, 1]])

    # (N_faces,3,3)
    U_rot = R @ U

    D_mat = np.diag([1.0 / (1.0 + options.alpha), 1.0, 1.0])

    # (N_faces,3,3)
    H = U_rot.transpose(0, 2, 1) @ D_mat @ U_rot

    eta = 1.0 - (options.tau**2)

    # (N_faces, 3)
    U2 = U_rot[:, 1, :]
    w = options.tau * U2

    # H @ w -> (N_faces,3)
    Hw = np.einsum("mij,mj->mi", H, w)
    wstar = -Hw / eta

    # (N_faces,3,3)
    Mstar = (eta * H + np.einsum("mi,mj->mij", Hw, Hw)) / (eta**2)

    # (N_faces,3,3)
    D_finsler = Mstar - np.einsum("mi,mj->mij", wstar, wstar)

    D_finsler[np.abs(D_finsler) < 1e-10] = 0.0
    D_finsler[np.abs(D_finsler - 1.0) < 1e-10] = 1.0

    return D_finsler
