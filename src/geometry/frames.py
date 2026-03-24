import igl
import numpy as np


def compute_local_frames(V, F):

    n = igl.per_face_normals(V, F, np.array([0.0, 0.0, 1.0]))

    pd1, pd2, pv1, pv2, bad_vertices = igl.principal_curvature(V, F)

    num_faces = F.shape[0]

    U = np.zeros((num_faces, 3, 3))

    for f_idx in range(num_faces):
        face = F[f_idx]
        n_f = n[f_idx]

        dir_max = pd1[face[0]]

        u_M_f = dir_max - np.dot(dir_max, n_f) * n_f

        norm = np.linalg.norm(u_M_f)
        if norm > 1e-8:
            u_M_f = u_M_f / norm
        else:
            edge = V[face[1]] - V[face[0]]
            u_M_f = edge - np.dot(edge, n_f) * n_f
            u_M_f = u_M_f / np.linalg.norm(u_M_f)

        u_m_f = np.cross(n_f, u_M_f)

        U[f_idx, :, :] = [u_M_f, u_m_f, n_f]

    return U
