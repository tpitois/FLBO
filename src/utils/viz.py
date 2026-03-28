import numpy as np
import pyvista as pv


def plot_mesh(V, F, face_func=None, vertex_func=None):
    V = np.asarray(V)
    F = np.asarray(F)

    faces_pv = np.hstack((np.full((F.shape[0], 1), 3), F)).ravel()
    mesh = pv.PolyData(V, faces_pv)

    plotter = pv.Plotter()

    mesh_kwargs = {
        "show_edges": True,
        "edge_color": "black",
        "line_width": 2,
        "cmap": "viridis",
    }

    if face_func is not None:
        mesh.cell_data["Intensity"] = face_func
        mesh_kwargs.update(
            {
                "scalars": "Intensity",
                "opacity": 1.0,
            }
        )

    elif vertex_func is not None:
        mesh.point_data["Intensity"] = vertex_func
        mesh_kwargs.update(
            {"scalars": "Intensity", "opacity": 1.0, "interpolate_before_map": True}
        )

    else:
        mesh_kwargs.update({"color": "lightblue", "opacity": 0.8})

    plotter.add_mesh(mesh, **mesh_kwargs)

    plotter.set_background("white")
    plotter.hide_axes()

    return plotter


def plot_diffusions_comparison(V, F, source, t, solve_heat_diffusion, options_list):
    V = np.asarray(V)
    F = np.asarray(F)
    num_plots = len(options_list)

    faces_pv = np.hstack((np.full((F.shape[0], 1), 3), F)).ravel()
    base_mesh = pv.PolyData(V, faces_pv)

    plotter = pv.Plotter(shape=(1, num_plots))

    all_ft = [
        solve_heat_diffusion(V, F, source, t, options=opt) for opt in options_list
    ]
    vmin, vmax = np.min(all_ft), np.max(all_ft)

    for i, opt in enumerate(options_list):
        plotter.subplot(0, i)

        title = f"alpha={opt.alpha}\nangle={opt.angle:.2f}, tau={opt.tau}"
        plotter.add_text(title, font_size=10, position="upper_edge", color="black")

        mesh = base_mesh.copy()
        mesh.point_data["Heat"] = all_ft[i]

        plotter.add_mesh(
            mesh,
            scalars="Heat",
            cmap="viridis",
            show_edges=True,
            edge_color="black",
            line_width=2,
            clim=[vmin, vmax],
            show_scalar_bar=(i == num_plots - 1),
        )

        plotter.set_background("white")
        plotter.hide_axes()

    plotter.link_views()

    return plotter
