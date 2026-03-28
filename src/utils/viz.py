import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt


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


def plot_pck_curve(errors_list, labels=None, ax=None, max_threshold=0.25, num_bins=100):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    thresholds = np.linspace(0, max_threshold, num_bins)

    if not isinstance(errors_list, list):
        errors_list = [errors_list]

    if labels is None:
        labels = [f"{i + 1}" for i in range(len(errors_list))]
    elif not isinstance(labels, list):
        labels = [labels]

    for i, errors in enumerate(errors_list):
        pck = []
        errors_arr = np.array(errors)

        for t in thresholds:
            correct_percentage = np.mean(errors_arr <= t) * 100
            pck.append(correct_percentage)

        ax.plot(thresholds, pck, linewidth=2, label=labels[i])

    ax.set_title("Correspondences Evaluation (PCK Curve)")
    ax.set_xlabel("Geodesic Error Threshold (D)")
    ax.set_ylabel("Percentage of Matches (%)")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_xlim([0, max_threshold])
    ax.set_ylim([0, 105])

    ax.legend()

    return ax


def plot_correspondence(V, F, p2p_pred):
    V = np.asarray(V)
    F = np.asarray(F)

    faces_pv = np.hstack((np.full((F.shape[0], 1), 3), F)).ravel()
    base_mesh = pv.PolyData(V, faces_pv)

    colors_gt = (V - V.min(axis=0)) / (V.max(axis=0) - V.min(axis=0))

    plotter = pv.Plotter(shape=(1, 2))

    plotter.subplot(0, 0)
    mesh_gt = base_mesh.copy()
    mesh_gt.point_data["RGB"] = colors_gt

    plotter.add_mesh(mesh_gt, scalars="RGB", rgb=True, show_edges=False)
    plotter.add_text("Ground Truth", font_size=12, color="black")
    plotter.set_background("white")

    plotter.subplot(0, 1)
    mesh_pred = base_mesh.copy()
    mesh_pred.point_data["RGB"] = colors_gt[p2p_pred]

    plotter.add_mesh(mesh_pred, scalars="RGB", rgb=True, show_edges=False)
    plotter.add_text("Prediction", font_size=12, color="black")
    plotter.set_background("white")

    plotter.link_views()
    plotter.view_isometric()

    return plotter


def plot_learning_curves(history):
    train_loss = history.get("train_loss", [])
    train_acc = history.get("train_acc", [])

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, "r-", marker="+", markersize=4, label="Train Loss")
    plt.title("Loss Evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, "b-", marker="+", markersize=4, label="Train Accuracy")
    plt.title("Accuracy Evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plt.tight_layout()

    plt.show()
