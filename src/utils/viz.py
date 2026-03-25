import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_mesh(V, F, face_func=None, vertex_func=None):
    V = np.asarray(V)
    F = np.asarray(F)

    mesh_kwargs = {
        'x': V[:, 0], 'y': V[:, 1], 'z': V[:, 2],
        'i': F[:, 0], 'j': F[:, 1], 'k': F[:, 2],
        'name': 'Faces'
    }

    if face_func is not None:
        mesh_kwargs.update({
            'intensity': face_func,
            'intensitymode': 'cell',
            'colorscale': 'Viridis',
            'flatshading': True,
            'opacity': 1.0
        })
    elif vertex_func is not None:
        mesh_kwargs.update({
            'intensity': vertex_func,
            'intensitymode': 'vertex',
            'colorscale': 'Viridis',
            'flatshading': False,
            'opacity': 1.0
        })
    else:
        mesh_kwargs.update({
            'color': 'lightblue',
            'opacity': 0.8,
            'flatshading': True
        })

    mesh_trace = go.Mesh3d(**mesh_kwargs)

    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    unique_edges = np.unique(np.sort(edges, axis=1), axis=0)

    x_edges = np.full(3 * len(unique_edges), np.nan)
    y_edges = np.full(3 * len(unique_edges), np.nan)
    z_edges = np.full(3 * len(unique_edges), np.nan)

    x_edges[0::3] = V[unique_edges[:, 0], 0]
    x_edges[1::3] = V[unique_edges[:, 1], 0]
    y_edges[0::3] = V[unique_edges[:, 0], 1]
    y_edges[1::3] = V[unique_edges[:, 1], 1]
    z_edges[0::3] = V[unique_edges[:, 0], 2]
    z_edges[1::3] = V[unique_edges[:, 1], 2]

    edge_trace = go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color='black', width=2),
        name='Arêtes',
        hoverinfo='skip'
    )

    fig = go.Figure(data=[mesh_trace, edge_trace])

    no_axis = dict(
        showbackground=False,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False,
        title=''
    )

    fig.update_layout(
        scene=dict(
            xaxis=no_axis,
            yaxis=no_axis,
            zaxis=no_axis,
            dragmode='orbit'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        dragmode='orbit'
    )

    return fig

def plot_diffusions_comparison(V, F, source, t, solve_heat_diffusion, options_list):
    V = np.asarray(V)
    F = np.asarray(F)
    num_plots = len(options_list)

    titles = [
        f"alpha={opt.alpha}<br>angle={opt.angle:.2f}, tau={opt.tau}"
        for opt in options_list
    ]

    fig = make_subplots(
        rows=1,
        cols=num_plots,
        specs=[[{"type": "scene"}] * num_plots],
        subplot_titles=titles,
        horizontal_spacing=0.01,
    )

    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    unique_edges = np.unique(np.sort(edges, axis=1), axis=0)

    x_edges, y_edges, z_edges = (
        np.empty(3 * len(unique_edges)),
        np.empty(3 * len(unique_edges)),
        np.empty(3 * len(unique_edges)),
    )
    x_edges[0::3], x_edges[1::3], x_edges[2::3] = (
        V[unique_edges[:, 0], 0],
        V[unique_edges[:, 1], 0],
        np.nan,
    )
    y_edges[0::3], y_edges[1::3], y_edges[2::3] = (
        V[unique_edges[:, 0], 1],
        V[unique_edges[:, 1], 1],
        np.nan,
    )
    z_edges[0::3], z_edges[1::3], z_edges[2::3] = (
        V[unique_edges[:, 0], 2],
        V[unique_edges[:, 1], 2],
        np.nan,
    )

    for i, opt in enumerate(options_list):
        ft = solve_heat_diffusion(V, F, source, t, options=opt)

        mesh_trace = go.Mesh3d(
            x=V[:, 0],
            y=V[:, 1],
            z=V[:, 2],
            i=F[:, 0],
            j=F[:, 1],
            k=F[:, 2],
            intensity=ft,
            intensitymode="vertex",
            colorscale="Viridis",
            opacity=1.0,
            flatshading=True,
            name=f"Finsler {i + 1}",
            showscale=(i == num_plots - 1),
        )

        edge_trace = go.Scatter3d(
            x=x_edges,
            y=y_edges,
            z=z_edges,
            mode="lines",
            line=dict(color="black", width=2),
            hoverinfo="skip",
            showlegend=False,
        )

        col_idx = i + 1
        fig.add_trace(mesh_trace, row=1, col=col_idx)
        fig.add_trace(edge_trace, row=1, col=col_idx)

    scene_layout = dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
    )

    fig.update_layout(
        **{f"scene{j + 1 if j > 0 else ''}": scene_layout for j in range(num_plots)},
        margin=dict(l=0, r=0, b=0, t=50),
        dragmode="orbit",
    )

    return fig
