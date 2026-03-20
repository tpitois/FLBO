import numpy as np
import plotly.graph_objects as go

def plot_mesh(V, F, face_func=None, vertex_func=None):
    V = np.asarray(V)
    F = np.asarray(F)

    # 1. Configuration des paramètres de base pour les faces
    mesh_kwargs = {
        'x': V[:, 0], 'y': V[:, 1], 'z': V[:, 2],
        'i': F[:, 0], 'j': F[:, 1], 'k': F[:, 2],
        'opacity': 1.0, # Mieux vaut 1.0 quand on utilise une carte de couleurs
        'flatshading': True,
        'name': 'Faces'
    }

    # Gestion des couleurs selon face_func ou vertex_func
    if face_func is not None:
        mesh_kwargs['intensity'] = face_func
        mesh_kwargs['intensitymode'] = 'cell'
        mesh_kwargs['colorscale'] = 'Viridis' # Vous pouvez changer la palette ici
    elif vertex_func is not None:
        mesh_kwargs['intensity'] = vertex_func
        mesh_kwargs['intensitymode'] = 'vertex'
        mesh_kwargs['colorscale'] = 'Viridis'
    else:
        # Couleur par défaut si aucune fonction n'est passée
        mesh_kwargs['color'] = 'lightblue'
        mesh_kwargs['opacity'] = 0.8

    mesh_trace = go.Mesh3d(**mesh_kwargs)

    # 2. Extraction des arêtes uniques
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    unique_edges = np.unique(np.sort(edges, axis=1), axis=0)

    # 3. Préparation des coordonnées des arêtes
    x_edges = np.empty(3 * len(unique_edges))
    y_edges = np.empty(3 * len(unique_edges))
    z_edges = np.empty(3 * len(unique_edges))

    x_edges[0::3] = V[unique_edges[:, 0], 0]
    x_edges[1::3] = V[unique_edges[:, 1], 0]
    x_edges[2::3] = np.nan

    y_edges[0::3] = V[unique_edges[:, 0], 1]
    y_edges[1::3] = V[unique_edges[:, 1], 1]
    y_edges[2::3] = np.nan

    z_edges[0::3] = V[unique_edges[:, 0], 2]
    z_edges[1::3] = V[unique_edges[:, 1], 2]
    z_edges[2::3] = np.nan

    # 4. Création de la trace pour les arêtes
    edge_trace = go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color='black', width=2),
        name='Arêtes',
        hoverinfo='skip'
    )

    # 5. Assemblage et affichage
    fig = go.Figure(data=[mesh_trace, edge_trace])
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    return fig