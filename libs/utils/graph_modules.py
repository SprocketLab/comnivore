import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
from libs.utils.metrics import shd, sid

def modify_single_edge(G, edge, value=None):
    G_modified = np.copy(G)
    if not value:
        G_modified[edge[0], edge[1]] = np.abs(G[edge[0], edge[1]] - 1)
    else:
        G_modified[edge[0], edge[1]] = value
    return G_modified

def get_ordered_edge_sets(G):
    vertex_indexes, _ = G.shape
    edges = []
    for row in range(vertex_indexes):
        for col in range(vertex_indexes):
            if row != col:
                edges.append((row, col))
    return edges

def store_graph(target_dir, filename, G):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    np.save(os.path.join(target_dir, filename), G)

def show_graph(adjacency_matrix, title="graph", save = False, figname= "", size=(2,2), labels = None, show=True):
    seed = 24
    np.random.seed(seed)
    G = nx.DiGraph(adjacency_matrix)
    pos = nx.spring_layout(G, seed=seed)
    plt.figure(figsize=size)
    with_labels = False if labels else True
    nx.draw(G, pos=pos, with_labels=with_labels, arrowsize=7, node_color = 'skyblue',)
    if with_labels == False:
        nx.draw_networkx_labels(G, pos, labels)
    plt.title(title)
    if save:
        plt.savefig(figname)
    if show:
        plt.show()

def compute_dist(G1, G2, metric=shd):
    return metric(G1, G2)