import networkx as nx
from cdt.metrics import retrieve_adjacency_matrix, SHD, SID

import numpy as np


def edge_errors(pred, target):
    """
    Counts all types of edge errors (false negatives, false positives, reversed edges)

    Parameters:
    -----------
    pred: nx.DiGraph or ndarray
        The predicted adjacency matrix
    target: nx.DiGraph or ndarray
        The true adjacency matrix

    Returns:
    --------
    fn, fp, rev

    """
    true_labels = retrieve_adjacency_matrix(target)
    predictions = retrieve_adjacency_matrix(pred, target.nodes() if isinstance(target, nx.DiGraph) else None)

    diff = true_labels - predictions

    rev = (((diff + diff.transpose()) == 0) & (diff != 0)).sum() / 2
    # Each reversed edge necessarily leads to one fp and one fn so we need to subtract those
    fn = (diff == 1).sum() - rev
    fp = (diff == -1).sum() - rev

    return fn, fp, rev


def edge_accurate(pred, target):
    """
    Counts the number of edge in ground truth DAG, true positives and the true
    negatives

    Parameters:
    -----------
    pred: nx.DiGraph or ndarray
        The predicted adjacency matrix
    target: nx.DiGraph or ndarray
        The true adjacency matrix

    Returns:
    --------
    total_edges, tp, tn

    """
    true_labels = retrieve_adjacency_matrix(target)
    predictions = retrieve_adjacency_matrix(pred, target.nodes() if isinstance(target, nx.DiGraph) else None)

    total_edges = (true_labels).sum()

    tp = ((predictions == 1) & (predictions == true_labels)).sum()
    tn = ((predictions == 0) & (predictions == true_labels)).sum()

    return total_edges, tp, tn

# @jit(target = "cuda")
def sid(pred, target):
    """
    Calculates Structural Intervention Distance (SID): https://arxiv.org/pdf/1306.1043.pdf
    :param pred:
    :param target:
    :return:
    """
    return SID(nx.DiGraph(target), nx.DiGraph(pred))

def shd(pred, target):
    """
    Calculates the structural hamming distance

    Parameters:
    -----------
    pred: nx.DiGraph or ndarray
        The predicted adjacency matrix
    target: nx.DiGraph or ndarray
        The true adjacency matrix

    Returns:
    --------
    shd

    """
    return(SHD(nx.DiGraph(target), nx.DiGraph(pred)))

def get_max_shd(g):
    max_dist_graph = np.zeros(g.shape)
    for n in range(g.shape[1]):
        incoming_edges = np.copy(g[n, :])
        outward_edges = np.copy(g[:, n])
        for i, edge in enumerate(incoming_edges):
            if edge == 1:
                max_dist_graph[n, i] = 0
                max_dist_graph[i, n] = 1
            else:
                max_dist_graph[n, i] = 1
                max_dist_graph[i, n] = 0
        for i, edge in enumerate(outward_edges):
            if edge == 1:
                max_dist_graph[i, n] = 0
                max_dist_graph[n, i] = 1
            else:
                max_dist_graph[i, n] = 1
                max_dist_graph[n, i] = 0
    return shd(g, max_dist_graph), max_dist_graph

def get_max_dist(g1, g2):
    max_shd_g1, _ = get_max_shd(g1)
    max_shd_g2, _ = get_max_shd(g2)
    return np.amax([max_shd_g1, max_shd_g2])