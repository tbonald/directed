#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu July 10 2018
@author: Nathan de Lara <ndelara@enst.fr>
"""

import numpy as np
from scipy import errstate, sqrt, isinf, sparse


def cocitation_modularity(partition, adjacency_matrix, resolution=1.0):
    """
    Compute the modularity of a node partition of a cocitation graph.
    Parameters
    ----------
    partition: dict
       The partition of the nodes.
       The keys of the dictionary correspond to the nodes and the values to the communities.
    adjacency_matrix: scipy.csr_matrix or np.ndarray
        The adjacency matrix of the graph (sparse or dense).
    resolution: double, optional
        The resolution parameter in the modularity function (default=1.).

    Returns
    -------
    modularity : float
       The modularity.
    """

    if type(adjacency_matrix) == sparse.csr_matrix:
        adj_matrix = adjacency_matrix
    elif type(adjacency_matrix) == np.ndarray:
        adj_matrix = sparse.csr_matrix(adjacency_matrix)
    else:
        raise TypeError(
            "The argument should be a NumPy array or a SciPy Compressed Sparse Row matrix.")

    n_nodes = adj_matrix.shape[0]
    out_degree = np.array(adj_matrix.sum(axis=1).flatten())
    in_degree = adj_matrix.sum(axis=0).flatten()
    total_weight = out_degree.sum()

    with errstate(divide='ignore'):
        in_degree_sqrt = 1.0 / sqrt(in_degree)
    in_degree_sqrt[isinf(in_degree_sqrt)] = 0
    in_degree_sqrt = sparse.spdiags(in_degree_sqrt, [0], adj_matrix.shape[1], adj_matrix.shape[1], format='csr')
    normalized_adjacency = (adj_matrix.dot(in_degree_sqrt)).T

    communities = lab2com(partition)
    mod = 0.

    for community in communities:
        indicator_vector = np.zeros(n_nodes)
        indicator_vector[list(community)] = 1
        mod += np.linalg.norm(normalized_adjacency.dot(indicator_vector)) ** 2
        mod -= (resolution / total_weight) * (np.dot(out_degree, indicator_vector)) ** 2

    return float(mod / total_weight)


def lab2com(labels):
    communities = {}

    if type(labels) == dict:
        for k, v in labels.items():
            try:
                communities[v].add(k)
            except KeyError:
                communities[v] = {k}
    elif type(labels) == np.ndarray:
        for k, v in enumerate(labels):
            try:
                communities[v].add(k)
            except KeyError:
                communities[v] = {k}
    else:
        raise TypeError(
            "The argument should be a dictionary or a NumPy array.")

    return sorted([v for k, v in communities.items()], reverse=True, key=len)
