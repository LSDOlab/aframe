"""Mesh generation helpers."""
import numpy as np

__all__ = ['mesh_from_points_and_edges']


def mesh_from_points_and_edges(points:np.ndarray, edges:np.ndarray, num_nodes:int)->np.ndarray:
    """
    Straight beam meshes along the edges of a point cloud.

    Parameters
    ----------
    points : np.ndarray
        Point coordinates, shape (num_points, 3).
    edges : np.ndarray
        Point index pairs (start, end), shape (num_edges, 2).
    num_nodes : int
        Number of evenly spaced nodes per edge (including both end points).

    Returns
    -------
    np.ndarray
        Node coordinates, shape (num_edges, num_nodes, 3).
    """
    points = np.asarray(points)
    edges = np.asarray(edges)
    return np.linspace(points[edges[:, 0]], points[edges[:, 1]], num=num_nodes, axis=1)
