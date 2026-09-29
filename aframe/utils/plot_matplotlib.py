"""
Matplotlib helpers for 3D beam plots.

``plot_box`` and ``plot_circle`` return per-element cross-section outlines as
vertex lists for ``mpl_toolkits.mplot3d.art3d.Poly3DCollection``; ``mesh`` is a
numpy array of node coordinates, shape (num_nodes, 3). Elements parallel to
the global z axis are not supported (their rotation axis is undefined).
"""
import numpy as np

__all__ = ['plot_mesh', 'rotation_matrix_from_axis_angle', 'plot_box', 'plot_circle']


def plot_mesh(ax, mesh, size, line_color, marker_color, edge_color):
    """draw the mesh nodes and the polyline through them on a 3D axis"""
    ax.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2], color=marker_color, edgecolor=edge_color, s=size, alpha=1, linewidth=0.5)
    ax.plot(mesh[:, 0], mesh[:, 1], mesh[:, 2], color=line_color, linewidth=1)


def rotation_matrix_from_axis_angle(axis, angle):
    """rotation matrix for a rotation of angle (rad) about the unit vector axis (Rodrigues)"""
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis
    return np.array([[t*x*x + c, t*x*y - z*s, t*x*z + y*s],
                     [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
                     [t*x*z - y*s, t*y*z + x*s, t*z*z + c]])


def _element_frames(mesh):
    """
    for each element: the rotation taking the global z axis onto the element
    axis, and the element midpoint
    """
    z_axis = np.array([0, 0, 1])
    for i in range(len(mesh) - 1):
        axis_vector = mesh[i + 1, :] - mesh[i, :]
        target_normal = axis_vector / np.linalg.norm(axis_vector)

        rotation_axis = np.cross(z_axis, target_normal)
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_angle = np.arccos(np.dot(z_axis, target_normal))

        yield rotation_matrix_from_axis_angle(rotation_axis, rotation_angle), mesh[i, :] + axis_vector / 2


def plot_box(mesh, width, height):
    """rectangle outlines (width x height) at each element midpoint, normal to the element"""
    vertices = []
    for i, (rotation_matrix, offset) in enumerate(_element_frames(mesh)):
        corners = np.array([[-width[i] / 2, height[i] / 2, 0],
                            [width[i] / 2, height[i] / 2, 0],
                            [width[i] / 2, -height[i] / 2, 0],
                            [-width[i] / 2, -height[i] / 2, 0]])
        points = corners @ rotation_matrix.T + offset
        vertices.append([list(map(tuple, points))])

    return vertices


def plot_circle(mesh, radius, num_circle):
    """circle outlines (num_circle points) at each element midpoint, normal to the element"""
    theta = np.linspace(0, 2 * np.pi, num_circle)

    vertices = []
    for i, (rotation_matrix, offset) in enumerate(_element_frames(mesh)):
        # circle in the x-y plane, rotated onto the element
        circle = np.column_stack([radius[i] * np.cos(theta), radius[i] * np.sin(theta), np.zeros(num_circle)])
        points = circle @ rotation_matrix.T + offset
        vertices.append([list(map(tuple, points))])

    return vertices
