"""
PyVista plotting helpers for beam meshes.

All functions draw into an existing ``pyvista.Plotter``; ``mesh`` is a numpy
array of node coordinates, shape (num_nodes, 3), and per-element data has
shape (num_nodes - 1,). Requires the optional ``plot`` dependencies.
"""
import numpy as np
import pyvista as pv
import matplotlib as mpl

__all__ = ['plot_box', 'plot_cyl', 'plot_mesh', 'plot_points']


def _colorize(cell_data, cmap, color, n):
    """per-element colors: cell_data normalized onto cmap, or a single color"""
    if cell_data is not None:
        n_cell_data = (cell_data - cell_data.min()) / (cell_data.max() - cell_data.min())
        return mpl.colormaps[cmap](n_cell_data)

    return [color] * n


def _polyline(mesh:np.ndarray)->pv.PolyData:
    """PolyData through the mesh nodes with one 2-point cell [2, i, i + 1] per element"""
    n = mesh.shape[0]
    cells = np.column_stack([np.full(n - 1, 2), np.arange(n - 1), np.arange(1, n)])
    return pv.PolyData(mesh, cells)


def plot_box(plotter, mesh, height, width, cell_data=None, cmap='viridis', color='lightblue'):
    """
    draw a rectangular box per element

    height, width : per-element box dimensions
    cell_data : optional per-element values used for coloring
    """
    n = mesh.shape[0]
    colors = _colorize(cell_data, cmap, color, n)

    up = np.array([0, 0, 1])
    for i in range(n - 1):
        start = mesh[i, :]
        end = mesh[i + 1, :]

        direction = end - start
        length = np.linalg.norm(direction)
        direction /= length

        # rotate the z-aligned cube onto the element axis
        if np.allclose(direction, up):
            axis = up
            angle = 0
        else:
            axis = np.cross(up, direction)
            angle = np.rad2deg(np.arccos(np.dot(up, direction)))

        midpoint = (start + end) / 2
        box = pv.Cube(center=midpoint, x_length=width[i], y_length=height[i], z_length=length)
        box.rotate_vector(vector=axis, angle=angle, point=midpoint, inplace=True)

        plotter.add_mesh(box, color=colors[i], show_edges=True)


def plot_cyl(plotter, mesh, radius, cell_data=None, cmap='viridis', color='lightblue'):
    """
    draw a cylinder per element

    radius : per-element cylinder radius
    cell_data : optional per-element values used for coloring
    """
    n = mesh.shape[0]
    colors = _colorize(cell_data, cmap, color, n)

    for i in range(n - 1):
        start = mesh[i, :]
        end = mesh[i + 1, :]

        cyl = pv.Cylinder(center=(start + end) / 2,
                          direction=end - start,
                          radius=radius[i],
                          height=np.linalg.norm(end - start))

        plotter.add_mesh(cyl, color=colors[i])


def plot_mesh(plotter,
              mesh,
              cell_data=None,
              plot_mesh=True,
              cmap='viridis',
              line_width=20,
              render_lines_as_tubes=True,
              color='red'):
    """
    draw the beam as a polyline, optionally colored by per-element cell_data
    (plot_mesh is unused and kept for backward compatibility)
    """
    polyline = _polyline(mesh)

    if cell_data is not None:
        # normalize the cell data
        polyline.cell_data['cell_data'] = (cell_data - cell_data.min()) / (cell_data.max() - cell_data.min())
        plotter.add_mesh(polyline,
                         scalars='cell_data',
                         render_lines_as_tubes=render_lines_as_tubes,
                         style='wireframe',
                         line_width=line_width,
                         cmap=cmap,
                         show_scalar_bar=False)
    else:
        plotter.add_mesh(polyline,
                         render_lines_as_tubes=render_lines_as_tubes,
                         style='wireframe',
                         line_width=line_width,
                         color=color,
                         show_scalar_bar=False)


def plot_points(plotter, mesh, color='red', point_size=50, render_points_as_spheres=True):
    """draw the mesh nodes as points"""
    plotter.add_points(np.asarray(mesh, dtype=float),
                       color=color,
                       point_size=point_size,
                       render_points_as_spheres=render_points_as_spheres)
