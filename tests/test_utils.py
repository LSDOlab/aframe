"""Meshing, load transfer and plotting utilities."""
import numpy as np
import pytest
import aframe as af
from aframe.utils import plot_matplotlib
from aframe.utils.aeroelastic_utils import NodalMap


def test_mesh_from_points_and_edges():
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 0], [1, -1, 1]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    mesh = af.mesh_from_points_and_edges(points, edges, 5)

    assert mesh.shape == (4, 5, 3)
    for (start, end), edge_mesh in zip(edges, mesh):
        np.testing.assert_array_equal(edge_mesh[0], points[start])
        np.testing.assert_array_equal(edge_mesh[-1], points[end])
        np.testing.assert_allclose(np.diff(edge_mesh, axis=0), np.tile((points[end] - points[start]) / 4, (4, 1)))


@pytest.mark.parametrize('method', ['rbf', 'idw'])
def test_nodal_map(method):
    mesh_in = np.zeros((10, 3))
    mesh_in[:, 0] = np.linspace(0, 10, 10)
    mesh_out = np.zeros((12, 3))
    mesh_out[:, 0] = np.linspace(0, 10, 12)
    mesh_out[:, 1] = 0.2

    nodal_map = NodalMap(mesh_in, mesh_out, method=method)
    # normalized weights map a constant field exactly
    shift = np.tile([0.5, -1.0, 2.0], (10, 1))
    np.testing.assert_allclose(nodal_map.evaluate(shift), mesh_out + shift[0])


def test_nodal_map_invalid_method():
    with pytest.raises(ValueError):
        NodalMap(np.zeros((2, 3)), np.ones((2, 3)), method='nope').evaluate(np.zeros((2, 3)))


def test_matplotlib_outlines():
    mesh = np.stack([np.linspace(0, 5, 6), np.linspace(-1, 2, 6) ** 2, np.linspace(0, 1, 6)], axis=1)
    radius = np.full(5, 0.3)
    circles = plot_matplotlib.plot_circle(mesh, radius, 12)
    boxes = plot_matplotlib.plot_box(mesh, np.full(5, 0.5), np.full(5, 0.2))
    assert len(circles) == len(boxes) == 5

    for i, circle in enumerate(circles):
        points = np.array(circle[0])
        axis = mesh[i + 1] - mesh[i]
        midpoint = (mesh[i] + mesh[i + 1]) / 2
        # circle points are at the radius from the midpoint, normal to the element
        np.testing.assert_allclose(np.linalg.norm(points - midpoint, axis=1), radius[i])
        np.testing.assert_allclose((points - midpoint) @ axis, 0, atol=1e-12)
        assert np.array(boxes[i][0]).shape == (4, 3)


def test_pyvista_helpers():
    pv = pytest.importorskip('pyvista')
    mesh = np.stack([np.linspace(0, 5, 6), np.linspace(0, 1, 6), np.zeros(6)], axis=1)
    plotter = pv.Plotter(off_screen=True)
    af.plot_mesh(plotter, mesh)
    af.plot_mesh(plotter, mesh, cell_data=np.arange(5.0))
    af.plot_points(plotter, mesh)
    af.plot_cyl(plotter, mesh, np.full(5, 0.1), cell_data=np.arange(5.0))
    af.plot_box(plotter, mesh, np.full(5, 0.2), np.full(5, 0.1))
    assert len(plotter.actors) == 13
    plotter.close()
