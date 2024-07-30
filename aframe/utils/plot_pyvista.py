import numpy as np
import pyvista as pv
import matplotlib as mpl



def _colorize(cell_data, cmap, color, n):

    if cell_data is not None:
        top = cell_data - cell_data.min()
        bot = cell_data.max() - cell_data.min()
        n_cell_data = top / bot

        colormap = mpl.colormaps[cmap]

        colors = colormap(n_cell_data)
    else:
        colors = [color] * n

    return colors


def plot_box(plotter,
             mesh,
             height,
             width,
             cell_data=None,
             cmap='viridis',
             color='lightblue',
             ):
    
    n = mesh.shape[0]

    colors = _colorize(cell_data, cmap, color, n)

    for i in range(n - 1):
        start = mesh[i, :]
        end = mesh[i + 1, :]

        direction = end - start
        length = np.linalg.norm(direction)
        direction /= length

        up = np.array([0, 0, 1])
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


def plot_cyl(plotter, 
             mesh,
             radius,
             cell_data=None, 
             cmap='viridis', 
             color='lightblue',
             ):
    
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
              color='red',
              ):

    n = mesh.shape[0]

    nodes = mesh

    edges = np.zeros((n - 1, 2)).astype(int)
    for i in range(n - 1): edges[i, :] = [i, i + 1]

    padding = np.empty(edges.shape[0], int) * 2
    padding[:] = 2
    edges_w_padding = np.vstack((padding, edges.T)).T

    mesh = pv.PolyData(nodes, edges_w_padding)

    if cell_data is not None:
        # normalize the cell data
        n_cell_data = (cell_data - cell_data.min()) / (cell_data.max() - cell_data.min())
        mesh.cell_data['cell_data'] = n_cell_data
        plotter.add_mesh(mesh,
                        scalars='cell_data',
                        render_lines_as_tubes=render_lines_as_tubes,
                        style='wireframe',
                        line_width=line_width,
                        cmap=cmap,
                        show_scalar_bar=False,
                        )
    else:
        plotter.add_mesh(mesh,
                        render_lines_as_tubes=render_lines_as_tubes,
                        style='wireframe',
                        line_width=line_width,
                        color=color,
                        show_scalar_bar=False,
                        )



def plot_points(plotter,
                mesh,
                color='red', 
                point_size=50,
                render_points_as_spheres=True):
    
    n = mesh.shape[0]
    nodes = mesh
    edges = np.zeros((n - 1, 2)).astype(int)
    for i in range(n - 1): edges[i, :] = [i, i + 1]
    padding = np.empty(edges.shape[0], int) * 2
    padding[:] = 2
    edges_w_padding = np.vstack((padding, edges.T)).T

    mesh = pv.PolyData(nodes, edges_w_padding)

    plotter.add_points(mesh.points, color=color,
                        point_size=point_size,
                        render_points_as_spheres=render_points_as_spheres,
                        )







if __name__ == '__main__':

    n = 10
    mesh = np.zeros((n, 3))
    mesh[:, 0] = np.linspace(0, 1, n)
    cell_data = np.linspace(0, 10, n - 1)

    plotter = pv.Plotter()
    # plot_mesh(plotter, mesh, cell_data)
    # cs_data = np.ones((n - 1)) * 0.2
    # plot_cyl(plotter, mesh, cell_data, cs_data)

    height = np.ones((n - 1)) * 0.2
    width = np.ones((n - 1)) * 0.2
    plot_box(plotter, mesh, height, width, cell_data)



    plotter.show()