import numpy as np



def rotation_matrix_from_axis_angle(axis, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis
    rotation_matrix = np.array([[t*x*x + c, t*x*y - z*s, t*x*z + y*s],
                                [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
                                [t*x*z - y*s, t*y*z + x*s, t*z*z + c]])
    return rotation_matrix


def plot_box(mesh, width, height):

    n = len(mesh)

    v1 = np.array([-width / 2, height / 2, np.zeros((n - 1))])
    v2 = np.array([width / 2, height / 2, np.zeros((n - 1))])
    v3 = np.array([width / 2, -height / 2, np.zeros((n - 1))])
    v4 = np.array([-width / 2, -height / 2, np.zeros((n - 1))])

    current_normal = np.array([0, 0, 1])

    vertices = []
    for i in range(n - 1):

        if i == 0: target_normal = (mesh[i+1, :] - mesh[i, :]) / np.linalg.norm(mesh[i+1, :] - mesh[i, :])
        else: target_normal = ((mesh[i+1, :] - mesh[i, :]) + (mesh[i, :] - mesh[i-1, :])) / np.linalg.norm((mesh[i+1, :] - mesh[i, :]) + (mesh[i, :] - mesh[i-1, :]))

        rotation_axis = np.cross(current_normal, target_normal)
        rotation_axis /= np.linalg.norm(rotation_axis)
        dot_product = np.dot(current_normal, target_normal)
        rotation_angle = np.arccos(dot_product)
        rotation_matrix = rotation_matrix_from_axis_angle(rotation_axis, rotation_angle)

        offset = mesh[i, :] + (mesh[i + 1, :] - mesh[i, :]) / 2

        nv1 = np.dot(rotation_matrix, v1[:, i]) + offset # v1
        nv2 = np.dot(rotation_matrix, v2[:, i]) + offset # v2
        nv3 = np.dot(rotation_matrix, v3[:, i]) + offset # v3
        nv4 = np.dot(rotation_matrix, v4[:, i]) + offset # v4
        # vertices.append((nv1, nv2, nv3, nv4))

        # format required for matplotlib plotting
        x = [nv1[0], nv2[0], nv3[0], nv4[0]]
        y = [nv1[1], nv2[1], nv3[1], nv4[1]]
        z = [nv1[2], nv2[2], nv3[2], nv4[2]]
        verts = [list(zip(x, y, z))]
        vertices.append(verts)

    return vertices





def plot_circle(mesh, radius, num_circle):
    n = len(mesh)

    theta = np.linspace(0, 2 * np.pi, num_circle)
    current_normal = np.array([0, 0, 1]) # assumes the circle starts in the x-y plane

    vertices = []
    for i in range(n - 1):

        if i == 0: target_normal = (mesh[i+1, :] - mesh[i, :]) / np.linalg.norm(mesh[i+1, :] - mesh[i, :])
        else: target_normal = ((mesh[i+1, :] - mesh[i, :]) + (mesh[i, :] - mesh[i-1, :])) / np.linalg.norm((mesh[i+1, :] - mesh[i, :]) + (mesh[i, :] - mesh[i-1, :]))

        rotation_axis = np.cross(current_normal, target_normal)
        rotation_axis /= np.linalg.norm(rotation_axis)
        dot_product = np.dot(current_normal, target_normal)
        rotation_angle = np.arccos(dot_product)
        rotation_matrix = rotation_matrix_from_axis_angle(rotation_axis, rotation_angle)

        offset = mesh[i, :] + (mesh[i + 1, :] - mesh[i, :]) / 2

        x, y, z = [], [], []
        for j in range(num_circle):
            coord = np.array([radius[i] * np.cos(theta[j]), radius[i] * np.sin(theta[j]), 0])
            new_coord = np.dot(rotation_matrix, coord) + offset
            x.append(new_coord[0])
            y.append(new_coord[1])
            z.append(new_coord[2])


        verts = [list(zip(x, y, z))]
        vertices.append(verts)

    return vertices







if __name__ == '__main__':
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    import matplotlib.pyplot as plt


    num_nodes = 6
    wing_mesh = np.zeros((num_nodes, 3))
    wing_mesh[:, 1] = np.linspace(-20, 20, num_nodes)
    wing_width = np.ones(num_nodes-1)*1
    wing_height = np.ones(num_nodes-1)*0.5
    wing_radius = np.ones(num_nodes-1)*1

    # vertices = plot_box(wing_mesh, wing_width, wing_height)
    vertices = plot_circle(wing_mesh, wing_radius, num_circle=20)


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.view_init(elev=35, azim=-10)
    ax.set_box_aspect((1, 4, 1))

    for i in range(num_nodes-1):
        ax.add_collection3d(Poly3DCollection(vertices[i], facecolors='cyan', linewidths=1, edgecolors='r', alpha=.20))

    ax.plot(wing_mesh[:, 0], wing_mesh[:, 1], wing_mesh[:, 2])
    ax.scatter(wing_mesh[:, 0], wing_mesh[:, 1], wing_mesh[:, 2])
    plt.show()