import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def normalize(v):
    return v / np.linalg.norm(v)

def midpoint(v1, v2):
    return normalize((v1 + v2) / 2)

def create_icosahedron():
    phi = (1 + np.sqrt(5)) / 2
    vertices = []

    # Create 12 vertices of icosahedron
    for coords in [
        (-1,  phi, 0), (1,  phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1,  phi), (0, 1,  phi), (0, -1, -phi), (0, 1, -phi),
        ( phi, 0, -1), ( phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)
    ]:
        vertices.append(normalize(np.array(coords)))

    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    return vertices, faces

def subdivide(vertices, faces, depth):
    midpoint_cache = {}

    def get_midpoint(i1, i2):
        key = tuple(sorted((i1, i2)))
        if key not in midpoint_cache:
            v1 = vertices[i1]
            v2 = vertices[i2]
            new_vertex = midpoint(v1, v2)
            midpoint_cache[key] = len(vertices)
            vertices.append(new_vertex)
        return midpoint_cache[key]

    for _ in range(depth):
        new_faces = []
        for tri in faces:
            a = get_midpoint(tri[0], tri[1])
            b = get_midpoint(tri[1], tri[2])
            c = get_midpoint(tri[2], tri[0])

            new_faces.extend([
                (tri[0], a, c),
                (tri[1], b, a),
                (tri[2], c, b),
                (a, b, c),
            ])
        faces = new_faces
    return vertices, faces

def plot_icosphere(vertices, faces):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    mesh = [[vertices[i] for i in face] for face in faces]
    poly3d = Poly3DCollection(mesh, facecolors='lightblue', edgecolors='k', linewidths=0.2, alpha=0.9)
    ax.add_collection3d(poly3d)

    ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
    ax.set_box_aspect([1,1,1])
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    depth = 2  # Adjust for smoother sphere
    vertices, faces = create_icosahedron()
    vertices, faces = subdivide(vertices, faces, depth)
    plot_icosphere(vertices, faces)
