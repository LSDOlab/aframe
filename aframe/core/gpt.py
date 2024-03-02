import numpy as np

def rotate_rectangle(vertices, target_normal):
    # Step 1: Calculate the current normal vector
    edge1 = vertices[1] - vertices[0]
    edge2 = vertices[2] - vertices[0]
    current_normal = np.cross(edge1, edge2)
    current_normal /= np.linalg.norm(current_normal)  # Normalize the current normal vector

    # Step 2: Calculate the rotation axis
    rotation_axis = np.cross(current_normal, target_normal)
    rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize the rotation axis

    # Step 3: Calculate the rotation angle
    dot_product = np.dot(current_normal, target_normal)
    rotation_angle = np.arccos(dot_product)

    # Step 4: Construct the rotation matrix
    rotation_matrix = rotation_matrix_from_axis_angle(rotation_axis, rotation_angle)

    # Step 5: Apply the rotation matrix to all vertices
    rotated_vertices = [np.dot(rotation_matrix, vertex) for vertex in vertices]

    return rotated_vertices

def rotation_matrix_from_axis_angle(axis, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis
    rotation_matrix = np.array([[t*x*x + c, t*x*y - z*s, t*x*z + y*s],
                                [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
                                [t*x*z - y*s, t*y*z + x*s, t*z*z + c]])
    return rotation_matrix

# Example usage
vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])  # Example rectangle vertices
target_normal = np.array([0, 0, 1])  # Target normal vector (e.g., pointing upwards)
rotated_vertices = rotate_rectangle(vertices, target_normal)
print("Rotated Vertices:")
for vertex in rotated_vertices:
    print(vertex)