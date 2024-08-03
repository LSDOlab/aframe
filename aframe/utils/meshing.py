import numpy as np



def mesh_from_points_and_edges(points, edges, num_nodes):
    """
    Create a 3D linspace mesh for each edge.

    Parameters:
    points (numpy array): Array of shape (num_points, 3) containing the coordinates of the points.
    edges (numpy array): Array of shape (num_edges, 2) containing the point connectivity.
    num_nodes (int): The number of nodes to generate along each edge.

    Returns:
    numpy array: Array of shape (num_edges, num_nodes, 3) containing the mesh points for each edge.
    """
    # Initialize the output array
    mesh_points = np.zeros((edges.shape[0], num_nodes, 3))
    
    # Loop through each edge to create the mesh
    for i, (start_idx, end_idx) in enumerate(edges):
        start_point = points[start_idx]
        end_point = points[end_idx]
        
        # Create linspace for each coordinate
        for j in range(3):
            mesh_points[i, :, j] = np.linspace(start_point[j], end_point[j], num=num_nodes)
    
    return mesh_points





if __name__ == '__main__':
    # Example usage:
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 0], [1, -1, 1]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    num_nodes = 5

    mesh = mesh_from_points_and_edges(points, edges, num_nodes)
    print(mesh)
