import numpy as np



class NodalMap:
    def __init__(self, mesh_in: np.ndarray, mesh_out: np.ndarray, method='rbf'):
        self.mesh_in = mesh_in
        self.mesh_out = mesh_out
        self.n, _ = mesh_in.shape
        self.m, _ = mesh_out.shape
        self.method = method


    def rbf_weighting(self, eps=1):
        # Compute the pairwise distances between each target point and all source points
        distances = np.linalg.norm(mesh_in[:, np.newaxis, :] - mesh_out[np.newaxis, :, :], axis=2)

        # Apply the radial basis function formula
        # Gaussian kernel
        weights = np.exp(-eps * distances**2)
        # Thin-plate spline kernel
        # weights = distances**4 * np.log(distances + 1e-6)
        weights /= weights.sum(axis=0, keepdims=True)  # Normalize the weights
        
        return weights


    def inverse_distance_weighting(self, power=2):
        # Compute the pairwise distances between each target point and all source points
        distances = np.linalg.norm(mesh_in[:, np.newaxis, :] - mesh_out[np.newaxis, :, :], axis=2)
        
        # Apply the inverse distance weighting formula
        with np.errstate(divide='ignore'):  # To handle division by zero
            weights = 1.0 / distances**power
            weights /= weights.sum(axis=0, keepdims=True)  # Normalize the weights
        
        return weights
    
    
    def evaluate(self, values):

        if self.method == 'rbf':
            weights = self.rbf_weighting()
        elif self.method == 'idw':
            weights = self.inverse_distance_weighting()
        else:
            raise ValueError(f"Invalid method: {self.method}")
        
        return self.mesh_out + weights.T @ values





if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n_in = 10
    mesh_in = np.zeros((n_in, 3))
    mesh_in[:, 0] = np.linspace(0, 10, n_in)

    n_out = 12
    mesh_out = np.zeros((n_out, 3))
    mesh_out[:, 0] = np.linspace(0, 10, n_out)
    mesh_out[:, 1] = 0.2


    disp = np.zeros((n_in, 3))
    disp[:, 1] = np.linspace(0, 2, n_in)
    def_mesh_in = mesh_in + disp

    nm = NodalMap(mesh_in, mesh_out, method='rbf')
    ans = nm.evaluate(disp)
    
    plt.plot(mesh_in[:, 0], mesh_in[:, 1], label='mesh_in')
    plt.scatter(mesh_in[:, 0], mesh_in[:, 1])
    plt.scatter(mesh_out[:, 0], mesh_out[:, 1])
    plt.plot(ans[:, 0], ans[:, 1], label='mapped_mesh')
    plt.plot(def_mesh_in[:, 0], def_mesh_in[:, 1], label='def_mesh_in')
    plt.scatter(ans[:, 0], ans[:, 1], color='red', s=70)

    plt.show()