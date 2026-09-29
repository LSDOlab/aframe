"""Transfer of nodal values between non-matching meshes (e.g. aerodynamic <-> structural)."""
import numpy as np

__all__ = ['NodalMap']


class NodalMap:
    """
    Map nodal values from mesh_in onto mesh_out with normalized distance
    weights (each output node's weights sum to 1).

    Parameters
    ----------
    mesh_in : np.ndarray
        Source node coordinates, shape (n, 3).
    mesh_out : np.ndarray
        Target node coordinates, shape (m, 3).
    method : {'rbf', 'idw'}
        Gaussian radial basis function or inverse distance weighting.
    """

    def __init__(self, mesh_in:np.ndarray, mesh_out:np.ndarray, method:str = 'rbf'):
        self.mesh_in = mesh_in
        self.mesh_out = mesh_out
        self.n, _ = mesh_in.shape
        self.m, _ = mesh_out.shape
        self.method = method
        self._weights = None


    def _distances(self)->np.ndarray:
        """pairwise distances between the source and target nodes, shape (n, m)"""
        return np.linalg.norm(self.mesh_in[:, np.newaxis, :] - self.mesh_out[np.newaxis, :, :], axis=2)


    def rbf_weighting(self, eps:float = 1)->np.ndarray:
        """Gaussian kernel weights exp(-eps d^2), shape (n, m)"""
        weights = np.exp(-eps * self._distances()**2)
        weights /= weights.sum(axis=0, keepdims=True)
        return weights


    def inverse_distance_weighting(self, power:float = 2)->np.ndarray:
        """inverse distance weights 1 / d^power, shape (n, m)"""
        # a coincident node gives an infinite weight
        with np.errstate(divide='ignore'):
            weights = 1.0 / self._distances()**power
            weights /= weights.sum(axis=0, keepdims=True)
        return weights


    def evaluate(self, values:np.ndarray)->np.ndarray:
        """
        map values (n, k) defined on mesh_in onto mesh_out and return
        mesh_out + mapped values (e.g. the displaced target mesh), shape (m, k)
        """
        if self._weights is None:
            if self.method == 'rbf':
                self._weights = self.rbf_weighting()
            elif self.method == 'idw':
                self._weights = self.inverse_distance_weighting()
            else:
                raise ValueError(f"Invalid method: {self.method}")

        return self.mesh_out + self._weights.T @ values
