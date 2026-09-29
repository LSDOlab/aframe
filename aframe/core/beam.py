"""3D Euler-Bernoulli beam elements: stiffness, mass and coordinate transforms."""
from __future__ import annotations
from typing import TYPE_CHECKING, List
import numpy as np
import csdl_alpha as csdl

if TYPE_CHECKING:
    from aframe.core.cs import CrossSection

__all__ = ['Beam']


class Beam:
    """
    A beam discretized into num_nodes - 1 two-node Euler-Bernoulli elements
    with 6 dofs per node (3 translations, 3 rotations).

    Element matrices are built in local coordinates (x along the element) and
    rotated to global coordinates. The mesh, material and cross-section are
    csdl Variables/constants, so everything is differentiable.

    Parameters
    ----------
    name : str
        Unique name, used as the key of the per-beam results in Frame.
    mesh : csdl.Variable
        Node coordinates, shape (num_nodes, 3).
    E, G : float
        Young's modulus and shear modulus.
    density : float
        Material density.
    cs : CrossSection
        Cross-section (CSTube, CSBox, ...) with per-element properties,
        shape (num_nodes - 1,).
    z : bool, optional
        Set True for a beam along the global z axis, where the default
        local-axis construction is singular.
    """

    def __init__(self,
                 name:str,
                 mesh:csdl.Variable,
                 E:float,
                 G:float,
                 density:float,
                 cs:CrossSection,
                 z:bool = False):

        self.name = name
        self.mesh = mesh
        self.E = E
        self.G = G
        self.density = density
        self.cs = cs
        self.z = z

        self.num_nodes = mesh.shape[0]
        self.num_elements = self.num_nodes - 1

        if cs.area.shape != (self.num_elements,):
            raise ValueError(f'the cross-section of beam {name!r} has properties of shape {cs.area.shape}, '
                             f'expected (num_nodes - 1,) = ({self.num_elements},)')

        self.loads = None
        self.extra_inertial_mass = None
        self.fixed_boundary_conditions: List[int] = []
        self.pinned_boundary_conditions: List[int] = []

        # element geometry and direction cosines
        self.lengths, self.ll, self.mm, self.nn, self.D = self._lengths(mesh)

        # element matrices in local and global coordinates
        self.local_stiffness = self._local_stiffness_matrices()
        self.local_mass = self._local_mass_matrices()
        self.transforms = self._vectorized_transforms()
        self.transformed_stiffness = self._transform_stiffness_matrices()
        self.transformed_mass = self._transform_mass_matrices()

        # mass properties
        element_masses = self.cs.area * self.lengths * self.density
        self.mass = csdl.sum(element_masses)

        cg2 = (self.mesh[1:, :] + self.mesh[:-1, :]) / 2
        element_masses_expanded = csdl.expand(element_masses, (self.num_elements, 3), action='i->ij')
        self.rmvec = csdl.sum(cg2 * element_masses_expanded, axes=(0,))
        self.cg = self.rmvec / self.mass


    def fix(self, node:int)->None:
        """clamp a node (all 6 dofs)"""
        if node < 0 or node > self.num_nodes - 1:
            raise ValueError('fixed nodes must be between 0 and num_nodes - 1')

        if node not in self.fixed_boundary_conditions and node not in self.pinned_boundary_conditions:
            self.fixed_boundary_conditions.append(node)


    def pin(self, node:int)->None:
        """pin a node (the 3 translations; rotations stay free)"""
        if node < 0 or node > self.num_nodes - 1:
            raise ValueError('pinned nodes must be between 0 and num_nodes - 1')

        if node not in self.pinned_boundary_conditions and node not in self.fixed_boundary_conditions:
            self.pinned_boundary_conditions.append(node)


    def add_inertial_mass(self, mass:csdl.Variable)->None:
        """
        add point masses at the nodes, shape (num_nodes,); they are resolved
        as inertial loads when the Frame has an acceleration
        """
        if mass.shape != (self.num_nodes,):
            raise ValueError('inertial mass must have shape (num_beam_nodes,)')

        self.extra_inertial_mass = mass


    def add_load(self, load:csdl.Variable)->None:
        """set the nodal loads [Fx, Fy, Fz, Mx, My, Mz], shape (num_nodes, 6)"""
        if load.shape != (self.num_nodes, 6):
            raise ValueError('load must have shape (num_beam_nodes, 6)')

        self.loads = load


    def _lengths(self, mesh:csdl.Variable)->tuple:
        """
        element lengths and direction cosines (ll, mm, nn) of the element axes,
        plus D = sqrt(ll^2 + mm^2) used by the transforms
        """
        diffs = mesh[1:] - mesh[:-1]
        lengths = csdl.norm(diffs, axes=(1,))
        exl = csdl.expand(lengths, (self.num_elements, 3), action='i->ij')
        cp = diffs / exl

        ll = cp[:, 0]
        mm = cp[:, 1]
        nn = cp[:, 2]
        D = (ll**2 + mm**2)**0.5

        return lengths, ll, mm, nn, D


    def _local_stiffness_matrices(self)->csdl.Variable:
        """Euler-Bernoulli element stiffness matrices, shape (num_elements, 12, 12)"""
        A = self.cs.area
        E, G = self.E, self.G
        Iz = self.cs.iz
        Iy = self.cs.iy
        J = self.cs.ix
        L = self.lengths

        AEL = A*E/L
        nAEL = -AEL
        GJL = G*J/L
        nGJL = -GJL

        # bending about local z (v, theta_z)
        EIzL = E*Iz/L
        EIzL2 = EIzL/L
        EIzL3 = EIzL2/L
        EIzL312 = 12*EIzL3
        nEIzL312 = -EIzL312
        EIzL26 = 6*EIzL2
        nEIzL26 = -EIzL26
        EIzL4 = 4*EIzL
        EIzL_2 = 2*EIzL

        # bending about local y (w, theta_y)
        EIyL = E*Iy/L
        EIyL2 = EIyL/L
        EIyL3 = EIyL2/L
        EIyL26 = 6*EIyL2
        nEIyL26 = -EIyL26
        EIyL312 = 12*EIyL3
        nEIyL312 = -EIyL312
        EIyL4 = 4*EIyL
        EIyL_2 = 2*EIyL

        # dof order per node: u, v, w, theta_x, theta_y, theta_z
        diag = [(0, AEL), (1, EIzL312), (2, EIyL312), (3, GJL), (4, EIyL4), (5, EIzL4),
                (6, AEL), (7, EIzL312), (8, EIyL312), (9, GJL), (10, EIyL4), (11, EIzL4)]

        # upper triangle; the lower triangle is mirrored
        off_diag = [(1, 5, EIzL26), (2, 4, nEIyL26), (0, 6, nAEL), (1, 7, nEIzL312),
                    (1, 11, EIzL26), (2, 8, nEIyL312), (2, 10, nEIyL26), (3, 9, nGJL),
                    (4, 8, EIyL26), (4, 10, EIyL_2), (5, 7, nEIzL26), (5, 11, EIzL_2),
                    (7, 11, nEIzL26), (8, 10, EIyL26)]

        return self._symmetric_matrices(diag, off_diag)


    def _local_mass_matrices(self)->csdl.Variable:
        """consistent element mass matrices, shape (num_elements, 12, 12)"""
        A = self.cs.area
        rho = self.density
        J = self.cs.ix
        L = self.lengths

        aa = L / 2
        aa2 = aa**2
        coef = rho * A * aa / 105
        coef70 = coef * 70
        coef78 = coef * 78
        coef35 = coef * 35
        ncoef35 = -coef35
        coef27 = coef * 27
        coef22aa = coef * 22 * aa
        ncoef22aa = -coef22aa
        coef13aa = coef * 13 * aa
        ncoef13aa = -coef13aa
        coef8aa2 = coef * 8 * aa2
        ncoef6aa2 = -coef * 6 * aa2
        # torsional inertia uses the polar radius of gyration squared
        rx2 = J / A
        coef70rx2 = coef70 * rx2
        ncoef35rx2 = ncoef35 * rx2

        diag = [(0, coef70), ([1, 2, 7, 8], coef78), (3, coef70rx2),
                ([4, 5, 10, 11], coef8aa2), (6, coef70), (9, coef70rx2)]

        # upper triangle; the lower triangle is mirrored
        off_diag = [(2, 4, ncoef22aa), (1, 5, coef22aa), (0, 6, coef35), (1, 7, coef27),
                    (5, 7, coef13aa), (2, 8, coef27), (4, 8, ncoef13aa), (3, 9, ncoef35rx2),
                    (2, 10, coef13aa), (4, 10, ncoef6aa2), (8, 10, coef22aa), (1, 11, ncoef13aa),
                    (5, 11, ncoef6aa2), (7, 11, ncoef22aa)]

        return self._symmetric_matrices(diag, off_diag)


    def _symmetric_matrices(self, diag:list, off_diag:list)->csdl.Variable:
        """
        build symmetric (num_elements, 12, 12) matrices from per-element values
        diag: (index or list of indices, value); off_diag: (row, col, value), upper triangle
        """
        n = self.num_elements

        diag_matrices = csdl.Variable(value=np.zeros((n, 12, 12)))
        for index, value in diag:
            if isinstance(index, list):
                value = value.expand((n, len(index)), action='i->ij')
            diag_matrices = diag_matrices.set(csdl.slice[:, index, index], value)

        upper = csdl.Variable(value=np.zeros((n, 12, 12)))
        for row, col, value in off_diag:
            upper = upper.set(csdl.slice[:, row, col], value)

        return diag_matrices + upper + csdl.einsum(upper, action='ijk->ikj')


    def _vectorized_transforms(self)->csdl.Variable:
        """
        global-to-local rotation matrices T, shape (num_elements, 12, 12):
        the same 3x3 direction-cosine block on each of the 4 diagonal blocks
        """
        n = self.num_elements

        if self.z:
            # element along global z: local x = global z
            zeros = csdl.Variable(value=np.zeros((n,)))
            ones = csdl.Variable(value=np.ones((n,)))
            block = [[zeros, zeros, ones],
                     [zeros, ones, None],
                     [-ones, zeros, zeros]]
        else:
            ll, mm, nn, D = self.ll, self.mm, self.nn, self.D
            nmmD = -mm / D
            llD = ll / D
            block = [[ll, mm, nn],
                     [nmmD, llD, None],
                     [-nn * llD, nn * nmmD, D]]

        T = csdl.Variable(value=np.zeros((n, 12, 12)))
        for r in range(3):
            for c in range(3):
                if block[r][c] is not None:
                    # entry (r, c) of all 4 diagonal blocks
                    rows = [r, r + 3, r + 6, r + 9]
                    cols = [c, c + 3, c + 6, c + 9]
                    T = T.set(csdl.slice[:, rows, cols], block[r][c].expand((n, 4), action='i->ij'))

        # kept for backward compatibility
        self.transformations_bookshelf = T

        return T


    def _transform(self, local_matrices:csdl.Variable)->csdl.Variable:
        """rotate local element matrices to global coordinates: T^T K T"""
        transforms = self.transforms
        T_transpose = csdl.einsum(transforms, action='ijk->ikj')
        T_transpose_K = csdl.einsum(T_transpose, local_matrices, action='ijk,ikl->ijl')
        return csdl.einsum(T_transpose_K, transforms, action='ijk,ikl->ijl')


    def _transform_stiffness_matrices(self)->csdl.Variable:
        """element stiffness matrices in global coordinates"""
        return self._transform(self.local_stiffness)


    def _transform_mass_matrices(self)->csdl.Variable:
        """element mass matrices in global coordinates"""
        return self._transform(self.local_mass)


    def element_loads(self, element_displacements:csdl.Variable)->csdl.Variable:
        """
        local element end loads K_local T u_e, shape (num_elements, 12), from the
        global element displacements u_e = [u_a, u_b], shape (num_elements, 12)
        """
        local_displacements = csdl.einsum(self.transforms, element_displacements, action='ijk,ik->ij')

        return csdl.einsum(self.local_stiffness, local_displacements, action='ijk,ik->ij')
