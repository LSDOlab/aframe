import numpy as np
import aframe as af
import csdl_alpha as csdl
from typing import List


class Beam:

    def __init__(self, name:str, 
                 mesh:csdl.Variable, 
                 material:'af.Material', 
                 cs:'af.cs',
                 z=False):
        
        self.name = name
        self.mesh = mesh
        self.material = material
        self.cs = cs
        self.z = z
        self.num_nodes = mesh.shape[0]
        self.num_elements = self.num_nodes - 1
        self.loads = None
        self.extra_inertial_mass = None
        self.fixed_boundary_conditions: List[int] = []
        self.pinned_boundary_conditions: List[int] = []
        # map the beam nodes to the global indices
        self.map: List[int] = []
        # precompute lengths
        self.lengths, self.ll, self.mm, self.nn, self.D = self._lengths(mesh)
        # beam-specific functions
        self.local_stiffness = self._local_stiffness_matrices()
        self.local_mass = self._local_mass_matrices()
        self.transforms = self._vectorized_transforms()
        self.transformed_stiffness = self._transform_stiffness_matrices()
        self.transformed_mass = self._transform_mass_matrices()


    def fix(self, node):

        if node not in self.fixed_boundary_conditions and node not in self.pinned_boundary_conditions:
            self.fixed_boundary_conditions.append(node)

    
    def pin(self, node):

        if node not in self.pinned_boundary_conditions and node not in self.fixed_boundary_conditions:
            self.pinned_boundary_conditions.append(node)


    def add_inertial_mass(self, mass, node):
        extra_mass = csdl.Variable(value=np.zeros(self.num_nodes))
        extra_mass = extra_mass.set(csdl.slice[node], mass)
        self.extra_inertial_mass = extra_mass


    def add_load(self, load):
        self.loads = load

    
    def _lengths(self, mesh)->tuple[csdl.Variable, csdl.Variable, csdl.Variable, csdl.Variable, csdl.Variable]:
        # Compute the squared differences
        diffs = mesh[1:] - mesh[:-1]
        # Sum the squared differences along the rows and take the square root
        lengths = csdl.norm(diffs, axes=(1,))
        exl = csdl.expand(lengths, (self.num_elements, 3), action='i->ij')
        cp = diffs / exl

        # precomps for transforms
        ll = cp[:, 0]
        mm = cp[:, 1]
        nn = cp[:, 2]
        D = (ll**2 + mm**2)**0.5


        return lengths, ll, mm, nn, D

        
    def _local_stiffness_matrices(self)->csdl.Variable:

        A = self.cs.area
        E, G = self.material.E, self.material.G
        Iz = self.cs.iz
        Iy = self.cs.iy
        J = self.cs.ix
        L = self.lengths

        # local_stiffness = csdl.Variable(value=np.zeros((self.num_elements, 12, 12)))
        diag = csdl.Variable(value=np.zeros((self.num_elements, 12, 12)))
        off_diag = csdl.Variable(value=np.zeros((self.num_elements, 12, 12)))

        # pre-computations for speed
        AEL = A*E/L
        nAEL = -AEL
        GJL = G*J/L
        nGJL = -GJL

        EIz = E*Iz
        EIzL = EIz/L
        EIzL2 = EIzL/L
        EIzL3 = EIzL2/L
        EIzL312 = 12*EIzL3
        nEIzL312 = -EIzL312
        EIzL26 = 6*EIzL2
        nEIzL26 = -EIzL26
        EIzL4 = 4*EIzL
        EIzL2 = 2*EIzL

        EIy = E*Iy
        EIyL = EIy/L
        EIyL2 = EIyL/L
        EIyL3 = EIyL2/L
        EIyL26 = 6*EIyL2
        nEIyL26 = -EIyL26
        EIyL312 = 12*EIyL3
        nEIyL312 = -EIyL312
        EIyL4 = 4*EIyL
        EIyL2 = 2*EIyL

        diag = diag.set(csdl.slice[:, 0, 0], AEL)
        diag = diag.set(csdl.slice[:, 1, 1], EIzL312)
        diag = diag.set(csdl.slice[:, 2, 2], EIyL312)
        diag = diag.set(csdl.slice[:, 3, 3], GJL)
        diag = diag.set(csdl.slice[:, 4, 4], EIyL4)
        diag = diag.set(csdl.slice[:, 5, 5], EIzL4)
        diag = diag.set(csdl.slice[:, 6, 6], AEL)
        diag = diag.set(csdl.slice[:, 7, 7], EIzL312)
        diag = diag.set(csdl.slice[:, 8, 8], EIyL312)
        diag = diag.set(csdl.slice[:, 9, 9], GJL)
        diag = diag.set(csdl.slice[:, 10, 10], EIyL4)
        diag = diag.set(csdl.slice[:, 11, 11], EIzL4)
        

        off_diag = off_diag.set(csdl.slice[:, 1, 5], EIzL26)
        # local_stiffness = local_stiffness.set(csdl.slice[:, 5, 1], EIzL26)

        off_diag = off_diag.set(csdl.slice[:, 2, 4], nEIyL26)
        # local_stiffness = local_stiffness.set(csdl.slice[:, 4, 2], nEIyL26)

        off_diag = off_diag.set(csdl.slice[:, 0, 6], nAEL)
        # local_stiffness = local_stiffness.set(csdl.slice[:, 6, 0], nAEL)

        off_diag = off_diag.set(csdl.slice[:, 1, 7], nEIzL312)
        # local_stiffness = local_stiffness.set(csdl.slice[:, 7, 1], nEIzL312)

        off_diag = off_diag.set(csdl.slice[:, 1, 11], EIzL26)
        # local_stiffness = local_stiffness.set(csdl.slice[:, 11, 1], EIzL26)

        off_diag = off_diag.set(csdl.slice[:, 2, 8], nEIyL312)
        # local_stiffness = local_stiffness.set(csdl.slice[:, 8, 2], nEIyL312)

        off_diag = off_diag.set(csdl.slice[:, 2, 10], nEIyL26)
        # local_stiffness = local_stiffness.set(csdl.slice[:, 10, 2], nEIyL26)

        off_diag = off_diag.set(csdl.slice[:, 3, 9], nGJL)
        # local_stiffness = local_stiffness.set(csdl.slice[:, 9, 3], nGJL)

        off_diag = off_diag.set(csdl.slice[:, 4, 8], EIyL26)
        # local_stiffness = local_stiffness.set(csdl.slice[:, 8, 4], EIyL26)

        off_diag = off_diag.set(csdl.slice[:, 4, 10], EIyL2)
        # local_stiffness = local_stiffness.set(csdl.slice[:, 10, 4], EIyL2)

        off_diag = off_diag.set(csdl.slice[:, 5, 7], nEIzL26)
        # local_stiffness = local_stiffness.set(csdl.slice[:, 7, 5], nEIzL26)

        off_diag = off_diag.set(csdl.slice[:, 5, 11], EIzL2)
        # local_stiffness = local_stiffness.set(csdl.slice[:, 11, 5], EIzL2)

        off_diag = off_diag.set(csdl.slice[:, 7, 11], nEIzL26)
        # local_stiffness = local_stiffness.set(csdl.slice[:, 11, 7], nEIzL26)

        off_diag = off_diag.set(csdl.slice[:, 8, 10], EIyL26)
        # local_stiffness = local_stiffness.set(csdl.slice[:, 10, 8], EIyL26)


        local_stiffness = diag + off_diag + csdl.einsum(off_diag, action='ijk->ikj') # symmetric



        return local_stiffness
    

    def _local_mass_matrices(self)->csdl.Variable:
        
        A = self.cs.area
        rho = self.material.density
        J = self.cs.ix
        L = self.lengths

        # coefficients
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
        rx2 = J / A
        ncoef35rx2 = ncoef35 * rx2

        # local_mass = csdl.Variable(value=np.zeros((self.num_elements, 12, 12)))
        mdiag = csdl.Variable(value=np.zeros((self.num_elements, 12, 12)))
        moff_diag = csdl.Variable(value=np.zeros((self.num_elements, 12, 12)))

        mdiag = mdiag.set(csdl.slice[:, 0, 0], coef70)
        mdiag = mdiag.set(csdl.slice[:, 1, 1], coef78)
        mdiag = mdiag.set(csdl.slice[:, 2, 2], coef78)
        # local_mass = local_mass.set(csdl.slice[:, 3, 3], coef78 * rx2)
        mdiag = mdiag.set(csdl.slice[:, 3, 3], coef70 * rx2)
        mdiag = mdiag.set(csdl.slice[:, 4, 4], coef8aa2)
        mdiag = mdiag.set(csdl.slice[:, 5, 5], coef8aa2)
        mdiag = mdiag.set(csdl.slice[:, 6, 6], coef70)
        mdiag = mdiag.set(csdl.slice[:, 7, 7], coef78)
        mdiag = mdiag.set(csdl.slice[:, 8, 8], coef78)
        mdiag = mdiag.set(csdl.slice[:, 9, 9], coef70 * rx2)
        mdiag = mdiag.set(csdl.slice[:, 10, 10], coef8aa2)
        mdiag = mdiag.set(csdl.slice[:, 11, 11], coef8aa2)


        moff_diag = moff_diag.set(csdl.slice[:, 2, 4], ncoef22aa)
        # local_mass = local_mass.set(csdl.slice[:, 4, 2], ncoef22aa)

        moff_diag = moff_diag.set(csdl.slice[:, 1, 5], coef22aa)
        # local_mass = local_mass.set(csdl.slice[:, 5, 1], coef22aa)

        moff_diag = moff_diag.set(csdl.slice[:, 0, 6], coef35)
        # local_mass = local_mass.set(csdl.slice[:, 6, 0], coef35)

        moff_diag = moff_diag.set(csdl.slice[:, 1, 7], coef27)
        # local_mass = local_mass.set(csdl.slice[:, 7, 1], coef27)

        moff_diag = moff_diag.set(csdl.slice[:, 5, 7], coef13aa)
        # local_mass = local_mass.set(csdl.slice[:, 7, 5], coef13aa)

        moff_diag = moff_diag.set(csdl.slice[:, 2, 8], coef27)
        # local_mass = local_mass.set(csdl.slice[:, 8, 2], coef27)

        moff_diag = moff_diag.set(csdl.slice[:, 4, 8], ncoef13aa)
        # local_mass = local_mass.set(csdl.slice[:, 8, 4], ncoef13aa)

        moff_diag = moff_diag.set(csdl.slice[:, 3, 9], ncoef35rx2)
        # local_mass = local_mass.set(csdl.slice[:, 9, 3], ncoef35rx2)

        moff_diag = moff_diag.set(csdl.slice[:, 2, 10], coef13aa)
        # local_mass = local_mass.set(csdl.slice[:, 10, 2], coef13aa)

        moff_diag = moff_diag.set(csdl.slice[:, 4, 10], ncoef6aa2)
        # local_mass = local_mass.set(csdl.slice[:, 10, 4], ncoef6aa2)

        moff_diag = moff_diag.set(csdl.slice[:, 8, 10], coef22aa)
        # local_mass = local_mass.set(csdl.slice[:, 10, 8], coef22aa)

        moff_diag = moff_diag.set(csdl.slice[:, 1, 11], ncoef13aa)
        # local_mass = local_mass.set(csdl.slice[:, 11, 1], ncoef13aa)

        moff_diag = moff_diag.set(csdl.slice[:, 5, 11], ncoef6aa2)
        # local_mass = local_mass.set(csdl.slice[:, 11, 5], ncoef6aa2)

        moff_diag = moff_diag.set(csdl.slice[:, 7, 11], ncoef22aa)
        # local_mass = local_mass.set(csdl.slice[:, 11, 7], ncoef22aa)

        local_mass = mdiag + moff_diag + csdl.einsum(moff_diag, action='ijk->ikj') # symmetric


        return local_mass


    def _transforms(self)->csdl.Variable:
        """
        no longer used
        use vectorized_transforms() instead
        """
        T = csdl.Variable(value=np.zeros((self.num_elements, 12, 12)))

        block = csdl.Variable(value=np.zeros((self.num_elements, 3, 3)))
        for i in range(self.num_elements):
            ll = self.ll[i]
            mm = self.mm[i]
            nn = self.nn[i]
            nmm = -mm # precomp for speed
            D = self.D[i]
            nmmD = nmm / D
            llD = ll / D

            if self.z:
                block = block.set(csdl.slice[i, 0, 2], 1)
                block = block.set(csdl.slice[i, 1, 1], 1)
                block = block.set(csdl.slice[i, 2, 0], -1)
            else:
                block = block.set(csdl.slice[i, 0, 0], ll)
                block = block.set(csdl.slice[i, 0, 1], mm)
                block = block.set(csdl.slice[i, 0, 2], nn)
                block = block.set(csdl.slice[i, 1, 0], nmmD)
                block = block.set(csdl.slice[i, 1, 1], llD)
                block = block.set(csdl.slice[i, 2, 0], -nn * llD)
                block = block.set(csdl.slice[i, 2, 1], nn * nmmD)
                block = block.set(csdl.slice[i, 2, 2], D)

        T = T.set(csdl.slice[:, 0:3, 0:3], block)
        T = T.set(csdl.slice[:, 3:6, 3:6], block)
        T = T.set(csdl.slice[:, 6:9, 6:9], block)
        T = T.set(csdl.slice[:, 9:12, 9:12], block)

        # self.transformations_bookshelf = T

        return T


    def _vectorized_transforms(self)->csdl.Variable:
        """
        a vectorized version of the transforms() method
        thanks Mark
        """
        ll = self.ll
        mm = self.mm
        nn = self.nn
        D = self.D
        T = csdl.Variable(value=np.zeros((self.num_elements, 12, 12)))

        if self.z:
            zeros = csdl.Variable(value=np.zeros((self.num_elements,)))
            ones = csdl.Variable(value=np.ones((self.num_elements,)))
            lls_concat = zeros
            mms_concat = zeros
            nns_concat = ones
            nmmDs_concat = zeros
            llDs_concat = ones
            nnllD_concant = -ones
            nnnmmD_concant = zeros
            Ds_concat = zeros
        else:
            lls_concat = ll
            mms_concat = mm
            nns_concat = nn
            nmmDs_concat = -mm / D
            llDs_concat = ll / D
            nnllD_concant = -nn * llDs_concat
            nnnmmD_concant = nn * nmmDs_concat
            Ds_concat = D

        T = T.set(csdl.slice[:, [0,3,6,9], [0,3,6,9]], lls_concat.expand((self.num_elements, 4), action='i->ij'))
        T = T.set(csdl.slice[:, [0,3,6,9], [1,4,7,10]], mms_concat.expand((self.num_elements, 4), action='i->ij'))
        T = T.set(csdl.slice[:, [0,3,6,9], [2,5,8,11]], nns_concat.expand((self.num_elements, 4), action='i->ij'))
        T = T.set(csdl.slice[:, [1,4,7,10], [0,3,6,9]], nmmDs_concat.expand((self.num_elements, 4), action='i->ij'))
        T = T.set(csdl.slice[:, [1,4,7,10], [1,4,7,10]], llDs_concat.expand((self.num_elements, 4), action='i->ij'))
        T = T.set(csdl.slice[:, [2,5,8,11], [0,3,6,9]], nnllD_concant.expand((self.num_elements, 4), action='i->ij'))
        T = T.set(csdl.slice[:, [2,5,8,11], [1,4,7,10]], nnnmmD_concant.expand((self.num_elements, 4), action='i->ij'))
        T = T.set(csdl.slice[:, [2,5,8,11], [2,5,8,11]], Ds_concat.expand((self.num_elements, 4), action='i->ij'))

        self.transformations_bookshelf = T

        return T


    def _transform_stiffness_matrices(self)->csdl.Variable:
        transforms = self.transforms
        local_stiffness_matrices = self.local_stiffness

        # transformed_stiffness_matrices = []
        # for i in range(self.num_elements):
        #     T = transforms[i]
        #     local_stiffness = local_stiffness_matrices[i, :, :]
        #     TKT = csdl.matmat(csdl.transpose(T), csdl.matmat(local_stiffness, T))
        #     transformed_stiffness_matrices.append(TKT)

        # Shape: (num_elements, n, n)
        T_transpose = csdl.einsum(transforms, action='ijk->ikj')
        # Shape: (num_elements, n, n)
        T_transpose_K = csdl.einsum(T_transpose, local_stiffness_matrices, action='ijk,ikl->ijl')
        # Shape: (num_elements, n, n)
        transformed_stiffness_matrices = csdl.einsum(T_transpose_K, transforms, action='ijk,ikl->ijl')


        return transformed_stiffness_matrices
    

    def _transform_mass_matrices(self)->csdl.Variable:
        transforms = self.transforms
        local_mass_matrices = self.local_mass

        # transformed_mass_matrices = []

        # for i in range(self.num_elements):
        #     T = transforms[i]
        #     local_mass = local_mass_matrices[i, :, :]
        #     TMT = csdl.matmat(csdl.transpose(T), csdl.matmat(local_mass, T))
        #     transformed_mass_matrices.append(TMT)

        # Shape: (num_elements, n, n)
        T_transpose = csdl.einsum(transforms, action='ijk->ikj')
        # Shape: (num_elements, n, n)
        T_transpose_M = csdl.einsum(T_transpose, local_mass_matrices, action='ijk,ikl->ijl')
        # Shape: (num_elements, n, n)
        transformed_mass_matrices = csdl.einsum(T_transpose_M, transforms, action='ijk,ikl->ijl')

        return transformed_mass_matrices
    

    def _recover_loads(self, U)->csdl.Variable:

        map = self.map
        displacements = csdl.Variable(value=np.zeros((self.num_elements, 12)))
        lsb = self.local_stiffness
        tb = self.transforms

        for i in range(self.num_elements):
            idxa, idxb = map[i], map[i+1]
            displacements = displacements.set(csdl.slice[i, 0:6], U[idxa:idxa+6])
            displacements = displacements.set(csdl.slice[i, 6:12], U[idxb:idxb+6])

        # Perform transformations
        transformed_displacements = csdl.einsum(tb, displacements, action='ijk,ik->ij')

        # Compute loads
        loads = csdl.einsum(lsb, transformed_displacements, action='ijk,ik->ij')

        return loads
    

    def _mass(self)->tuple[csdl.Variable, csdl.Variable]:

        lengths = self.lengths
        rho = self.material.density
        area = self.cs.area

        element_masses = area * lengths * rho
        beam_mass = csdl.sum(element_masses)

        rmvec = 0
        for i in range(self.num_elements):
            cg = (self.mesh[i + 1, :] + self.mesh[i, :]) / 2
            rmvec += cg * element_masses[i]

        return beam_mass, rmvec

