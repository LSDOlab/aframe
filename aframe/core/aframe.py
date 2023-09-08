import numpy as np
import csdl
import python_csdl_backend
from aframe.core.massprop import MassPropModule as MassProp
from aframe.core.model import Model
from aframe.core.buckle import Buckle
from aframe.core.stress import StressTube, StressBox
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL


# TODO: define an assembel model method and call that

class Aframe(ModuleCSDL):

    def initialize(self):
        self.parameters.declare('beams', default={})
        self.parameters.declare('joints', default={})
        self.parameters.declare('bounds', default={})
        self.parameters.declare('mesh_units', default='m')


    def tube(self, element_name, t, r):
        r1, r2 = r - t, r
        A = np.pi * (r2**2 - r1**2)
        Iy = np.pi * (r2**4 - r1**4) / 4.0
        Iz = np.pi * (r2**4 - r1**4) / 4.0
        J = np.pi * (r2**4 - r1**4) / 2.0

        self.register_output(element_name + '_A', A)
        self.register_output(element_name + '_Ix', 1*J)
        self.register_output(element_name + '_Iy', Iy)
        self.register_output(element_name + '_Iz', Iz)
        self.register_output(element_name + '_J', J)


    def box(self, element_name, w, h, tweb, tcap):
        w_i = w - 2*tweb
        h_i = h - 2*tcap
        # A = (w*h) - (w_i*h_i)
        A = (((w*h) - (w_i*h_i))**2 + 1E-14)**0.5 # for robustness
        Iz = ((w**3)*h - (w_i**3)*h_i)/12
        Iy = (w*(h**3) - w_i*(h_i**3))/12
        J = (w*h*(h**2 + w**2)/12) - (w_i*h_i*(h_i**2 + w_i**2)/12)
        # Q = 2*(h/2)*tweb*(h/4) + (w - 2*tweb)*tcap*((h/2) - (tcap/2))

        Q = (A/2)*(h/4)

        self.register_output(element_name + '_A', A)
        self.register_output(element_name + '_Ix', 1*J) # I think J is the same as Ix...
        self.register_output(element_name + '_Iy', Iy)
        self.register_output(element_name + '_Iz', Iz)
        self.register_output(element_name + '_J', J)
        self.register_output(element_name + '_Q', Q)


    def local_mass(self, element_name, E, G, node_dict, node_index, dim, i, element_density_list):
        A = self.declare_variable(element_name + '_A')
        Ix = self.declare_variable(element_name + '_Ix')
        Iy = self.declare_variable(element_name + '_Iy')
        Iz = self.declare_variable(element_name + '_Iz')
        J = self.declare_variable(element_name + '_J')

        node_a = self.declare_variable(element_name + 'node_a', shape=(3))
        node_b = self.declare_variable(element_name + 'node_b', shape=(3))
        L = csdl.pnorm(node_b - node_a, pnorm_type=2) + 1E-12

        a = L/2
        rho = element_density_list[i]
        coef = rho*A*a/105
        rx2 = Ix/A
        
        mp = self.create_output(element_name + 'mp', shape=(12,12), val=0)

        mp[0,0] = csdl.reshape(coef*70, (1,1))
        mp[1,1] = csdl.reshape(coef*78, (1,1))
        mp[2,2] = csdl.reshape(coef*78, (1,1))
        mp[3,3] = csdl.reshape(coef*78*rx2, (1,1))
        mp[2,4] = csdl.reshape(coef*-22*a, (1,1))
        mp[4,2] = csdl.reshape(coef*-22*a, (1,1))
        mp[4,4] = csdl.reshape(coef*8*a**2, (1,1))
        mp[1,5] = csdl.reshape(coef*22*a, (1,1))
        mp[5,1] = csdl.reshape(coef*22*a, (1,1))
        mp[5,5] = csdl.reshape(coef*8*a**2, (1,1))
        mp[0,6] = csdl.reshape(coef*35, (1,1))
        mp[6,0] = csdl.reshape(coef*35, (1,1))
        mp[6,6] = csdl.reshape(coef*70, (1,1))
        mp[1,7] = csdl.reshape(coef*27, (1,1))
        mp[7,1] = csdl.reshape(coef*27, (1,1))
        mp[5,7] = csdl.reshape(coef*13*a, (1,1))
        mp[7,5] = csdl.reshape(coef*13*a, (1,1))
        mp[7,7] = csdl.reshape(coef*78, (1,1))
        mp[2,8] = csdl.reshape(coef*27, (1,1))
        mp[8,2] = csdl.reshape(coef*27, (1,1))
        mp[4,8] = csdl.reshape(coef*-13*a, (1,1))
        mp[8,4] = csdl.reshape(coef*-13*a, (1,1))
        mp[8,8] = csdl.reshape(coef*78, (1,1))
        mp[3,9] = csdl.reshape(coef*-35*rx2, (1,1))
        mp[9,3] = csdl.reshape(coef*-35*rx2, (1,1))
        mp[9,9] = csdl.reshape(coef*70*rx2, (1,1))
        mp[2,10] = csdl.reshape(coef*13*a, (1,1))
        mp[10,2] = csdl.reshape(coef*13*a, (1,1))
        mp[4,10] = csdl.reshape(coef*-6*a**2, (1,1))
        mp[10,4] = csdl.reshape(coef*-6*a**2, (1,1))
        mp[8,10] = csdl.reshape(coef*22*a, (1,1))
        mp[10,8] = csdl.reshape(coef*22*a, (1,1))
        mp[10,10] = csdl.reshape(coef*8*a**2, (1,1))
        mp[1,11] = csdl.reshape(coef*-13*a, (1,1))
        mp[11,1] = csdl.reshape(coef*-13*a, (1,1))
        mp[5,11] = csdl.reshape(coef*-6*a**2, (1,1))
        mp[11,5] = csdl.reshape(coef*-6*a**2, (1,1))
        mp[7,11] = csdl.reshape(coef*-22*a, (1,1))
        mp[11,7] = csdl.reshape(coef*-22*a, (1,1))
        mp[11,11] = csdl.reshape(coef*8*a**2, (1,1))

        # transform the local mass matrix to global coordinates:
        cp = (node_b - node_a)/csdl.expand(L, (3))
        ll, mm, nn = cp[0], cp[1], cp[2]
        D = (ll**2 + mm**2)**0.5

        block = self.create_output(element_name + 'mass_block',shape=(3,3),val=0)
        block[0,0] = csdl.reshape(ll, (1,1))
        block[0,1] = csdl.reshape(mm, (1,1))
        block[0,2] = csdl.reshape(nn, (1,1))
        block[1,0] = csdl.reshape(-mm/D, (1,1))
        block[1,1] = csdl.reshape(ll/D, (1,1))
        block[2,0] = csdl.reshape(-ll*nn/D, (1,1))
        block[2,1] = csdl.reshape(-mm*nn/D, (1,1))
        block[2,2] = csdl.reshape(D, (1,1))

        # rebranded the transformation matrix T to MT:
        MT = self.create_output(element_name + 'MT', shape=(12,12), val=0)
        MT[0:3,0:3] = 1*block
        MT[3:6,3:6] = 1*block
        MT[6:9,6:9] = 1*block
        MT[9:12,9:12] = 1*block
        tmt = csdl.matmat(csdl.transpose(MT), csdl.matmat(mp, MT))

        # expand the transformed mass matrix to the global dimensions:
        element_mass_matrix = self.create_output(element_name + 'element_mass_matrix', shape=(dim,dim), val=0)

        # parse tmt:
        m11 = tmt[0:6,0:6] # upper left
        m12 = tmt[0:6,6:12] # upper right
        m21 = tmt[6:12,0:6] # lower left
        m22 = tmt[6:12,6:12] # lower right

        # assign the four block matrices to their respective positions in mass_matrix:
        node_a_index = node_index[node_dict[i]]
        node_b_index = node_index[node_dict[i + 1]]

        row_i = node_a_index*6
        row_f = node_a_index*6 + 6
        col_i = node_a_index*6
        col_f = node_a_index*6 + 6
        element_mass_matrix[row_i:row_f, col_i:col_f] = m11

        row_i = node_a_index*6
        row_f = node_a_index*6 + 6
        col_i = node_b_index*6
        col_f = node_b_index*6 + 6
        element_mass_matrix[row_i:row_f, col_i:col_f] = m12

        row_i = node_b_index*6
        row_f = node_b_index*6 + 6
        col_i = node_a_index*6
        col_f = node_a_index*6 + 6
        element_mass_matrix[row_i:row_f, col_i:col_f] = m21

        row_i = node_b_index*6
        row_f = node_b_index*6 + 6
        col_i = node_b_index*6
        col_f = node_b_index*6 + 6
        element_mass_matrix[row_i:row_f, col_i:col_f] = m22


    def lumped_local_mass(self, element_name, E, G, node_dict, node_index, dim, i, element_density_list):
        A = self.declare_variable(element_name + '_A')
        Ix = self.declare_variable(element_name + '_Ix')
        Iy = self.declare_variable(element_name + '_Iy')
        Iz = self.declare_variable(element_name + '_Iz')
        J = self.declare_variable(element_name + '_J')

        node_a = self.declare_variable(element_name + 'node_a', shape=(3))
        node_b = self.declare_variable(element_name + 'node_b', shape=(3))
        L = csdl.pnorm(node_b - node_a, pnorm_type=2)

        a = L/2
        rho = element_density_list[i]
        coef = rho*A*a/105
        rx2 = Ix/A
        
        mp = self.create_output(element_name + 'mp', shape=(12,12), val=0)

        mp[0,0] = csdl.reshape(coef*70, (1,1))
        mp[1,1] = csdl.reshape(coef*78, (1,1))
        mp[2,2] = csdl.reshape(coef*78, (1,1))
        mp[3,3] = csdl.reshape(coef*78*rx2, (1,1))
        mp[2,4] = csdl.reshape(coef*-22*a, (1,1))
        mp[4,2] = csdl.reshape(coef*-22*a, (1,1))
        mp[4,4] = csdl.reshape(coef*8*a**2, (1,1))
        mp[1,5] = csdl.reshape(coef*22*a, (1,1))
        mp[5,1] = csdl.reshape(coef*22*a, (1,1))
        mp[5,5] = csdl.reshape(coef*8*a**2, (1,1))
        mp[0,6] = csdl.reshape(coef*35, (1,1))
        mp[6,0] = csdl.reshape(coef*35, (1,1))
        mp[6,6] = csdl.reshape(coef*70, (1,1))
        mp[1,7] = csdl.reshape(coef*27, (1,1))
        mp[7,1] = csdl.reshape(coef*27, (1,1))
        mp[5,7] = csdl.reshape(coef*13*a, (1,1))
        mp[7,5] = csdl.reshape(coef*13*a, (1,1))
        mp[7,7] = csdl.reshape(coef*78, (1,1))
        mp[2,8] = csdl.reshape(coef*27, (1,1))
        mp[8,2] = csdl.reshape(coef*27, (1,1))
        mp[4,8] = csdl.reshape(coef*-13*a, (1,1))
        mp[8,4] = csdl.reshape(coef*-13*a, (1,1))
        mp[8,8] = csdl.reshape(coef*78, (1,1))
        mp[3,9] = csdl.reshape(coef*-35*rx2, (1,1))
        mp[9,3] = csdl.reshape(coef*-35*rx2, (1,1))
        mp[9,9] = csdl.reshape(coef*70*rx2, (1,1))
        mp[2,10] = csdl.reshape(coef*13*a, (1,1))
        mp[10,2] = csdl.reshape(coef*13*a, (1,1))
        mp[4,10] = csdl.reshape(coef*-6*a**2, (1,1))
        mp[10,4] = csdl.reshape(coef*-6*a**2, (1,1))
        mp[8,10] = csdl.reshape(coef*22*a, (1,1))
        mp[10,8] = csdl.reshape(coef*22*a, (1,1))
        mp[10,10] = csdl.reshape(coef*8*a**2, (1,1))
        mp[1,11] = csdl.reshape(coef*-13*a, (1,1))
        mp[11,1] = csdl.reshape(coef*-13*a, (1,1))
        mp[5,11] = csdl.reshape(coef*-6*a**2, (1,1))
        mp[11,5] = csdl.reshape(coef*-6*a**2, (1,1))
        mp[7,11] = csdl.reshape(coef*-22*a, (1,1))
        mp[11,7] = csdl.reshape(coef*-22*a, (1,1))
        mp[11,11] = csdl.reshape(coef*8*a**2, (1,1))

        # transform the local mass matrix to global coordinates:
        cp = (node_b - node_a)/csdl.expand(L, (3))
        ll, mm, nn = cp[0], cp[1], cp[2]
        D = (ll**2 + mm**2)**0.5

        block = self.create_output(element_name + 'mass_block',shape=(3,3),val=0)
        block[0,0] = csdl.reshape(ll, (1,1))
        block[0,1] = csdl.reshape(mm, (1,1))
        block[0,2] = csdl.reshape(nn, (1,1))
        block[1,0] = csdl.reshape(-mm/D, (1,1))
        block[1,1] = csdl.reshape(ll/D, (1,1))
        block[2,0] = csdl.reshape(-ll*nn/D, (1,1))
        block[2,1] = csdl.reshape(-mm*nn/D, (1,1))
        block[2,2] = csdl.reshape(D, (1,1))

        # rebranded the transformation matrix T to MT:
        MT = self.create_output(element_name + 'MT', shape=(12,12), val=0)
        MT[0:3,0:3] = 1*block
        MT[3:6,3:6] = 1*block
        MT[6:9,6:9] = 1*block
        MT[9:12,9:12] = 1*block
        tmt = csdl.matmat(csdl.transpose(MT), csdl.matmat(mp, MT))

        # expand the transformed mass matrix to the global dimensions:
        element_mass_matrix = self.create_output(element_name + 'element_mass_matrix', shape=(dim,dim), val=0)

        # parse tmt:
        m11 = tmt[0:6,0:6] # upper left
        m12 = tmt[0:6,6:12] # upper right
        m21 = tmt[6:12,0:6] # lower left
        m22 = tmt[6:12,6:12] # lower right

        # assign the four block matrices to their respective positions in mass_matrix:
        node_a_index = node_index[node_dict[i]]
        node_b_index = node_index[node_dict[i + 1]]

        row_i = node_a_index*6
        row_f = node_a_index*6 + 6
        col_i = node_a_index*6
        col_f = node_a_index*6 + 6
        element_mass_matrix[row_i:row_f, col_i:col_f] = m11

        row_i = node_a_index*6
        row_f = node_a_index*6 + 6
        col_i = node_b_index*6
        col_f = node_b_index*6 + 6
        element_mass_matrix[row_i:row_f, col_i:col_f] = m12

        row_i = node_b_index*6
        row_f = node_b_index*6 + 6
        col_i = node_a_index*6
        col_f = node_a_index*6 + 6
        element_mass_matrix[row_i:row_f, col_i:col_f] = m21

        row_i = node_b_index*6
        row_f = node_b_index*6 + 6
        col_i = node_b_index*6
        col_f = node_b_index*6 + 6
        element_mass_matrix[row_i:row_f, col_i:col_f] = m22

    def local_stiffness(self, element_name, E, G, node_dict, node_index, dim, i):
        A = self.declare_variable(element_name + '_A')
        Iy = self.declare_variable(element_name + '_Iy')
        Iz = self.declare_variable(element_name + '_Iz')
        J = self.declare_variable(element_name + '_J')

        node_a = self.declare_variable(element_name + 'node_a', shape=(3))
        node_b = self.declare_variable(element_name + 'node_b', shape=(3))

        L = self.register_output(element_name + 'L', csdl.pnorm(node_b - node_a, pnorm_type=2)) + 1E-12

        kp = self.create_output(element_name + 'kp', shape=(12,12), val=0)
        # the upper left block
        kp[0,0] = csdl.reshape(A*E/L, (1,1))
        kp[1,1] = csdl.reshape(12*E*Iz/L**3, (1,1))
        kp[1,5] = csdl.reshape(6*E*Iz/L**2, (1,1))
        kp[5,1] = csdl.reshape(6*E*Iz/L**2, (1,1))
        kp[2,2] = csdl.reshape(12*E*Iy/L**3, (1,1))
        kp[2,4] = csdl.reshape(-6*E*Iy/L**2, (1,1))
        kp[4,2] = csdl.reshape(-6*E*Iy/L**2, (1,1))
        kp[3,3] = csdl.reshape(G*J/L, (1,1))
        kp[4,4] = csdl.reshape(4*E*Iy/L, (1,1))
        kp[5,5] = csdl.reshape(4*E*Iz/L, (1,1))
        # the upper right block
        kp[0,6] = csdl.reshape(-A*E/L, (1,1))
        kp[1,7] = csdl.reshape(-12*E*Iz/L**3, (1,1))
        kp[1,11] = csdl.reshape(6*E*Iz/L**2, (1,1))
        kp[2,8] = csdl.reshape(-12*E*Iy/L**3, (1,1))
        kp[2,10] = csdl.reshape(-6*E*Iy/L**2, (1,1))
        kp[3,9] = csdl.reshape(-G*J/L, (1,1))
        kp[4,8] = csdl.reshape(6*E*Iy/L**2, (1,1))
        kp[4,10] = csdl.reshape(2*E*Iy/L, (1,1))
        kp[5,7] = csdl.reshape(-6*E*Iz/L**2, (1,1))
        kp[5,11] = csdl.reshape(2*E*Iz/L, (1,1))
        # the lower left block
        kp[6,0] = csdl.reshape(-A*E/L, (1,1))
        kp[7,1] = csdl.reshape(-12*E*Iz/L**3, (1,1))
        kp[7,5] = csdl.reshape(-6*E*Iz/L**2, (1,1))
        kp[8,2] = csdl.reshape(-12*E*Iy/L**3, (1,1))
        kp[8,4] = csdl.reshape(6*E*Iy/L**2, (1,1))
        kp[9,3] = csdl.reshape(-G*J/L, (1,1))
        kp[10,2] = csdl.reshape(-6*E*Iy/L**2, (1,1))
        kp[10,4] = csdl.reshape(2*E*Iy/L, (1,1))
        kp[11,1] = csdl.reshape(6*E*Iz/L**2, (1,1))
        kp[11,5] = csdl.reshape(2*E*Iz/L, (1,1))
        # the lower right block
        kp[6,6] = csdl.reshape(A*E/L, (1,1))
        kp[7,7] = csdl.reshape(12*E*Iz/L**3, (1,1))
        kp[7,11] = csdl.reshape(-6*E*Iz/L**2, (1,1))
        kp[11,7] = csdl.reshape(-6*E*Iz/L**2, (1,1))
        kp[8,8] = csdl.reshape(12*E*Iy/L**3, (1,1))
        kp[8,10] = csdl.reshape(6*E*Iy/L**2, (1,1))
        kp[10,8] = csdl.reshape(6*E*Iy/L**2, (1,1))
        kp[9,9] = csdl.reshape(G*J/L, (1,1))
        kp[10,10] = csdl.reshape(4*E*Iy/L, (1,1))
        kp[11,11] = csdl.reshape(4*E*Iz/L, (1,1))

        # transform the local stiffness to global coordinates:
        cp = (node_b - node_a)/csdl.expand(L, (3))
        ll, mm, nn = cp[0], cp[1], cp[2]
        D = (ll**2 + mm**2)**0.5

        block = self.create_output(element_name + 'block',shape=(3,3),val=0)
        block[0,0] = csdl.reshape(ll, (1,1))
        block[0,1] = csdl.reshape(mm, (1,1))
        block[0,2] = csdl.reshape(nn, (1,1))
        block[1,0] = csdl.reshape(-mm/D, (1,1))
        block[1,1] = csdl.reshape(ll/D, (1,1))
        block[2,0] = csdl.reshape(-ll*nn/D, (1,1))
        block[2,1] = csdl.reshape(-mm*nn/D, (1,1))
        block[2,2] = csdl.reshape(D, (1,1))

        T = self.create_output(element_name + 'T',shape=(12,12),val=0)
        T[0:3,0:3] = 1*block
        T[3:6,3:6] = 1*block
        T[6:9,6:9] = 1*block
        T[9:12,9:12] = 1*block

        tkt = csdl.matmat(csdl.transpose(T), csdl.matmat(kp, T))

        # expand the transformed stiffness matrix to the global dimensions:
        k = self.create_output(element_name + 'k', shape=(dim,dim), val=0)

        # parse tkt:
        k11 = tkt[0:6,0:6] # upper left
        k12 = tkt[0:6,6:12] # upper right
        k21 = tkt[6:12,0:6] # lower left
        k22 = tkt[6:12,6:12] # lower right

        # assign the four block matrices to their respective positions in k:
        node_a_index = node_index[node_dict[i]]
        node_b_index = node_index[node_dict[i + 1]]

        row_i = node_a_index*6
        row_f = node_a_index*6 + 6
        col_i = node_a_index*6
        col_f = node_a_index*6 + 6
        k[row_i:row_f, col_i:col_f] = k11

        row_i = node_a_index*6
        row_f = node_a_index*6 + 6
        col_i = node_b_index*6
        col_f = node_b_index*6 + 6
        k[row_i:row_f, col_i:col_f] = k12

        row_i = node_b_index*6
        row_f = node_b_index*6 + 6
        col_i = node_a_index*6
        col_f = node_a_index*6 + 6
        k[row_i:row_f, col_i:col_f] = k21

        row_i = node_b_index*6
        row_f = node_b_index*6 + 6
        col_i = node_b_index*6
        col_f = node_b_index*6 + 6
        k[row_i:row_f, col_i:col_f] = k22


    def global_loads(self, b_index_list, num_unique_nodes, node_dict, beams, node_index):

        nodal_loads = self.create_output('nodal_loads', shape=(len(beams), num_unique_nodes, 6), val=0)
        for i, beam_name in enumerate(beams):
            n = len(beams[beam_name]['nodes'])
            
            forces = self.declare_variable(beam_name + '_forces', shape=(n,3), val=0)
            moments = self.declare_variable(beam_name + '_moments', shape=(n,3), val=0)

            # concatenate the forces and moments:
            loads = self.create_output(f'{beam_name}_loads', shape=(n,6), val=0)
            loads[:,0:3], loads[:, 3:6] = 1*forces, 1*moments

            for j, bnode in enumerate(node_dict[beam_name]):
                for k in range(6):
                    if (node_index[bnode]*6 + k) not in b_index_list:
                        nodal_loads[i,node_index[bnode],k] = csdl.reshape(loads[j,k], (1,1,1))


        total_loads = csdl.sum(nodal_loads, axes=(0,))

        # flatten the total loads matrix to a vector:
        Fi = self.register_output('Fi', csdl.reshape(total_loads, new_shape=(6*num_unique_nodes)))
        return Fi


    def add_beam(self, beam_name, nodes, cs, e, g, rho, node_dict, node_index, dim, element_density_list):
        mesh_units = self.parameters['mesh_units']
        n = len(nodes)

        default_val = np.zeros((n, 3))
        default_val[:,1] = np.linspace(0,n,n)
        # mesh = self.declare_variable(name + '_mesh', shape=(n,3), val=default_val)
        mesh_in = self.register_module_input(beam_name + '_mesh', shape=(n,3), promotes=True, val=default_val)

        if mesh_units == 'm': mesh = 1*mesh_in
        elif mesh_units == 'ft': mesh = 0.304*mesh_in
        
        # self.print_var(mesh)
        
        # iterate over each element:
        for i in range(n - 1):
            element_name = beam_name + '_element_' + str(i)
            node_a = csdl.reshape(mesh[i, :], (3))
            node_b = csdl.reshape(mesh[i + 1, :], (3))
            self.register_output(element_name + 'node_a', node_a)
            self.register_output(element_name + 'node_b', node_b)


        if cs == 'tube':
            # t = self.declare_variable(beam_name + '_t', shape=(n-1), val=0.001)
            # r = self.declare_variable(beam_name + '_r', shape=(n-1), val=0.1)
            t = self.register_module_input(beam_name + '_t', shape=(n-1), val=0.001)
            r_in = self.register_module_input(beam_name + '_r', shape=(n-1), val=0.1)

            if mesh_units == 'm': r = r_in
            elif mesh_units == 'ft': r = 0.304*r_in

            for i in range(n - 1):
                element_name = beam_name + '_element_' + str(i)
                self.tube(element_name=element_name, t=t[i], r=r[i])


        elif cs == 'box':
            # w = self.declare_variable(beam_name + '_w', shape=(n-1))
            # h = self.declare_variable(beam_name + '_h', shape=(n-1))
            width = self.register_module_input(beam_name + '_width', shape=(n), promotes=True)
            height = self.register_module_input(beam_name + '_height', shape=(n), promotes=True)

            w = self.create_output(beam_name + '_w', shape=(n - 1), val=0)
            h = self.create_output(beam_name + '_h', shape=(n - 1), val=0)
            for i in range(n - 1):
                element_name = beam_name + '_element_' + str(i)

                if mesh_units == 'm':
                    w[i] = (width[i] + width[i + 1])/2
                    h[i] = (height[i] + height[i + 1])/2
                elif mesh_units == 'ft':
                    w[i] = 0.304*(width[i] + width[i + 1])/2
                    h[i] = 0.304*(height[i] + height[i + 1])/2

                self.register_output(element_name + '_h', 1*h[i])
                self.register_output(element_name + '_w', 1*w[i])

            #self.print_var(w)
            #self.print_var(h)

            # tweb = self.declare_variable(beam_name + '_tweb', shape=(n-1))
            # tcap = self.declare_variable(beam_name + '_tcap', shape=(n-1))
            tweb = self.register_module_input(beam_name + '_tweb', shape=(n - 1))
            tcap = self.register_module_input(beam_name + '_tcap', shape=(n - 1))

            self.print_var(tweb)
            self.print_var(tcap)

            for i in range(n - 1):
                element_name = beam_name + '_element_' + str(i)
                self.box(element_name=element_name, w=w[i], h=h[i], tweb=tweb[i], tcap=tcap[i])

                self.register_output(element_name + '_tweb', tweb[i])
                self.register_output(element_name + '_tcap', tcap[i])
        
        else: raise NotImplementedError('Error: cs type for' + beam_name + 'is not implemented')



        # calculate the stiffness matrix and the mass matrix for each element:
        for i in range(n - 1):
            element_name = beam_name + '_element_' + str(i)

            self.local_stiffness(element_name=element_name, 
                                 E=e, 
                                 G=g, 
                                 node_dict=node_dict, 
                                 node_index=node_index, 
                                 dim=dim,
                                 i=i)
            
            self.local_mass(element_name=element_name, 
                                 E=e, 
                                 G=g, 
                                 node_dict=node_dict, 
                                 node_index=node_index, 
                                 dim=dim,
                                 i=i,
                                 element_density_list=element_density_list)


    def define(self):
        beams = self.parameters['beams']
        joints = self.parameters['joints']
        bounds = self.parameters['bounds']
        mesh_units = self.parameters['mesh_units']


        if not beams: raise Exception('Error: empty beam dictionary')
        if not bounds: raise Exception('Error: no boundary conditions specified')
        

        # automated beam node assignment:
        node_dict = {}
        # start by populating the nodes dictionary as if there aren't any joints:
        index = 0
        for beam_name in beams:
            node_dict[beam_name] = np.arange(index, index + len(beams[beam_name]['nodes']))
            index += len(beams[beam_name]['nodes'])

        # assign nodal indices in the global system:
        for joint_name in joints:
            joint_beam_list = joints[joint_name]['beams']
            joint_node_list = joints[joint_name]['nodes']
            joint_node_a = node_dict[joint_beam_list[0]][joint_node_list[0]]
            for i, beam_name in enumerate(joint_beam_list):
                if i != 0: node_dict[beam_name][joint_node_list[i]] = joint_node_a



        node_set = set(node_dict[beam_name][i] for beam_name in beams for i in range(len(beams[beam_name]['nodes'])))
        num_unique_nodes = len(node_set)
        dim = num_unique_nodes*6
        node_index = {list(node_set)[i]: i for i in range(num_unique_nodes)}



        # create a list of element names:
        elements, element_density_list = [], []
        num_elements = 0
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            num_elements += n - 1
            for i in range(n - 1): 
                elements.append(beam_name + '_element_' + str(i))
                element_density_list.append(beams[beam_name]['rho'])



        for beam_name in beams:
            self.add_beam(beam_name=beam_name, 
                          nodes=beams[beam_name]['nodes'], 
                          cs=beams[beam_name]['cs'], 
                          e=beams[beam_name]['E'],
                          g=beams[beam_name]['G'],
                          rho=beams[beam_name]['rho'],
                          node_dict=node_dict[beam_name],
                          node_index=node_index,
                          dim=dim,
                          element_density_list=element_density_list)
        

        # compute the global stiffness matrix and the global mass matrix:
        helper = self.create_output('helper', shape=(num_elements,dim,dim), val=0)
        mass_helper = self.create_output('mass_helper', shape=(num_elements,dim,dim), val=0)
        for i, element_name in enumerate(elements):
            helper[i,:,:] = csdl.reshape(self.declare_variable(element_name + 'k', shape=(dim,dim)), (1,dim,dim))
            mass_helper[i,:,:] = csdl.reshape(self.declare_variable(element_name + 'element_mass_matrix', shape=(dim,dim)), (1,dim,dim))

        sum_k = csdl.sum(helper, axes=(0, ))
        sum_m = csdl.sum(mass_helper, axes=(0, ))

        b_index_list = []
        for b_name in bounds:
            fpos = bounds[b_name]['node']
            fdim = bounds[b_name]['fdim']
            b_node_index = node_index[node_dict[bounds[b_name]['beam']][fpos]]
            # add the constrained dof index to the b_index_list:
            for i, fdim in enumerate(fdim):
                if fdim == 1: b_index_list.append(b_node_index*6 + i)



        mask = self.create_output('mask', shape=(dim,dim), val=np.eye(dim))
        mask_eye = self.create_output('mask_eye', shape=(dim,dim), val=0)
        zero, one = self.create_input('zero', shape=(1,1), val=0), self.create_input('one', shape=(1,1), val=1)
        [(mask.__setitem__((i,i),1*zero), mask_eye.__setitem__((i,i),1*one)) for i in range(dim) if i in b_index_list]

        # modify the global stiffness matrix and the global mass matrix with boundary conditions:
        # first remove the row/column with a boundary condition, then add a 1:
        K = self.register_output('K', csdl.matmat(csdl.matmat(mask, sum_k), mask) + mask_eye)
        mass_matrix = self.register_output('mass_matrix', csdl.matmat(csdl.matmat(mask, sum_m), mask) + mask_eye)



        # compute the mass properties:
        self.add(MassProp(elements=elements, element_density_list=element_density_list), name='MassProp')


        Fi = self.global_loads(b_index_list=b_index_list,
                              num_unique_nodes=num_unique_nodes,
                              node_dict=node_dict,
                              beams=beams,
                              node_index=node_index)
        

        # solve the linear system
        solve_res = self.create_implicit_operation(Model(dim=dim))
        solve_res.declare_state(state='U', residual='R')
        solve_res.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=False,maxiter=100,iprint=False,atol=1E-7,)
        solve_res.linear_solver = csdl.ScipyKrylov()
        U = solve_res(K, Fi)




        # recover the local elemental forces/moments:
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            for i in range(n - 1):
                element_name = beam_name + '_element_' + str(i)
                node_a_id, node_b_id = node_index[node_dict[beam_name][i]], node_index[node_dict[beam_name][i + 1]]
                # get the nodal displacements for the current element:
                disp_a, disp_b = 1*U[node_a_id*6:node_a_id*6 + 6], 1*U[node_b_id*6:node_b_id*6 + 6]
                # concatenate the nodal displacements:
                d = self.create_output(element_name + 'd', shape=(12), val=0)
                d[0:6], d[6:12] = disp_a, disp_b
                kp = self.declare_variable(element_name + 'kp',shape=(12,12))
                T = self.declare_variable(element_name + 'T',shape=(12,12))
                # element local loads output (required for the stress recovery):
                local_loads = self.register_output(element_name + 'local_loads', csdl.matvec(kp,csdl.matvec(T,d)))

        



        # parse the displacements to get the new nodal coordinates:
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            for i in range(n - 1):
                element_name = beam_name + '_element_' + str(i)
                node_a_position = self.declare_variable(element_name + 'node_a', shape=(3))
                node_b_position = self.declare_variable(element_name + 'node_b', shape=(3))
                a, b =  node_index[node_dict[beam_name][i]], node_index[node_dict[beam_name][i + 1]]
                # get the nodal displacements for the current element:
                dn1, dn2 = U[a*6:a*6 + 3], U[b*6:b*6 + 3]
                self.register_output(element_name + 'node_a_def', node_a_position + dn1)
                self.register_output(element_name + 'node_b_def', node_b_position + dn2)


        # the displacement outputs for each beam:
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            d = self.create_output(beam_name + '_displacement', shape=(n,3), val=0)
            for i in range(n - 1):
                element_name = beam_name + '_element_' + str(i)
                a, b =  node_index[node_dict[beam_name][i]], node_index[node_dict[beam_name][i + 1]]
                dna, dnb = U[a*6:a*6 + 3], U[b*6:b*6 + 3]
                d[i,:] = csdl.reshape(dna, (1,3))
            d[n - 1,:] = csdl.reshape(dnb, (1,3))



        # get the rotations:
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])

            # define the axis-wise unit vectors:
            ex = self.create_input('ex', shape=(3), val=[1,0,0])
            ey = self.create_input('ey', shape=(3), val=[0,1,0])
            ez = self.create_input('ez', shape=(3), val=[0,0,1])

            r = self.create_output(beam_name + 'r', shape=(n - 1,3), val=0)
            for i in range(n - 1):
                element_name = beam_name + '_element_' + str(i)
                node_a = self.declare_variable(element_name + 'node_a_def', shape=(3))
                node_b = self.declare_variable(element_name + 'node_b_def', shape=(3))
                v = node_b - node_a
                mag = csdl.pnorm(v)

                r[i,0] = csdl.reshape(csdl.arccos(csdl.dot(v, ex)/mag), (1,1))
                r[i,1] = csdl.reshape(csdl.arccos(csdl.dot(v, ey)/mag), (1,1))
                r[i,2] = csdl.reshape(csdl.arccos(csdl.dot(v, ez)/mag), (1,1))



        # perform a stress recovery:
        boxflag = False
        for beam_name in beams:
            if beams[beam_name]['cs'] == 'box': boxflag = True

        if boxflag: new_stress = self.create_output('new_stress', shape=(len(elements),5), val=0)

        stress = self.create_output('vm_stress', shape=(len(elements)), val=0)

        index = 0
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])

            if beams[beam_name]['cs'] == 'tube':
                for i in range(n - 1):
                    element_name = beam_name + '_element_' + str(i)
                    self.add(StressTube(name=element_name), name=element_name + 'StressTube')

                    stress[index] = self.declare_variable(element_name + '_stress')
                    index += 1

            elif beams[beam_name]['cs'] == 'box':
                for i in range(n - 1):
                    element_name = beam_name + '_element_' + str(i)
                    self.add(StressBox(name=element_name), name=element_name + 'StressBox')
                    
                    stress[index] = self.declare_variable(element_name + '_stress')
                    new_stress[index,:] = csdl.reshape(self.declare_variable(element_name + '_stress_array', shape=(5)), new_shape=(1,5)) # no ks max

                    index += 1



        # buckling:
        bkl = self.create_output('bkl', shape=(len(elements)))
        index = 0
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            Modulus = beams[beam_name]['E']
            if beams[beam_name]['cs'] == 'box':
                for i in range(n - 1):
                    element_name = beam_name + '_element_' + str(i)

                    self.add(Buckle(element_name=element_name,E=Modulus), name=element_name + 'Buckle')
                    bkl_ratio = self.declare_variable(element_name + 'bkl_ratio')
                    bkl[index] = 1*bkl_ratio

                    index += 1




        
        # output dummy forces and moments for CADDEE:
        zero = self.declare_variable('zero_vec', shape=(3), val=0)
        self.register_output('F', 1*zero)
        self.register_output('M', 1*zero)





class AframeResidual(Aframe):
    def add_beam(self, beam_name, nodes, cs, e, g, rho, node_dict, node_index, dim, element_density_list):
        n = len(nodes)

        default_val = np.zeros((n, 3))
        default_val[:,1] = np.linspace(0,n,n)
        # mesh = self.declare_variable(name + '_mesh', shape=(n,3), val=default_val)
        mesh = self.register_module_input(beam_name + '_mesh', shape=(n,3), promotes=True, val=default_val)
        
        # iterate over each element:
        for i in range(n - 1):
            element_name = beam_name + '_element_' + str(i)
            node_a = csdl.reshape(mesh[i, :], (3))
            node_b = csdl.reshape(mesh[i + 1, :], (3))
            self.register_output(element_name + 'node_a', node_a)
            self.register_output(element_name + 'node_b', node_b)


        if cs == 'tube':
            # t = self.declare_variable(beam_name + '_t', shape=(n-1), val=0.001)
            # r = self.declare_variable(beam_name + '_r', shape=(n-1), val=0.1)
            t = self.register_module_input(beam_name + '_t', shape=(n-1), val=0.001)
            r = self.register_module_input(beam_name + '_r', shape=(n-1), val=0.1)

            for i in range(n - 1):
                element_name = beam_name + '_element_' + str(i)
                self.tube(element_name=element_name, t=t[i], r=r[i])


        elif cs == 'box':
            # w = self.declare_variable(beam_name + '_w', shape=(n-1))
            # h = self.declare_variable(beam_name + '_h', shape=(n-1))
            w = self.register_module_input(beam_name + '_w', shape=(n - 1), promotes=True)
            h = self.register_module_input(beam_name + '_h', shape=(n - 1), promotes=True)
            # tweb = self.declare_variable(beam_name + '_tweb', shape=(n-1))
            # tcap = self.declare_variable(beam_name + '_tcap', shape=(n-1))
            tweb = self.register_module_input(beam_name + '_tweb', shape=(n - 1))
            tcap = self.register_module_input(beam_name + '_tcap', shape=(n - 1))

            for i in range(n - 1):
                element_name = beam_name + '_element_' + str(i)
                self.box(element_name=element_name, w=w[i], h=h[i], tweb=tweb[i], tcap=tcap[i])
        
        else: raise NotImplementedError('Error: cs type for' + beam_name + 'is not implemented')



        # calculate the stiffness matrix and the mass matrix for each element:
        for i in range(n - 1):
            element_name = beam_name + '_element_' + str(i)

            self.local_stiffness(element_name=element_name, 
                                 E=e, 
                                 G=g, 
                                 node_dict=node_dict, 
                                 node_index=node_index, 
                                 dim=dim,
                                 i=i)
            
            self.lumped_local_mass(element_name=element_name, 
                                 E=e, 
                                 G=g, 
                                 node_dict=node_dict, 
                                 node_index=node_index, 
                                 dim=dim,
                                 i=i,
                                 element_density_list=element_density_list)
            
    def define(self):
        beams = self.parameters['beams']
        joints = self.parameters['joints']
        bounds = self.parameters['bounds']


        if not beams: raise Exception('Error: empty beam dictionary')
        if not bounds: raise Exception('Error: no boundary conditions specified')
        

        # automated beam node assignment:
        node_dict = {}
        # start by populating the nodes dictionary as if there aren't any joints:
        index = 0
        for beam_name in beams:
            node_dict[beam_name] = np.arange(index, index + len(beams[beam_name]['nodes']))
            index += len(beams[beam_name]['nodes'])

        # assign nodal indices in the global system:
        for joint_name in joints:
            joint_beam_list = joints[joint_name]['beams']
            joint_node_list = joints[joint_name]['nodes']
            joint_node_a = node_dict[joint_beam_list[0]][joint_node_list[0]]
            for i, beam_name in enumerate(joint_beam_list):
                if i != 0: node_dict[beam_name][joint_node_list[i]] = joint_node_a



        node_set = set(node_dict[beam_name][i] for beam_name in beams for i in range(len(beams[beam_name]['nodes'])))
        num_unique_nodes = len(node_set)
        dim = num_unique_nodes*6
        node_index = {list(node_set)[i]: i for i in range(num_unique_nodes)}



        # create a list of element names:
        elements, element_density_list = [], []
        num_elements = 0
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            num_elements += n - 1
            for i in range(n - 1): 
                elements.append(beam_name + '_element_' + str(i))
                element_density_list.append(beams[beam_name]['rho'])



        for beam_name in beams:
            self.add_beam(beam_name=beam_name, 
                          nodes=beams[beam_name]['nodes'], 
                          cs=beams[beam_name]['cs'], 
                          e=beams[beam_name]['E'],
                          g=beams[beam_name]['G'],
                          rho=beams[beam_name]['rho'],
                          node_dict=node_dict[beam_name],
                          node_index=node_index,
                          dim=dim,
                          element_density_list=element_density_list)


        # compute the global stiffness matrix and the global mass matrix:
        helper = self.create_output('helper', shape=(num_elements,dim,dim), val=0)
        mass_helper = self.create_output('mass_helper', shape=(num_elements,dim,dim), val=0)
        for i, element_name in enumerate(elements):
            helper[i,:,:] = csdl.reshape(self.declare_variable(element_name + 'k', shape=(dim,dim)), (1,dim,dim))
            mass_helper[i,:,:] = csdl.reshape(self.declare_variable(element_name + 'element_mass_matrix', shape=(dim,dim)), (1,dim,dim))

        sum_k = csdl.sum(helper, axes=(0, ))
        sum_m = csdl.sum(mass_helper, axes=(0, ))

        b_index_list = []
        for b_name in bounds:
            fpos = bounds[b_name]['node']
            fdim = bounds[b_name]['fdim']
            b_node_index = node_index[node_dict[bounds[b_name]['beam']][fpos]]
            # add the constrained dof index to the b_index_list:
            for i, fdim in enumerate(fdim):
                if fdim == 1: b_index_list.append(b_node_index*6 + i)



        mask = self.create_output('mask', shape=(dim,dim), val=np.eye(dim))
        mask_eye = self.create_output('mask_eye', shape=(dim,dim), val=0)
        zero, one = self.create_input('zero', shape=(1,1), val=0), self.create_input('one', shape=(1,1), val=1)
        [(mask.__setitem__((i,i),1*zero), mask_eye.__setitem__((i,i),1*one)) for i in range(dim) if i in b_index_list]

        # modify the global stiffness matrix and the global mass matrix with boundary conditions:
        # first remove the row/column with a boundary condition, then add a 1:
        K = self.register_output('K', csdl.matmat(csdl.matmat(mask, sum_k), mask) + mask_eye)
        mass_matrix = self.register_output('mass_matrix', csdl.matmat(csdl.matmat(mask, sum_m), mask) + mask_eye)

        # compute inverse mass matrix
        mass_matrix_inverse = self.create_output('mass_matrix_inverse', shape=mass_matrix.shape)
        for i in range(mass_matrix.shape[0]):
            mass_matrix_inverse[i,i] = 1/mass_matrix[i,i]


        # compute the mass properties:
        self.add(MassProp(elements=elements, element_density_list=element_density_list), name='MassProp')


        Fi = self.global_loads(b_index_list=b_index_list,
                              num_unique_nodes=num_unique_nodes,
                              node_dict=node_dict,
                              beams=beams,
                              node_index=node_index)
        
        F_ssr = self.create_output('F_extended', shape = (2*dim,))
        for i in range(dim):
            F_ssr[i] = Fi[i]
            F_ssr[i+dim] = 0
        A_ssr = self.create_output('A_ssr', shape = (2*dim,2*dim))
        for i in range(dim):
            for j in range(dim):
                A_ssr[i,j] = 0
                A_ssr[i+dim,j+dim] = 0
                if i == j:
                    A_ssr[i+dim,j] = 1
                else:
                    A_ssr[i+dim,j] = 0
        A_ssr[0:dim,dim:2*dim] = -csdl.matmat(K*mass_matrix_inverse)

        delta = self.create_output('delta', shape = (2*dim,))
        delta[0:dim] = self.declare_variable('velocities', shape=(dim,), val=0)
        delta[dim:2*dim] = self.declare_variable('displacements', shape=(dim,), val=0)

        residual = self.register_output('residual', F_ssr + csdl.matvec(A_ssr, delta))


class AframeResidualJacobian(AframeResidual):
    def define(self):
        beams = self.parameters['beams']
        joints = self.parameters['joints']
        bounds = self.parameters['bounds']


        if not beams: raise Exception('Error: empty beam dictionary')
        if not bounds: raise Exception('Error: no boundary conditions specified')
        

        # automated beam node assignment:
        node_dict = {}
        # start by populating the nodes dictionary as if there aren't any joints:
        index = 0
        for beam_name in beams:
            node_dict[beam_name] = np.arange(index, index + len(beams[beam_name]['nodes']))
            index += len(beams[beam_name]['nodes'])

        # assign nodal indices in the global system:
        for joint_name in joints:
            joint_beam_list = joints[joint_name]['beams']
            joint_node_list = joints[joint_name]['nodes']
            joint_node_a = node_dict[joint_beam_list[0]][joint_node_list[0]]
            for i, beam_name in enumerate(joint_beam_list):
                if i != 0: node_dict[beam_name][joint_node_list[i]] = joint_node_a



        node_set = set(node_dict[beam_name][i] for beam_name in beams for i in range(len(beams[beam_name]['nodes'])))
        num_unique_nodes = len(node_set)
        dim = num_unique_nodes*6
        node_index = {list(node_set)[i]: i for i in range(num_unique_nodes)}



        # create a list of element names:
        elements, element_density_list = [], []
        num_elements = 0
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            num_elements += n - 1
            for i in range(n - 1): 
                elements.append(beam_name + '_element_' + str(i))
                element_density_list.append(beams[beam_name]['rho'])



        for beam_name in beams:
            self.add_beam(beam_name=beam_name, 
                          nodes=beams[beam_name]['nodes'], 
                          cs=beams[beam_name]['cs'], 
                          e=beams[beam_name]['E'],
                          g=beams[beam_name]['G'],
                          rho=beams[beam_name]['rho'],
                          node_dict=node_dict[beam_name],
                          node_index=node_index,
                          dim=dim,
                          element_density_list=element_density_list)


        # compute the global stiffness matrix and the global mass matrix:
        helper = self.create_output('helper', shape=(num_elements,dim,dim), val=0)
        mass_helper = self.create_output('mass_helper', shape=(num_elements,dim,dim), val=0)
        for i, element_name in enumerate(elements):
            helper[i,:,:] = csdl.reshape(self.declare_variable(element_name + 'k', shape=(dim,dim)), (1,dim,dim))
            mass_helper[i,:,:] = csdl.reshape(self.declare_variable(element_name + 'element_mass_matrix', shape=(dim,dim)), (1,dim,dim))

        sum_k = csdl.sum(helper, axes=(0, ))
        sum_m = csdl.sum(mass_helper, axes=(0, ))

        b_index_list = []
        for b_name in bounds:
            fpos = bounds[b_name]['node']
            fdim = bounds[b_name]['fdim']
            b_node_index = node_index[node_dict[bounds[b_name]['beam']][fpos]]
            # add the constrained dof index to the b_index_list:
            for i, fdim in enumerate(fdim):
                if fdim == 1: b_index_list.append(b_node_index*6 + i)



        mask = self.create_output('mask', shape=(dim,dim), val=np.eye(dim))
        mask_eye = self.create_output('mask_eye', shape=(dim,dim), val=0)
        zero, one = self.create_input('zero', shape=(1,1), val=0), self.create_input('one', shape=(1,1), val=1)
        [(mask.__setitem__((i,i),1*zero), mask_eye.__setitem__((i,i),1*one)) for i in range(dim) if i in b_index_list]

        # modify the global stiffness matrix and the global mass matrix with boundary conditions:
        # first remove the row/column with a boundary condition, then add a 1:
        K = self.register_output('K', csdl.matmat(csdl.matmat(mask, sum_k), mask) + mask_eye)
        mass_matrix = self.register_output('mass_matrix', csdl.matmat(csdl.matmat(mask, sum_m), mask) + mask_eye)
        
        # compute inverse mass matrix
        mass_matrix_inverse = self.create_output('mass_matrix_inverse', shape=mass_matrix.shape)
        for i in range(mass_matrix.shape[0]):
            mass_matrix_inverse[i,i] = 1/mass_matrix[i,i]


        # compute the mass properties:
        self.add(MassProp(elements=elements, element_density_list=element_density_list), name='MassProp')


        Fi = self.global_loads(b_index_list=b_index_list,
                              num_unique_nodes=num_unique_nodes,
                              node_dict=node_dict,
                              beams=beams,
                              node_index=node_index)
        
        # TODO: fix the below and do it properly
        zeros = self.declare_variable('zeros', val=np.zeros((dim,dim)), shape=(dim,dim))
        eye = self.declare_variable('eye', val=np.eye(dim), shape=(dim,dim))

        F_ssr = self.create_output('F_extended', shape = (2*dim,), val=0)
        for i in range(dim):
            F_ssr[i] = Fi[i]
            #F_ssr[i+dim] = zero
        A_ssr = self.create_output('displacement_jacobian', shape = (2*dim,2*dim))
        # for i in range(dim):
        #     for j in range(dim):
        #         A_ssr[i,j] = zero
        #         A_ssr[i+dim,j+dim] = zero
        #         if i == j:
        #             A_ssr[i+dim,j] = one
        #         else:
        #             A_ssr[i+dim,j] = zero
        A_ssr[0:dim,0:dim] = 1*zeros
        A_ssr[dim:2*dim,dim:2*dim] = 1*zeros
        A_ssr[dim:2*dim,0:dim] = eye
        A_ssr[0:dim,dim:2*dim] = -csdl.matmat(K,mass_matrix_inverse)

        # delta = self.create_output('delta', shape = (2*dim,))
        # delta[0:dim] = self.declare_variable('velocities', shape=(dim,), val=0)
        # delta[dim:2*dim] = self.declare_variable('displacements', shape=(dim,), val=0)

        # self.register_output('output_residual_jacobian', A_ssr)
        # TODO: add input_residual_jacobian





if __name__ == '__main__':

    beams, bounds, joints = {}, {}, {}
    beams['wing'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(10))}
    beams['boom'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(10))}
    joints['joint'] = {'beams': ['wing', 'boom'],'nodes': [4, 4]}
    bounds['root'] = {'beam': 'wing','node': 0,'fdim': [1,1,1,1,1,1]}

    sim = python_csdl_backend.Simulator(Aframe(beams=beams, joints=joints, bounds=bounds))

    f = np.zeros((10,3))
    f[:,2] = 100
    sim['wing_forces'] = f

    sim.run()



    # plotting:
    import matplotlib.pyplot as plt
    plt.rcParams.update(plt.rcParamsDefault)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for beam_name in beams:
        n = len(beams[beam_name]['nodes'])
        for i in range(n - 1):
            element_name = beam_name + '_element_' + str(i)
            na = sim[element_name+'node_a_def']
            nb = sim[element_name+'node_b_def']

            x = np.array([na[0], nb[0]])
            y = np.array([na[1], nb[1]])
            z = np.array([na[2], nb[2]])

            ax.plot(x,y,z,color='k',label='_nolegend_',linewidth=2)
            ax.scatter(na[0], na[1], na[2],color='yellow',edgecolors='black',linewidth=1,zorder=10,label='_nolegend_',s=30)
            ax.scatter(nb[0], nb[1], nb[2],color='yellow',edgecolors='black',linewidth=1,zorder=10,label='_nolegend_',s=30)


    ax.set_xlim(-1,1)
    ax.set_ylim(0,10)
    ax.set_zlim(-1,1)
    plt.show()