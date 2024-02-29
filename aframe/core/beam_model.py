import numpy as np
import csdl
from aframe.core.dataclass import Beam, CSProp

class BeamModel(csdl.Model):

    def initialize(self):
        self.beams = self.parameters.declare('beams', default=[])
        self.parameters.declare('boundary_conditions', default=[])
        self.parameters.declare('joints', default=[])


    def stiffness(self, beam, mesh, cs, dimension, node_dictionary, index):

        # iterate over the beam elements
        beam_stiffness = 0
        for i in range(beam.num_nodes - 1):
            E, G = beam.material.E, beam.material.G

            L = csdl.pnorm(mesh[i + 1, :] - mesh[i, :])

            A, Ix, Iy, Iz, J, Q = cs.A[i], cs.Ix[i], cs.Iy[i], cs.Iz[i], cs.J[i], cs.Q[i]

            kp = self.create_output(beam.name + str(i) + 'kp', shape=(12,12), val=0)
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
            cp = (mesh[i + 1, :] - mesh[i, :]) / csdl.expand(L, (1, 3))
            ll, mm, nn = cp[0, 0], cp[0, 1], cp[0, 2]
            D = (ll**2 + mm**2)**0.5

            block = self.create_output(beam.name + str(i) + 'block', shape=(3,3), val=0)
            block[0,0] = ll
            block[0,1] = mm
            block[0,2] = nn
            block[1,0] = -mm/D
            block[1,1] = ll/D
            block[2,0] = -ll*nn/D
            block[2,1] = -mm*nn/D
            block[2,2] = D

            T = self.create_output(beam.name + str(i) + 'T', shape=(12,12), val=0)
            T[0:3,0:3], T[3:6,3:6], T[6:9,6:9], T[9:12,9:12] = 1*block, 1*block, 1*block, 1*block

            tkt = csdl.matmat(csdl.transpose(T), csdl.matmat(kp, T))
            k11, k12, k21, k22 = tkt[0:6,0:6], tkt[0:6,6:12], tkt[6:12,0:6], tkt[6:12,6:12]

            # expand the transformed stiffness matrix to the global dimensions:
            k = self.create_output(beam.name + str(i) + 'k', shape=(dimension, dimension), val=0)

            # assign the four block matrices to their respective positions in k:
            node_a_index = index[node_dictionary[beam.name][i]]
            node_b_index = index[node_dictionary[beam.name][i + 1]]

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

            beam_stiffness = beam_stiffness + k

        return beam_stiffness


    def define(self):
        beams = self.parameters['beams']
        boundary_conditions = self.parameters['boundary_conditions']
        joints = self.parameters['joints']


        # automated beam node assignment
        node_dictionary = {}
        # start by populating the nodes dictionary as if there aren't any joints
        val = 0
        for beam in beams:
            node_dictionary[beam.name] = np.arange(val, val + beam.num_nodes)
            val += beam.num_nodes

        # assign nodal indices in the global system using the joints
        for joint in joints:
            joint_beams, joint_nodes = joint.beams, joint.nodes
            node_a = node_dictionary[joint_beams[0]][joint_nodes[0]]
            for i, beam in enumerate(joint_beams):
                if i != 0: node_dictionary[beam.name][joint_nodes[i]] = node_a

        node_set = set(node_dictionary[beam.name][i] for beam in beams for i in range(beam.num_nodes))
        num_unique_nodes = len(node_set)
        dimension = num_unique_nodes * 6
        index = {list(node_set)[i]: i for i in range(num_unique_nodes)}


        # construct the stiffness matrix
        global_stiffness_matrix = 0
        for beam in beams:
            mesh = self.declare_variable(beam.name + '_mesh', shape=(beam.num_nodes, 3))

            if beam.cs == 'tube':
                radius = self.declare_variable(beam.name + '_radius', shape=(beam.num_nodes))
                thickness = self.declare_variable(beam.name + '_thickness', shape=(beam.num_nodes))
                inner_radius, outer_radius = radius - thickness, radius

                A = np.pi * (outer_radius**2 - inner_radius**2)
                Iy = np.pi * (outer_radius**4 - inner_radius**4) / 4.0
                Iz = np.pi * (outer_radius**4 - inner_radius**4) / 4.0
                Ix = J = Q = np.pi * (outer_radius**4 - inner_radius**4) / 2.0
                cs = CSProp(A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, Q=Q)

                beam_stiffness = self.stiffness(beam=beam, mesh=mesh, cs=cs, dimension=dimension, node_dictionary=node_dictionary, index=index)
                global_stiffness_matrix = global_stiffness_matrix + csdl.reshape(beam_stiffness, (dimension, dimension))


        # self.print_var(global_stiffness_matrix)
                
        # deal with the boundary conditions
        bound_node_index_list = []
        for bound in boundary_conditions:
            bound_node, dof = bound.node, bound.dof
            bound_node_index = index[node_dictionary[bound.beam.name][bound_node]]
            # add the constrained dof index to the bound_node_index_list
            for i, degree in enumerate(dof):
                if degree: bound_node_index_list.append(bound_node_index*6 + i)

        mask, mask_eye = self.create_output('mask', shape=(dimension, dimension), val=np.eye(dimension)), self.create_output('mask_eye', shape=(dimension, dimension), val=0)
        zero, one = self.create_input('zero', shape=(1, 1), val=0), self.create_input('one', shape=(1, 1), val=1)
        [(mask.__setitem__((i,i),1*zero), mask_eye.__setitem__((i,i),1*one)) for i in range(dimension) if i in bound_node_index_list]

        # modify the global stiffness matrix with boundary conditions
        # first remove the row/column with a boundary condition, then add a 1
        K = csdl.matmat(csdl.matmat(mask, global_stiffness_matrix), mask) + mask_eye