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
        local_stiffness = self.create_output(beam.name + '_local_stiffness', shape=(beam.num_elements, 12, 12))
        transformations = self.create_output(beam.name + '_transformations', shape=(beam.num_elements, 12, 12))
        for i in range(beam.num_elements):
            E, G = beam.material.E, beam.material.G

            L = csdl.pnorm(mesh[i + 1, :] - mesh[i, :])

            A, Ix, Iy, Iz, J, Q = cs.A[i], cs.Ix[i], cs.Iy[i], cs.Iz[i], cs.J[i], cs.Q[i]

            kp = self.create_output(beam.name + str(i) + 'kp', shape=(12, 12), val=0)
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

            local_stiffness[i, :, :] = csdl.reshape(kp, (1, 12, 12))

            # transform the local stiffness to global coordinates:
            cp = (mesh[i + 1, :] - mesh[i, :]) / csdl.expand(L, (1, 3))
            ll, mm, nn = cp[0, 0], cp[0, 1], cp[0, 2]
            D = (ll**2 + mm**2)**0.5

            block = self.create_output(beam.name + str(i) + '_block', shape=(3, 3), val=0)
            block[0,0] = ll
            block[0,1] = mm
            block[0,2] = nn
            block[1,0] = -mm/D
            block[1,1] = ll/D
            block[2,0] = -ll*nn/D
            block[2,1] = -mm*nn/D
            block[2,2] = D

            T = self.create_output(beam.name + str(i) + '_T', shape=(12, 12), val=0)
            T[0:3,0:3], T[3:6,3:6], T[6:9,6:9], T[9:12,9:12] = 1*block, 1*block, 1*block, 1*block
            transformations[i, :, :] = csdl.reshape(T, (1, 12, 12))

            tkt = csdl.matmat(csdl.transpose(T), csdl.matmat(kp, T))
            k11, k12, k21, k22 = tkt[0:6,0:6], tkt[0:6,6:12], tkt[6:12,0:6], tkt[6:12,6:12]

            # expand the transformed stiffness matrix to the global dimensions:
            k = self.create_output(beam.name + str(i) + 'k', shape=(dimension, dimension), val=0)

            # assign the four block matrices to their respective positions in k:
            node_a_index, node_b_index = index[node_dictionary[beam.name][i]], index[node_dictionary[beam.name][i + 1]]

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

        return beam_stiffness, local_stiffness, transformations
    


    def mass_properties(self, beam, mesh, cs):
        rho = beam.material.rho

        beam_mass, beam_rmvec = 0, 0
        for i in range(beam.num_elements):
            A = cs.A[i]
            L = csdl.pnorm(mesh[i + 1, :] - mesh[i, :])
            element_mass = A * L * rho
            element_cg = (mesh[i + 1, :] + mesh[i, :]) / 2

            beam_mass = beam_mass + element_mass
            beam_rmvec = beam_rmvec + element_cg * csdl.expand(element_mass, (1, 3))

        return beam_mass, beam_rmvec



    def define(self):
        beams = self.parameters['beams']
        boundary_conditions = self.parameters['boundary_conditions']
        joints = self.parameters['joints']


        # automated beam node assignment
        node_dictionary = {}
        # start by populating the nodes dictionary as if there aren't any joints
        val, num_elements = 0, 0
        for beam in beams:
            node_dictionary[beam.name] = np.arange(val, val + beam.num_nodes)
            val += beam.num_nodes
            num_elements += beam.num_elements

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




        # construct the stiffness matrix and the mass properties
        global_stiffness_matrix, mass, rmvec, element_index = 0, 0, 0, 0
        local_stiffness_storage = self.create_output('local_stiffness_storage', shape=(num_elements, 12, 12), val=0)
        transformations_storage = self.create_output('transformations_storage', shape=(num_elements, 12, 12), val=0)
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

            beam_stiffness, local_stiffness, transformations = self.stiffness(beam=beam, mesh=mesh, cs=cs, dimension=dimension, node_dictionary=node_dictionary, index=index)
            global_stiffness_matrix = global_stiffness_matrix + csdl.reshape(beam_stiffness, (dimension, dimension))
            local_stiffness_storage[element_index:element_index + beam.num_nodes - 1, :, :] = local_stiffness
            transformations_storage[element_index:element_index + beam.num_nodes - 1, :, :] = local_stiffness

            beam_mass, beam_rmvec = self.mass_properties(beam, mesh, cs)
            mass = mass + beam_mass
            rmvec = rmvec + beam_rmvec

            element_index += beam.num_elements

        undeformed_cg = self.register_output('undeformed_cg', rmvec / csdl.expand(mass, (1, 3)))



                
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
        [(mask.__setitem__((i, i), 1*zero), mask_eye.__setitem__((i, i), 1*one)) for i in range(dimension) if i in bound_node_index_list]

        # modify the global stiffness matrix with boundary conditions
        # first remove the row/column with a boundary condition, then add a 1
        K = csdl.matmat(csdl.matmat(mask, global_stiffness_matrix), mask) + mask_eye






        # create the global loads vector
        nodal_loads = self.create_output('nodal_loads', shape=(len(beams), num_unique_nodes, 6), val=0)
        for i, beam in enumerate(beams):

            forces = self.declare_variable(beam.name + '_forces', shape=(beam.num_nodes, 3), val=0)
            moments = self.declare_variable(beam.name + '_moments', shape=(beam.num_nodes, 3), val=0)

            beam_loads = self.create_output(beam.name + '_loads', shape=(beam.num_nodes, 6), val=0)
            beam_loads[:, 0:3], beam_loads[:, 3:6] = 1*forces, 1*moments

            for j, node in enumerate(node_dictionary[beam.name]):
                for k in range(6):
                    if (index[node]*6 + k) not in bound_node_index_list:
                        nodal_loads[i, index[node], k] = csdl.reshape(beam_loads[j, k], (1, 1, 1))

        loads = csdl.sum(nodal_loads, axes=(0, ))
        F = self.register_output('F', csdl.reshape(loads, new_shape=(6*num_unique_nodes))) # flatten loads to a vector




        # solve the linear system
        U = csdl.solve(K, F)






        # parse the displacements to get the deformed mesh
        for beam in beams:
            mesh = self.declare_variable(beam.name + '_mesh', shape=(beam.num_nodes, 3))
            deformed_mesh = self.create_output(beam.name + '_deformed_mesh', shape=(beam.num_nodes, 3), val=0)

            for i in range(beam.num_nodes):
                node_id = index[node_dictionary[beam.name][i]]
                deformed_mesh[i, :] = mesh[i, :] + csdl.reshape(U[node_id*6:node_id*6 + 3], (1, 3))








        # recover the elemental forces/moments
        element_index = 0
        for beam in beams:
            # element_loads = self.create_output(beam.name + '_element_loads', shape=(beam.num_nodes - 1, 6), val=0)
            local_stiffness = local_stiffness_storage[element_index:element_index + beam.num_nodes - 1, :, :]
            transformations = transformations_storage[element_index:element_index + beam.num_nodes - 1, :, :]
            element_index += beam.num_elements

            for i in range(beam.num_elements):
                node_a_id, node_b_id = index[node_dictionary[beam.name][i]], index[node_dictionary[beam.name][i + 1]]

                d = self.create_output(beam.name + str(i) + '_d', shape=(12), val=0)
                d[0:6], d[6:12] = 1*U[node_a_id*6:node_a_id*6 + 6], 1*U[node_b_id*6:node_b_id*6 + 6]
                kp = csdl.reshape(local_stiffness[i, :, :], (12, 12))
                T = csdl.reshape(transformations[i, :, :], (12, 12))

                # element local loads output (required for the stress recovery)
                element_loads = csdl.matvec(kp, csdl.matvec(T, d))
                # self.register_output(element_name + 'local_loads', element_loads)
                # element_loads[i,:] = csdl.reshape(element_loads[0:6], (1,6))



