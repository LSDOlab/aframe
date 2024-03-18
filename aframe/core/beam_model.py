import numpy as np
import csdl
from aframe.core.dataclass import Beam, CSProp, CSPropTube, CSPropBox, CSPropEllipse

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
        # print('shapes in the beam code')
        # print('beam_stiffness',beam_stiffness.shape)
        # print('local_stiffness',local_stiffness.shape)
        # print('transformations',transformations.shape)

        for i in range(beam.num_elements):
            E, G = beam.material.E, beam.material.G

            L = csdl.pnorm(mesh[i + 1, :] - mesh[i, :])

            A, Iy, Iz, J = cs.A[i], cs.Iy[i], cs.Iz[i], cs.J[i]

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
    


    def mass_matrix(self, beam, mesh, cs, dimension, node_dictionary, index):

        transformations = self.create_output(beam.name + '_mass_transformations', shape=(beam.num_elements, 12, 12))
        beam_mass = 0
        for i in range(beam.num_elements):
            E, G, rho = beam.material.E, beam.material.G, beam.material.rho
            L = csdl.pnorm(mesh[i + 1, :] - mesh[i, :])
            A, Iy, Iz, J = cs.A[i], cs.Iy[i], cs.Iz[i], cs.J[i]

            a = L / 2
            coef = rho * A * a / 105
            rx2 = J / A # Ix / A

            mp = self.create_output(beam.name + str(i) + 'mp', shape=(12, 12), val=0)
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
            cp = (mesh[i + 1, :] - mesh[i, :]) / csdl.expand(L, (1, 3))
            ll, mm, nn = cp[0, 0], cp[0, 1], cp[0, 2]
            D = (ll**2 + mm**2)**0.5

            block = self.create_output(beam.name + str(i) + '_mass_block', shape=(3, 3), val=0)
            block[0,0] = ll
            block[0,1] = mm
            block[0,2] = nn
            block[1,0] = -mm/D
            block[1,1] = ll/D
            block[2,0] = -ll*nn/D
            block[2,1] = -mm*nn/D
            block[2,2] = D

            T = self.create_output(beam.name + str(i) + '_mass_T', shape=(12, 12), val=0)
            T[0:3,0:3], T[3:6,3:6], T[6:9,6:9], T[9:12,9:12] = 1*block, 1*block, 1*block, 1*block
            transformations[i, :, :] = csdl.reshape(T, (1, 12, 12))

            tmt = csdl.matmat(csdl.transpose(T), csdl.matmat(mp, T))
            m11, m12, m21, m22 = tmt[0:6,0:6], tmt[0:6,6:12], tmt[6:12,0:6], tmt[6:12,6:12]

            # expand the transformed stiffness matrix to the global dimensions:
            m = self.create_output(beam.name + str(i) + 'm', shape=(dimension, dimension), val=0)

            # assign the four block matrices to their respective positions in m:
            node_a_index, node_b_index = index[node_dictionary[beam.name][i]], index[node_dictionary[beam.name][i + 1]]

            row_i = node_a_index*6
            row_f = node_a_index*6 + 6
            col_i = node_a_index*6
            col_f = node_a_index*6 + 6
            m[row_i:row_f, col_i:col_f] = m11

            row_i = node_a_index*6
            row_f = node_a_index*6 + 6
            col_i = node_b_index*6
            col_f = node_b_index*6 + 6
            m[row_i:row_f, col_i:col_f] = m12

            row_i = node_b_index*6
            row_f = node_b_index*6 + 6
            col_i = node_a_index*6
            col_f = node_a_index*6 + 6
            m[row_i:row_f, col_i:col_f] = m21

            row_i = node_b_index*6
            row_f = node_b_index*6 + 6
            col_i = node_b_index*6
            col_f = node_b_index*6 + 6
            m[row_i:row_f, col_i:col_f] = m22

            beam_mass = beam_mass + m

        return beam_mass

    


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


        # throw an error if there are empty beam or boundary condition lists
        if not beams: raise Exception('Aframe error: empty beam list')
        if not boundary_conditions: raise Exception('Aframe error: no boundary conditions specified')


        # automated beam node assignment
        node_dictionary = {}
        # start by populating the nodes dictionary as if there aren't any joints
        val = 0
        for beam in beams:
            node_dictionary[beam.name] = np.arange(val, val + beam.num_nodes)
            val += beam.num_nodes

        # assign nodal indices in the global system using the joints (remove duplicate assignment for joints)
        for joint in joints:
            joint_beams, joint_nodes = joint.beams, joint.nodes
            node_a = node_dictionary[joint_beams[0].name][joint_nodes[0]]
            for i, beam in enumerate(joint_beams):
                if i != 0: node_dictionary[beam.name][joint_nodes[i]] = node_a

        # helpers
        node_set = set(node_dictionary[beam.name][i] for beam in beams for i in range(beam.num_nodes))
        num_unique_nodes = len(node_set)
        dimension = num_unique_nodes * 6
        index = {list(node_set)[i]: i for i in range(num_unique_nodes)}


        # construct the stiffness matrix and get the undeformed mass properties
        global_stiffness_matrix, global_mass_matrix, mass, rmvec = 0, 0, 0, 0
        cs_storage, mesh_storage, transformations_storage, local_stiffness_storage = [], [], [], []
        for beam in beams:
            mesh = self.declare_variable(beam.name + '_mesh', shape=(beam.num_nodes, 3))
            mesh_storage.append(mesh)

            if beam.cs == 'tube':
                radius = self.declare_variable(beam.name + '_radius', shape=(beam.num_elements))
                thickness = self.declare_variable(beam.name + '_thickness', shape=(beam.num_elements))

                inner_radius, outer_radius = radius - thickness, radius

                A = np.pi * (outer_radius**2 - inner_radius**2)
                Iy = np.pi * (outer_radius**4 - inner_radius**4) / 4.0
                Iz = np.pi * (outer_radius**4 - inner_radius**4) / 4.0
                Ix = J = np.pi * (outer_radius**4 - inner_radius**4) / 2.0
                cs = CSPropTube(A=A, Iy=Iy, Iz=Iz, J=J, radius=radius, thickness=thickness)
                cs_storage.append(cs)

            elif beam.cs == 'box':
                width = self.declare_variable(beam.name + '_width', shape=(beam.num_elements))
                height = self.declare_variable(beam.name + '_height', shape=(beam.num_elements))

                tweb = self.declare_variable(beam.name + '_tweb', shape=(beam.num_elements))
                ttop = self.declare_variable(beam.name + '_ttop', shape=(beam.num_elements))
                tbot = self.declare_variable(beam.name + '_tbot', shape=(beam.num_elements))

                # average nodal inputs to get elemental inputs
                # width = (width_in[i] + width_in[i + 1]) / 2
                # height = (height_in[i] + height_in[i + 1]) / 2
                # tweb = (tweb_in[i] + tweb_in[i + 1]) / 2
                # ttop = (ttop_in[i] + ttop_in[i + 1]) / 2
                # tbot = (tbot_in[i] + tbot_in[i + 1]) / 2

                tcap_avg = (ttop + tbot)/2

                # compute the box-beam cs properties
                # w_i, h_i = w - 2*tweb, h - 2*tcap
                w_i, h_i = width - 2*tweb, height - ttop - tbot
                A = width*height - w_i*h_i
                Iy = (width*(height**3) - w_i*(h_i**3)) / 12
                Iz = ((width**3)*height - (w_i**3)*h_i) / 12
                J = Ix = (2*tweb*tcap_avg*(width - tweb)**2*(height - tcap_avg)**2) / (width*tweb + height*tcap_avg - tweb**2 - tcap_avg**2) # Darshan's formula
                cs = CSPropBox(A=A, Iy=Iy, Iz=Iz, J=J, width=width, height=height, tweb=tweb, ttop=ttop, tbot=tbot)
                cs_storage.append(cs)

            if beam.cs == 'ellipse':
                a = self.declare_variable(beam.name + '_semi_major_axis', shape=(beam.num_elements))
                b = self.declare_variable(beam.name + '_semi_minor_axis', shape=(beam.num_elements))

                A = np.pi * a * b
                Iy = np.pi / 4 * a * b**3
                Iz = np.pi / 4 * a**3 * b
                # Compute beta for the approximation of J
                beta = 1 / ((1 + (b/a)**2)**0.5)
                J = (np.pi / 2) * a * b**3 * beta
                
                cs = CSPropEllipse(A=A, Iy=Iy, Iz=Iz, J=J, major_axis=a, minor_axis=b)
                cs_storage.append(cs)




            beam_stiffness, local_stiffness, transformations = self.stiffness(beam=beam, mesh=mesh, cs=cs, dimension=dimension, node_dictionary=node_dictionary, index=index)
            global_stiffness_matrix = global_stiffness_matrix + csdl.reshape(beam_stiffness, (dimension, dimension))
            local_stiffness_storage.append(local_stiffness)
            transformations_storage.append(transformations)

            beam_mass_matrix = self.mass_matrix(beam=beam, mesh=mesh, cs=cs, dimension=dimension, node_dictionary=node_dictionary, index=index)
            global_mass_matrix = global_mass_matrix + csdl.reshape(beam_mass_matrix, (dimension, dimension))

            beam_mass, beam_rmvec = self.mass_properties(beam, mesh, cs)
            mass = mass + beam_mass
            rmvec = rmvec + beam_rmvec


        self.register_output('global_stiffness_matrix', global_stiffness_matrix) # for Jiayao and Andrew
        self.register_output('global_mass_matrix', global_mass_matrix) # for Jiayao and Andrew

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
        F = csdl.reshape(loads, new_shape=(6*num_unique_nodes)) # flatten loads to a vector
        self.register_output('global_loads_vector', F) # for Jiayao and Andrew




        # solve the linear system
        U = csdl.solve(K, F)





        # parse the displacements to get the deformed mesh for each beam
        for j, beam in enumerate(beams):
            mesh = mesh_storage[j]
            deformed_mesh = self.create_output(beam.name + '_deformed_mesh', shape=(beam.num_nodes, 3), val=0)

            for i in range(beam.num_nodes):
                node_id = index[node_dictionary[beam.name][i]]
                deformed_mesh[i, :] = mesh[i, :] + csdl.reshape(U[node_id*6:node_id*6 + 3], (1, 3))


        # recover the elemental forces/moments
        element_loads_storage = []
        for j, beam in enumerate(beams):
            local_stiffness, transformations = local_stiffness_storage[j], transformations_storage[j]

            element_beam_loads = self.create_output(beam.name + '_element_beam_loads', shape=(beam.num_elements, 12))
            for i in range(beam.num_elements):
                node_a_id, node_b_id = index[node_dictionary[beam.name][i]], index[node_dictionary[beam.name][i + 1]]

                d = self.create_output(beam.name + str(i) + '_d', shape=(12), val=0)
                d[0:6], d[6:12] = 1*U[node_a_id*6:node_a_id*6 + 6], 1*U[node_b_id*6:node_b_id*6 + 6]
                kp = csdl.reshape(local_stiffness[i, :, :], (12, 12))
                T = csdl.reshape(transformations[i, :, :], (12, 12))

                element_loads = csdl.matvec(kp, csdl.matvec(T, d))
                element_beam_loads[i, :] = csdl.reshape(element_loads, (1, 12))

            element_loads_storage.append(element_beam_loads)



        # stress recovery
        for i, beam in enumerate(beams):
            beam_element_loads, cs = element_loads_storage[i], cs_storage[i]

            # average the opposite loads at each end of the elements (odd but it works)
            F_x = (beam_element_loads[:, 0] - beam_element_loads[:, 6]) / 2
            F_y = (beam_element_loads[:, 1] - beam_element_loads[:, 7]) / 2
            F_z = (beam_element_loads[:, 2] - beam_element_loads[:, 8]) / 2
            M_x = (beam_element_loads[:, 3] - beam_element_loads[:, 9]) / 2
            M_y = (beam_element_loads[:, 4] - beam_element_loads[:, 10]) / 2
            M_z = (beam_element_loads[:, 5] - beam_element_loads[:, 11]) / 2

            if beam.cs == 'tube':
                # beam cs properties
                radius = csdl.reshape(cs.radius, (beam.num_elements, 1))
                A = csdl.reshape(cs.A, (beam.num_elements, 1))
                Iy = csdl.reshape(cs.Iy, (beam.num_elements, 1))
                J = csdl.reshape(cs.J, (beam.num_elements, 1))

                axial_stress = F_x / A
                # torsional_stress = M_x / A
                torsional_stress = M_x * radius / J
                MAX_MOMENT = (M_y**2 + M_z**2 + 1E-12)**0.5
                bending_stress = MAX_MOMENT * radius / Iy

                tensile_stress = axial_stress + bending_stress
                shear_stress = torsional_stress

                von_mises = (tensile_stress**2 + 3*shear_stress**2 + 1E-12)**0.5
                beam_stress = self.register_output(beam.name + '_stress', von_mises)
            if beam.cs == 'ellipse':
                # beam cs properties
                a = csdl.reshape(cs.major_axis, (beam.num_elements, 1))
                b = csdl.reshape(cs.minor_axis, (beam.num_elements, 1))
                A = csdl.reshape(cs.A, (beam.num_elements, 1))
                Iy = csdl.reshape(cs.Iy, (beam.num_elements, 1))
                Iz = csdl.reshape(cs.Iz, (beam.num_elements, 1))
                J = csdl.reshape(cs.J, (beam.num_elements, 1))

                axial_stress = F_x / A
                torsional_stress = M_x * ((a**2 + b**2)**0.5) / J  # Using the mean radius for simplicity
                
                # For bending stress, assuming bending occurs about the major axis for Iy and minor axis for Iz
                # Adjust as necessary for your specific loading conditions
                MAX_MOMENT = (M_y**2 + M_z**2 + 1E-12)**0.5
                bending_stress_major = M_y * b / Iy  # Bending about the major axis, stress at the minor axis
                bending_stress_minor = M_z * a / Iz  # Bending about the minor axis, stress at the major axis
                bending_stress = (bending_stress_major**2 + bending_stress_minor**2)**0.5  # Resultant bending stress
                
                tensile_stress = axial_stress + bending_stress
                shear_stress = torsional_stress

                von_mises = (tensile_stress**2 + 3*shear_stress**2 + 1E-12)**0.5
                beam_stress = self.register_output(beam.name + '_stress', von_mises)


            elif beam.cs == 'box':
                """ the stress for box beams is evaluated at four points:
                    0 ------------------------------------- 1
                      -                y                  -
                      -                |                  -
                      4                -->  z             -
                      -                                   -
                      -                                   -
                    3 ------------------------------------- 2
                """
                # beam cs properties
                A = csdl.reshape(cs.A, (beam.num_elements, 1))
                J = csdl.reshape(cs.J, (beam.num_elements, 1))
                Iy = csdl.reshape(cs.Iy, (beam.num_elements, 1))
                Iz = csdl.reshape(cs.Iz, (beam.num_elements, 1))
                width = csdl.reshape(cs.width, (beam.num_elements, 1))
                height = csdl.reshape(cs.height, (beam.num_elements, 1))
                tweb = csdl.reshape(cs.tweb, (beam.num_elements, 1))
                ttop = csdl.reshape(cs.ttop, (beam.num_elements, 1))
                tbot = csdl.reshape(cs.tbot, (beam.num_elements, 1))

                # the stress evaluation point coordinates
                zero = self.create_input(beam.name + '_zero', shape=(beam.num_elements, 1), val=0)
                coordinate_list = []
                coordinate_list.append((-width / 2, height / 2)) # point 0
                coordinate_list.append((width / 2, height / 2)) # point 1
                coordinate_list.append((width / 2, -height / 2)) # point 2
                coordinate_list.append((-width / 2, -height / 2)) # point 3
                coordinate_list.append((-width / 2, zero)) # point 4

                # first moment of area (Q) at point 4
                Q = csdl.reshape(width * ttop * (height / 2) + 2 * (height / 2) * tweb * (height / 4), (beam.num_elements, 1))

                # box beam signum function for buckling computations
                my_delta = M_y / ((M_y**2 + 1E-6)**0.5) # signum function


                beam_stress = self.create_output(beam.name + '_stress', shape=(beam.num_elements, 5), val=0)
                s4bkl_top, s4bkl_bot = 0, 0
                for i in range(5):
                    coordinate = coordinate_list[i]
                    z, y = coordinate[0], coordinate[1]
                    p = (z**2 + y**2)**0.5

                    normal_stress = F_x / A
                    torsional_stress = M_x * p / J
                    bending_stress_y = M_y * y / Iy
                    bending_stress_z = M_z * z / Iz

                    axial_stress = normal_stress + bending_stress_y + bending_stress_z

                    # ********************** shear stress stuff for point 4 *******************
                    if i == 4: shear_stress = F_z * Q / (Iy * 2 * tweb)
                    else: shear_stress = 0

                    tau = torsional_stress + shear_stress

                    von_mises = (axial_stress**2 + 3*tau**2 + 1E-12)**0.5
                    beam_stress[:, i] = von_mises

                    # ************ signed buckling stress calculation *******************
                    if i == 0 or i == 1: # average across the top two eval points
                        s4bkl_top = s4bkl_top + 0.5 * (my_delta * ((axial_stress + bending_stress_y + bending_stress_z)**2)**0.5)

                    if i == 2 or i == 3: # average across the bottom two eval points
                        s4bkl_bot = s4bkl_bot + 0.5 * (-1 * my_delta * ((axial_stress + bending_stress_y + bending_stress_z)**2)**0.5)

                # self.print_var(beam_stress)

                # Roark's simply-supported panel buckling
                k = 6.3
                critical_stress_top = k * beam.material.E * (ttop / width)**2 / (1 - beam.material.v**2)
                critical_stress_bot = k * beam.material.E * (tbot / width)**2 / (1 - beam.material.v**2)

                top_bkl = s4bkl_top / critical_stress_top # greater than 1 means the beam buckles
                self.register_output(beam.name + '_top_buckle', top_bkl)

                bot_bkl = s4bkl_bot / critical_stress_bot # greater than 1 means the beam buckles
                self.register_output(beam.name + '_bot_buckle', bot_bkl)

                # self.print_var(top_bkl)
                # self.print_var(bot_bkl)
                # self.print_var(s4bkl_top)
                


