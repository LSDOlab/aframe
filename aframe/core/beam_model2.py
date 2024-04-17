import numpy as np
import csdl_alpha as csdl


class Frame:
    def __init__(self):
        self.beams = []
        self.joints = []

    def add_beam(self, beam):
        self.beams.append(beam)

    def add_joint(self, beams, nodes):
        self.joints.append({'beams': beams, 'nodes': nodes})

    def _stiffness_matrix(self, beam, dimension, index, node_dictionary):

        local_stiffness = csdl.Variable(value=np.zeros((beam.num_elements, 12, 12)))
        tkt_storage = csdl.Variable(value=np.zeros((beam.num_elements, 12, 12)))
        transformations = csdl.Variable(value=np.zeros((beam.num_elements, 12, 12)))

        mesh = beam.mesh
        area = beam.cs.area
        iy = beam.cs.iy
        iz = beam.cs.iz
        ix = beam.cs.ix
        E, G = beam.material.E, beam.material.G

        # print(area.value, E, iy.value, iz.value, ix.value)
        print(mesh.value)

        for i in range(beam.num_elements):
            L = csdl.norm(mesh[i + 1, :] - mesh[i, :])
            print(L.value)
            A, Iy, Iz, J = area[i], iy[i], iz[i], ix[i]

            kp = csdl.Variable(value=np.zeros((12, 12)))


            s = csdl.slice
            # the upper left block
            kp = kp.set(csdl.slice[0, 0], A * E / L)
            kp = kp.set(csdl.slice[1, 1], 12 * E * Iz / L**3)
            kp = kp.set(csdl.slice[1, 5], 6 * E * Iz / L**2)
            kp = kp.set(csdl.slice[5, 1], 6 * E * Iz / L**2)
            kp = kp.set(csdl.slice[2, 2], 12 * E * Iy / L**3)
            kp = kp.set(csdl.slice[2, 4], -6 * E * Iy / L**2)
            kp = kp.set(csdl.slice[4, 2], -6 * E * Iy / L**2)
            kp = kp.set(csdl.slice[3, 3], G * J / L)
            kp = kp.set(csdl.slice[4, 4], 4 * E * Iy / L)
            kp = kp.set(csdl.slice[5, 5], 4 * E * Iz / L)

            # print(kp.value)

            # the upper right block
            print('before', kp[0,0].value)
            kp = kp.set(csdl.slice[0, 6], -A * E / L)
            kp = kp.set(csdl.slice[0, 6], -A * E / L)
            print('after', kp[0,0].value)
            kp = kp.set(csdl.slice[1, 7], -12 * E * Iz / L**3)
            kp = kp.set(csdl.slice[1, 11], 6 * E * Iz / L**2)
            kp = kp.set(csdl.slice[2, 8], -12 * E * Iy / L**3)
            kp = kp.set(csdl.slice[2, 10], -6 * E * Iy / L**2)
            kp = kp.set(csdl.slice[3, 9], -G * J / L)
            kp = kp.set(csdl.slice[4, 8], 6 * E * Iy / L**2)
            kp = kp.set(csdl.slice[4, 10], 2 * E * Iy / L)
            kp = kp.set(csdl.slice[5, 7], -6 * E * Iz / L**2)
            kp = kp.set(csdl.slice[5, 11], 2 * E * Iz / L)

            # the lower left block
            kp = kp.set(csdl.slice[6, 0], -A * E / L)
            kp = kp.set(csdl.slice[7, 1], -12 * E * Iz / L**3)
            kp = kp.set(csdl.slice[7, 5], -6 * E * Iz / L**2)
            kp = kp.set(csdl.slice[8, 2], -12 * E * Iy / L**3)
            kp = kp.set(csdl.slice[8, 4], 6 * E * Iy / L**2)
            kp = kp.set(csdl.slice[9, 3], -G * J / L)
            kp = kp.set(csdl.slice[10, 2], -6 * E * Iy / L**2)
            kp = kp.set(csdl.slice[10, 4], 2 * E * Iy / L)
            kp = kp.set(csdl.slice[11, 1], 6 * E * Iz / L**2)
            kp = kp.set(csdl.slice[11, 5], 2 * E * Iz / L)

            # the lower right block
            kp = kp.set(csdl.slice[6, 6], A * E / L)
            kp = kp.set(csdl.slice[7, 7], 12 * E * Iz / L**3)
            kp = kp.set(csdl.slice[7, 11], -6 * E * Iz / L**2)
            kp = kp.set(csdl.slice[11, 7], -6 * E * Iz / L**2)
            kp = kp.set(csdl.slice[8, 8], 12 * E * Iy / L**3)
            kp = kp.set(csdl.slice[8, 10], 6 * E * Iy / L**2)
            kp = kp.set(csdl.slice[10, 8], 6 * E * Iy / L**2)
            kp = kp.set(csdl.slice[9, 9], G * J / L)
            kp = kp.set(csdl.slice[10, 10], 4 * E * Iy / L)
            kp = kp.set(csdl.slice[11, 11], 4 * E * Iz / L)

            local_stiffness = local_stiffness.set(csdl.slice[i, :, :], kp)
            # print(kp.value)

            cp = (mesh[i + 1, :] - mesh[i, :]) / L
            ll, mm, nn = cp[0], cp[1], cp[2]
            D = (ll**2 + mm**2)**0.5

            block = csdl.Variable(value=np.zeros((3, 3)))
            block = block.set([0, 0], ll)
            block = block.set([0, 1], mm)
            block = block.set([0, 2], nn)
            block = block.set([1, 0], -mm / D)
            block = block.set([1, 1], ll / D)
            block = block.set([2, 0], -ll * nn / D)
            block = block.set([2, 1], -mm * nn / D)
            block = block.set([2, 2], D)

            T = csdl.Variable(value=np.zeros((12, 12)))
            T = T.set(csdl.slice[0:3, 0:3], block)
            T = T.set(csdl.slice[3:6, 3:6], block)
            T = T.set(csdl.slice[6:9, 6:9], block)
            T = T.set(csdl.slice[9:12, 9:12], block)
            transformations = transformations.set(csdl.slice[i, :, :], T)

            tkt = csdl.matmat(csdl.transpose(T), csdl.matmat(kp, T))
            tkt_storage = tkt_storage.set(csdl.slice[i, :, :], tkt)


        beam_stiffness = 0
        for i in range(beam.num_elements):
            tkt = tkt_storage[i, :, :]
            k11, k12, k21, k22 = tkt[0:6,0:6], tkt[0:6,6:12], tkt[6:12,0:6], tkt[6:12,6:12]

            # expand the transformed stiffness matrix to the global dimensions
            k = csdl.Variable(value=np.zeros((dimension, dimension)))

            # assign the four block matrices to their respective positions in k
            node_a_index = index[node_dictionary[beam.name][i]]
            node_b_index = index[node_dictionary[beam.name][i + 1]]

            row_i = node_a_index * 6
            row_f = node_a_index * 6 + 6
            col_i = node_a_index * 6
            col_f = node_a_index * 6 + 6
            k = k.set(csdl.slice[row_i:row_f, col_i:col_f], k11)

            row_i = node_a_index*6
            row_f = node_a_index*6 + 6
            col_i = node_b_index*6
            col_f = node_b_index*6 + 6
            k = k.set(csdl.slice[row_i:row_f, col_i:col_f], k12)

            row_i = node_b_index*6
            row_f = node_b_index*6 + 6
            col_i = node_a_index*6
            col_f = node_a_index*6 + 6
            k = k.set(csdl.slice[row_i:row_f, col_i:col_f], k21)

            row_i = node_b_index*6
            row_f = node_b_index*6 + 6
            col_i = node_b_index*6
            col_f = node_b_index*6 + 6
            k = k.set(csdl.slice[row_i:row_f, col_i:col_f], k22)

            beam_stiffness = beam_stiffness + k

        # print(beam_stiffness[0, 0].value)
        # print(tkt_storage[0, 0, 0].value)
        print(local_stiffness[0, :, :].value)
        # print(A.value, E, L.value, Iy.value, Iz.value, J.value)

        return beam_stiffness, local_stiffness, transformations



    def _mass_matrix(self):
        pass

    def _mass_properties(self):
        pass

    def evaluate(self):
        
        # check for beams
        if not self.beams: 
            raise Exception('Error: beam(s) must be added to the frame')
        
        # create boundary conditions dictionary
        num_bc = 0
        boundary_conditions = []
        for beam in self.beams: 
            num_bc += len(beam.bc)
            for i in range(len(beam.bc)):
                boundary_conditions.append(beam.bc[i])

        # check for boundary conditions    
        if num_bc == 0: 
            raise Exception('Error: no boundary conditions')

        # check for kinematic constraints
        if len(self.beams) - num_bc > len(self.joints):
            raise Exception('Error: not enough kinematic constraints')
        
        # automated beam node assignment
        node_dictionary = {}

        # start by populating the nodes dictionary without joints
        val = 0
        for beam in self.beams:
            node_dictionary[beam.name] = np.arange(val, val + beam.num_nodes)
            val += beam.num_nodes

        # assign nodal indices in the global system with joints
        for joint in self.joints:
            joint_beams, joint_nodes = joint['beams'], joint['nodes']
            node_a = node_dictionary[joint_beams[0].name][joint_nodes[0]]
            for i, beam in enumerate(joint_beams):
                if i != 0: node_dictionary[beam.name][joint_nodes[i]] = node_a

        # helpers
        node_set = set()
        for beam in self.beams:
            for i in range(beam.num_nodes):
                node_set.add(node_dictionary[beam.name][i])

        num_unique_nodes = len(node_set)
        dimension = num_unique_nodes * 6
        index = {list(node_set)[i]: i for i in range(num_unique_nodes)}

        # construct the global stiffness matrix
        global_stiffness_matrix = 0
        transformations_storage, local_stiffness_storage = [], []
        for beam in self.beams:

            beam_stiffness, local_stiffness, transformations = self._stiffness_matrix(beam, dimension, index, node_dictionary)
            global_stiffness_matrix = global_stiffness_matrix + beam_stiffness
            local_stiffness_storage.append(local_stiffness)
            transformations_storage.append(transformations)

            # beam_mass_matrix = self.mass_matrix(beam=beam, dimension=dimension, node_dictionary=node_dictionary, index=index)
            # global_mass_matrix = global_mass_matrix + csdl.reshape(beam_mass_matrix, (dimension, dimension))

            # beam_mass, beam_rmvec = self.mass_properties(beam)
            # mass = mass + beam_mass
            # rmvec = rmvec + beam_rmvec

        # print(global_stiffness_matrix.value)

        # deal with the boundary conditions
        bound_node_index_list = []
        for bc_dict in boundary_conditions:
            bound_node, dof = bc_dict['node'], bc_dict['dof']
            bound_node_index = index[node_dictionary[bc_dict['name']][bound_node]]

            # add the constrained dof index to the bound_node_index_list
            for i, degree in enumerate(dof):
                if degree: bound_node_index_list.append(bound_node_index*6 + i)

        mask = csdl.Variable(value=np.eye(dimension))
        mask_eye = csdl.Variable(value=np.zeros((dimension, dimension)))
        for i in range(dimension):
            if i in bound_node_index_list:
                mask = mask.set([i, i], 0)
                mask_eye = mask_eye.set([i, i], 1)

        # modify the global stiffness matrix with boundary conditions
        # first remove the row/column with a boundary condition, then add a 1
        K = csdl.matmat(csdl.matmat(mask, global_stiffness_matrix), mask) + mask_eye
        # print(K[0,0].value)

        # create the global loads vector
        loads = csdl.Variable(value=np.zeros((len(self.beams), num_unique_nodes, 6)))
        for i, beam in enumerate(self.beams):
            beam_loads = beam.loads
            for j, node in enumerate(node_dictionary[beam.name]):
                loads = loads.set(csdl.slice[i, index[node], :], beam_loads[j, :])

                # I changed this bit for new csdl ********************************************
                for k in range(6):
                    if (index[node]*6 + k) in bound_node_index_list:
                        loads = loads.set(csdl.slice[i, index[node], k], 0)

        # F = csdl.reshape(loads, new_shape=(6*num_unique_nodes)) # flatten loads to a vector
        F = csdl.sum(loads, axes=(0, )).flatten() # changed for new csdl ******************
        # print(F.value)
        # solve the linear system
        # U = csdl.solve_linear(K, F)

           