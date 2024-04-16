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

        # # generate the nodal indices for the beam
        # node_a_index_array = csdl.Variable(value=np.zeros((beam.num_elements)))
        # node_b_index_array = csdl.Variable(value=np.zeros((beam.num_elements)))
        # for i in range(beam.num_elements):
        #     node_a_index_array = node_a_index_array.set([i], index[node_dictionary[beam.name][i]])
        #     node_b_index_array = node_b_index_array.set([i], index[node_dictionary[beam.name][i + 1]])

        local_stiffness = csdl.Variable(value=np.zeros((beam.num_elements, 12, 12)))
        for i in csdl.frange(beam.num_elements):
            E, G = beam.material.E, beam.material.G
            L = csdl.norm(beam.mesh[i + 1, :] - beam.mesh[i, :])
            A, Iy, Iz, J = beam.cs.area[i], beam.cs.iy[i], beam.cs.iz[i], beam.cs.ix[i]

            kp = csdl.Variable(value=np.zeros((12, 12)))

            # the upper left block
            kp = kp.set([0, 0], A * E / L)
            kp = kp.set([1, 1], 12 * E * Iz / L**3)
            kp = kp.set([1, 5], 6 * E * Iz / L**2)
            kp = kp.set([5, 1], 6 * E * Iz / L**2)
            kp = kp.set([2, 2], 12 * E * Iy / L**3)
            kp = kp.set([2, 4], -6 * E * Iy / L**2)
            kp = kp.set([4, 2], -6 * E * Iy / L**2)
            kp = kp.set([3, 3], G * J / L)
            kp = kp.set([4, 4], 4 * E * Iy / L)
            kp = kp.set([5, 5], 4 * E * Iz / L)

            # the upper right block
            kp = kp.set([0, 6], -A * E / L)
            kp = kp.set([1, 7], -12 * E * Iz / L**3)
            kp = kp.set([1, 11], -6 * E * Iz / L**2)
            kp = kp.set([2, 8], -12 * E * Iy / L**3)
            kp = kp.set([2, 10], 6 * E * Iy / L**2)
            kp = kp.set([3, 9], -G * J / L)
            kp = kp.set([4, 8], 6 * E * Iy / L**2)
            kp = kp.set([5, 11], 2 * E * Iz / L)

            # the lower left block
            kp = kp.set([6, 0], -A * E / L)
            kp = kp.set([7, 1], -12 * E * Iz / L**3)
            kp = kp.set([7, 5], -6 * E * Iz / L**2)
            kp = kp.set([8, 2], -12 * E * Iy / L**3)
            kp = kp.set([8, 4], 6 * E * Iy / L**2)
            kp = kp.set([9, 3], -G * J / L)
            kp = kp.set([10, 2], -6 * E * Iy / L**2)
            kp = kp.set([10, 4], 2 * E * Iy / L)
            kp = kp.set([11, 1], 6 * E * Iz / L**2)
            kp = kp.set([11, 5], 2 * E * Iz / L)

            # the lower right block
            kp = kp.set([6, 6], A * E / L)
            kp = kp.set([7, 7], 12 * E * Iz / L**3)
            kp = kp.set([7, 11], -6 * E * Iz / L**2)
            kp = kp.set([11, 7], -6 * E * Iz / L**2)
            kp = kp.set([8, 8], 12 * E * Iy / L**3)
            kp = kp.set([8, 10], 6 * E * Iy / L**2)
            kp = kp.set([10, 8], 6 * E * Iy / L**2)
            kp = kp.set([9, 9], G * J / L)
            kp = kp.set([10, 10], 4 * E * Iy / L)
            kp = kp.set([11, 11], 4 * E * Iz / L)

            local_stiffness = local_stiffness.set(csdl.slice[i, :, :], kp)

            # cp = (beam.mesh[i + 1, :] - beam.mesh[i, :]) / L
            # ll, mm, nn = cp[0], cp[1], cp[2]
            # D = (ll**2 + mm**2)**0.5

            # block = csdl.Variable(value=np.zeros((3, 3)))
            # block = block.set([0, 0], ll)
            # block = block.set([0, 1], mm)
            # block = block.set([0, 2], nn)
            # block = block.set([1, 0], -mm / D)
            # block = block.set([1, 1], ll / D)
            # block = block.set([2, 0], -ll * nn / D)
            # block = block.set([2, 1], -mm * nn / D)
            # block = block.set([2, 2], D)

            # T = csdl.Variable(value=np.zeros((12, 12)))
            # T = T.set(csdl.slice[0:3, 0:3], block)
            # T = T.set(csdl.slice[3:6, 3:6], block)
            # T = T.set(csdl.slice[6:9, 6:9], block)
            # T = T.set(csdl.slice[9:12, 9:12], block)

            # tkt = csdl.matmat(csdl.transpose(T), csdl.matmat(kp, T))
            # k11, k12, k21, k22 = tkt[0:6,0:6], tkt[0:6,6:12], tkt[6:12,0:6], tkt[6:12,6:12]

            # # expand the transformed stiffness matrix to the global dimensions
            # k = csdl.Variable(value=np.zeros((dimension, dimension)))

            # # assign the four block matrices to their respective positions in k
            # node_a_index = index[node_dictionary[beam.name][i]]
            # node_b_index = index[node_dictionary[beam.name][i + 1]]

            # # node_a_index = node_a_index_array[i]
            # # node_b_index = node_b_index_array[i]

            # row_i = node_a_index * 6
            # row_f = node_a_index * 6 + 6
            # col_i = node_a_index * 6
            # col_f = node_a_index * 6 + 6
            # # k[row_i:row_f, col_i:col_f] = k11
            # k = k.set(csdl.slice[row_i:row_f, col_i:col_f], k11)



    def _mass_matrix(self):
        pass

    def _mass_properties(self):
        pass

    def evaluate(self):
        
        # check for beams
        if not self.beams: 
            raise Exception('Error: beam(s) must be added to the frame')
        
        # check for boundary conditions
        num_bc = 0
        for beam in self.beams: num_bc += len(beam.boundary_conditions)
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
        for beam in self.beams:
            mesh = beam.mesh

            null = self._stiffness_matrix(beam, dimension, index, node_dictionary)
           