import numpy as np
import csdl_alpha as csdl
import aframe as af
import pickle

class Frame:
    def __init__(self):
        self.beams = []
        self.joints = []

    def add_beam(self, beam):
        self.beams.append(beam)

    def add_joint(self, beams, nodes):
        self.joints.append({'beams': beams, 'nodes': nodes})

    def _stiffness_matrix(self, beam, dimension, index, node_dictionary):

        element_stiffness = csdl.Variable(value=np.zeros((beam.num_elements, 12, 12)))
        tkt_storage = csdl.Variable(value=np.zeros((beam.num_elements, 12, 12)))
        transformations = csdl.Variable(value=np.zeros((beam.num_elements, 12, 12)))

        mesh = beam.mesh
        area = A = beam.cs.area
        iy, iz, ix = beam.cs.iy, beam.cs.iz, beam.cs.ix
        E, G = beam.material.E, beam.material.G

        L = csdl.Variable(value=np.zeros(beam.num_elements))
        for i in csdl.frange(beam.num_elements):
            L = L.set(csdl.slice[i], csdl.norm(mesh[i + 1, :] - mesh[i, :]))
        
        # kpp = csdl.Variable(value=np.zeros((beam.num_elements, 12, 12)))
        # values = [A * E / L, , , , ,]
        # indices = [, , , , ]

        for i in csdl.frange(beam.num_elements):
            L = csdl.norm(mesh[i + 1, :] - mesh[i, :])
            A, Iy, Iz, J = area[i], iy[i], iz[i], ix[i]

            kp = csdl.Variable(value=np.zeros((12, 12)))

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

            # the upper right block
            kp = kp.set(csdl.slice[0, 6], -A * E / L)
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

            element_stiffness = element_stiffness.set(csdl.slice[i, :, :], kp)

            cp = (mesh[i + 1, :] - mesh[i, :]) / L
            ll, mm, nn = cp[0], cp[1], cp[2]
            D = (ll**2 + mm**2)**0.5

            block = csdl.Variable(value=np.zeros((3, 3)))
            block = block.set(csdl.slice[0, 0], ll)
            block = block.set(csdl.slice[0, 1], mm)
            block = block.set(csdl.slice[0, 2], nn)
            block = block.set(csdl.slice[1, 0], -mm / D)
            block = block.set(csdl.slice[1, 1], ll / D)
            block = block.set(csdl.slice[2, 0], -ll * nn / D)
            block = block.set(csdl.slice[2, 1], -mm * nn / D)
            block = block.set(csdl.slice[2, 2], D)

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

        return beam_stiffness, element_stiffness, transformations



    def _mass_matrix(self, beam, dimension, index, node_dictionary):
        
        E, G, rho = beam.material.E, beam.material.G, beam.material.rho
        area = beam.cs.area
        iy, iz, ix = beam.cs.iy, beam.cs.iz, beam.cs.ix
        mesh = beam.mesh

        for i in csdl.frange(beam.num_elements):
            L = csdl.norm(mesh[i + 1, :] - mesh[i, :])
            A, Iy, Iz, J = area[i], iy[i], iz[i], ix[i]

            a = L / 2
            coef = rho * A * a / 105
            rx2 = J / A

            mp = csdl.Variable(value=np.zeros((12, 12)))
            mp = mp.set(csdl.slice[0, 0], coef * 70)
            mp = mp.set(csdl.slice[1, 1], coef * 78)
            mp = mp.set(csdl.slice[2, 2], coef * 78)
            mp = mp.set(csdl.slice[3, 3], coef * 78 * rx2)
            mp = mp.set(csdl.slice[2, 4], coef * -22 * a)
            mp = mp.set(csdl.slice[4, 2], coef * -22 * a)
            mp = mp.set(csdl.slice[4, 4], coef * 8 * a**2)
            mp = mp.set(csdl.slice[1, 5], coef * 22 * a)
            mp = mp.set(csdl.slice[5, 1], coef * 22 * a)
            mp = mp.set(csdl.slice[5, 5], coef * 8 * a**2)
            mp = mp.set(csdl.slice[0, 6], coef * 35)
            mp = mp.set(csdl.slice[6, 0], coef * 35)
            mp = mp.set(csdl.slice[6, 6], coef * 70)
            mp = mp.set(csdl.slice[1, 7], coef * 27)
            mp = mp.set(csdl.slice[7, 1], coef * 27)
            mp = mp.set(csdl.slice[5, 7], coef * 13 * a)
            mp = mp.set(csdl.slice[7, 5], coef * 13 * a)
            mp = mp.set(csdl.slice[7, 7], coef * 78)
            mp = mp.set(csdl.slice[2, 8], coef * 27)

    def _mass_properties(self):
        pass

    def evaluate(self):
        
        # check for beams
        if not self.beams: 
            raise Exception('Error: beam(s) must be added to the frame')
        
        # create boundary conditions dictionary
        num_bc = 0
        for beam in self.beams: num_bc += len(beam.bc)

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
        K = 0
        transformations_storage, local_stiffness_storage = [], []
        for beam in self.beams:

            beam_stiffness, element_stiffness, transformations = self._stiffness_matrix(beam, dimension, index, node_dictionary)
            K = K + beam_stiffness
            local_stiffness_storage.append(element_stiffness)
            transformations_storage.append(transformations)

            # beam_mass_matrix = self.mass_matrix(beam=beam, dimension=dimension, node_dictionary=node_dictionary, index=index)
            # global_mass_matrix = global_mass_matrix + csdl.reshape(beam_mass_matrix, (dimension, dimension))

            # beam_mass, beam_rmvec = self.mass_properties(beam)
            # mass = mass + beam_mass
            # rmvec = rmvec + beam_rmvec


        # construct the global loads vector
        F = 0
        for i, beam in enumerate(self.beams):
            beam_loads = beam.loads
            loads = csdl.Variable(value=np.zeros((num_unique_nodes, 6)))

            for j, node in enumerate(node_dictionary[beam.name]):
                loads = loads.set(csdl.slice[index[node], :], beam_loads[j, :])

            F = F + loads
            
        F = F.flatten()
        

        # boundary conditions
        for beam in self.beams:
            for bc in beam.bc:
                node, dof = bc['node'], bc['dof']
                node_index = index[node_dictionary[beam.name][node]] * 6

                for i in range(6):
                    if dof[i] == 1:
                        # zero the row/column then put a 1 in the diagonal
                        K = K.set(csdl.slice[node_index + i, :], 0) # row
                        K = K.set(csdl.slice[:, node_index + i], 0) # column
                        K = K.set(csdl.slice[node_index + i, node_index + i], 1)
                        # zero the corresponding load index as well
                        F = F.set(csdl.slice[node_index + i], 0)

        # with open('matrix2.pkl', 'wb') as f:
        #     pickle.dump(K.value, f)

        # solve the linear system
        U = csdl.solve_linear(K, F)


        # create the beam displacements dictionary
        displacement = {}

        # parse the displacements to get the deformed mesh for each beam
        for j, beam in enumerate(self.beams):
            mesh = beam.mesh
            def_mesh = csdl.Variable(value=np.zeros((beam.num_nodes, 3)))
            for i in range(beam.num_nodes):
                node_index = index[node_dictionary[beam.name][i]] * 6
                def_mesh = def_mesh.set(csdl.slice[i, :], mesh[i, :] + U[node_index:node_index + 3])

            displacement[beam.name] = def_mesh


        # recover the elemental loads
        element_loads_storage = []
        for j, beam in enumerate(self.beams):
            local_stiffness, transformations = local_stiffness_storage[j], transformations_storage[j]

            element_beam_loads = csdl.Variable(value=np.zeros((beam.num_elements, 12)))
            for i in range(beam.num_elements):
                node_a_index = index[node_dictionary[beam.name][i]] * 6
                node_b_index = index[node_dictionary[beam.name][i + 1]] * 6

                element_disp = csdl.Variable(value=np.zeros((12)))
                element_disp = element_disp.set(csdl.slice[0:6], U[node_a_index:node_a_index + 6])
                element_disp = element_disp.set(csdl.slice[6:12], U[node_b_index:node_b_index + 6])

                kp = local_stiffness[i, :, :]
                T = transformations[i, :, :]

                element_loads = csdl.matvec(kp, csdl.matvec(T, element_disp))
                element_beam_loads = element_beam_loads.set(csdl.slice[i, :], element_loads)

            element_loads_storage.append(element_beam_loads)
           












           
        
        return af.Solution(displacement=displacement)