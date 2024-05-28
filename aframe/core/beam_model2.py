import numpy as np
import csdl_alpha as csdl
import aframe as af
# import pickle

class Frame:
    def __init__(self):
        self.beams = []
        self.joints = []

    def add_beam(self, beam):
        self.beams.append(beam)

    def add_joint(self, joint_beams, joint_nodes):

        for i, beam in enumerate(joint_beams):
            if joint_nodes[i] > beam.num_nodes - 1:
                raise Exception(f'joint node index {joint_nodes[i]} is out of range for {beam.name}')

        self.joints.append({'beams': joint_beams, 'nodes': joint_nodes})

    def _stiffness_matrix(self, beam, dimension, index, node_dictionary):

        element_stiffness = csdl.Variable(value=np.zeros((beam.num_elements, 12, 12)))
        tkt_storage = csdl.Variable(value=np.zeros((beam.num_elements, 12, 12)))
        transformations = csdl.Variable(value=np.zeros((beam.num_elements, 12, 12)))

        mesh = beam.mesh
        area = A = beam.cs.area
        iy, iz, ix = beam.cs.iy, beam.cs.iz, beam.cs.ix
        E, G = beam.material.E, beam.material.G

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
            a_ind = index[node_dictionary[beam.name][i]] * 6
            b_ind = index[node_dictionary[beam.name][i + 1]] * 6

            k = k.set(csdl.slice[a_ind:a_ind + 6, a_ind:a_ind + 6], k11)
            k = k.set(csdl.slice[a_ind:a_ind + 6, b_ind:b_ind + 6], k12)
            k = k.set(csdl.slice[b_ind:b_ind + 6, a_ind:a_ind + 6], k21)
            k = k.set(csdl.slice[b_ind:b_ind + 6, b_ind:b_ind + 6], k22)

            beam_stiffness = beam_stiffness + k

        return beam_stiffness, element_stiffness, transformations



    def _mass_matrix(self, beam, dimension, index, node_dictionary):
        
        rho = beam.material.rho
        area = beam.cs.area
        ix = beam.cs.ix
        mesh = beam.mesh

        tmt_storage = csdl.Variable(value=np.zeros((beam.num_elements, 12, 12)))

        for i in csdl.frange(beam.num_elements):
            L = csdl.norm(mesh[i + 1, :] - mesh[i, :])
            A, J = area[i], ix[i]

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
            mp = mp.set(csdl.slice[8, 2], coef * 27)
            mp = mp.set(csdl.slice[4, 8], coef * -13 * a)
            mp = mp.set(csdl.slice[8, 4], coef * -13 * a)
            mp = mp.set(csdl.slice[8, 8], coef * 78)
            mp = mp.set(csdl.slice[3, 9], coef * -35 * rx2)
            mp = mp.set(csdl.slice[9, 3], coef * -35 * rx2)
            mp = mp.set(csdl.slice[9, 9], coef * 70 * rx2)
            mp = mp.set(csdl.slice[2, 10], coef * 13 * a)
            mp = mp.set(csdl.slice[10, 2], coef * 13 * a)
            mp = mp.set(csdl.slice[4, 10], coef * -6 * a**2)
            mp = mp.set(csdl.slice[10, 4], coef * -6 * a**2)
            mp = mp.set(csdl.slice[8, 10], coef * 22 * a)
            mp = mp.set(csdl.slice[10, 8], coef * 22 * a)
            mp = mp.set(csdl.slice[10, 10], coef * 8 * a**2)
            mp = mp.set(csdl.slice[1, 11], coef * -13 * a)
            mp = mp.set(csdl.slice[11, 1], coef * -13 * a)
            mp = mp.set(csdl.slice[5, 11], coef * -6 * a**2)
            mp = mp.set(csdl.slice[11, 5], coef * -6 * a**2)
            mp = mp.set(csdl.slice[7, 11], coef * -22 * a)
            mp = mp.set(csdl.slice[11, 7], coef * -22 * a)
            mp = mp.set(csdl.slice[11, 11], coef * 8 * a**2)

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

            tmt = csdl.matmat(csdl.transpose(T), csdl.matmat(mp, T))
            tmt_storage = tmt_storage.set(csdl.slice[i, :, :], tmt)


        beam_mass = 0
        for i in range(beam.num_elements):
            tmt = tmt_storage[i, :, :]
            m11, m12, m21, m22 = tmt[0:6,0:6], tmt[0:6,6:12], tmt[6:12,0:6], tmt[6:12,6:12]

            # expand the transformed mass matrix to the global dimensions
            m = csdl.Variable(value=np.zeros((dimension, dimension)))

            # assign the four block matrices to their respective positions in m
            a_ind = index[node_dictionary[beam.name][i]] * 6
            b_ind = index[node_dictionary[beam.name][i + 1]] * 6

            m = m.set(csdl.slice[a_ind:a_ind + 6, a_ind:a_ind + 6], m11)
            m = m.set(csdl.slice[a_ind:a_ind + 6, b_ind:b_ind + 6], m12)
            m = m.set(csdl.slice[b_ind:b_ind + 6, a_ind:a_ind + 6], m21)
            m = m.set(csdl.slice[b_ind:b_ind + 6, b_ind:b_ind + 6], m22)

            beam_mass = beam_mass + m

        return beam_mass


    def _mass_properties(self, beam, mesh):

        rho = beam.material.rho
        area = beam.cs.area
        # mesh = beam.mesh

        beam_mass, beam_rmvec = 0, 0
        for i in range(beam.num_elements):

            L = csdl.norm(mesh[i + 1, :] - mesh[i, :])
            element_mass = area[i] * L * rho

            # element cg is the nodal average
            element_cg = (mesh[i + 1, :] + mesh[i, :]) / 2

            beam_mass = beam_mass + element_mass
            beam_rmvec = beam_rmvec + element_cg * element_mass

        return beam_mass, beam_rmvec
    


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
        K, M = 0, 0
        mass, rmvec = 0, 0
        transformations_storage, local_stiffness_storage = [], []

        for beam in self.beams:

            beam_stiffness, element_stiffness, transformations = self._stiffness_matrix(beam, dimension, index, node_dictionary)
            K = K + beam_stiffness
            local_stiffness_storage.append(element_stiffness)
            transformations_storage.append(transformations)

            beam_mass_matrix = self._mass_matrix(beam=beam, dimension=dimension, node_dictionary=node_dictionary, index=index)
            M = M + beam_mass_matrix

            beam_mass, beam_rmvec = self._mass_properties(beam, beam.mesh)
            mass = mass + beam_mass
            rmvec = rmvec + beam_rmvec

        # compute the undeformed cg for the frame
        cg = rmvec / mass



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
                        # and mask the mass matrix also
                        M = M.set(csdl.slice[node_index + i, :], 0) # row
                        M = M.set(csdl.slice[:, node_index + i], 0) # column
                        M = M.set(csdl.slice[node_index + i, node_index + i], 1)

        # with open('matrix2.pkl', 'wb') as f:
        #     pickle.dump(K.value, f)

        # solve the linear system
        U = csdl.solve_linear(K, F)


        # create the beam displacements dictionary
        displacement = {}
        deformed_mesh = {}

        # parse the displacements to get the deformed mesh for each beam
        for j, beam in enumerate(self.beams):
            mesh = beam.mesh

            def_mesh = csdl.Variable(value=np.zeros((beam.num_nodes, 3)))
            disp = csdl.Variable(value=np.zeros((beam.num_nodes, 3)))
            for i in range(beam.num_nodes):
                node_index = index[node_dictionary[beam.name][i]] * 6
                def_mesh = def_mesh.set(csdl.slice[i, :], mesh[i, :] + U[node_index:node_index + 3])
                disp = disp.set(csdl.slice[i, :], U[node_index:node_index + 3])

            displacement[beam.name] = disp
            deformed_mesh[beam.name] = def_mesh


        # compute the deformed cg for the frame
        def_rmvec = 0
        for i, beam in enumerate(self.beams):
            _, def_beam_rmvec = self._mass_properties(beam, displacement[beam.name])
            def_rmvec = def_rmvec + def_beam_rmvec
        
        # get the deformed cg
        dcg = def_rmvec / mass


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



        # create the beam stress & buckle dicts
        stress = {}
        bkl = {}
        
        # stress recovery
        for i, beam in enumerate(self.beams):
            beam_element_loads = element_loads_storage[i]

            # average the opposite loads at each end of the elements (odd but it works)
            F_x = (beam_element_loads[:, 0] - beam_element_loads[:, 6]) / 2
            F_y = (beam_element_loads[:, 1] - beam_element_loads[:, 7]) / 2
            F_z = (beam_element_loads[:, 2] - beam_element_loads[:, 8]) / 2
            M_x = (beam_element_loads[:, 3] - beam_element_loads[:, 9]) / 2
            M_y = (beam_element_loads[:, 4] - beam_element_loads[:, 10]) / 2
            M_z = (beam_element_loads[:, 5] - beam_element_loads[:, 11]) / 2

            if beam.cs.type == 'tube':
                radius = beam.cs.radius
                axial_stress = F_x / beam.cs.area
                torsional_stress = M_x / beam.cs.area
                torsional_stress = M_x * radius / beam.cs.ix
                MAX_MOMENT = (M_y**2 + M_z**2 + 1E-12)**0.5
                bending_stress = MAX_MOMENT * radius / beam.cs.iy

                tensile_stress = axial_stress + bending_stress
                shear_stress = torsional_stress

                von_mises = (tensile_stress**2 + 3*shear_stress**2 + 1E-12)**0.5
                stress[beam.name] = von_mises

            elif beam.cs.type == 'box':
                width = beam.cs.width
                height = beam.cs.height
                tweb = beam.cs.tweb
                ttop = beam.cs.ttop
                tbot = beam.cs.tbot

                # the stress evaluation point coordinates
                coordinate_list = []
                coordinate_list.append((-width / 2, height / 2)) # point 0
                coordinate_list.append((width / 2, height / 2)) # point 1
                coordinate_list.append((width / 2, -height / 2)) # point 2
                coordinate_list.append((-width / 2, -height / 2)) # point 3
                coordinate_list.append((-width / 2, 0)) # point 4

                # first moment of area (Q) at point 4
                Q = width * ttop * (height / 2) + 2 * (height / 2) * tweb * (height / 4)

                # box beam signum function for buckling computations
                my_delta = M_y / ((M_y**2 + 1E-6)**0.5) # signum function

                beam_stress = csdl.Variable(value=np.zeros((beam.num_elements, 5)))
                s4bkl_top, s4bkl_bot = 0, 0
                for i in range(5):
                    coordinate = coordinate_list[i]
                    z, y = coordinate[0], coordinate[1]
                    p = (z**2 + y**2)**0.5

                    normal_stress = F_x / beam.cs.area
                    torsional_stress = M_x * p / beam.cs.ix
                    bending_stress_y = M_y * y / beam.cs.iy
                    bending_stress_z = M_z * z / beam.cs.iz

                    axial_stress = normal_stress + bending_stress_y + bending_stress_z

                    # ********************** shear stress stuff for point 4 *******************
                    if i == 4: 
                        shear_stress = F_z * Q / (beam.cs.iy * 2 * tweb)
                    else: 
                        shear_stress = 0

                    tau = torsional_stress + shear_stress

                    von_mises = (axial_stress**2 + 3*tau**2 + 1E-12)**0.5
                    # beam_stress[:, i] = von_mises
                    beam_stress = beam_stress.set(csdl.slice[:, i], von_mises)

                    # ************ signed buckling stress calculation *******************
                    if i == 0 or i == 1: # average across the top two eval points
                        s4bkl_top = s4bkl_top + 0.5 * (my_delta * ((axial_stress + bending_stress_y + bending_stress_z)**2)**0.5)

                    if i == 2 or i == 3: # average across the bottom two eval points
                        s4bkl_bot = s4bkl_bot + 0.5 * (-1 * my_delta * ((axial_stress + bending_stress_y + bending_stress_z)**2)**0.5)

                # add the beam stress to the stress dictionary
                stress[beam.name] = beam_stress

                # Roark's simply-supported panel buckling
                k = 6.3
                critical_stress_top = k * beam.material.E * (ttop / width)**2 / (1 - beam.material.v**2)
                critical_stress_bot = k * beam.material.E * (tbot / width)**2 / (1 - beam.material.v**2)

                top_bkl = s4bkl_top / critical_stress_top # greater than 1 means the beam buckles

                bot_bkl = s4bkl_bot / critical_stress_bot # greater than 1 means the beam buckles

                # add the box-beam buckling to the buckle dictionary
                bkl[beam.name] = {'top': top_bkl, 'bot': bot_bkl}
           












           
        
        return af.Solution(displacement=displacement,
                           mesh=deformed_mesh,
                           stress=stress,
                           bkl=bkl, 
                           cg=cg,
                           dcg=dcg)