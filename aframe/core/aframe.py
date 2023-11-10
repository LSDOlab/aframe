import numpy as np
import csdl
from aframe.core.model import Model
from aframe.core.stress import StressBox
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL




class Aframe(ModuleCSDL):

    def initialize(self):
        self.parameters.declare('beams', default={})
        self.parameters.declare('joints', default={})
        self.parameters.declare('bounds', default={})
        self.parameters.declare('mesh_units', default='m')


    def define(self):
        mesh_units = self.parameters['mesh_units']
        beams, joints, bounds = self.parameters['beams'], self.parameters['joints'], self.parameters['bounds']
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
            joint_beam_list, joint_node_list = joints[joint_name]['beams'], joints[joint_name]['nodes']
            joint_node_a = node_dict[joint_beam_list[0]][joint_node_list[0]]
            for i, beam_name in enumerate(joint_beam_list):
                if i != 0: node_dict[beam_name][joint_node_list[i]] = joint_node_a

        node_set = set(node_dict[beam_name][i] for beam_name in beams for i in range(len(beams[beam_name]['nodes'])))
        num_unique_nodes = len(node_set)
        dim = num_unique_nodes*6
        node_index = {list(node_set)[i]: i for i in range(num_unique_nodes)}



        # create a list of element names:
        elements, element_density_list, num_elements = [], [], 0
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            num_elements += n - 1
            for i in range(n - 1): 
                elements.append(beam_name + '_element_' + str(i))
                element_density_list.append(beams[beam_name]['rho'])

        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            cs=beams[beam_name]['cs']
            E = beams[beam_name]['E']
            G = beams[beam_name]['G']

            default_val = np.zeros((n, 3))
            default_val[:,1] = np.linspace(0,n,n)
            # mesh = self.declare_variable(name + '_mesh', shape=(n,3), val=default_val)
            mesh_in = self.register_module_input(beam_name + '_mesh', shape=(n,3), promotes=True, val=default_val)
            # self.print_var(mesh_in)
            if mesh_units == 'm': mesh = 1*mesh_in
            elif mesh_units == 'ft': mesh = 0.304*mesh_in


            if cs == 'box':
                width = self.register_module_input(beam_name + '_width', shape=(n), promotes=True)
                height = self.register_module_input(beam_name + '_height', shape=(n), promotes=True)
                tweb_in = self.register_module_input(beam_name + '_tweb', shape=(n))
                tcap_in = self.register_module_input(beam_name + '_tcap', shape=(n))

                # create elemental outputs
                w_vec, h_vec = self.create_output(beam_name + '_w', shape=(n - 1), val=0), self.create_output(beam_name + '_h', shape=(n - 1), val=0)
                iyo, izo, jo = self.create_output(beam_name + '_iyo', shape=(n - 1), val=0), self.create_output(beam_name + '_izo', shape=(n - 1), val=0), self.create_output(beam_name + '_jo', shape=(n - 1), val=0)

                for i in range(n - 1):
                    element_name = beam_name + '_element_' + str(i)
                    if mesh_units == 'm': converted_width, converted_height = 1*width, 1*height
                    elif mesh_units == 'ft': converted_width, converted_height = 0.304*width, 0.304*height

                    w_vec[i] = (converted_width[i] + converted_width[i + 1])/2
                    h_vec[i] = (converted_height[i] + converted_height[i + 1])/2
                    h = self.register_output(element_name + '_h', h_vec[i])
                    w = self.register_output(element_name + '_w', w_vec[i])

                    tweb = self.register_output(element_name + '_tweb', (tweb_in[i]+tweb_in[i+1])/2)
                    tcap = self.register_output(element_name + '_tcap', (tcap_in[i]+tcap_in[i+1])/2)



                    # compute the box-beam cs properties
                    w_i, h_i = w - 2*tweb, h - 2*tcap
                    A = self.register_output(element_name + '_A', (((w*h) - (w_i*h_i))**2 + 1E-14)**0.5)
                    iyo[i] = Iy = self.register_output(element_name + '_Iy', (w*(h**3) - w_i*(h_i**3))/12)
                    izo[i] = Iz = self.register_output(element_name + '_Iz', ((w**3)*h - (w_i**3)*h_i)/12)
                    # jo[i] = J = self.register_output(element_name + '_J', (w*h*(h**2 + w**2)/12) - (w_i*h_i*(h_i**2 + w_i**2)/12))
                    jo[i] = J = self.register_output(element_name + '_J', (2*tweb*tcap*(w-tweb)**2*(h-tcap)**2)/(w*tweb+h*tcap-tweb**2-tcap**2)) # Darshan's formula
                    # Q = 2*(h/2)*tweb*(h/4) + (w - 2*tweb)*tcap*((h/2) - (tcap/2))
                    Q = self.register_output(element_name + '_Q', (A/2)*(h/4))
                    self.register_output(element_name + '_Ix', 1*J) # I think J is the same as Ix...

                    node_a = self.register_output(beam_name + '_element_' + str(i) + 'node_a', csdl.reshape(mesh[i, :], (3)))
                    node_b = self.register_output(beam_name + '_element_' + str(i) + 'node_b', csdl.reshape(mesh[i + 1, :], (3)))
                    
                    # calculate the stiffness matrix for each element:
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
                    node_a_index = node_index[node_dict[beam_name][i]]
                    node_b_index = node_index[node_dict[beam_name][i + 1]]

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

        # compute the global stiffness matrix and the global mass matrix:
        helper = self.create_output('helper', shape=(num_elements,dim,dim), val=0)
        for i, element_name in enumerate(elements):
            helper[i,:,:] = csdl.reshape(self.declare_variable(element_name + 'k', shape=(dim,dim)), (1,dim,dim))

        sum_k = csdl.sum(helper, axes=(0, ))

        b_index_list = []
        for b_name in bounds:
            fpos, fdim = bounds[b_name]['node'], bounds[b_name]['fdim']
            b_node_index = node_index[node_dict[bounds[b_name]['beam']][fpos]]
            # add the constrained dof index to the b_index_list:
            for i, fdim in enumerate(fdim):
                if fdim == 1: b_index_list.append(b_node_index*6 + i)

        mask, mask_eye = self.create_output('mask', shape=(dim,dim), val=np.eye(dim)), self.create_output('mask_eye', shape=(dim,dim), val=0)
        zero, one = self.create_input('zero', shape=(1,1), val=0), self.create_input('one', shape=(1,1), val=1)
        [(mask.__setitem__((i,i),1*zero), mask_eye.__setitem__((i,i),1*one)) for i in range(dim) if i in b_index_list]

        # modify the global stiffness matrix with boundary conditions:
        # first remove the row/column with a boundary condition, then add a 1:
        K = self.register_output('K', csdl.matmat(csdl.matmat(mask, sum_k), mask) + mask_eye)











        # compute the mass properties:
        rm_vec = self.create_output('rm_vec', shape=(len(elements),3), val=0)
        m_vec = self.create_output('m_vec', shape=(len(elements)), val=0)

        for i, element_name in enumerate(elements):
            rho = element_density_list[i]

            A = self.declare_variable(element_name + '_A')
            L = self.declare_variable(element_name + 'L')

            # compute the element mass:
            m = self.register_output(element_name + 'm', (A*L)*rho)

            # get the (undeformed) position vector of the cg for each element:
            r_a = self.declare_variable(element_name + 'node_a', shape=(3))
            r_b = self.declare_variable(element_name + 'node_b', shape=(3))

            r_cg = self.register_output(element_name+'r_cg', (r_a + r_b)/2)\
            
            # assign r_cg to the r*mass vector:
            rm_vec[i,:] = csdl.reshape(r_cg*csdl.expand(m, (3)), new_shape=(1,3))
            # assign the mass to the mass vector:
            m_vec[i] = m
        
        # compute the center of gravity for the entire structure:
        total_mass = self.register_module_output('mass', csdl.sum(m_vec))
        self.register_output('struct_mass', 1*total_mass)

        # self.print_var(total_mass)
        
        cg = csdl.sum(rm_vec, axes=(0,))/csdl.expand(total_mass, (3))
        self.register_output('cg_vector', cg)

        self.register_output('cgx', cg[0])
        self.register_output('cgy', cg[1]*0) # zeroed to make opt converge better and stuff
        self.register_output('cgz', cg[2])

        # compute moments of inertia:
        eixx = self.create_output('eixx',shape=(len(elements)),val=0)
        eiyy = self.create_output('eiyy',shape=(len(elements)),val=0)
        eizz = self.create_output('eizz',shape=(len(elements)),val=0)
        eixz = self.create_output('eixz',shape=(len(elements)),val=0)
        for i, element_name in enumerate(elements):
            m = m_vec[i]

            # get the position vector:
            r = self.declare_variable(element_name + 'r_cg', shape=(3))
            x, y, z = r[0], r[1], r[2]
            rxx = y**2 + z**2
            ryy = x**2 + z**2
            rzz = x**2 + y**2
            rxz = x*z
            eixx[i] = m*rxx
            eiyy[i] = m*ryy
            eizz[i] = m*rzz
            eixz[i] = m*rxz
            
        # sum the m*r vector to get the moi:
        Ixx, Iyy, Izz, Ixz = csdl.sum(eixx), csdl.sum(eiyy), csdl.sum(eizz), csdl.sum(eixz)

        inertia_tensor = self.register_module_output('inertia_tensor', shape=(3, 3), val=0)
        inertia_tensor[0, 0] = csdl.reshape(Ixx, (1, 1))
        inertia_tensor[0, 2] = csdl.reshape(Ixz, (1, 1))
        inertia_tensor[1, 1] = csdl.reshape(Iyy, (1, 1))
        inertia_tensor[2, 0] = csdl.reshape(Ixz, (1, 1))
        inertia_tensor[2, 2] = csdl.reshape(Izz, (1, 1))

        self.register_output('ixx', Ixx)
        self.register_output('iyy', Iyy)
        self.register_output('izz', Izz)
        self.register_output('ixz', Ixz)

        # self.print_var(m_vec)







        # create the global loads vector:
        nodal_loads = self.create_output('nodal_loads', shape=(len(beams), num_unique_nodes, 6), val=0)
        for i, beam_name in enumerate(beams):
            n = len(beams[beam_name]['nodes'])
          
            forces = self.declare_variable(beam_name + '_forces', shape=(n,3), val=0)
            moments = self.declare_variable(beam_name + '_moments', shape=(n,3), val=0)

            # concatenate the forces and moments:
            loads = self.create_output(f'{beam_name}_loads', shape=(n,6), val=0)
            loads[:, 0:3], loads[:, 3:6] = 1*forces, 1*moments

            for j, bnode in enumerate(node_dict[beam_name]):
                for k in range(6):
                    if (node_index[bnode]*6 + k) not in b_index_list:
                        nodal_loads[i,node_index[bnode],k] = csdl.reshape(loads[j,k], (1,1,1))

        total_loads = csdl.sum(nodal_loads, axes=(0,))

        # flatten the total loads matrix to a vector:
        Fi = self.register_output('Fi', csdl.reshape(total_loads, new_shape=(6*num_unique_nodes)))








        # solve the linear system:
        solve_res = self.create_implicit_operation(Model(dim=dim))
        solve_res.declare_state(state='U', residual='R')
        solve_res.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=False,maxiter=10,iprint=False,atol=1E-8,)
        solve_res.linear_solver = csdl.DirectSolver()
        U = solve_res(K, Fi)








        # recover the elemental forces/moments:
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            element_loads = self.create_output(beam_name + '_element_loads', shape=(n-1,6), val=0)
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
                ans = csdl.matvec(kp,csdl.matvec(T,d))
                self.register_output(element_name + 'local_loads', ans)
                element_loads[i,:] = csdl.reshape(ans[0:6], (1,6))



        # parse the displacements to get the new nodal coordinates:
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            d = self.create_output(beam_name + '_displacement', shape=(n,3), val=0)

            for i in range(n - 1):
                element_name = beam_name + '_element_' + str(i)

                node_a_position = self.declare_variable(element_name + 'node_a', shape=(3))
                node_b_position = self.declare_variable(element_name + 'node_b', shape=(3))
                a, b =  node_index[node_dict[beam_name][i]], node_index[node_dict[beam_name][i + 1]]

                # get the nodal displacements for the current element:
                dna, dnb = U[a*6:a*6 + 3], U[b*6:b*6 + 3]
                self.register_output(element_name + 'node_a_def', node_a_position + dna)
                self.register_output(element_name + 'node_b_def', node_b_position + dnb)

                d[i,:] = csdl.reshape(dna, (1,3))
            d[n - 1,:] = csdl.reshape(dnb, (1,3))


        

        # perform a stress recovery:
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])

            element_stress = self.create_output(beam_name + '_element_stress', shape=(n-1,5), val=0)
            element_axial_stress = self.create_output(beam_name + '_element_axial_stress', shape=(n-1,5), val=0)
            element_shear_stress = self.create_output(beam_name + '_element_shear_stress', shape=(n-1), val=0)
            element_torsional_stress = self.create_output(beam_name + '_element_torsional_stress', shape=(n-1,5), val=0)

            fwd = self.create_output(beam_name + '_fwd', shape=(n,5), val=0)
            rev = self.create_output(beam_name + '_rev', shape=(n,5), val=0)

            for i in range(n - 1):
                element_name = beam_name + '_element_' + str(i)

                self.add(StressBox(name=element_name), name=element_name + 'StressBox')
                element_stress[i,:] = csdl.reshape(self.declare_variable(element_name + '_stress_array', shape=(5)), new_shape=(1,5))
                fwd[i,:] = csdl.reshape(self.declare_variable(element_name + '_stress_array', shape=(5)), new_shape=(1,5))
                rev[i+1,:] = csdl.reshape(self.declare_variable(element_name + '_stress_array', shape=(5)), new_shape=(1,5))
                element_axial_stress[i,:] = csdl.reshape(self.declare_variable(element_name + '_axial_stress', shape=(5)), new_shape=(1,5))
                element_shear_stress[i] = self.declare_variable(element_name + '_shear_stress', shape=(1))
                element_torsional_stress[i,:] = csdl.reshape(self.declare_variable(element_name + '_torsional_stress', shape=(5)), new_shape=(1,5))

            stress = (fwd + rev)/2
            self.register_output(beam_name + '_stress', stress)





        # buckling:
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            E = beams[beam_name]['E']
            v = 0.33 # Poisson's ratio
            k = 3.0

            bkl = self.create_output(beam_name + '_bkl', shape=(n - 1), val=0)
            for i in range(n - 1):
                element_name = beam_name + '_element_' + str(i)

                wb = self.declare_variable(element_name + '_w')
                hb = self.declare_variable(element_name + '_h')
                tcapb = self.declare_variable(element_name + '_tcap')
                #a = self.declare_variable(element_name + 'L')

                critical_stress = k*E*(tcapb/wb)**2/(1 - v**2) # Roark's simply-supported panel buckling
                #self.print_var(critical_stress)

                actual_stress_array = self.declare_variable(element_name + '_stress_array', shape=(5))
                actual_stress = (actual_stress_array[0] + actual_stress_array[1])/2

                bkl[i] = actual_stress/critical_stress # greater than 1 = bad





        # output dummy forces and moments for CADDEE:
        zero = self.declare_variable('zero_vec', shape=(3), val=0)
        self.register_output('F', 1*zero)
        self.register_output('M', 1*zero)