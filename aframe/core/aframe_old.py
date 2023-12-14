import numpy as np
import csdl
from aframe.core.massprop import MassPropModule as MassProp
from aframe.core.model import Model
from aframe.core.buckle import Buckle
from aframe.core.nodal_stress import NodalStressBox
from aframe.core.stress import StressBox





class Aframe(csdl.Model):

    def initialize(self):
        self.parameters.declare('beams', default={})
        self.parameters.declare('joints', default={})
        self.parameters.declare('bounds', default={})
        self.parameters.declare('mesh_units', default='m')


    def box(self, name, w, h, tweb, tcap):
        w_i = w - 2*tweb
        h_i = h - 2*tcap
        # A = (w*h) - (w_i*h_i)
        A = (((w*h) - (w_i*h_i))**2 + 1E-14)**0.5 # for robustness
        Iz = ((w**3)*h - (w_i**3)*h_i)/12
        Iy = (w*(h**3) - w_i*(h_i**3))/12
        J = (w*h*(h**2 + w**2)/12) - (w_i*h_i*(h_i**2 + w_i**2)/12)
        # J = ( 2*tweb*tcap*(w-tweb)**2*(h-tcap)**2 ) / ( w*tweb + h*tcap - tweb**2 -tcap**2 )
        # Q = 2*(h/2)*tweb*(h/4) + (w - 2*tweb)*tcap*((h/2) - (tcap/2))

        Q = (A/2)*(h/4)

        self.register_output(name + '_A', A)
        self.register_output(name + '_Ix', 1*J) # I think J is the same as Ix...
        self.register_output(name + '_Iy', Iy)
        self.register_output(name + '_Iz', Iz)
        self.register_output(name + '_J', J)
        self.register_output(name + '_Q', Q)


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
        mesh_in = self.declare_variable(beam_name + '_mesh', shape=(n,3), val=default_val)
        # self.print_var(mesh_in)
        if mesh_units == 'm': mesh = 1*mesh_in
        elif mesh_units == 'ft': mesh = 0.304*mesh_in

        # iterate over each element:
        for i in range(n - 1):
            self.register_output(beam_name + '_element_' + str(i) + 'node_a', csdl.reshape(mesh[i, :], (3)))
            self.register_output(beam_name + '_element_' + str(i) + 'node_b', csdl.reshape(mesh[i + 1, :], (3)))


        if cs == 'box':
            width = self.declare_variable(beam_name + '_width', shape=(n))
            height = self.declare_variable(beam_name + '_height', shape=(n))
            tweb = self.declare_variable(beam_name + '_tweb', shape=(n))
            tcap = self.declare_variable(beam_name + '_tcap', shape=(n))


            # parse elemental outputs
            w, h = self.create_output(beam_name + '_w', shape=(n - 1), val=0), self.create_output(beam_name + '_h', shape=(n - 1), val=0)

            iyo, izo, jo = self.create_output(beam_name + '_iyo', shape=(n - 1), val=0), self.create_output(beam_name + '_izo', shape=(n - 1), val=0), self.create_output(beam_name + '_jo', shape=(n - 1), val=0)

            for i in range(n - 1):
                element_name = beam_name + '_element_' + str(i)
                if mesh_units == 'm': converted_width, converted_height = 1*width, 1*height
                elif mesh_units == 'ft': converted_width, converted_height = 0.304*width, 0.304*height

                w[i], h[i] = (converted_width[i] + converted_width[i + 1])/2, (converted_height[i] + converted_height[i + 1])/2
                self.register_output(element_name + '_h', h[i])
                self.register_output(element_name + '_w', w[i])

                # compute the box-beam cs properties
                self.box(name=element_name, w=w[i], h=h[i], tweb=(tweb[i]+tweb[i+1])/2, tcap=(tcap[i]+tcap[i+1])/2)

                self.register_output(element_name + '_tweb', (tweb[i] + tweb[i+1])/2)
                self.register_output(element_name + '_tcap', (tcap[i] + tcap[i+1])/2)


                iyo[i] = self.declare_variable(element_name + '_Iy')
                izo[i] = self.declare_variable(element_name + '_Iz')
                jo[i] = self.declare_variable(element_name + '_J')


            # parse nodal outputs
            for i in range(n):
                name = beam_name + str(i)
                self.box(name=name, w=width[i], h=height[i], tweb=tweb[i], tcap=tcap[i])
                self.register_output(name + '_h', height[i])
                self.register_output(name + '_w', width[i])
                self.register_output(name + '_tweb', tweb[i])
        


        # calculate the stiffness matrix for each element:
        for i in range(n - 1): self.local_stiffness(element_name=beam_name + '_element_' + str(i), E=e, G=g, node_dict=node_dict, node_index=node_index, dim=dim, i=i)










    def define(self):
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

        sum_k, sum_m = csdl.sum(helper, axes=(0, )), csdl.sum(mass_helper, axes=(0, ))

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

        # modify the global stiffness matrix and the global mass matrix with boundary conditions:
        # first remove the row/column with a boundary condition, then add a 1:
        K = self.register_output('K', csdl.matmat(csdl.matmat(mask, sum_k), mask) + mask_eye)
        # mass_matrix = self.register_output('mass_matrix', csdl.matmat(csdl.matmat(mask, sum_m), mask) + mask_eye)



        # compute the mass properties:
        self.add(MassProp(elements=elements, element_density_list=element_density_list), name='MassProp')

        # create the global loads vector:
        Fi = self.global_loads(b_index_list=b_index_list, num_unique_nodes=num_unique_nodes, node_dict=node_dict, beams=beams, node_index=node_index)

        # solve the linear system
        solve_res = self.create_implicit_operation(Model(dim=dim))
        solve_res.declare_state(state='U', residual='R')
        solve_res.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=False,maxiter=10,iprint=False,atol=1E-8,)
        solve_res.linear_solver = csdl.ScipyKrylov()
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

            """
            # get the rotations:
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
            """




        # perform a nodal stress recovery
        """
        for beam_name in beams:
            #boxflag = False
            #if beams[beam_name]['cs'] == 'box': boxflag = True
            n = len(beams[beam_name]['nodes'])
            test_nodal_loads = self.declare_variable(beam_name + '_test_nodal_loads', shape=(n-1,6))
            stress = self.create_output(beam_name + '_stress', shape=(n-1,5), val=0)
            for i in range(n - 1):
                # name = beam_name + str(i)
                name = beam_name + '_element_' + str(i)
                n_load = self.register_output(name + 'n_load', csdl.reshape(test_nodal_loads[i,:], (6)))
                self.add(NodalStressBox(name=name), name=name + 'NodalStressBox')
                node_stress = self.declare_variable(name + 'stress', shape=(5))
                stress[i,:] = csdl.reshape(node_stress, (1,5))
        """

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
        bkl = self.create_output('bkl', shape=(len(elements)))
        index = 0
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            Modulus = beams[beam_name]['E']

            s_cap = self.create_output(beam_name + '_sp_cap', shape=(n-1), val=0)
            if beams[beam_name]['cs'] == 'box':
                for i in range(n - 1):
                    element_name = beam_name + '_element_' + str(i)

                    self.add(Buckle(element_name=element_name,E=Modulus), name=element_name + 'Buckle')
                    bkl_ratio = self.declare_variable(element_name + 'bkl_ratio')
                    s_cap[i] = self.declare_variable(element_name + '_sp_cap')
                    bkl[index] = 1*bkl_ratio
                    index += 1




        
        # output dummy forces and moments for CADDEE:
        zero = self.declare_variable('zero_vec', shape=(3), val=0)
        self.register_output('F', 1*zero)
        self.register_output('M', 1*zero)




