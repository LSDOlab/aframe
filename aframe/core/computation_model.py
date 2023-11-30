import csdl
import numpy as np
from aframe.core.stress import StressBox
from aframe.core.model import Model
from aframe.core.moment_of_inertia import MomentofInertia


class ComputationModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_elements', types=int)
        self.parameters.declare('elements')
        self.parameters.declare('bounds')
        self.parameters.declare('dim', types=int)
        self.parameters.declare('node_dict', types=dict)
        self.parameters.declare('node_index', types=dict)
        self.parameters.declare('element_density_list', types=list)
        self.parameters.declare('beams', types=dict)
        self.parameters.declare('num_unique_nodes', types=int)
    
    def define(self):
        elements = self.parameters['elements']
        num_elements = self.parameters['num_elements']
        dim = self.parameters['dim']
        bounds = self.parameters['bounds']
        node_dict = self.parameters['node_dict']
        node_index = self.parameters['node_index']
        element_density_list = self.parameters['element_density_list']
        beams = self.parameters['beams']
        num_unique_nodes = self.parameters['num_unique_nodes']

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
        total_mass = self.register_output('mass', csdl.sum(m_vec))
        self.register_output('struct_mass', 1*total_mass)

        # self.print_var(total_mass)
        
        cg = csdl.sum(rm_vec, axes=(0,))/csdl.expand(total_mass, (3))
        self.register_output('cg_vector', cg)

        self.register_output('cgx', cg[0])
        self.register_output('cgy', cg[1]*0) # zeroed to make opt converge better and stuff
        self.register_output('cgz', cg[2])

        # compute moments of inertia:
        moment_of_inertia = MomentofInertia(
            elements=elements,
        )
        self.add(moment_of_inertia, 'moments_of_inertia')
       
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
            k = 6.3
            num_ribs = 12

            length_helper = self.create_output(beam_name + '_length_helper', shape=(n - 1), val=0)
            for i in range(n - 1):
                element_name = beam_name + '_element_' + str(i)
                length_helper[i] = self.declare_variable(element_name + 'L')
            total_beam_length = csdl.sum(length_helper)
            length_per_rib = total_beam_length/num_ribs

            # bkl = self.create_output(beam_name + '_bkl', shape=(n - 1), val=0)
            top_bkl = self.create_output(beam_name + '_top_bkl', shape=(n - 1), val=0)
            bot_bkl = self.create_output(beam_name + '_bot_bkl', shape=(n - 1), val=0)
            for i in range(n - 1):
                element_name = beam_name + '_element_' + str(i)

                wb = self.declare_variable(element_name + '_w')
                # tcapb = self.declare_variable(element_name + '_tcap')
                ttopb = self.declare_variable(element_name + '_ttop')
                tbotb = self.declare_variable(element_name + '_tbot')

                critical_stress_top = k*E*(ttopb/wb)**2/(1 - v**2) # Roark's simply-supported panel buckling
                critical_stress_bot = k*E*(tbotb/wb)**2/(1 - v**2)

                #actual_stress_array = self.declare_variable(element_name + '_stress_array', shape=(5))
                #actual_stress = (actual_stress_array[0] + actual_stress_array[1])/2
                axial_stress_array = self.declare_variable(element_name + '_axial_stress', shape=(5))
                top_stress = (axial_stress_array[0] + axial_stress_array[1])/2
                bot_stress = (axial_stress_array[2] + axial_stress_array[3])/2

                # bkl[i] = actual_stress/critical_stress # greater than 1 = bad
                top_bkl[i] = top_stress/critical_stress_top
                bot_bkl[i] = bot_stress/critical_stress_bot

            # for i in range(num_ribs - 1): # iterate over the panels
                





        # # output dummy forces and moments for CADDEE:
        # zero = self.declare_variable('zero_vec', shape=(3), val=0)
        # self.register_output('F', 1*zero)
        # self.register_output('M', 1*zero)

        # # buckling:
        # for beam_name in beams:
        #     n = len(beams[beam_name]['nodes'])
        #     E = beams[beam_name]['E']
        #     v = 0.33 # Poisson's ratio
        #     k = 3.0

        #     bkl = self.create_output(beam_name + '_bkl', shape=(n - 1), val=0)
        #     for i in range(n - 1):
        #         element_name = beam_name + '_element_' + str(i)

        #         wb = self.declare_variable(element_name + '_w')
        #         hb = self.declare_variable(element_name + '_h')
        #         tcapb = self.declare_variable(element_name + '_tcap')
        #         #a = self.declare_variable(element_name + 'L')

        #         critical_stress = k*E*(tcapb/wb)**2/(1 - v**2) # Roark's simply-supported panel buckling
        #         #self.print_var(critical_stress)

        #         actual_stress_array = self.declare_variable(element_name + '_stress_array', shape=(5))
        #         actual_stress = (actual_stress_array[0] + actual_stress_array[1])/2

        #         bkl[i] = actual_stress/critical_stress # greater than 1 = bad





        # # output dummy forces and moments for CADDEE:
        # # zero = self.declare_variable('zero_vec', shape=(3), val=0)
        # # self.register_output('F', 1*zero)
        # # self.register_output('M', 1*zero)