import numpy as np
import csdl
from aframe.core.model import Model
from aframe.core.stress import StressBox
from aframe.core.computation_model import ComputationModel




class Aframe(csdl.Model):

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
            mesh_in = self.declare_variable(beam_name + '_mesh', shape=(n,3), val=default_val)
            # self.print_var(mesh_in)
            if mesh_units == 'm': mesh = 1*mesh_in
            elif mesh_units == 'ft': mesh = 0.304*mesh_in


            if cs == 'box':
                width = self.register_module_input(beam_name + '_width', shape=(n), promotes=True)
                height = self.register_module_input(beam_name + '_height', shape=(n), promotes=True)
                # tweb_in = self.register_module_input(beam_name + '_tweb', shape=(n))
                # tcap_in = self.register_module_input(beam_name + '_tcap', shape=(n))
                tweb_in = self.register_module_input(beam_name + '_tweb', shape=(n))
                tbot_in = self.register_module_input(beam_name + '_tbot', shape=(n))
                ttop_in = self.register_module_input(beam_name + '_ttop', shape=(n))

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
                    #tcap = self.register_output(element_name + '_tcap', (tcap_in[i]+tcap_in[i+1])/2)
                    ttop = self.register_output(element_name + '_ttop', (ttop_in[i]+ttop_in[i+1])/2)
                    tbot = self.register_output(element_name + '_tbot', (tbot_in[i]+tbot_in[i+1])/2)

                    tcap_avg = (ttop + tbot)/2

                    # compute the box-beam cs properties
                    # w_i, h_i = w - 2*tweb, h - 2*tcap
                    w_i, h_i = w - 2*tweb, h - ttop - tbot
                    A = self.register_output(element_name + '_A', (((w*h) - (w_i*h_i))**2 + 1E-14)**0.5)
                    iyo[i] = Iy = self.register_output(element_name + '_Iy', (w*(h**3) - w_i*(h_i**3))/12)
                    izo[i] = Iz = self.register_output(element_name + '_Iz', ((w**3)*h - (w_i**3)*h_i)/12)
                    # jo[i] = J = self.register_output(element_name + '_J', (w*h*(h**2 + w**2)/12) - (w_i*h_i*(h_i**2 + w_i**2)/12))
                    jo[i] = J = self.register_output(element_name + '_J', (2*tweb*tcap_avg*(w-tweb)**2*(h-tcap_avg)**2)/(w*tweb+h*tcap_avg-tweb**2-tcap_avg**2)) # Darshan's formula
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

        comp_model = ComputationModel(
            num_elements=num_elements,
            elements=elements,
            bounds=bounds,
            dim=dim, 
            node_dict=node_dict,
            node_index=node_index,
            element_density_list=element_density_list,
            beams=beams,
            num_unique_nodes=num_unique_nodes,
        )
        self.add(comp_model, 'comp_model')
