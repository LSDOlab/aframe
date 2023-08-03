import numpy as np
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL


class MassPropModule(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('elements')
        self.parameters.declare('element_density_list')

    def define(self):
        elements = self.parameters['elements']
        element_density_list = self.parameters['element_density_list']

        # calculate the mass and the position of each element:
        rm_vec = self.create_output('rm_vec',shape=(len(elements),3),val=0)
        m_vec = self.create_output('m_vec',shape=(len(elements)),val=0)
        
        for i, element_name in enumerate(elements):
            rho = element_density_list[i]

            A = self.register_module_input(element_name + '_A')
            L = self.register_module_input(element_name + 'L')

            # compute the element volume:
            V = A*L

            # compute the element mass:
            m = V*rho
            self.register_output(element_name + 'm', m)


            # get the (undeformed) position vector of the cg for each element:
            r_a = self.register_module_input(element_name + 'node_a', shape=(3))
            r_b = self.register_module_input(element_name + 'node_b', shape=(3))

            r_cg = (r_a + r_b)/2
            self.register_output(element_name+'r_cg', r_cg)

            # assign r_cg to the r*mass vector:
            rm_vec[i,:] = csdl.reshape(r_cg*csdl.expand(m, (3)), new_shape=(1,3))
            # assign the mass to the mass vector:
            m_vec[i] = m



        # compute the center of gravity for the entire structure:
        total_mass = csdl.sum(m_vec)
        self.register_module_output('mass', total_mass)
        self.register_output('struct_mass', 1*total_mass)
        #self.print_var(total_mass)
        
        sum_rm = csdl.sum(rm_vec,axes=(0,))

        cg = sum_rm/csdl.expand(total_mass, (3))
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

            # get the mass:
            m = m_vec[i]

            # get the position vector:
            r = self.register_module_input(element_name+'r_cg',shape=(3))
            x = r[0]
            y = r[1]
            z = r[2]

            rxx = y**2 + z**2
            ryy = x**2 + z**2
            rzz = x**2 + y**2
            rxz = x*z

            eixx[i] = m*rxx
            eiyy[i] = m*ryy
            eizz[i] = m*rzz
            eixz[i] = m*rxz
            
        
        # sum the m*r vector to get the moi:
        Ixx = csdl.sum(eixx)
        Iyy = csdl.sum(eiyy)
        Izz = csdl.sum(eizz)
        Ixz = csdl.sum(eixz)

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



        
            

        


"""
class MassProp(csdl.Model):
    def initialize(self):
        self.parameters.declare('elements')
        self.parameters.declare('element_density_list')

    def define(self):
        elements = self.parameters['elements']
        element_density_list = self.parameters['element_density_list']

        # calculate the mass and the position of each element:
        rm_vec = self.create_output('rm_vec',shape=(len(elements),3),val=0)
        m_vec = self.create_output('m_vec',shape=(len(elements)),val=0)
        
        for i, element_name in enumerate(elements):
            rho = element_density_list[i]

            A = self.declare_variable(element_name + '_A')
            L = self.declare_variable(element_name + 'L')

            # compute the element volume:
            V = A*L

            # compute the element mass:
            m = V*rho
            self.register_output(element_name + 'm', m)


            # get the (undeformed) position vector of the cg for each element:
            r_a = self.declare_variable(element_name + 'node_a', shape=(3))
            r_b = self.declare_variable(element_name + 'node_b', shape=(3))

            r_cg = (r_a + r_b)/2
            self.register_output(element_name+'r_cg', r_cg)

            # assign r_cg to the r*mass vector:
            rm_vec[i,:] = csdl.reshape(r_cg*csdl.expand(m, (3)), new_shape=(1,3))
            # assign the mass to the mass vector:
            m_vec[i] = m



        # compute the center of gravity for the entire structure:
        total_mass = csdl.sum(m_vec)
        self.register_output('mass', total_mass)
        self.register_output('struct_mass', 1*total_mass)
        #self.print_var(total_mass)
        
        sum_rm = csdl.sum(rm_vec,axes=(0,))

        cg = sum_rm/csdl.expand(total_mass, (3))
        self.register_output('cg',cg)


        self.register_output('cgx', cg[0])
        self.register_output('cgy', cg[1]*0)
        self.register_output('cgz', cg[2])



        
        # compute moments of inertia:
        eixx = self.create_output('eixx',shape=(len(elements)),val=0)
        eiyy = self.create_output('eiyy',shape=(len(elements)),val=0)
        eizz = self.create_output('eizz',shape=(len(elements)),val=0)
        eixz = self.create_output('eixz',shape=(len(elements)),val=0)
        for i, element_name in enumerate(elements):

            # get the mass:
            m = m_vec[i]

            # get the position vector:
            r = self.declare_variable(element_name+'r_cg',shape=(3))
            x = r[0]
            y = r[1]
            z = r[2]

            rxx = y**2 + z**2
            ryy = x**2 + z**2
            rzz = x**2 + y**2
            rxz = x*z

            eixx[i] = m*rxx
            eiyy[i] = m*ryy
            eizz[i] = m*rzz
            eixz[i] = m*rxz
            
        
        # sum the m*r vector to get the moi:
        Ixx = csdl.sum(eixx)
        Iyy = csdl.sum(eiyy)
        Izz = csdl.sum(eizz)
        Ixz = csdl.sum(eixz)
        self.register_output('ixx', Ixx)
        self.register_output('iyy', Iyy)
        self.register_output('izz', Izz)
        self.register_output('ixz', Ixz)



"""
            

        