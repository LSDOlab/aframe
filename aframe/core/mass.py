import numpy as np
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
import m3l

# this file contains an entirely separate mass computation model for aframe
# it requires the same inputs as an ordinary aframe beam model 
# only valid for box beams



class Mass(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('struct_solver', True)
        self.parameters.declare('compute_mass_properties', default=True, types=bool)

        self.parameters.declare('beams', default={})
        self.parameters.declare('mesh_units', default='ft')
        self.num_nodes = None

    def assign_attributes(self):
        self.component = self.parameters['component']
        self.mesh = self.parameters['mesh']
        self.struct_solver = self.parameters['struct_solver']
        self.compute_mass_properties = self.parameters['compute_mass_properties']

        self.beams = self.parameters['beams']
        self.mesh_units = self.parameters['mesh_units']

    def compute(self):
        beams = self.parameters['beams']
        mesh_units = self.parameters['mesh_units']

        csdl_model = MassCSDL(
            module=self,
            beams=beams,
            mesh_units=mesh_units,)
        
        return csdl_model

    def evaluate(self):

        self.name = 'mass_model'
        self.arguments = {}
        
        mass = m3l.Variable('mass', shape=(1,), operation=self)

        return mass







class MassMesh(Module):
    def initialize(self, kwargs):
        self.parameters.declare('meshes', types=dict)
        self.parameters.declare('mesh_units', default='ft')






class MassCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('beams', default={})
        self.parameters.declare('mesh_units', default='ft')

    def define(self):
        beams = self.parameters['beams']
        mesh_units = self.parameters['mesh_units']



        # create a list of element names:
        elements, element_density_list, num_elements = [], [], 0
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            num_elements += n - 1
            for i in range(n - 1): 
                elements.append(beam_name + '_element_' + str(i))
                element_density_list.append(beams[beam_name]['rho'])



        


        m_vec = self.create_output('m_vec',shape=(len(beams)),val=0) # stores the mass for each beam
        for j, beam_name in enumerate(beams):
            n = len(beams[beam_name]['nodes'])
            rho = beams[beam_name]['rho']

            

            # get the mesh:
            mesh_in = self.register_module_input(beam_name + '_mesh', shape=(n,3), promotes=True)
            if mesh_units == 'm': mesh = 1*mesh_in
            elif mesh_units == 'ft': mesh = 0.304*mesh_in
            # get the width and height meshes:
            width = self.register_module_input(beam_name + '_width', shape=(n), promotes=True)
            height = self.register_module_input(beam_name + '_height', shape=(n), promotes=True)
            w = self.create_output(beam_name + '_w', shape=(n - 1), val=0)
            h = self.create_output(beam_name + '_h', shape=(n - 1), val=0)
            # take averages of nodal meshes to get elemental meshes and convert units:
            for i in range(n - 1):
                if mesh_units == 'm':
                    w[i] = (width[i] + width[i + 1])/2
                    h[i] = (height[i] + height[i + 1])/2
                elif mesh_units == 'ft':
                    w[i] = 0.304*(width[i] + width[i + 1])/2
                    h[i] = 0.304*(height[i] + height[i + 1])/2

            # the box-beam thicknesses:
            #tweb = self.declare_variable(beam_name + '_tweb', shape=(n - 1))
            #tcap = self.declare_variable(beam_name + '_tcap', shape=(n - 1))
            tweb_in = self.register_module_input(beam_name + '_tweb', shape=(n), computed_upstream=False)
            tcap_in = self.register_module_input(beam_name + '_tcap', shape=(n), computed_upstream=False)

            tweb = self.create_output('marius_tweb', shape=(n-1), val=0)
            tcap = self.create_output('marius_tcap', shape=(n-1), val=0)
            for i in range(n - 1):
                tweb[i] = (tweb_in[i]+tweb_in[i+1])/2
                tcap[i] = (tcap_in[i]+tcap_in[i+1])/2

            # get cs area:
            w_i = w - 2*tweb
            h_i = h - 2*tcap
            A = (((w*h) - (w_i*h_i))**2 + 1E-14)**0.5

            # iterate over the elements:
            em_vec = self.create_output(beam_name + '_em_vec',shape=(n - 1),val=0)
            for i in range(n - 1):

                node_a = csdl.reshape(mesh[i, :], (3))
                node_b = csdl.reshape(mesh[i + 1, :], (3))
                L = csdl.pnorm(node_b - node_a, pnorm_type=2) + 1E-12

                V = A[i]*L
                em_vec[i] = V*rho
        
            beam_mass = csdl.sum(em_vec)
            m_vec[j] = beam_mass







        # compute the cg and moi:
        dup_m_vec = self.create_output('dup_m_vec', shape=(len(elements)), val=0)
        rm_vec = self.create_output('rm_vec', shape=(len(elements),3), val=0)
        for i, element_name in enumerate(elements):
            rho = element_density_list[i]

            A = self.declare_variable(element_name + '_A')
            L = self.declare_variable(element_name + 'L')

            # compute the element mass:
            m = self.register_output(element_name + 'm', (A*L)*rho)

            # get the (undeformed) position vector of the cg for each element:
            r_a = self.declare_variable(element_name + 'node_a', shape=(3))
            r_b = self.declare_variable(element_name + 'node_b', shape=(3))

            r_cg = self.register_output(element_name+'r_cg', (r_a + r_b)/2)
            
            # assign r_cg to the r*mass vector:
            rm_vec[i,:] = csdl.reshape(r_cg*csdl.expand(m, (3)), new_shape=(1,3))
            dup_m_vec[i] = m

        
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
            m = dup_m_vec[i]

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








        total_mass = csdl.sum(m_vec) # sums the beam masses
        self.register_module_output('mass', total_mass)
        self.print_var(total_mass)