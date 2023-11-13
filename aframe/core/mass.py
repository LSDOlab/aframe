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

            #self.print_var(tweb)
            #self.print_var(tcap)

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



        total_mass = csdl.sum(m_vec) # sums the beam masses
        self.register_module_output('mass', total_mass)

        self.print_var(total_mass)