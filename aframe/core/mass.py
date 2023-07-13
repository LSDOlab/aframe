import numpy as np
import csdl
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL


# this file contains an entirely separate mass computation model for aframe
# it requires the same inputs as an ordinary aframe beam model 
# only valid for box beams


class Mass(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('beams', default={})
        self.parameters.declare('mesh_units', default='m')

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
            tweb = self.register_module_input(beam_name + '_tweb', shape=(n - 1))
            tcap = self.register_module_input(beam_name + '_tcap', shape=(n - 1))

            # get cs area:
            w_i = w - 2*tweb
            h_i = h - 2*tcap
            A = (((w*h) - (w_i*h_i))**2 + 1E-14)**0.5

            # iterate over the elements:
            em_vec = self.create_output('em_vec',shape=(n - 1),val=0)
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