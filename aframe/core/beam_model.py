import numpy as np
import csdl
from aframe.core.dataclass import Beam, CSProp

class BeamModel(csdl.Model):

    def initialize(self):
        self.beams = self.parameters.declare('beams', default=[])
        self.parameters.declare('boundary_conditions', default=[])
        self.parameters.declare('joints', default=[])


    def stiffness(self, beam, mesh, cs):

        # iterate over the beam elements
        for i in range(beam.num_nodes - 1):
            E, G = beam.material.E, beam.material.G

            L = csdl.pnorm(mesh[i + 1, :] - mesh[i, :])

            A = cs.A[i]
            Ix = cs.Ix[i]
            Iy = cs.Iy[i]
            Iz = cs.Iz[i]
            J = cs.J[i]
            Q = cs.Q[i]

            kp = self.create_output(beam.name + str(i) + 'kp', shape=(12,12), val=0)
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

        return kp


    def define(self):
        beams = self.parameters['beams']
        boundary_conditions = self.parameters['boundary_conditions']
        joints = self.parameters['joints']



        for beam in beams:
            mesh = self.declare_variable(beam.name + '_mesh', shape=(beam.num_nodes, 3))
            # self.register_output(beam.name + '_mesh_out', 1*mesh)

            if beam.cs == 'tube':
                radius = self.declare_variable(beam.name + '_radius', shape=(beam.num_nodes))
                thickness = self.declare_variable(beam.name + '_thickness', shape=(beam.num_nodes))
                inner_radius, outer_radius = radius - thickness, radius

                A = np.pi * (outer_radius**2 - inner_radius**2)
                Iy = np.pi * (outer_radius**4 - inner_radius**4) / 4.0
                Iz = np.pi * (outer_radius**4 - inner_radius**4) / 4.0
                Ix = J = Q = np.pi * (outer_radius**4 - inner_radius**4) / 2.0
                cs = CSProp(A=A, Ix=Ix, Iy=Iy, Iz=Iz, J=J, Q=Q)

                kp = self.stiffness(beam=beam, mesh=mesh, cs=cs)
                self.print_var(kp)