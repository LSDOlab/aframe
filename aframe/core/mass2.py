import numpy as np
import csdl_alpha as csdl
import aframe as af
from dataclasses import dataclass


@dataclass
class MassProperties:
    mass: csdl.Variable
    cg: csdl.Variable

    # @property
    # def cg(self):
    #     return self.cg
    
    # @property
    # def mass(self):
    #     return self.mass



class FrameMass():
    def __init__(self):
        self.beams = []
        self.joints = []

    def add_beam(self, beam):
        self.beams.append(beam)

    def evaluate(self):


        mass, rmvec = 0, 0
        for beam in self.beams:
            rho = beam.material.density
            area = beam.cs.area
            mesh = beam.mesh

            beam_mass, beam_rmvec = 0, 0
            for i in range(beam.num_elements):

                L = csdl.norm(mesh[i + 1, :] - mesh[i, :])
                element_mass = area[i] * L * rho

                # element cg is the nodal average
                element_cg = (mesh[i + 1, :] + mesh[i, :]) / 2

                beam_mass = beam_mass + element_mass
                beam_rmvec = beam_rmvec + element_cg * element_mass

            mass = mass + beam_mass
            rmvec = rmvec + beam_rmvec


        # compute the undeformed cg for the frame
        cg = rmvec / mass


        return MassProperties(mass=mass, cg=cg)