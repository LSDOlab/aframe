import numpy as np
import csdl_alpha as csdl
from dataclasses import dataclass
# import aframe as af

@dataclass
class Material:
    name: str
    E: float
    G: float
    rho: float
    v: float

    def __str__(self):
        return f'{self.name}: E={self.E} Pa, G={self.G} Pa, rho={self.rho} kg/m^3'

    
@dataclass
class CSTube:
    radius: csdl.Variable
    thickness: csdl.Variable

    @property
    def type(self):
        return 'tube'

    @property
    def area(self):
        inner_radius, outer_radius = self.radius - self.thickness, self.radius
        return np.pi * (outer_radius**2 - inner_radius**2)
    
    @property
    def ix(self):
        inner_radius, outer_radius = self.radius - self.thickness, self.radius
        return np.pi * (outer_radius**4 - inner_radius**4) / 2

    @property
    def iy(self):
        inner_radius, outer_radius = self.radius - self.thickness, self.radius
        return np.pi * (outer_radius**4 - inner_radius**4) / 4
    
    @property
    def iz(self):
        inner_radius, outer_radius = self.radius - self.thickness, self.radius
        return np.pi * (outer_radius**4 - inner_radius**4) / 4
    
    def __post_init__(self):

        if type(self.radius) != csdl.Variable:
            print('radius type is not csdl.Variable')
        
        if type(self.thickness) != csdl.Variable:
            print('thickness type is not csdl.Variable')


# @dataclass
# class Beam:
#     name: str
#     mesh: csdl.Variable
#     material: Material
#     cs: CSTube

#     def add_boundary_condition(self, node):
#         self.bc_node = node

#     def add_load(self, loads):
#         self.loads = loads

#     def __post_init__(self):

#         if type(self.mesh) != csdl.Variable:
#             print('mesh is not a csdl.Variable')
        
#         if type(self.material) != Material:
#             print('material is not a Material dataclass')

class Beam:
    def __init__(self, name, mesh, material, cs):

        # required
        self.name = name
        self.mesh = mesh
        self.material = material
        self.cs = cs
        self.num_nodes = len(mesh.value)
        self.num_elements = self.num_nodes - 1

        # optional
        self.bc = []
        self.loads = np.zeros((self.num_nodes, 6))

        if type(self.mesh) != csdl.Variable:
            print('mesh type is not csdl.Variable')

        if cs.area.shape != (self.num_elements,):
            raise ValueError('CS shape does not match number of elements')

    def add_boundary_condition(self, node, dof):
        bc_dict = {'node': node, 'dof': dof, 'beam_name': self.name}
        self.bc.append(bc_dict)

    def add_load(self, loads):
        self.loads += loads





class Solution:
    def __init__(self, displacement, stress):

        self.displacement = displacement
        self.stress = stress

    def get_displacement(self, beam):
        return self.displacement[beam.name]
    
    def get_stress(self, beam):
        return self.stress[beam.name]