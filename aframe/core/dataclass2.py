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



@dataclass
class CSBox:
    ttop: csdl.Variable
    tbot: csdl.Variable
    tweb: csdl.Variable
    height: csdl.Variable
    width: csdl.Variable

    @property
    def type(self):
        return 'box'

    @property
    def area(self):
        w_i = self.width - 2 * self.tweb
        h_i = self.height - self.ttop - self.tbot
        return self.width * self.height - w_i * h_i
    
    @property
    def ix(self):
        tcap = (self.ttop + self.tbot) / 2
        numerator = 2 * self.tweb * tcap * (self.width - self.tweb)**2 * (self.height - tcap)**2
        denominator = self.width * self.tweb + self.height * tcap - self.tweb**2 - tcap**2
        return numerator / denominator

    @property
    def iy(self):
        w_i = self.width - 2 * self.tweb
        h_i = self.height - self.ttop - self.tbot
        return (self.width * self.height**3 - w_i * h_i**3) / 12
    
    @property
    def iz(self):
        w_i = self.width - 2 * self.tweb
        h_i = self.height - self.ttop - self.tbot
        return (self.width**3 * self.height - w_i**3 * h_i) / 12
    
    def __post_init__(self):

        if type(self.ttop) != csdl.Variable or type(self.tbot) != csdl.Variable:
            print('ttop/tbot type is not csdl.Variable')
        
        if type(self.height) != csdl.Variable or type(self.width) != csdl.Variable:
            print('height/width type is not csdl.Variable')


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

        if self.mesh.shape != (self.num_nodes, 3):
            raise ValueError('incorrect mesh shape (should be num_nodes by 3)')

        if cs.area.shape != (self.num_elements,):
            raise ValueError('CS shape does not match the number of elements')

    def add_boundary_condition(self, node, dof):
        bc_dict = {'node': node, 'dof': dof, 'beam_name': self.name}
        self.bc.append(bc_dict)

    def add_load(self, loads):

        if loads.shape != (self.num_nodes, 6):
            raise ValueError('incorrect loads shape (should be num_nodes by 6)')
        
        self.loads += loads





class Solution:
    def __init__(self, displacement, stress, bkl, cg):

        self.displacement = displacement
        self.stress = stress
        self.cg = cg
        self.bkl = bkl

    def get_displacement(self, beam):
        return self.displacement[beam.name]
    
    def get_stress(self, beam):
        return self.stress[beam.name]
    
    def get_bkl(self, beam):

        if beam.cs.type == 'tube':
            raise NotImplementedError('bkl not available for tubes')
        
        return self.bkl[beam.name]