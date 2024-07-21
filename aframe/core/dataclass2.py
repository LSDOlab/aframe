import numpy as np
import csdl_alpha as csdl
from typing import Union
from dataclasses import dataclass
import aframe as af
from aframe.core.materials import Material



# no longer used
# @dataclass
# class Material:
#     name: str
#     E: float
#     G: float
#     density: float
#     nu: float

#     @property
#     def type(self):
#         return 'isotropic'



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
    def neutral_axis(self):
        Atop = self.ttop * (self.width - 2 * self.tweb)
        Abot = self.tbot * (self.width - 2 * self.tweb)

        centroid_z = (Atop - Abot) / self.width
        centroid_y = 0

        return (centroid_y, centroid_z)
    
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
        ix = self.iy + self.iz
        return ix

        # Older version
        # tcap = (self.ttop + self.tbot) / 2
        # numerator = 2 * self.tweb * tcap * (self.width - self.tweb)**2 * (self.height - tcap)**2
        # denominator = self.width * self.tweb + self.height * tcap - self.tweb**2 - tcap**2
        # return numerator / denominator

    @property
    def iy(self):
        neutral_axis = self.neutral_axis
        cy_neutral = neutral_axis[0]
        cz_neutral = neutral_axis[1]

        # top
        area_top = self.ttop * self.width
        iy_top = self.ttop ** 3 * self.width / 12
        cz_top = self.height / 2 - self.ttop / 2
        d_top = cz_top - cz_neutral
        iy_top_centroid = iy_top + area_top * d_top**2

        # bottom
        area_bot = self.tbot * self.width
        iy_bot = self.tbot ** 3 * self.width / 12
        cz_bot = -self.height / 2 + self.tbot / 2
        d_bot = cz_bot - cz_neutral
        iy_bot_centroid = iy_bot + area_bot * d_bot**2

        # front/ rear
        area_front = area_rear = self.tweb * (self.height - self.ttop - self.tbot)
        iy_front = iy_rear = (self.height - self.ttop - self.tbot) ** 3 * self.tweb / 12

        d_rear = cz_neutral
        d_front = cz_neutral

        iy_rear_centroid = iy_rear + area_rear * d_rear**2
        iy_front_centroid = iy_front + area_front * d_front**2

        iy_total = iy_top_centroid + iy_bot_centroid + iy_rear_centroid + iy_front_centroid

        # Older version
        # w_i = self.width - 2 * self.tweb
        # h_i = self.height - self.ttop - self.tbot
        # return (self.width * self.height**3 - w_i * h_i**3) / 12

        return iy_total
    
    @property
    def iz(self):
        neutral_axis = self.neutral_axis
        cy_neutral = neutral_axis[0]
        cz_neutral = neutral_axis[1]

        # top / bottom
        area_top = self.ttop * self.width
        iz_top = self.ttop * self.width**3 / 12

        area_bot = self.tbot * self.width
        iz_bot = self.tbot * self.width**3 / 12

        d_top = cy_neutral
        d_bot = cy_neutral

        iz_top_centroid = iz_top +  area_top * d_top**2
        iz_bot_centroid = iz_bot +  area_bot * d_bot**2

        # front/ rear
        area_front = area_rear = self.tweb * (self.height - self.ttop - self.tbot)
        iz_front = iz_rear = (self.height - self.ttop - self.tbot) * self.tweb**3 / 12

        cy_front = -self.width / 2 + self.tweb / 2
        cy_rear = self.width / 2 - self.tweb / 2

        d_front = cy_front - cy_neutral
        d_rear = cy_rear - cy_neutral

        iz_front_centroid = iz_front + area_front * d_front**2
        iz_rear_centroid = iz_rear + area_rear * d_rear**2

        iz_total = iz_top_centroid + iz_bot_centroid + iz_front_centroid + iz_rear_centroid

        return iz_total

        # Older version
        # w_i = self.width - 2 * self.tweb
        # h_i = self.height - self.ttop - self.tbot
        # return (self.width**3 * self.height - w_i**3 * h_i) / 12
    
    def __post_init__(self):

        if type(self.ttop) != csdl.Variable or type(self.tbot) != csdl.Variable:
            print('ttop/tbot type is not csdl.Variable')
        
        if type(self.height) != csdl.Variable or type(self.width) != csdl.Variable:
            print('height/width type is not csdl.Variable')




@dataclass
class CSEllipse:
    semi_major_axis: csdl.Variable
    semi_minor_axis: csdl.Variable

    @property
    def type(self):
        return 'ellipse'

    @property
    def area(self):
        return np.pi * self.semi_major_axis * self.semi_minor_axis
    
    @property
    def ix(self):
        beta = 1 / ((1 + (self.semi_minor_axis / self.semi_major_axis)**2)**0.5)
        return (np.pi / 2) * self.semi_major_axis * self.semi_minor_axis**3 * beta

    @property
    def iy(self):
        return np.pi / 4 * self.semi_major_axis * self.semi_minor_axis**3
    
    @property
    def iz(self):
        return np.pi / 4 * self.semi_major_axis**3 * self.semi_minor_axis
    
    def __post_init__(self):

        if type(self.semi_major_axis) != csdl.Variable:
            print('semi_major_axis type is not csdl.Variable')
        
        if type(self.semi_minor_axis) != csdl.Variable:
            print('semi_minor_axis type is not csdl.Variable')





class Beam:
    def __init__(self, name:str, mesh:csdl.Variable, material:Material, cs:Union[CSBox, CSTube]):
        """Initialize a beam.

        Parameters
        ----------
        name : str
            The name of the instance.
        mesh : csdl.Variable
            The mesh variable representing the geometry.
        material : Material
            The material of the instance.
        cs : Union[CSBox, CSTube]
            The cross-section of the instance.

        Raises
        ------
        ValueError
            If the mesh type is not csdl.Variable.
        ValueError
            If the mesh shape is incorrect (should be num_nodes by 3).
        ValueError
            If the CS shape does not match the number of elements.
        """
        # required
        self.name = name
        self.mesh = mesh
        self.material = material
        self.cs = cs
        self.num_nodes = mesh.shape[0]
        # self.num_nodes = len(mesh.value)
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

        if len(dof) != 6:
            raise ValueError('dof should have length 6 e.g., [1, 1, 1, 1, 1, 1]')
        
        if not isinstance(node, int):
            raise ValueError('BC node should be an integer')
        
        if node > self.num_nodes - 1 or node < 0:
            raise ValueError('BC node out of range for ', self.name)

        bc_dict = {'node': node, 'dof': dof, 'beam_name': self.name}
        self.bc.append(bc_dict)

    def add_load(self, loads):

        if loads.shape != (self.num_nodes, 6):
            raise ValueError('incorrect loads shape (should be num_nodes by 6)')
        
        self.loads += loads





class Solution:
    def __init__(self, displacement: dict, mesh: dict, stress: dict, 
                 bkl: dict, cg: dict, dcg: dict, M: csdl.Variable,
                 K: csdl.Variable, F: csdl.Variable, u0:csdl.Variable,
                 node_dictionary: dict, index: dict):

        self.displacement = displacement
        self.mesh = mesh
        self.stress = stress
        self.cg = cg
        self.dcg = dcg
        self.bkl = bkl
        self.M = M
        self.K = K
        self.F = F
        self.u0 = u0
        self.node_dictionary = node_dictionary
        self.index = index

    def get_displacement(self, beam: Beam):
        return self.displacement[beam.name]
    
    def get_stress(self, beam: Beam):
        return self.stress[beam.name]
    
    def get_mesh(self, beam: Beam):
        return self.mesh[beam.name]
    
    def get_bkl(self, beam: Beam):

        if beam.cs.type == 'tube':
            raise NotImplementedError('bkl not available for tubes')
        
        return self.bkl[beam.name]