import aframe as af
import numpy as np
import csdl_alpha as csdl
# from dataclasses import dataclass
# from typing import Optional


class CSTube:
    def __init__(self, 
                 radius:csdl.Variable,
                 thickness:csdl.Variable
                 ):
        
        self.radius = radius
        self.thickness = thickness
        self.inner_radius = radius - thickness
        self.outer_radius = radius
        self.precomp = np.pi * (self.outer_radius**4 - self.inner_radius**4)

    @property
    def area(self):
        return np.pi * (self.outer_radius**2 - self.inner_radius**2)
    
    @property
    def ix(self):
        return self.precomp / 2
    
    @property
    def iy(self):
        return self.precomp / 4
    
    @property
    def iz(self):
        return self.precomp / 4
    

    def stress(self, element_loads):

        F_x1 = element_loads[:, 0]
        # F_y1 = element_loads[:, 1]
        # F_z1 = element_loads[:, 2]
        M_x1 = element_loads[:, 3]
        M_y1 = element_loads[:, 4]
        M_z1 = element_loads[:, 5]

        F_x2 = element_loads[:, 6]
        # F_y2 = element_loads[:, 7]
        # F_z2 = element_loads[:, 8]
        M_x2 = element_loads[:, 9]
        M_y2 = element_loads[:, 10]
        M_z2 = element_loads[:, 11]


        # average the nodal loads
        F_x = (F_x1 + F_x2) / 2
        # F_y = (F_y1 + F_y2) / 2
        # F_z = (F_z1 + F_z2) / 2
        M_x = (M_x1 + M_x2) / 2
        M_y = (M_y1 + M_y2) / 2
        M_z = (M_z1 + M_z2) / 2

        axial_stress = F_x / self.area
        shear_stress = M_x * self.radius / self.ix

        max_moment = (M_y**2 + M_z**2 + 1E-12) ** 0.5
        bending_stress = max_moment * self.radius / self.iy

        tensile_stress = axial_stress + bending_stress

        eps = 1E-12
        von_mises_stress = (tensile_stress**2 + 3*shear_stress**2 + eps) ** 0.5

        
        return von_mises_stress
    




class CSCircle:
    def __init__(self, 
                 radius:csdl.Variable,
                 ):
        
        self.radius = radius
        self.precomp = np.pi * self.radius**4

        self.area = self._area()
        self.ix = self._ix()
        self.iy = self._iy()
        self.iz = self._iz()

    # @property
    def _area(self):
        return np.pi * self.radius**2
    
    # @property
    def _ix(self):
        return (1 / 2) * self.precomp
    
    # @property
    def _iy(self):
        return (1 / 4) * self.precomp
    
    # @property
    def _iz(self):
        return (1 / 4) * self.precomp
    

    def stress(self, element_loads):

        pass







class CSBox:
    def __init__(self, 
                 ttop:csdl.Variable,
                 tbot:csdl.Variable,
                 tweb:csdl.Variable,
                 height:csdl.Variable,
                 width:csdl.Variable
                 ):
        
        self.ttop = ttop
        self.tbot = tbot
        self.tweb = tweb
        self.height = height
        self.width = width

        self.neutral_axis = self._neutral_axis()
        self.area = self._area()
        self.iy = self._iy()
        self.iz = self._iz()
        self.ix = self._ix()

    # @property
    def _neutral_axis(self):
        Atop = self.ttop * (self.width - 2 * self.tweb)
        Abot = self.tbot * (self.width - 2 * self.tweb)

        centroid_z = (Atop - Abot) / self.width
        centroid_y = 0

        return (centroid_y, centroid_z)
    
    # @property
    def type(self):
        return 'box'

    # @property
    def _area(self):
        w_i = self.width - 2 * self.tweb
        h_i = self.height - self.ttop - self.tbot
        return self.width * self.height - w_i * h_i
    
    # @property
    def _ix(self):
        ix = self.iy + self.iz
        return ix

        # Older version
        # tcap = (self.ttop + self.tbot) / 2
        # numerator = 2 * self.tweb * tcap * (self.width - self.tweb)**2 * (self.height - tcap)**2
        # denominator = self.width * self.tweb + self.height * tcap - self.tweb**2 - tcap**2
        # return numerator / denominator

    # @property
    def _iy(self):
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
    
    # @property
    def _iz(self):
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

    def stress(self, element_loads):
        pass