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