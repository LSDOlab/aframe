import numpy as np
import csdl_alpha as csdl
from typing import Union
from dataclasses import dataclass


@dataclass
class AnisotropicMaterial:
    name: str
    Exx: float
    Eyy: float
    Ezz: float
    Exy: float
    Exx: float
    Exx: float

    rho: float
    v: float

    @property
    def type(self):
        return 'anisotropic'


@dataclass
class CSCompositeBox:
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