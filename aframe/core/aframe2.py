import numpy as np
import csdl
from aframe.core.massprop import MassPropModule as MassProp
from aframe.core.model import Model
from aframe.core.buckle import Buckle
from aframe.core.nodal_stress import NodalStressBox
from aframe.core.stress import StressBox
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL




class Aframe(ModuleCSDL):

    def initialize(self):
        self.parameters.declare('beams', default={})
        self.parameters.declare('joints', default={})
        self.parameters.declare('bounds', default={})
        self.parameters.declare('mesh_units', default='m')