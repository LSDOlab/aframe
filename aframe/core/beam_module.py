from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
from caddee.caddee_core.system_model.design_scenario.design_condition.mechanics_group.mechanics_model.mechanics_model import MechanicsModel
from aframe.aframe import Aframe
import numpy as np

import csdl
import m3l
import array_mapper as am

class LinearBeam(m3l.Model):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('struct_solver', True)
        self.parameters.declare('compute_mass_properties', default=True, types=bool)

        self.parameters.declare('beams', default={})
        self.parameters.declare('bounds', default={})
        self.parameters.declare('joints', default={})
        self.num_nodes = None

    def construct_force_map(self, nodal_force):
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
        oml_mesh = nodal_force.mesh.value.reshape((-1, 3))
        force_map = self.fmap(mesh.value.reshape((-1,3)), oml=oml_mesh)
        return force_map
    
    def construct_moment_map(self, nodal_moment):
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
        oml_mesh = nodal_moment.mesh.value.reshape((-1, 3))
        moment_map = self.fmap(mesh.value.reshape((-1,3)), oml=oml_mesh)
        return moment_map

    def construct_displacement_map(self, nodal_outputs_mesh):
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
        oml_mesh = nodal_outputs_mesh.value.reshape((-1, 3))
        displacement_map = self.umap(mesh.value.reshape((-1,3)), oml=oml_mesh)

        return displacement_map
    
    def construct_rotation_map(self, nodal_outputs_mesh):
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
        oml_mesh = nodal_outputs_mesh.value.reshape((-1, 3))
        # rotation_map = self.mmap(mesh.value, oml=oml_mesh)

        rotation_map = np.zeros((oml_mesh.shape[0],mesh.shape[0]))

        return rotation_map
    
    def construct_invariant_matrix(self):
        pass

    def evaluate(self, nodal_outputs_mesh:am.MappedArray, nodal_force:m3l.FunctionValues=None, nodal_moment:m3l.FunctionValues=None):
        '''
        Evaluates the model.

        Parameters
        ----------
        nodal_outputs_mesh : am.MappedArray
            The mesh or point cloud representing the locations at which the nodal displacements and rotations will be returned.
        nodal_force : m3l.FunctionValues
            The nodal forces that will be mapped onto the beam.
        nodal_moment : m3l.FunctionValues
            The nodal moments that will be mapped onto the beam.
        
        Returns
        -------
        nodal_displacement : m3l.FunctionValues
            The displacements evaluated at the locations specified by nodal_outputs_mesh
        nodal_rotation : m3l.FunctionValues
            The rotations evluated at the locations specified by the nodal_outputs_mesh
        '''

        # NOTE: This is assuming one mesh. To handle multiple meshes, a method must be developed to figure out how mappings work.
        beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        component_name = self.parameters['component'].name

        input_modules = []
        if nodal_force is not None:
            force_map = self.construct_force_map(nodal_force=nodal_force)
            force_input_module = m3l.ModelInputModule(name='force_input_module', 
                                                  module_input=nodal_force, map=force_map, model_input_name=f'{beam_name}_forces')
            input_modules.append(force_input_module)

        if nodal_moment is not None:
            moment_map = self.construct_moment_map(nodal_moment=nodal_moment)
            moment_input_module = m3l.ModelInputModule(name='moment_input_module', 
                                                   module_input=nodal_moment, map=moment_map, model_input_name=f'{beam_name}_moments')
            input_modules.append(moment_input_module)


        displacement_map = self.construct_displacement_map(nodal_outputs_mesh=nodal_outputs_mesh)
        rotation_map = self.construct_rotation_map(nodal_outputs_mesh=nodal_outputs_mesh)

        displacement_output_module = m3l.ModelOutputModule(name='displacement_output_module',
                                                    model_output_name=f'{beam_name}_displacement',
                                                    map=displacement_map, module_output_name=f'beam_nodal_displacement_{component_name}',
                                                    module_output_mesh=nodal_outputs_mesh)
        rotation_output_module = m3l.ModelOutputModule(name='rotation_output_module',
                                                    model_output_name=f'{beam_name}_rotation',
                                                    map=rotation_map, module_output_name=f'beam_nodal_rotation_{component_name}',
                                                    module_output_mesh=nodal_outputs_mesh)

        nodal_displacement, nodal_rotation = self.construct_module_csdl(
                         model_map=self._assemble_csdl(), 
                         input_modules=input_modules,
                         output_modules=[displacement_output_module, rotation_output_module]
                         )
        
        return nodal_displacement, nodal_rotation


    def _assemble_csdl(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']

        csdl_model = LinearBeamCSDL(
            module=self,
            beams=beams,  
            bounds=bounds,
            joints=joints)

        return csdl_model

    def umap(self, mesh, oml):
        # Up = W*Us

        x, y = mesh.copy(), oml.copy()
        n, m = len(mesh), len(oml)

        d = np.zeros((m,2))
        for i in range(m):
            dist = np.sum((x - y[i,:])**2, axis=1)
            d[i,:] = np.argsort(dist)[:2]

        # create the weighting matrix:
        weights = np.zeros((m,n))
        for i in range(m):
            ia, ib = int(d[i,0]), int(d[i,1])
            a, b = x[ia,:], x[ib,:]
            p = y[i,:]

            length = np.linalg.norm(b - a)
            norm = (b - a)/length
            t = np.dot(p - a, norm)
            # c is the closest point on the line segment (a,b) to point p:
            c =  a + t*norm

            ac, bc = np.linalg.norm(c - a), np.linalg.norm(c - b)
            l = max(length, bc)
            
            weights[i, ia] = (l - ac)/length
            weights[i, ib] = (l - bc)/length

        return weights
    


    def fmap(self, mesh, oml):
        # Fs = W*Fp

        x, y = mesh.copy(), oml.copy()
        n, m = len(mesh), len(oml)

        d = np.zeros((m,2))
        for i in range(m):
            dist = np.sum((x - y[i,:])**2, axis=1)
            d[i,:] = np.argsort(dist)[:2]

        # create the weighting matrix:
        weights = np.zeros((n, m))
        for i in range(m):
            ia, ib = int(d[i,0]), int(d[i,1])
            a, b = x[ia,:], x[ib,:]
            p = y[i,:]

            length = np.linalg.norm(b - a)
            norm = (b - a)/length
            t = np.dot(p - a, norm)
            # c is the closest point on the line segment (a,b) to point p:
            c =  a + t*norm

            ac, bc = np.linalg.norm(c - a), np.linalg.norm(c - b)
            l = max(length, bc)
            
            weights[ia, i] = (l - ac)/length
            weights[ib, i] = (l - bc)/length

        return weights




class LinearBeamMesh(Module):
    def initialize(self, kwargs):
        self.parameters.declare('meshes', types=dict)
        self.parameters.declare('mesh_units', default='m')



class LinearBeamCSDL(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('beams')
        self.parameters.declare('bounds')
        self.parameters.declare('joints')
    
    def define(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']

        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            cs = beams[beam_name]['cs']

            if cs == 'box':
                xweb = self.register_module_input(beam_name+'t_web_in',shape=(n-1), computed_upstream=False)
                xcap = self.register_module_input(beam_name+'t_cap_in',shape=(n-1), computed_upstream=False)
                self.register_output(beam_name+'_tweb',1*xweb)
                self.register_output(beam_name+'_tcap',1*xcap)
                
            elif cs == 'tube':
                thickness = self.register_module_input(beam_name+'thickness_in',shape=(n-1), computed_upstream=False)
                radius = self.register_module_input(beam_name+'radius_in',shape=(n-1), computed_upstream=False)
                self.register_output(beam_name+'_t', 1*thickness)
                self.register_output(beam_name+'_r', 1*radius)

        # solve the beam group:
        self.add_module(Aframe(beams=beams, bounds=bounds, joints=joints), name='Aframe')