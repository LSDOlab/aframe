from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from lsdo_modules.module.module import Module
from caddee.caddee_core.system_model.design_scenario.design_condition.mechanics_group.mechanics_model.mechanics_model import MechanicsModel
from aframe.core.aframe import Aframe, AframeResidual, AframeResidualJacobian
import numpy as np

import csdl
import m3l
import array_mapper as am


import m3l

class BeamM3LDisplacement(m3l.ImplicitOperation):
    def initialize(self):        
        # self.parameters.declare('connectivity', types=list)        
        # self.parameters.declare('out_name', types=str)

        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('struct_solver', True)
        self.parameters.declare('compute_mass_properties', default=True, types=bool)

        self.parameters.declare('beams', default={})
        self.parameters.declare('bounds', default={})
        self.parameters.declare('joints', default={})
        self.num_nodes = None

    def evaluate_residuals(self, inputs, outputs, residuals, num_nodes):
        geo_mesh = inputs['geo_mesh']
        thickness_mesh = inputs['thickness_mesh']
        forces = inputs['forces']
        displacement = outputs['displacement']
        residuals['displacement'] = LinearBeamResidualCSDL(geo_mesh, thickness_mesh, forces, displacement, num_nodes)

    # required for dynamic, not needed for static
    def compute_derivatives(self, inputs, outputs, derivatives, num_nodes):
        geo_mesh = inputs['geo_mesh']
        thickness_mesh = inputs['thickness_mesh']
        forces = inputs['forces']
        displacement = outputs['displacement']
        
        derivatives['displacement', 'forces'] = AframeResidualJacobian(geo_mesh, thickness_mesh, forces, displacement, num_nodes)
        derivatives['displacement', 'displacement'] = AframeResidualJacobian(geo_mesh, thickness_mesh, forces, displacement, num_nodes)

    # optional method
    def solve_residual_equations(self, inputs, outputs):
        pass


class EBBeam(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('component', default=None)
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('struct_solver', True)
        self.parameters.declare('compute_mass_properties', default=True, types=bool)

        self.parameters.declare('beams', default={})
        self.parameters.declare('bounds', default={})
        self.parameters.declare('joints', default={})
        self.num_nodes = None

    def compute(self):
        '''
        Creates a CSDL model to compute the solver outputs.

        Returns
        -------
        csdl_model : csdl.Model
            The csdl model which computes the outputs (the normal solver)
        '''
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']

        csdl_model = LinearBeamCSDL(
            module=self,
            beams=beams,  
            bounds=bounds,
            joints=joints)
        
        return csdl_model

    # # optional - for dynamic or conservative coupling
    # def compute_derivatives(self, inputs, derivatives):
    #     '''
    #     Creates a CSDL model to compute the derivatives of the solver outputs.


    #     '''
    #     geo_mesh = inputs['geo_mesh']
    #     displacement_nodes = inputs['displacement_nodes']
    #     derivatives['nodal_displacement', 'displacement'] = displacement_map(geo_mesh, displacement_nodes)

    def evaluate(self, forces:m3l.Variable=None, moments:m3l.Variable=None) -> m3l.Variable:
        '''
        Evaluates the beam model.
        
        Parameters
        ----------
        forces : m3l.Variable = None
            The forces on the mesh nodes.
        moments : m3l.Variable = None
            The moments on the mesh nodes.

        Returns
        -------
        displacements : m3l.Variable
            The displacements of the mesh nodes.
        rotations : m3l.Variable
            The rotations of the mesh nodes.

        '''

        # Assembles the CSDL model
        operation_csdl = self.compute()

        # Gets information for naming/shapes
        beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.

        arguments = {}
        if forces is not None:
            arguments[f'{beam_name}_forces'] = forces
        if moments is not None:
            arguments[f'{beam_name}_moments'] = moments

        # Create the M3L graph operation
        beam_operation = m3l.CSDLOperation(name='eb_beam_model', arguments=arguments, operation_csdl=operation_csdl)
        # Create the M3L variables that are being output
        displacements = m3l.Variable(name=f'{beam_name}_displacement', shape=mesh.shape, operation=beam_operation)
        rotations = m3l.Variable(name=f'{beam_name}_rotation', shape=mesh.shape, operation=beam_operation)

        return displacements, rotations


class EBBeamForces(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('beams', default={})

    def compute(self, nodal_forces:m3l.Variable, nodal_forces_mesh:am.MappedArray) -> csdl.Model:
        beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.

        csdl_model = ModuleCSDL()

        force_map = self.fmap(mesh.value.reshape((-1,3)), oml=nodal_forces_mesh.value.reshape((-1,3)))

        flattened_nodal_forces_shape = (np.prod(nodal_forces.shape[:-1]), nodal_forces.shape[-1])
        nodal_forces = csdl_model.register_module_input(name='nodal_forces', shape=nodal_forces.shape)
        flattened_nodal_forces = csdl.reshape(nodal_forces, new_shape=flattened_nodal_forces_shape)
        force_map_csdl = csdl_model.create_input(f'nodal_to_{beam_name}_forces_map', val=force_map)
        flatenned_beam_mesh_forces = csdl.matmat(force_map_csdl, flattened_nodal_forces)
        output_shape = tuple(mesh.shape[:-1]) + (nodal_forces.shape[-1],)
        beam_mesh_forces = csdl.reshape(flatenned_beam_mesh_forces, new_shape=output_shape)
        csdl_model.register_module_output(f'{beam_name}_forces', beam_mesh_forces)

        return csdl_model

    def evaluate(self, nodal_forces:m3l.Variable, nodal_forces_mesh:am.MappedArray) -> m3l.Variable:
        '''
        Maps nodal forces from arbitrary locations to the mesh nodes.
        
        Parameters
        ----------
        nodal_forces : m3l.Variable
            The forces to be mapped to the mesh nodes.
        nodal_forces_mesh : m3l.Variable
            The mesh that the nodal forces are currently defined over.

        Returns
        -------
        mesh_forces : m3l.Variable
            The forces on the mesh.
        '''
        operation_csdl = self.compute(nodal_forces=nodal_forces, nodal_forces_mesh=nodal_forces_mesh)

        beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.

        arguments = {'nodal_forces': nodal_forces}
        force_map_operation = m3l.CSDLOperation(name='ebbeam_force_map', arguments=arguments, operation_csdl=operation_csdl)
        output_shape = tuple(mesh.shape[:-1]) + (nodal_forces.shape[-1],)
        beam_forces = m3l.Variable(name=f'{beam_name}_forces', shape=output_shape, operation=force_map_operation)
        return beam_forces


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
    

class EBBeamMoments(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('beams', default={})

    def compute(self, nodal_moments, nodal_moments_mesh):
        beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.

        csdl_model = ModuleCSDL()

        force_map = self.fmap(mesh.value.reshape((-1,3)), oml=nodal_moments_mesh.value.reshape((-1,3)))

        nodal_moments = csdl_model.register_module_input(name='nodal_moments', shape=nodal_moments.shape)
        moment_map_csdl = csdl_model.create_input(f'nodal_to_{beam_name}_moments_map', val=force_map)
        beam_moments = csdl.matmat(moment_map_csdl, nodal_moments)
        csdl_model.register_module_output(f'{beam_name}_moments', beam_moments)

        return csdl_model


    def evaluate(self, nodal_moments:m3l.Variable, nodal_moments_mesh:am.MappedArray) -> m3l.Variable:
        '''
        Maps nodal moments from arbitrary locations to the mesh nodes.
        
        Parameters
        ----------
        nodal_moments : m3l.Variable
            The moments to be mapped to the mesh nodes.
        nodal_moments_mesh : m3l.Variable
            The mesh that the nodal moments are currently defined over.

        Returns
        -------
        mesh_moments : m3l.Variable
            The moments on the mesh.
        '''
        operation_csdl = self.compute(nodal_moments=nodal_moments, nodal_moments_mesh=nodal_moments_mesh)

        beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.

        arguments = {'nodal_moments': nodal_moments}
        moment_map_operation = m3l.CSDLOperation(name='ebbeam_moment_map', arguments=arguments, operation_csdl=operation_csdl)
        beam_moments = m3l.Variable(name=f'{beam_name}_moments', shape=mesh.shape, operation=moment_map_operation)
        return beam_moments
    

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


class EBBeamNodalDisplacements(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('mesh', default=None)
        self.parameters.declare('beams', default={})

    def compute(self, beam_displacements:m3l.Variable, nodal_displacements_mesh:am.MappedArray)->csdl.Model:
        beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
        mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.

        csdl_model = ModuleCSDL()

        displacement_map = self.umap(mesh.value.reshape((-1,3)), oml=nodal_displacements_mesh.value.reshape((-1,3)))

        beam_displacements = csdl_model.register_module_input(name=f'{beam_name}_displacement', shape=beam_displacements.shape)
        displacement_map_csdl = csdl_model.create_input(f'{beam_name}_displacements_to_nodal_displacements', val=displacement_map)
        nodal_displacements = csdl.matmat(displacement_map_csdl, beam_displacements)
        csdl_model.register_module_output(f'{beam_name}_nodal_displacement', nodal_displacements)

        return csdl_model

    def evaluate(self, beam_displacements:m3l.Variable, nodal_displacements_mesh:am.MappedArray) -> m3l.Variable:
        '''
        Maps nodal forces and moments from arbitrary locations to the mesh nodes.
        
        Parameters
        ----------
        beam_displacements : m3l.Variable
            The displacements to be mapped from the beam mesh to the desired mesh.
        nodal_displacements_mesh : m3l.Variable
            The mesh to evaluate the displacements over.

        Returns
        -------
        nodal_displacements : m3l.Variable
            The displacements on the given nodal displacements mesh.
        '''
        operation_csdl = self.compute(beam_displacements, nodal_displacements_mesh)

        beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.

        arguments = {f'{beam_name}_displacement': beam_displacements}
        displacement_map_operation = m3l.CSDLOperation(name='eb_beam_displacement_map', arguments=arguments, operation_csdl=operation_csdl)
        nodal_displacements = m3l.Variable(name=f'{beam_name}_nodal_displacement', shape=nodal_displacements_mesh.shape, 
                                           operation=displacement_map_operation)
        return nodal_displacements


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
    

class BeamM3LStrain(m3l.ExplicitOperation):
    def compute():
        pass

    def compute_derivatives(): # optional
        pass

class BeamM3LStress(m3l.ExplicitOperation):
    def compute():
        pass

    def compute_derivatives(): # optional
        pass



# ## OLD
# class LinearBeam(m3l.Model):
#     def initialize(self, kwargs):
#         self.parameters.declare('component', default=None)
#         self.parameters.declare('mesh', default=None)
#         self.parameters.declare('struct_solver', True)
#         self.parameters.declare('compute_mass_properties', default=True, types=bool)

#         self.parameters.declare('beams', default={})
#         self.parameters.declare('bounds', default={})
#         self.parameters.declare('joints', default={})
#         self.num_nodes = None

#     def construct_force_map(self, nodal_force, nodal_force_mesh):
#         mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
#         # oml_mesh = nodal_force.mesh.value.reshape((-1, 3))
#         force_map = self.fmap(mesh.value.reshape((-1,3)), oml=nodal_force_mesh.value.reshape((-1,3)))
#         return force_map
    
#     def construct_moment_map(self, nodal_moment, nodal_moment_mesh):
#         mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
#         # oml_mesh = nodal_moment.mesh.value.reshape((-1, 3))
#         moment_map = self.fmap(mesh.value.reshape((-1,3)), oml=nodal_moment_mesh.value.reshape((-1,3)))
#         return moment_map

#     def construct_displacement_map(self, nodal_outputs_mesh):
#         mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
#         oml_mesh = nodal_outputs_mesh.value.reshape((-1, 3))
#         displacement_map = self.umap(mesh.value.reshape((-1,3)), oml=oml_mesh)

#         return displacement_map
    
#     def construct_rotation_map(self, nodal_outputs_mesh):
#         mesh = list(self.parameters['mesh'].parameters['meshes'].values())[0]   # this is only taking the first mesh added to the solver.
#         oml_mesh = nodal_outputs_mesh.value.reshape((-1, 3))
#         # rotation_map = self.mmap(mesh.value, oml=oml_mesh)

#         rotation_map = np.zeros((oml_mesh.shape[0],mesh.shape[0]))

#         return rotation_map
    
#     def construct_invariant_matrix(self):
#         pass

#     def evaluate(self, nodal_outputs_mesh:am.MappedArray, nodal_force:m3l.Variable=None, nodal_force_mesh:am.MappedArray=None,
#                  nodal_moment:m3l.Variable=None, nodal_moment_mesh:am.MappedArray=None):
#         '''
#         Evaluates the model.

#         Parameters
#         ----------
#         nodal_outputs_mesh : am.MappedArray
#             The mesh or point cloud representing the locations at which the nodal displacements and rotations will be returned.
#         nodal_force : m3l.FunctionValues
#             The nodal forces that will be mapped onto the beam.
#         nodal_moment : m3l.FunctionValues
#             The nodal moments that will be mapped onto the beam.
        
#         Returns
#         -------
#         nodal_displacement : m3l.FunctionValues
#             The displacements evaluated at the locations specified by nodal_outputs_mesh
#         nodal_rotation : m3l.FunctionValues
#             The rotations evluated at the locations specified by the nodal_outputs_mesh
#         '''

#         # NOTE: This is assuming one mesh. To handle multiple meshes, a method must be developed to figure out how mappings work.
#         beam_name = list(self.parameters['beams'].keys())[0]   # this is only taking the first mesh added to the solver.
#         component_name = self.parameters['component'].name

#         input_modules = []
#         if nodal_force is not None:
#             force_map = self.construct_force_map(nodal_force=nodal_force, nodal_force_mesh=nodal_force_mesh)
#             force_input_module = m3l.ModelInputModule(name='force_input_module', 
#                                                   module_input=nodal_force, map=force_map, model_input_name=f'{beam_name}_forces')
#             input_modules.append(force_input_module)

#         if nodal_moment is not None:
#             moment_map = self.construct_moment_map(nodal_moment=nodal_moment, nodal_moment_mesh=nodal_moment_mesh)
#             moment_input_module = m3l.ModelInputModule(name='moment_input_module', 
#                                                    module_input=nodal_moment, map=moment_map, model_input_name=f'{beam_name}_moments')
#             input_modules.append(moment_input_module)


#         displacement_map = self.construct_displacement_map(nodal_outputs_mesh=nodal_outputs_mesh)
#         rotation_map = self.construct_rotation_map(nodal_outputs_mesh=nodal_outputs_mesh)

#         displacement_output_module = m3l.ModelOutputModule(name='displacement_output_module',
#                                                     model_output_name=f'{beam_name}_displacement',
#                                                     map=displacement_map, module_output_name=f'beam_nodal_displacement_{component_name}',
#                                                     module_output_mesh=nodal_outputs_mesh)
#         rotation_output_module = m3l.ModelOutputModule(name='rotation_output_module',
#                                                     model_output_name=f'{beam_name}_rotation',
#                                                     map=rotation_map, module_output_name=f'beam_nodal_rotation_{component_name}',
#                                                     module_output_mesh=nodal_outputs_mesh)

#         nodal_displacement, nodal_rotation = self.construct_module_csdl(
#                          model_map=self._assemble_csdl(), 
#                          input_modules=input_modules,
#                          output_modules=[displacement_output_module, rotation_output_module]
#                          )
        
#         return nodal_displacement, nodal_rotation


#     def _assemble_csdl(self):
#         beams = self.parameters['beams']
#         bounds = self.parameters['bounds']
#         joints = self.parameters['joints']

#         csdl_model = LinearBeamCSDL(
#             module=self,
#             beams=beams,  
#             bounds=bounds,
#             joints=joints)

#         return csdl_model

#     def umap(self, mesh, oml):
#         # Up = W*Us

#         x, y = mesh.copy(), oml.copy()
#         n, m = len(mesh), len(oml)

#         d = np.zeros((m,2))
#         for i in range(m):
#             dist = np.sum((x - y[i,:])**2, axis=1)
#             d[i,:] = np.argsort(dist)[:2]

#         # create the weighting matrix:
#         weights = np.zeros((m,n))
#         for i in range(m):
#             ia, ib = int(d[i,0]), int(d[i,1])
#             a, b = x[ia,:], x[ib,:]
#             p = y[i,:]

#             length = np.linalg.norm(b - a)
#             norm = (b - a)/length
#             t = np.dot(p - a, norm)
#             # c is the closest point on the line segment (a,b) to point p:
#             c =  a + t*norm

#             ac, bc = np.linalg.norm(c - a), np.linalg.norm(c - b)
#             l = max(length, bc)
            
#             weights[i, ia] = (l - ac)/length
#             weights[i, ib] = (l - bc)/length

#         return weights
    


#     def fmap(self, mesh, oml):
#         # Fs = W*Fp

#         x, y = mesh.copy(), oml.copy()
#         n, m = len(mesh), len(oml)

#         d = np.zeros((m,2))
#         for i in range(m):
#             dist = np.sum((x - y[i,:])**2, axis=1)
#             d[i,:] = np.argsort(dist)[:2]

#         # create the weighting matrix:
#         weights = np.zeros((n, m))
#         for i in range(m):
#             ia, ib = int(d[i,0]), int(d[i,1])
#             a, b = x[ia,:], x[ib,:]
#             p = y[i,:]

#             length = np.linalg.norm(b - a)
#             norm = (b - a)/length
#             t = np.dot(p - a, norm)
#             # c is the closest point on the line segment (a,b) to point p:
#             c =  a + t*norm

#             ac, bc = np.linalg.norm(c - a), np.linalg.norm(c - b)
#             l = max(length, bc)
            
#             weights[ia, i] = (l - ac)/length
#             weights[ib, i] = (l - bc)/length

#         return weights




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

class LinearBeamResidualCSDL(ModuleCSDL):
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
        self.add_module(AframeResidual(beams=beams, bounds=bounds, joints=joints), name='Aframe')