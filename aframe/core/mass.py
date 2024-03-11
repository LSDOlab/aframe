import numpy as np
import csdl
import m3l
from dataclasses import dataclass

# this file contains an entirely separate mass computation model for aframe
# it requires the same inputs as an ordinary aframe beam model 
# only valid for box beams

@dataclass
class BeamMassProps:
    mass : m3l.Variable
    cg_vector : m3l.Variable
    inertia_tensor : m3l.Variable


class BeamMassModel(m3l.ExplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('name', types=str)
        self.parameters.declare('mesh', default=None)

        self.parameters.declare('beams', default={})
        self.parameters.declare('mesh_units', default='ft')
        self.num_nodes = None

    def assign_attributes(self):
        self.mesh = self.parameters['mesh']
        self.name = self.parameters['name']
        self.beams = self.parameters['beams']
        self.mesh_units = self.parameters['mesh_units']

    def compute(self):
        beams = self.parameters['beams']
        mesh_units = self.parameters['mesh_units']

        csdl_model = MassCSDL(
            beams=beams,
            mesh_units=mesh_units,)
        
        return csdl_model

    def evaluate(self, width, height, t_top, t_bot, t_web, beam_nodes) -> BeamMassProps:
        self.arguments = {}
        beams = self.parameters['beams']

        for j, beam_name in enumerate(beams):
            self.arguments[f'{beam_name}_mesh'] = beam_nodes
            self.arguments[f'{beam_name}_width'] = width
            self.arguments[f'{beam_name}_height'] = height
            self.arguments[f'{beam_name}_tweb'] = t_web
            self.arguments[f'{beam_name}_ttop'] = t_top
            self.arguments[f'{beam_name}_tbot'] = t_bot

        mass = m3l.Variable(name='mass', shape=(1,), operation=self)
        cg_vector = m3l.Variable(name='cg_vector', shape=(3, ), operation=self)
        inertia_tensor = m3l.Variable(name='inertia_tensor', shape=(3, 3), operation=self)

        beam_mass_props = BeamMassProps(
            mass=mass,
            cg_vector=cg_vector,
            inertia_tensor=inertia_tensor,
        )

        return beam_mass_props







class MassCSDL(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams', default={})
        self.parameters.declare('mesh_units', default='ft')

    def define(self):
        beams = self.parameters['beams']
        mesh_units = self.parameters['mesh_units']


        total_mass, total_sum_rm, Ixx, Iyy, Izz, Ixz = 0, 0, 0, 0, 0, 0

        for j, beam_name in enumerate(beams):
            n = len(beams[beam_name]['nodes'])
            rho = beams[beam_name]['rho']
            cs = beams[beam_name]['cs']

            mesh_in = self.declare_variable(beam_name + '_mesh', shape=(n,3))
            if mesh_units == 'm': mesh = 1*mesh_in
            elif mesh_units == 'ft': mesh = 0.304*mesh_in

            
            
            beam_mass, beam_sum_r_times_m, beam_eixx, beam_eiyy, beam_eizz, beam_eixz = 0, 0, 0, 0, 0, 0

            if cs == 'box':
                width = self.declare_variable(beam_name + '_width', shape=(n))
                height = self.declare_variable(beam_name + '_height', shape=(n))
                w = self.create_output(beam_name + '_width' + '_avg', shape=(n - 1), val=0)
                h = self.create_output(beam_name + '_height' + '_avg', shape=(n - 1), val=0)
                for i in range(n - 1):
                    if mesh_units == 'm':
                        w[i], h[i] = (width[i] + width[i + 1])/2, (height[i] + height[i + 1])/2
                    elif mesh_units == 'ft':
                        w[i], h[i] = 0.304*(width[i] + width[i + 1])/2, 0.304*(height[i] + height[i + 1])/2

                tweb_in = self.declare_variable(beam_name + '_tweb', shape=(n))
                tbot_in = self.declare_variable(beam_name + '_tbot', shape=(n))
                ttop_in = self.declare_variable(beam_name + '_ttop', shape=(n))
                tweb = self.create_output(beam_name + '_tweb' + '_avg', shape=(n-1), val=0)
                ttop = self.create_output(beam_name + '_ttop' + '_avg', shape=(n-1), val=0)
                tbot = self.create_output(beam_name + '_tbot' + '_avg', shape=(n-1), val=0)
                for i in range(n - 1):
                    tweb[i] = (tweb_in[i]+tweb_in[i+1])/2
                    ttop[i] = (ttop_in[i]+ttop_in[i+1])/2
                    tbot[i] = (tbot_in[i]+tbot_in[i+1])/2

                w_i, h_i = w - 2*tweb, h - ttop - tbot
                A = (((w*h) - (w_i*h_i))**2 + 1E-14)**0.5


            elif cs == 'tube':
                radius_in = self.declare_variable(beam_name + '_radius', shape=(n))
                thick_in = self.declare_variable(beam_name + '_thick', shape=(n))

                if mesh_units == 'm': converted_radius = 1*radius_in
                elif mesh_units == 'ft': converted_radius = 0.304*radius_in

                radius = self.create_output(beam_name + '_radius' + '_avg', shape=(n - 1), val=0)
                thick = self.create_output(beam_name + '_thick' + '_avg', shape=(n - 1), val=0)
                for i in range(n - 1):
                    radius[i] = (converted_radius[i] + converted_radius[i + 1]) / 2
                    thick[i] = (thick_in[i]+thick_in[i+1]) / 2

                r1, r2 = radius - thick, radius
                A = np.pi * (r2**2 - r1**2)

            

            # compute the mass properties
            for i in range(n - 1):
                L = csdl.pnorm(mesh[i + 1, :] - mesh[i, :], pnorm_type=2) + 1E-12
                V = L * A[i]
                element_mass = V * rho
                beam_mass = beam_mass + element_mass

                r_cg = csdl.reshape((mesh[i + 1, :] + mesh[i, :]) / 2, (3))
                r_times_m = r_cg * csdl.expand(element_mass, (3))
                beam_sum_r_times_m = beam_sum_r_times_m + r_times_m

                x, y, z = r_cg[0], r_cg[1], r_cg[2]
                rxx = y**2 + z**2
                ryy = x**2 + z**2
                rzz = x**2 + y**2
                rxz = x*z

                beam_eixx = beam_eixx + element_mass*rxx
                beam_eiyy = beam_eiyy + element_mass*ryy
                beam_eizz = beam_eizz + element_mass*rzz
                beam_eixz = beam_eixz + element_mass*rxz




            total_mass = total_mass + beam_mass
            total_sum_rm = total_sum_rm + beam_sum_r_times_m
            Ixx = Ixx + beam_eixx
            Iyy = Iyy + beam_eiyy
            Izz = Izz + beam_eizz
            Ixz = Ixz + beam_eixz
        


        self.register_output('mass', total_mass)

        cg = total_sum_rm / csdl.expand(total_mass, (3))
        self.register_output('cg_vector', cg)
        self.register_output('cgx', cg[0])
        self.register_output('cgy', cg[1])
        self.register_output('cgz', cg[2])


        inertia_tensor = self.create_output('inertia_tensor', shape=(3, 3), val=0)
        inertia_tensor[0, 0] = csdl.reshape(Ixx, (1, 1))
        inertia_tensor[0, 2] = csdl.reshape(Ixz, (1, 1))
        inertia_tensor[1, 1] = csdl.reshape(Iyy, (1, 1))
        inertia_tensor[2, 0] = csdl.reshape(Ixz, (1, 1))
        inertia_tensor[2, 2] = csdl.reshape(Izz, (1, 1))

        self.register_output('ixx', Ixx)
        self.register_output('iyy', Iyy)
        self.register_output('izz', Izz)
        self.register_output('ixz', Ixz)


# class MassCSDL(csdl.Model):
#     def initialize(self):
#         self.parameters.declare('beams', default={})
#         self.parameters.declare('mesh_units', default='ft')

#     def define(self):
#         beams = self.parameters['beams']
#         mesh_units = self.parameters['mesh_units']



#         # create a list of element names:
#         elements, element_density_list, num_elements = [], [], 0
#         for beam_name in beams:
#             n = len(beams[beam_name]['nodes'])
#             num_elements += n - 1
#             for i in range(n - 1): 
#                 elements.append(beam_name + '_element_' + str(i))
#                 element_density_list.append(beams[beam_name]['rho'])

#         m_vec = self.create_output('m_vec',shape=(len(beams)),val=0) # stores the mass for each beam
#         for j, beam_name in enumerate(beams):
#             n = len(beams[beam_name]['nodes'])
#             rho = beams[beam_name]['rho']

#             # get the mesh:
#             mesh_in = self.declare_variable(beam_name + '_mesh', shape=(n,3))
#             if mesh_units == 'm': mesh = 1*mesh_in
#             elif mesh_units == 'ft': mesh = 0.304*mesh_in
#             # get the width and height meshes:
#             width = self.declare_variable(beam_name + '_width', shape=(n))
#             height = self.declare_variable(beam_name + '_height', shape=(n))
#             w = self.create_output(beam_name + '_w', shape=(n - 1), val=0)
#             h = self.create_output(beam_name + '_h', shape=(n - 1), val=0)
#             # take averages of nodal meshes to get elemental meshes and convert units:
#             for i in range(n - 1):
#                 if mesh_units == 'm':
#                     w[i] = (width[i] + width[i + 1])/2
#                     h[i] = (height[i] + height[i + 1])/2
#                 elif mesh_units == 'ft':
#                     w[i] = 0.304*(width[i] + width[i + 1])/2
#                     h[i] = 0.304*(height[i] + height[i + 1])/2

#             # the box-beam thicknesses:
#             #tweb = self.declare_variable(beam_name + '_tweb', shape=(n - 1))
#             #tcap = self.declare_variable(beam_name + '_tcap', shape=(n - 1))
#             tweb_in = self.declare_variable(beam_name + '_tweb', shape=(n))
#             # tcap_in = self.declare_variable(beam_name + '_tcap', shape=(n), computed_upstream=False)
#             tbot_in = self.declare_variable(beam_name + '_tbot', shape=(n))
#             ttop_in = self.declare_variable(beam_name + '_ttop', shape=(n))


#             tweb = self.create_output(beam_name + 'marius_tweb', shape=(n-1), val=0)
#             # tcap = self.create_output('marius_tcap', shape=(n-1), val=0)
#             ttop = self.create_output(beam_name + 'marius_ttop', shape=(n-1), val=0)
#             tbot = self.create_output(beam_name + 'marius_tbot', shape=(n-1), val=0)
#             for i in range(n - 1):
#                 tweb[i] = (tweb_in[i]+tweb_in[i+1])/2
#                 # tcap[i] = (tcap_in[i]+tcap_in[i+1])/2
#                 ttop[i] = (ttop_in[i]+ttop_in[i+1])/2
#                 tbot[i] = (tbot_in[i]+tbot_in[i+1])/2

#             # get cs area:
#             w_i = w - 2*tweb
#             # h_i = h - 2*tcap
#             h_i = h - ttop - tbot
#             A = (((w*h) - (w_i*h_i))**2 + 1E-14)**0.5

#             # iterate over the elements:
#             em_vec = self.create_output(beam_name + '_em_vec',shape=(n - 1),val=0)
#             for i in range(n - 1):
#             # for i, element_name in enumerate(elements):
#                 element_name = beam_name + '_element_' + str(i)

#                 node_a = csdl.reshape(mesh[i, :], (3))
#                 node_b = csdl.reshape(mesh[i + 1, :], (3))
#                 L = csdl.pnorm(node_b - node_a, pnorm_type=2) + 1E-12
#                 self.register_output(element_name + 'node_a', node_a)
#                 self.register_output(element_name + 'node_b', node_b)
#                 self.register_output(element_name + 'L', L)
#                 self.register_output(element_name + '_A', A[i])

#                 V = A[i]*L
#                 em_vec[i] = V*rho
        
#             beam_mass = csdl.sum(em_vec)
#             m_vec[j] = beam_mass

#         total_mass = csdl.sum(m_vec) # sums the beam masses
#         self.register_output('mass', total_mass)
#         self.print_var(total_mass)

#         self.add(MassPostProcess(
#             elements=elements,
#             element_density_list=element_density_list,
#         ), 'mass_post_process_model')

#         self.add(MassPostProcess2(
#             elements=elements,
#         ), 'mass_post_process_model_2')
        


# class MassPostProcess(csdl.Model):
#     def initialize(self):
#         self.parameters.declare('elements')
#         self.parameters.declare('element_density_list')

#     def define(self):
#         elements = self.parameters['elements']
#         element_density_list = self.parameters['element_density_list']
        
#         # compute the cg and moi:
#         dup_m_vec = self.create_output('dup_m_vec', shape=(len(elements)), val=0)
#         rm_vec = self.create_output('rm_vec', shape=(len(elements),3), val=0)
#         for i, element_name in enumerate(elements):
#             rho = element_density_list[i]

#             A = self.declare_variable(element_name + '_A')
#             L = self.declare_variable(element_name + 'L')

#             # compute the element mass:
#             m = self.register_output(element_name + 'm', (A*L)*rho)

#             # get the (undeformed) position vector of the cg for each element:
#             r_a = self.declare_variable(element_name + 'node_a', shape=(3))
#             r_b = self.declare_variable(element_name + 'node_b', shape=(3))

#             r_cg = self.register_output(element_name+'r_cg', (r_a + r_b)/2)
            
#             # assign r_cg to the r*mass vector:
#             rm_vec[i,:] = csdl.reshape(r_cg*csdl.expand(m, (3)), new_shape=(1,3))
#             dup_m_vec[i] = m

#         total_mass = self.declare_variable('mass', shape=(1, ))
        
#         cg = csdl.sum(rm_vec, axes=(0,))/csdl.expand(total_mass, (3))
#         self.register_output('cg_vector', cg)

#         self.register_output('cgx', cg[0])
#         self.register_output('cgy', cg[1]*0) # zeroed to make opt converge better and stuff
#         self.register_output('cgz', cg[2])

           
            

# class MassPostProcess2(csdl.Model):
#     def initialize(self):
#         self.parameters.declare('elements')
        

#     def define(self):
#         elements = self.parameters['elements']
            
#             # compute moments of inertia:
#         eixx = self.create_output('eixx',shape=(len(elements)),val=0)
#         eiyy = self.create_output('eiyy',shape=(len(elements)),val=0)
#         eizz = self.create_output('eizz',shape=(len(elements)),val=0)
#         eixz = self.create_output('eixz',shape=(len(elements)),val=0)
#         dup_m_vec = self.declare_variable('dup_m_vec', shape=(len(elements)), val=0)
        
#         for i, element_name in enumerate(elements):
#             m = dup_m_vec[i]

#             # get the position vector:
#             r = self.declare_variable(element_name + 'r_cg', shape=(3))
#             x, y, z = r[0], r[1], r[2]
#             rxx = y**2 + z**2
#             ryy = x**2 + z**2
#             rzz = x**2 + y**2
#             rxz = x*z
#             eixx[i] = m*rxx
#             eiyy[i] = m*ryy
#             eizz[i] = m*rzz
#             eixz[i] = m*rxz
            
#         # sum the m*r vector to get the moi:
#         Ixx, Iyy, Izz, Ixz = csdl.sum(eixx), csdl.sum(eiyy), csdl.sum(eizz), csdl.sum(eixz)

#         inertia_tensor = self.create_output('inertia_tensor', shape=(3, 3), val=0)
#         inertia_tensor[0, 0] = csdl.reshape(Ixx, (1, 1))
#         inertia_tensor[0, 2] = csdl.reshape(Ixz, (1, 1))
#         inertia_tensor[1, 1] = csdl.reshape(Iyy, (1, 1))
#         inertia_tensor[2, 0] = csdl.reshape(Ixz, (1, 1))
#         inertia_tensor[2, 2] = csdl.reshape(Izz, (1, 1))

#         self.register_output('ixx', Ixx)
#         self.register_output('iyy', Iyy)
#         self.register_output('izz', Izz)
#         self.register_output('ixz', Ixz)








        