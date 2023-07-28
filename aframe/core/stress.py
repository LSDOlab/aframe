import numpy as np
import csdl




class StressTube(csdl.Model):
    def initialize(self):
        self.parameters.declare('name')
    def define(self):
        name = self.parameters['name']

        A = self.declare_variable(name+'_A')
        J = self.declare_variable(name+'_J')
        Iy = self.declare_variable(name+'_Iy')
        r = self.declare_variable(name+'_r')

        # get the local loads:
        local_loads = self.declare_variable(name+'local_loads',shape=(12))
        loads_a = local_loads[0:6]
        loads_b = local_loads[6:12]

        #self.print_var(local_loads)

        # compute the normal stress:
        s_normal_a = loads_a[0]/A
        s_normal_b = loads_b[0]/A

        # compute the torsional stress:
        tau_a = loads_a[3]*r/J
        tau_b = loads_b[3]*r/J

        # compute the bending stress:
        moment_1_a = loads_a[4]
        moment_2_a = loads_a[5]
        moment_1_b = loads_b[4]
        moment_2_b = loads_b[5]

        moment_a = (moment_1_a**2 + moment_2_a**2 + 1E-16)**0.5
        moment_b = (moment_1_b**2 + moment_2_b**2 + 1E-16)**0.5

        s_bend_a = moment_a*r/Iy # note: Iy = Iz for a tube
        s_bend_b = moment_b*r/Iy

        # sum the bending and normal stresses:
        s_axial_a = s_normal_a + s_bend_a
        s_axial_b = s_normal_b + s_bend_b

        # compute the maximum von-mises stress
        stress_a = (s_axial_a**2 + 3*tau_a**2)**0.5
        stress_b = (s_axial_b**2 + 3*tau_b**2)**0.5

        stress_ab = self.create_output(name + 'stress_ab', shape=(2), val=0)
        stress_ab[0] = stress_a
        stress_ab[1] = stress_b

        # self.register_output(name + '_stress', csdl.max(1E-3*stress_ab)/1E-3)
        self.register_output(name + '_stress', (stress_a + stress_b)/2)







"""
the stress for box beams is evaluated at four points:
    0 ------------------------------------- 1
      -                y                  -
      -                |                  -
      4                --> x              -
      -                                   -
      -                                   -
    3 ------------------------------------- 2
"""
class StressBox(csdl.Model):
    def initialize(self):
        self.parameters.declare('name')
    def define(self):
        name = self.parameters['name']

        A = self.declare_variable(name+'_A')
        J = self.declare_variable(name+'_J')
        Iy = self.declare_variable(name+'_Iy') # height axis
        Iz = self.declare_variable(name+'_Iz') # width axis

        w = self.declare_variable(name+'_w')
        h = self.declare_variable(name+'_h')

        # get the local loads:
        local_loads = self.declare_variable(name+'local_loads',shape=(12))
        # self.print_var(local_loads)

        loads_a = local_loads[0:6]
        loads_b = local_loads[6:12]

        # create the point coordinate matrix
        x_coord = self.create_output(name+'x_coord',shape=(5),val=0)
        y_coord = self.create_output(name+'y_coord',shape=(5),val=0)
        # point 1
        x_coord[0] = -w/2
        y_coord[0] = h/2
        # point 2
        x_coord[1] = w/2
        y_coord[1] = h/2
        # point 3
        x_coord[2] = w/2
        y_coord[2] = -h/2
        # point 4
        x_coord[3] = -w/2
        y_coord[3] = -h/2
        # point 5
        x_coord[4] = -w/2




        # compute the stress at each point:
        shear_a = self.create_output(name + 'shear_a',shape=(5), val=0)
        shear_b = self.create_output(name + 'shear_b',shape=(5), val=0)
        stress_a = self.create_output(name + 'stress_a', shape=(5), val=0)
        stress_b = self.create_output(name + 'stress_b', shape=(5), val=0)
        stress_a_comp = self.create_output(name + 'stress_a_comp', shape=(5), val=0)
        stress_b_comp = self.create_output(name + 'stress_b_comp', shape=(5), val=0)
        
        # hard-coding values for quasi-isotropic laminate
        T = self.declare_variable('T_composite', val=690.*1e+6)
        C = self.declare_variable('C_composite', val=520.*1e+6)

        tweb = self.declare_variable(name + '_tweb')
        Q = self.declare_variable(name + '_Q')

        # self.print_var(tweb)

        for point in range(5):
            x = x_coord[point]
            y = y_coord[point]
            r = (x**2 + y**2)**0.5

            s_axial_a = (loads_a[0]/A) + (loads_a[4]*y/Iy) + (loads_a[5]*x/Iz)
            s_axial_b = (loads_b[0]/A) + (loads_b[4]*y/Iy) + (loads_b[5]*x/Iz)
            s_torsional_a = loads_a[3]*r/J
            s_torsional_b = loads_b[3]*r/J

            if point == 4: # the max shear at the neutral axis:
                shear_a[point] = loads_a[2]*Q/(Iy*2*tweb)
                shear_b[point] = loads_b[2]*Q/(Iy*2*tweb)
                # self.print_var(transverse_shear_stress_a)

            tau_a = s_torsional_a + shear_a[point]
            tau_b = s_torsional_b + shear_b[point]

            stress_a[point] = (s_axial_a**2 + 3*tau_a**2 + 1E-14)**0.5
            stress_b[point] = (s_axial_b**2 + 3*tau_b**2 + 1E-14)**0.5
            stress_a_comp[point] = (1/T - 1/C)*s_axial_a + 1/(T*C)*s_axial_a**2 + 3/(T*C)*tau_a**2
            stress_b_comp[point] = (1/T - 1/C)*s_axial_b + 1/(T*C)*s_axial_b**2 + 3/(T*C)*tau_b**2

            

        max_stress_a = csdl.max(1E-3*stress_a)/1E-3
        max_stress_b = csdl.max(1E-3*stress_b)/1E-3

        #stress_ab = self.create_output(name + 'stress_ab', shape=(2), val=0)
        #stress_ab[0] = max_stress_a
        #stress_ab[1] = max_stress_b

        # self.register_output(name + '_stress', csdl.max(1E-3*stress_ab)/1E-3)
        self.register_output(name + '_stress', (max_stress_a + max_stress_b)/2)
        

        self.register_output(name + '_stress_array', (stress_a + stress_b)/2) # a more reliable stress constraint
        self.register_output(name + '_comp_stress_array', (stress_a_comp + stress_b_comp)/2)