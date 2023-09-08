import numpy as np
import csdl



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
class NodalStressBox(csdl.Model):
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
        load = self.declare_variable(name + 'n_load', shape=(6))

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
        shear = self.create_output(name + 'shear', shape=(5), val=0)
        stress = self.create_output(name + 'stress', shape=(5), val=0)

        tweb = self.declare_variable(name + '_tweb')
        Q = self.declare_variable(name + '_Q')

        for point in range(5):
            x = x_coord[point]
            y = y_coord[point]
            r = (x**2 + y**2)**0.5

            s_axial = (load[0]/A) + (load[4]*y/Iy) + (load[5]*x/Iz)
            s_torsional = load[3]*r/J

            if point == 4: # the max shear at the neutral axis:
                shear[point] = load[2]*Q/(Iy*2*tweb)

            tau = s_torsional + shear[point]

            stress[point] = (s_axial**2 + 3*tau**2 + 1E-14)**0.5

        #self.print_var(stress)
