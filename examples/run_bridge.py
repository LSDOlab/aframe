import numpy as np
import csdl
import python_csdl_backend
from aframe.core.aframe import Aframe
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)




n = 5

class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams',default={})
        self.parameters.declare('bounds',default={})
        self.parameters.declare('joints',default={})
    def define(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']


        # dummy mesh generation code:
        for beam_name in beams:
            num_beam_nodes = len(beams[beam_name]['nodes'])
            # get the beam start/stop coordinates
            a = self.create_input(beam_name+'a',shape=(3),val=beams[beam_name]['a'])
            b = self.create_input(beam_name+'b',shape=(3),val=beams[beam_name]['b'])
            ds = (b - a)/(num_beam_nodes - 1)

            mesh = self.create_output(beam_name+'_mesh', shape=(num_beam_nodes,3), val=0)
            for i in range(num_beam_nodes):
                node_i = a + ds*i
                mesh[i,:] = csdl.reshape(node_i, (1,3))


        f = np.zeros((n,3))
        f[-1,1] = 1500000
        self.create_input('b4_forces', shape=(n,3), val=f)

        self.add(Aframe(beams=beams, bounds=bounds, joints=joints), name='Aframe')






if __name__ == '__main__':
    
    beams, bounds, joints = {}, {}, {}

    beams['b1'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(n)),'a': [0,0,0],'b': [1,0,0]}
    beams['b2'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(n)),'a': [0,0,0],'b': [0.5,1,0]}
    beams['b3'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(n)),'a': [0.5,1,0],'b': [1,0,0]}
    beams['b4'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(n)),'a': [0.5,1,0],'b': [1.5,1,0]}
    beams['b5'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(n)),'a': [1,0,0],'b': [1.5,1,0]}
    beams['b6'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(n)),'a': [1,0,0],'b': [2,0,0]}
    beams['b7'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(n)),'a': [1.5,1,0],'b': [2,0,0]}
    beams['b8'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(n)),'a': [1.5,1,0],'b': [2.5,1,0]}
    beams['b9'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(n)),'a': [2,0,0],'b': [2.5,1,0]}
    beams['b10'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(n)),'a': [2,0,0],'b': [3,0,0]}
    beams['b11'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(n)),'a': [2.5,1,0],'b': [3,0,0]}

    bounds['fixed_left'] = {'beam': 'b1','node': 0,'fdim': [1,1,1,1,1,1]}
    bounds['fixed_right'] = {'beam': 'b10','node': n-1,'fdim': [1,1,1,1,1,1]}

    joints['c1'] = {'beams': ['b1','b2'],'nodes': [0,0]}
    joints['c2'] = {'beams': ['b2','b3','b4'],'nodes': [n-1,0,0]}
    joints['c3'] = {'beams': ['b1','b3','b5','b6'],'nodes': [n-1,n-1,0,0]}
    joints['c4'] = {'beams': ['b4','b5','b7','b8'],'nodes': [n-1,n-1,0,0]}
    joints['c5'] = {'beams': ['b6','b7','b9','b10'],'nodes': [n-1,n-1,0,0]}
    joints['c6'] = {'beams': ['b8','b9','b11'],'nodes': [n-1,n-1,0]}
    joints['c7'] = {'beams': ['b10','b11'],'nodes': [n-1,n-1]}




    sim = python_csdl_backend.Simulator(Run(beams=beams,bounds=bounds,joints=joints))
    sim.run()

    # plot the undeformed bridge:
    for beam_name in beams:
        n = len(beams[beam_name]['nodes'])
        for i in range(n - 1):
            element_name = beam_name + '_element_' + str(i)
            a = sim[element_name+'node_a']
            b = sim[element_name+'node_b']

            xu = np.array([a[0],b[0]])
            yu = np.array([a[1],b[1]])
            zu = np.array([a[2],b[2]])
            plt.plot(xu,yu,color='silver',linewidth=4)
            plt.scatter(xu,yu,s=50,color='silver',edgecolors='black',linewidth=0.5,zorder=5)

    for beam_name in beams:
        n = len(beams[beam_name]['nodes'])
        for i in range(n - 1):
            element_name = beam_name + '_element_' + str(i)
            na = sim[element_name+'node_a_def']
            nb = sim[element_name+'node_b_def']

            x = np.array([na[0], nb[0]])
            y = np.array([na[1], nb[1]])
            z = np.array([na[2], nb[2]])
            plt.plot(x,y,color='k',zorder=7)
            plt.scatter(x,y,s=50,zorder=10,color='yellow',edgecolors='black',linewidth=1)

    # plot the applied force arrow:
    plt.arrow(1.5,1,0,0.2,width=0.04,color='red')

    # plot the cg:
    plt.scatter(sim['cg'][0],sim['cg'][1],color='blue',s=50,edgecolors='black')

    plt.show()