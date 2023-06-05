import numpy as np
import csdl
import python_csdl_backend
from aframe.core.aframe import Aframe
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)


n = 21
mesh = np.zeros((n,3))
mesh[:,1] = np.linspace(-10,10,n)

forces = np.zeros((n,3))
forces[:,2] = 100

class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams',default={})
        self.parameters.declare('bounds',default={})
        self.parameters.declare('joints',default={})
    def define(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']

        
        self.create_input('wing_mesh', shape=(n,3), val=mesh)
        self.create_input('wing_t', shape=(n - 1), val=0.001)
        self.create_input('wing_r', shape=(n - 1), val=0.1)
        self.create_input('wing_forces', shape=(n,3), val=forces)
        
        # solve the beam group:
        self.add(Aframe(beams=beams, bounds=bounds, joints=joints), name='Aframe')


        self.add_constraint('stress', upper=450E6, scaler=1E-8)
        self.add_design_variable('wing_t', lower=0.0001 ,scaler=100)
        self.add_objective('mass',scaler=1E-2)
        
        



if __name__ == '__main__':

    joints, bounds, beams = {}, {}, {}
    beams['wing'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(n))}
    bounds['root'] = {'beam': 'wing','node': 10,'fdim': [1,1,1,1,1,1]}




    sim = python_csdl_backend.Simulator(Run(beams=beams,bounds=bounds,joints=joints))
    sim.run()


    # prob = CSDLProblem(problem_name='run_opt', simulator=sim)
    # optimizer = SLSQP(prob, maxiter=1000, ftol=1E-12)
    # optimizer.solve()
    # optimizer.print_results()


    stress = sim['stress']
    print(stress)

    t = sim['wing_t']
    print(t)

    print(sim['mass'])

    # plotting:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for beam_name in beams:
        n = len(beams[beam_name]['nodes'])
        for i in range(n - 1):
            element_name = beam_name + '_element_' + str(i)
            na = sim[element_name+'node_a_def']
            nb = sim[element_name+'node_b_def']

            x = np.array([na[0], nb[0]])
            y = np.array([na[1], nb[1]])
            z = np.array([na[2], nb[2]])

            ax.plot(x,y,z,color='k',label='_nolegend_',linewidth=2)
            ax.scatter(na[0], na[1], na[2],color='yellow',edgecolors='black',linewidth=1,zorder=10,label='_nolegend_',s=30)
            ax.scatter(nb[0], nb[1], nb[2],color='yellow',edgecolors='black',linewidth=1,zorder=10,label='_nolegend_',s=30)



    ax.set_xlim(-1,1)
    ax.set_ylim(-10,10)
    ax.set_zlim(-1,1)
    plt.show()

    
