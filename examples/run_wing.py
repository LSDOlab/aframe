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
mesh[:,1] = np.linspace(-20,20,n)

forces = np.zeros((n,3))
forces[:,2] = 1000
forces[:,0] = 0

h = np.ones(n)*0.375
w = np.ones(n)*1.5



class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams',default={})
        self.parameters.declare('bounds',default={})
        self.parameters.declare('joints',default={})
    def define(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']


        self.create_input('wing_mesh', shape=(len(mesh), 3), val=mesh)
        self.create_input('wing_height', shape=(len(mesh)), val=h)
        self.create_input('wing_width', shape=(len(mesh)), val=w)
        self.create_input('wing_tcap', shape=(len(mesh)), val=0.01)
        self.create_input('wing_tweb', shape=(len(mesh)), val=0.001)
        self.create_input('wing_forces', shape=(len(mesh),3), val=forces)

        
        # solve the beam group:
        self.add(Aframe(beams=beams, bounds=bounds, joints=joints, mesh_units='ft'), name='Aframe')


        self.add_constraint('wing_stress', upper=450E6, scaler=1E-8)
        self.add_design_variable('wing_tcap', lower=0.00001, upper=0.5, scaler=1E1)
        self.add_design_variable('wing_tweb', lower=0.00001, upper=0.5, scaler=1E2)
        self.add_objective('mass', scaler=1E-2)

        mass = self.declare_variable('mass')
        self.print_var(mass)
        
        






if __name__ == '__main__':

    joints, bounds, beams = {}, {}, {}
    beams['wing'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'box','nodes': list(range(len(mesh)))}
    bounds['root'] = {'beam': 'wing','node': 10,'fdim': [1,1,1,1,1,1]}


    sim = python_csdl_backend.Simulator(Run(beams=beams,bounds=bounds,joints=joints))
    sim.run()

    
    #prob = CSDLProblem(problem_name='run_opt', simulator=sim)
    #optimizer = SLSQP(prob, maxiter=1000, ftol=1E-8)
    #optimizer.solve()
    #optimizer.print_results()


    print('tcap: ', sim['wing_tcap'])
    print('tweb: ', sim['wing_tweb'])

    print('bkl: ', sim['wing_bkl'])

    cg = sim['cg_vector']


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


    ax.scatter(cg[0],cg[1],cg[2],color='blue',s=50,edgecolors='black')


    ax.set_xlim(-5,5)
    ax.set_ylim(-10,10)
    ax.set_zlim(0,5)
    plt.show()


    #print(sim['wing_stress'])
    plt.plot(sim['wing_stress'])
    plt.show()