import numpy as np
import csdl
import python_csdl_backend
from aframe.core.aframe import Aframe
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
import scipy.io as sio


axis_nodes_dict = sio.loadmat('examples/data/axis_nodes.mat')
axis_nodes = axis_nodes_dict['axis_nodes']/39.3700787
# add the initial node:
axis_nodes = np.concatenate([np.array([[9.86,0,1.042]]),axis_nodes])
airfoil_ribs_points_dict = sio.loadmat('examples/data/ribs_oml_points.mat')
airfoil_ribs_points = airfoil_ribs_points_dict['airfoil_ribs_points']/39.3700787


w, h = np.zeros((len(axis_nodes))), np.zeros((len(axis_nodes)))
for i in range(1,len(axis_nodes) - 1):
    top_left = airfoil_ribs_points[0,:,i]
    top_right = airfoil_ribs_points[9,:,i]
    bot_left = airfoil_ribs_points[19,:,i]
    bot_right = airfoil_ribs_points[10,:,i]
    w_top = np.linalg.norm(top_right - top_left)
    w_bot = np.linalg.norm(bot_right - bot_left)
    h_front = np.linalg.norm(top_left - bot_left)
    h_back = np.linalg.norm(top_right - bot_right)

    w[i] = (w_top + w_bot)/2
    h[i] = (h_front + h_back)/2

w[0] = w[1]
h[0] = h[1]

loads_dict = sio.loadmat('examples/data/loads_2p5g_n1g_aero_static.mat')
static_forces = loads_dict['forces']
static_moments = loads_dict['moments']

forces, moments = np.zeros((len(axis_nodes),3)), np.zeros((len(axis_nodes),3))
for i in range(len(axis_nodes) - 2):
    forces[i+1,:] = static_forces[0,i,:]
    moments[i+1,:] = static_moments[0,i,:]




class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams',default={})
        self.parameters.declare('bounds',default={})
        self.parameters.declare('joints',default={})
    def define(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']


        self.create_input('wing_mesh', shape=(len(axis_nodes),3), val=axis_nodes)

        self.create_input('wing_h', shape=(len(axis_nodes)-1), val=h[0:-1])
        self.create_input('wing_w', shape=(len(axis_nodes)-1), val=w[0:-1])
        self.create_input('wing_tcap', shape=(len(axis_nodes) - 1), val=0.003)
        self.create_input('wing_tweb', shape=(len(axis_nodes) - 1), val=0.003)
        self.create_input('wing_forces', shape=(len(axis_nodes),3), val=forces)
        # self.create_input('wing_moments', shape=(len(axis_nodes),3), val=moments)

        
        # solve the beam group:
        self.add(Aframe(beams=beams, bounds=bounds, joints=joints), name='Aframe')


        self.add_constraint('stress', upper=450E6/1, scaler=1E-8)
        self.add_design_variable('wing_tcap', lower=0.001, scaler=1E2)
        self.add_design_variable('wing_tweb', lower=0.001, scaler=1E3)
        self.add_objective('mass', scaler=1E-2)

        mass = self.declare_variable('mass')
        self.print_var(mass)
        
        






if __name__ == '__main__':

    joints, bounds, beams = {}, {}, {}
    beams['wing'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'box','nodes': list(range(len(axis_nodes)))}
    bounds['root'] = {'beam': 'wing','node': 0,'fdim': [1,1,1,1,1,1]}


    sim = python_csdl_backend.Simulator(Run(beams=beams,bounds=bounds,joints=joints))
    sim.run()


    # prob = CSDLProblem(problem_name='run_opt', simulator=sim)
    # optimizer = SLSQP(prob, maxiter=1000, ftol=1E-8)
    # optimizer.solve()
    # optimizer.print_results()


    print('displacement: ', sim['wing_displacement'])
    print('stress', sim['stress'])


    print(sim['wing_tcap'])
    print(sim['wing_tweb'])


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



    ax.set_xlim(0,20)
    ax.set_ylim(0,15)
    ax.set_zlim(0,5)
    plt.show()