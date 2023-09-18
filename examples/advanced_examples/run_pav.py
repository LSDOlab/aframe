import numpy as np
import csdl
import python_csdl_backend
from aframe.core.aframe import Aframe
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)


# region Data
nodes = np.array([[ 2.94811022e+00,  4.26720000e+00,  6.04990180e-01],
       [ 2.97757859e+00,  3.84045955e+00,  6.04825502e-01],
       [ 3.00704575e+00,  3.41373953e+00,  6.04644661e-01],
       [ 3.04885750e+00,  2.98700842e+00,  6.04464035e-01],
       [ 3.09431722e+00,  2.56028844e+00,  6.04283473e-01],
       [ 3.13977694e+00,  2.13356845e+00,  6.04102911e-01],
       [ 3.18523667e+00,  1.70684846e+00,  6.03922349e-01],
       [ 3.23069639e+00,  1.28012847e+00,  6.03741787e-01],
       [ 3.28160445e+00,  8.74003116e-01,  6.03561200e-01],
       [ 3.36157619e+00,  4.26719858e-01,  6.03381338e-01],
       [ 3.36177427e+00,  1.50947138e-13,  6.03197467e-01],
       [ 3.36156802e+00, -4.26719800e-01,  6.03378820e-01],
       [ 3.28159740e+00, -8.73977233e-01,  6.03558999e-01],
       [ 3.23068979e+00, -1.28011567e+00,  6.03739749e-01],
       [ 3.18523061e+00, -1.70683566e+00,  6.03920480e-01],
       [ 3.13977143e+00, -2.13355565e+00,  6.04101210e-01],
       [ 3.09431225e+00, -2.56027564e+00,  6.04281941e-01],
       [ 3.04885308e+00, -2.98699562e+00,  6.04462671e-01],
       [ 3.00704183e+00, -3.41373123e+00,  6.04643452e-01],
       [ 2.97757502e+00, -3.84045126e+00,  6.04824402e-01],
       [ 2.94810822e+00, -4.26717128e+00,  6.04995104e-01]])

width = np.array([15.02832841,16.68571462,18.3431398,20.69488606,23.25181241,25.80873883,28.36566531,30.92259183,33.78853197,38.20994261,38.22183283,
                        38.2101895,33.7887308,30.92279163,28.36584859,25.80890559,23.25196265,20.69501978,18.34325832,16.68582243,15.0283895])*0.0254
height = np.array([4.829655269,5.36233072,5.894975046,6.6507877,7.472511573,8.294235446,9.11595932,9.937683193,10.78952744,12.27959289,
                        12.29249872,12.27348164,10.7828073,9.9327594,9.111444436,8.290129472,7.468814508,6.647499545,5.892050209,5.359670941,4.827291672])*0.0254
fz = np.array([178.5249412,225.7910602,255.3444864,254.0378545,264.3659094,274.6239472,281.8637954,292.5067646,318.2693761,325.1311971,0,
                        324.954771,318.1305384,292.5699649,281.8552967,274.6799369,264.4083816,254.1059857,255.3734613,225.8430446,178.5818996])*4.44822162
tcap = 0.05 * 0.0254 * np.ones(len(nodes))
tweb = 0.05 * 0.0254 * np.ones(len(nodes))

forces = np.zeros((len(nodes),3))
forces[:,2] = fz
# endregion

class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams',default={})
        self.parameters.declare('bounds',default={})
        self.parameters.declare('joints',default={})
    def define(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']

        self.create_input('wing_mesh', shape=(len(nodes),3), val=nodes)
        self.create_input('wing_height', shape=(len(nodes)), val=height)
        self.create_input('wing_width', shape=(len(nodes)), val=width)
        self.create_input('wing_tcap', shape=(len(nodes)), val=tcap)
        self.create_input('wing_tweb', shape=(len(nodes)), val=tweb)
        self.create_input('wing_forces', shape=(len(nodes),3), val=forces)
        
        # solve the PAV beam model:
        self.add(Aframe(beams=beams, bounds=bounds, joints=joints), name='Aframe')

        #self.add_constraint('wing_stress', upper=450E6, scaler=1E-8)
        #self.add_design_variable('wing_tcap', lower=0.001, upper=0.2, scaler=1E2)
        #self.add_design_variable('wing_tweb', lower=0.001, upper=0.2, scaler=1E3)
        #self.add_objective('mass', scaler=1E-2)
        
        




if __name__ == '__main__':

    joints, bounds, beams = {}, {}, {}
    beams['wing'] = {'E': 7.31E10,'G': 26E9,'rho': 2768,'cs': 'box','nodes': list(range(len(nodes)))}
    bounds['root'] = {'beam': 'wing','node': 10,'fdim': [1,1,1,1,1,1]}


    sim = python_csdl_backend.Simulator(Run(beams=beams,bounds=bounds,joints=joints))
    sim.run()


    stress = sim['wing_stress']
    disp = sim['wing_displacement']

    print('stress (psi): ', np.max(stress, axis=1)/6894.75729)
    print('displacement (in): ', disp*39.3700787)



    plt.plot(stress)
    plt.show()