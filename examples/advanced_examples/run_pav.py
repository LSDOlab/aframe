import numpy as np
import csdl
import python_csdl_backend
from aframe.core.aframe import Aframe
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import scipy
plt.rcParams.update(plt.rcParamsDefault)


m2ft = 3.288084
m2in = 39.3701
N2lbf = 0.224809
Npm22psi = 0.000145038
Nm2lbft = 0.73756214728
m42in4 = 2402509.607720009

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
        self.parameters.declare('beams', default={})
        self.parameters.declare('bounds', default={})
        self.parameters.declare('joints', default={})
        self.parameters.declare('optimization_flag', default=False)

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

        if self.parameters['optimization_flag']:
            self.add_constraint('wing_stress', upper=2.16e8, scaler=1E-8)
            self.add_design_variable('wing_tcap', lower=0.001, upper=0.2, scaler=1E2)
            self.add_design_variable('wing_tweb', lower=0.001, upper=0.2, scaler=1E3)
            self.add_objective('mass', scaler=1E-2)
        
        




if __name__ == '__main__':

    fileObj = open('PAVsymbbeam_data.p', 'rb')
    PAVsymbbeam_data = pickle.load(fileObj)
    fileObj.close()

    PAVsymbbeam_data['LoadNodes'].to_csv('PavLoadNodesFromRade.csv')
    PAVsymbbeam_data['AxisNodes'].to_csv('PavAxisNodesFromRade.csv')
    PAVsymbbeam_data['WidthHeight'].to_csv('PavWidthHeightFromRade.csv')

    scipy.io.savemat('PavRibsPointsFromRade.mat',
                     {'Airfoil': PAVsymbbeam_data['AirfoilRibsPoints'],
                      'Corner': PAVsymbbeam_data['CornerRibsPoints']}
                     )

    joints, bounds, beams = {}, {}, {}
    beams['wing'] = {'E': 7.31E10,'G': 26E9,'rho': 2768,'cs': 'box','nodes': list(range(len(nodes)))}
    bounds['root'] = {'beam': 'wing','node': 10,'fdim': [1,1,1,1,1,1]}


    sim = python_csdl_backend.Simulator(
        Run(beams=beams,
            bounds=bounds,
            joints=joints,
            optimization_flag=False), 
        analytics=True
        )
    sim.run()

    spanwise_location_ft = sim['wing_mesh'][:, 1]*m2ft
    width_in = sim['wing_width']*m2in
    height_in = sim['wing_height']*m2in
    tcap_in = sim['wing_tcap']*m2in
    tweb_in = sim['wing_tweb']*m2in
    nodal_forces_lbf = sim['wing_forces']*N2lbf
    
    stress_psi = np.max(sim['wing_stress'], axis=1)*Npm22psi
    disp_in = sim['wing_displacement'][:, 2]*m2in

    element_loads = sim['wing_element_loads'] # (n-1,6) [fx,fy,fz,mx,my,mz]
    element_axial_stress = np.max(np.abs(sim['wing_element_axial_stress']), axis=1) # (n-1,5)
    element_shear_stress = np.abs(sim['wing_element_shear_stress']) # (n-1), evaluated at the center of the web
    element_torsional_stress = np.max(np.abs(sim['wing_element_torsional_stress']), axis=1)
    element_sp_cap = sim['wing_sp_cap'] # (n-1) critical stress for the spar cap
    element_Iy_out = sim['wing_iyo'] # (n-1)
    element_Iz_out = sim['wing_izo'] # (n-1)
    element_J_out = sim['wing_jo'] # (n-1)

    beamNodalDf = pd.DataFrame(
        data={
            'Spanwise y-location (ft)': spanwise_location_ft,
            'Width (in)': width_in,
            'Height (in)': height_in,
            'Web thickness (in)': tweb_in,
            'Cap thickness (in)': tcap_in,
            'Nodal forces (lbf)': nodal_forces_lbf[:, 2],
            'Displacement (in)': disp_in,
            'Stress (psi)': stress_psi
        },
    )

    beamElementDf = pd.DataFrame(
        data={
            'Spanwise y-location (ft)': np.average([spanwise_location_ft[:-1], spanwise_location_ft[1:]], axis=0),
            'Axial stress (psi)': element_axial_stress*Npm22psi,
            'Shear stress (psi)': element_shear_stress*Npm22psi,
            'Torsional stress (psi)': element_torsional_stress*Npm22psi,
            'Fz (lbf)': element_loads[:, 2]*N2lbf,
            'My (lbf-ft)': element_loads[:, 4]*Nm2lbft,
            'Iy (in^4)': element_Iy_out * m42in4,
            'Iz (in^4)': element_Iz_out * m42in4,
            'J (in^4)': element_J_out * m42in4,
        },
    )
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(beamNodalDf)
        print(beamElementDf)

    beamNodalDf.to_excel('pav_3g_results_nodal.xlsx')
    beamElementDf.to_excel('pav_3g_results_element.xlsx')
    print('Max stress (psi): ', np.max(stress_psi))
    print('displacement (in): ', np.max(disp_in))

    plt.plot(stress_psi)
    plt.show()

    plt.plot(element_loads[:,4])
    plt.show()