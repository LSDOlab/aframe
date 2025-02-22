import csdl_alpha as csdl
import numpy as np
import aframe as af



# Aurora PAV V&V case
# uses box beams


# start recorder
recorder = csdl.Recorder(inline=True)
recorder.start()

# create a 1D beam mesh
num_nodes = 21
mesh = np.zeros((num_nodes, 3))
mesh[:, 1] = np.array([14.03091,12.62775,11.22466,9.821535,8.418443,7.015352,5.612261,
                            4.20917,2.873796,1.403091,4.96E-13,-1.40309,-2.87371,-4.20913,
                            -5.61222,-7.01531,-8.4184,-9.82149,-11.2246,-12.6277,-14.0308]) / 3.2808399
beam_mesh = csdl.Variable(value=mesh)

nodal_width = np.array([15.02834,16.68572,18.34315,20.6949,23.25182,25.80875,28.36568,
                        30.92261,33.78855,38.20996,38.22185,38.21021,33.78875,30.92281,
                        28.36586,25.80892,23.25198,20.69503,18.34327,16.68583,15.0284]) / 39.3700787
nodal_height = np.array([4.829658,5.362334,5.894978,6.650791,7.472516,8.29424,9.115964,
                        9.937689,10.78953,12.2796,12.29251,12.27349,10.78281,9.932765,
                        9.111449,8.290134,7.468819,6.647503,5.892053,5.359674,4.827294]) / 39.3700787

element_width = np.zeros((num_nodes - 1))
element_height = np.zeros((num_nodes - 1))
for i in range(num_nodes - 1):
    element_width[i] = (nodal_width[i] + nodal_width[i + 1]) / 2
    element_height[i] = (nodal_height[i] + nodal_height[i + 1]) / 2

beam_width = csdl.Variable(value=element_width)
beam_height = csdl.Variable(value=element_height)

tweb = csdl.Variable(value=np.ones(num_nodes - 1) * 0.05 / 39.3700787)
tcap = csdl.Variable(value=np.ones(num_nodes - 1) * 0.05 / 39.3700787)

nodal_loads = np.zeros((num_nodes, 6))
nodal_loads[:, 2] = np.array([178.525,225.7911,255.3446,254.0379,264.366,274.624,281.8639,
                            292.5068,318.2695,325.1313,0,324.9549,318.1306,292.57,281.8554,
                            274.68,264.4084,254.1061,255.3735,225.8431,178.5819]) * 4.44822162
loads = csdl.Variable(value=nodal_loads)

aluminum = af.Material(name='aluminum', E=7.3E10, G=27E9, density=2768)


beam_cs = af.CSBox(ttop=tcap, tbot=tcap, tweb=tweb, height=beam_height, width=beam_width)


beam = af.Beam(name='beam_1', mesh=beam_mesh, material=aluminum, cs=beam_cs)
beam.fix(node=10)
beam.add_load(loads)


frame = af.Frame(beams=[beam])

# solve the linear system
frame.solve()

# get the displacement
displacement = frame.displacement[beam.name]

# return stress at all the stress evaluation points for each element
"""
        0-----------------1
        |                 |
        |                 |
        4                 |
        |                 |
        |                 |
        3-----------------2
"""
stress = frame.compute_stress()[beam.name]


recorder.stop()

print('result: ', displacement.value * 39.3700787)

print('stress: ', stress.value)