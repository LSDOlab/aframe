import csdl_alpha as csdl
import numpy as np
import aframe as af


recorder = csdl.Recorder(inline=True)
recorder.start()

num_beam_nodes = 9

# Setting the beam nodes (num_nodes, 3) where 3 refers to x, y, z coordinates
# beam_nodes = csdl.Variable(
#     value=np.array([
#         [-3.53927E+00,2.96057E-05,-2.63734E+00],
#         [-3.55440E+00,9.24183E-01,-2.60233E+00],
#         [-3.57151E+00,1.88572E+00,-2.56586E+00],
#         [-3.58867E+00,2.84852E+00,-2.52935E+00],
#         [-3.60592E+00,3.81277E+00,-2.49278E+00],
#         [-3.62204E+00,4.77378E+00,-2.45601E+00],
#         [-3.64281E+00,5.73453E+00,-2.41846E+00],
#         [-3.67890E+00,6.69733E+00,-2.38051E+00],
#         [-3.92518E+00,7.69578E+00,-2.32876E+00],
#     ])
# )

beam_nodes = csdl.Variable(
    value=np.array([
        [0.,0.,0.],
        [0.,1.,0.],
        [0.,2.,0.],
        [0.,3.,0.],
        [0.,4.,0.],
        [0.,5.,0.],
        [0.,6.,0.],
        [0.,7.,0.],
        [0.,8.,0.],
    ])
)

# Thicknesses (num_beam_nodes-1)
tbot = csdl.Variable(value=np.array([
    4.7226865E-03,
    4.3909194E-03,
    3.9121813E-03,
    3.3915973E-03,
    2.9390926E-03,
    2.0629110E-03,
    1.3926143E-03,
    3.0000000E-04,
]))

ttop = csdl.Variable(value=np.array([
    8.298922E-03,
    7.689502E-03,
    6.751313E-03,
    5.704943E-03,
    4.375919E-03,
    2.971346E-03,
    1.925665E-03,
    3.000000E-04,
]))

tweb = csdl.Variable(value=3.00E-04 * np.ones((num_beam_nodes-1, )))

# beam width
# width = csdl.Variable(value=np.array([
#     8.087064E-01,
#     7.670528E-01,
#     7.242612E-01,
#     6.813902E-01,
#     6.374276E-01,
#     5.865347E-01,
#     5.257780E-01,
#     2.467123E-01,
# ]))

# beam height
# height = csdl.Variable(value=np.array([
#     2.6502895E-01,
#     2.5133992E-01,
#     2.3728960E-01,
#     2.2321557E-01,
#     2.0880246E-01,
#     1.9208410E-01,
#     1.7181877E-01,
#     1.0655984E-01,
# ]))

height = csdl.Variable(
    shape=(8, ),
    value=0.5
)

width = csdl.Variable(
    shape=(8, ),
    value=0.5
)

# Beam nodal forces (num_nodes, 3) (x, y, z)
# beam_forces = np.array([
#     [3.028972E+02,1.940171E+02,-4.005064E+03],
#     [5.575601E+02,3.279875E+02,-6.741551E+03],
#     [6.721802E+02,3.420607E+02,-6.970990E+03],
#     [7.267794E+02,3.272591E+02,-6.610402E+03],
#     [7.387596E+02,3.090851E+02,-6.139581E+03],
#     [7.166787E+02,2.950833E+02,-5.558692E+03],
#     [6.552210E+02,2.881245E+02,-4.848959E+03],
#     [4.970194E+02,2.987922E+02,-3.685410E+03],
#     [1.633403E+02,1.808189E+02,-1.475417E+03],
# ])

beam_forces = np.array([
    [0,0,-5000.],
    [0,0,-5000.],
    [0,0,-5000.],
    [0,0,-5000.],
    [0,0,-5000.],
    [0,0,-5000.],
    [0,0,-5000.],
    [0,0,-5000.],
    [0,0,-5000.],
])

areas = csdl.Variable(shape=(8, ), value=0.0099)
iz = csdl.Variable(shape=(8, ), value=0.000404332)
iy = csdl.Variable(shape=(8, ), value=0.000404332)
ix = csdl.Variable(shape=(8, ), value=0.000609781)


# Setting the loads (num_beam_nodes, 6) where the first three loads are the forces and the last three are moments;
#  here we don't apply any moments so the [:, 3:] section of the loads array will be zero
beam_loads = np.zeros((num_beam_nodes, 6))
beam_loads[:, 0:3] = beam_forces

beam_loads_csdl = csdl.Variable(value=beam_loads)

beam_cs = af.CSBoxMarius(
    height=height, 
    width=width, 
    ttop=ttop, 
    tbot=tbot, 
    tweb=tweb,
    area=areas,
    ix=ix,
    iy=iy,
    iz=iz,
)

aluminum = af.Material(name='aluminum', E=69E9, G=26E9, density=2700)
beam = af.Beam(name='beam', mesh=beam_nodes, material=aluminum, cs=beam_cs)
beam.fix(node=0)
beam.add_load(beam_loads_csdl)

frame = af.Frame()
frame.add_beam(beam)

# evaluating the frame model returns a solution dataclass
frame.solve()

stress = frame.compute_stress()
beam_stress = stress['beam']

print("stress", np.max(beam_stress.value, axis=1))
# print("displacement", beam_displacement.value[:, 2])
# print("max stress", np.max(beam_stress.value, axis=1))
# print("stress 1", beam_stress.value)
# print("stress 2", beam_stress.value[:, 1].flatten())
# print("stress 3", beam_stress.value[:, 2].flatten())
# print("stress 4", beam_stress.value[:, 3].flatten())
# print("stress 5", beam_stress.value[:, 4].flatten())
# print("bot_bkl", bot_bkl.value)
# print("top_bkl", bot_bkl.value)