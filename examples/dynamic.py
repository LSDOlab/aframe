import csdl_alpha as csdl
import numpy as np
import aframe as af


# start recorder
recorder = csdl.Recorder(inline=True)
recorder.start()

# create a 1D beam 1 mesh
num_nodes_1 = 11
beam_1_mesh = np.zeros((num_nodes_1, 3))
beam_1_mesh[:, 1] = np.linspace(0, 10, num_nodes_1)
beam_1_mesh = csdl.Variable(value=beam_1_mesh)

# create beam 1 loads
beam_1_loads = np.zeros((num_nodes_1, 6))
beam_1_loads[:, 2] = 20000
beam_1_loads = csdl.Variable(value=beam_1_loads)

# create a material
aluminum = af.IsotropicMaterial(name='aluminum', E=69E9, G=26E9, density=2700)

# create cs properties for beam 1
beam_1_radius = csdl.Variable(value=np.ones(num_nodes_1 - 1) * 0.5)
beam_1_thickness = csdl.Variable(value=np.ones(num_nodes_1 - 1) * 0.001)
beam_1_cs = af.CSTube(radius=beam_1_radius, thickness=beam_1_thickness)

# create beam 1 with boundary conditions and loads
beam_1 = af.Beam(name='beam_1', mesh=beam_1_mesh, material=aluminum, cs=beam_1_cs)
beam_1.add_boundary_condition(node=0, dof=[1, 1, 1, 1, 1, 1])
beam_1.add_load(beam_1_loads)

# instantiate the frame model and add all beams and joints
frame = af.Frame()
frame.add_beam(beam_1)

# evaluating the frame model returns a solution dataclass
solution = frame.evaluate()

# displacement
beam_1_displacement = solution.get_displacement(beam_1)

# displaced mesh
beam_1_def_mesh = solution.get_mesh(beam_1)

# stress
beam_1_stress = solution.get_stress(beam_1)

M = solution.M
K = solution.K
F = solution.F
u0 = solution.u0
node_dictionary = solution.node_dictionary
index = solution.index

# finish up
recorder.stop()


from scipy.integrate import solve_ivp

# Define the matrices M, K, and the vector F
M = M.value
K = K.value
F = F.value
y0 = u0.value

nu = len(y0)

def ode(t, y, nu):
    u = y[0:nu]
    u_dot = y[nu:-1]
    u_ddot = np.linalg.solve(M, F - K @ u)
    return np.concatenate((u_dot, u_ddot))

nt = 100
t_span = (0, 0.01)  # start and end time
t_eval = np.linspace(t_span[0], t_span[1], nt)  # times at which to store the computed solution

# Solve the system of ODEs
sol = solve_ivp(ode, t_span, y0, t_eval=t_eval, args=(nu,), method='Radau') # 'LSODA' works well also

t = sol.t
u = sol.y # 66 x 10

mesh = beam_1.mesh.value

def_mesh = np.zeros((beam_1.num_nodes, 3, nt))

for i in range(nt):
    for j in range(beam_1.num_nodes):
        node_index = index[node_dictionary[beam_1.name][j]] * 6
        def_mesh[j, :, i] = mesh[j, :] + u[node_index:node_index + 3, i]

from matplotlib import pyplot as plt

plt.scatter(def_mesh[:, 1, :], def_mesh[:, 2, :])
plt.show()

