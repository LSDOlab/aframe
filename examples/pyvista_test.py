import csdl_alpha as csdl
import numpy as np
import aframe as af


recorder = csdl.Recorder(inline=True)
# recorder = csdl.Recorder()
recorder.start()

# create a 1D beam 1 mesh
num_nodes_1 = 21
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

beam_1_displacement = solution.get_displacement(beam_1)
beam_1_def_mesh = solution.get_mesh(beam_1)
beam_1_stress = solution.get_stress(beam_1)

recorder.stop()


import pyvista as pv

plotter = pv.Plotter()

for beam in frame.beams:
    mesh0 = beam.mesh.value
    mesh1 = solution.get_mesh(beam).value

    stress = beam_1_stress.value

    af.plot_mesh(plotter, mesh0, color='lightblue', line_width=10)
    # plot_mesh(plotter, mesh1, cell_data=stress, cmap='viridis', line_width=20)
    # plot_points(plotter, mesh1, color='blue', point_size=30)

    # plot_cyl(plotter, mesh1, cell_data=stress, cs_data=beam_1_cs.radius.value)

    height = np.ones((beam.num_elements)) * 0.2
    width = np.ones((beam.num_elements)) * 0.2
    af.plot_box(plotter, mesh1, height, width, cell_data=stress)

plotter.show()