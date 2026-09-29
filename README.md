# aframe
![aframe](https://github.com/user-attachments/assets/02c7aae8-5a49-4ff5-8a16-f2b6295dab02)

[![Tests](https://github.com/LSDOlab/aframe/actions/workflows/actions.yml/badge.svg)](https://github.com/LSDOlab/aframe/actions/workflows/actions.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)

**aframe** is a linear finite-element solver for 3D beams and frames, written in
[CSDL](https://github.com/LSDOlab/CSDL_alpha). The code is written entirely in CSDL, so it is fully differentiable and well-suited for gradient-based design optimization and analysis.

**Features**
- 3D Euler-Bernoulli beam elements with 6 dofs per node and consistent mass matrices
- Frames of many beams connected by rigid joints, with fixed and pinned supports
- Nodal forces and moments, and inertial loads from a frame acceleration (e.g. gravity) and point masses
- Cross-sections: tube, box (wing box), solid circle, ellipse, or your own properties
- Stress recovery (von Mises) for tube and box sections
- Exact derivatives through CSDL, with the NumPy (inline) or JAX backend

# Installation
aframe requires Python 3.9+ and [CSDL_alpha](https://github.com/LSDOlab/CSDL_alpha), which is not on PyPI, so install it first:
```bash
pip install git+https://github.com/LSDOlab/CSDL_alpha.git
git clone https://github.com/LSDOlab/aframe.git
```

# Quick start: a cantilever beam
A 10 m aluminum tube, clamped at the root, with a 1 kN tip load:

```python
import numpy as np
import csdl_alpha as csdl
import aframe as af

# record the model; inline=True evaluates values as the model is built
recorder = csdl.Recorder(inline=True)
recorder.start()

# a 10 m cantilever along x with 21 nodes (20 elements)
num_nodes = 21
mesh = np.zeros((num_nodes, 3))
mesh[:, 0] = np.linspace(0, 10, num_nodes)

# aluminum tube cross-section, one value per element
radius = csdl.Variable(value=np.full(num_nodes - 1, 0.1), name='radius')
thickness = csdl.Variable(value=np.full(num_nodes - 1, 0.005), name='thickness')
cs = af.CSTube(radius=radius, thickness=thickness)

beam = af.Beam(name='cantilever', mesh=csdl.Variable(value=mesh),
               E=69e9, G=26e9, density=2700, cs=cs)
beam.fix(0)  # clamp the root node

# nodal loads [Fx, Fy, Fz, Mx, My, Mz]: 1 kN downward at the tip
loads = np.zeros((num_nodes, 6))
loads[-1, 2] = -1000
beam.add_load(csdl.Variable(value=loads))

frame = af.Frame(beams=[beam])
frame.solve()

tip_deflection = frame.displacement['cantilever'][-1, 2]
stress = frame.compute_stress()['cantilever']  # von Mises, one value per element

# derivatives of any output with respect to any input
d_deflection_d_radius = csdl.derivative(tip_deflection, radius)

recorder.stop()

print('tip deflection (m):', tip_deflection.value)
print('max stress (Pa):', stress.value.max())
```

```
tip deflection (m): [-0.33159692]
max stress (Pa): 66924549.00143065
```

Both match beam theory: the tip deflection is PL³/3EI, and the maximum stress is the
bending stress M r / I at the midpoint of the root element (stresses are evaluated at
element midpoints). `d_deflection_d_radius` has shape (1, 20): the sensitivity of the
tip deflection to the radius of each element.

# Building models
### Beams
`af.Beam(name, mesh, E, G, density, cs, z=False)` is a beam discretized into
`num_nodes - 1` elements between the nodes of `mesh` (shape `(num_nodes, 3)`). `E`, `G`
and `density` are the Young's modulus, shear modulus and density; use any consistent
units (the examples use SI). Beam names must be unique within a frame, since results
are keyed by name.

Each element's local x axis points along the element. The standard construction of the
local y and z axes is singular for elements parallel to the global z axis, so **vertical
beams need `z=True`**.

### Cross-sections
Section properties are per element, i.e. arrays of shape `(num_nodes - 1,)`, and can be
design variables.

| Cross-section | Parameters | `stress()` |
|---|---|---|
| `af.CSTube` | `radius`, `thickness` | von Mises at the outer fiber, shape `(num_elements,)` |
| `af.CSBox` | `ttop`, `tbot`, `tweb`, `height`, `width` | von Mises at 4 corners and the web mid-height, shape `(num_elements, 5)` |
| `af.CSBox.from_properties` | box dimensions plus `area`, `ix`, `iy`, `iz` | as `CSBox` |
| `af.CSCircle` | `radius` | none |
| `af.CSEllipse` | `semi_major_axis`, `semi_minor_axis` | none |
| `af.CrossSection` | `area`, `ix`, `iy`, `iz` | none |

`ix` is the torsion constant and `iy`, `iz` are the second moments of area about the
element's local y and z axes. To add a section type, subclass `af.CrossSection`, pass the
computed properties to `super().__init__`, and override `stress`.

### Supports and loads
- `beam.fix(node)` clamps a node (all 6 dofs); `beam.pin(node)` fixes its translations only.
- `beam.add_load(loads)` sets the nodal loads, shape `(num_nodes, 6)`, ordered
  `[Fx, Fy, Fz, Mx, My, Mz]` in global coordinates. Distributed loads must be lumped
  to the nodes.
- `beam.add_inertial_mass(masses)` adds point masses at the nodes, shape `(num_nodes,)`.
- `af.Frame(..., acc=acc)` applies a frame acceleration `[ax, ay, az, αx, αy, αz]`,
  shape `(6,)`, as inertial loads `M @ acc` on the structure and on the point masses.
  Use `[0, 0, -9.81, 0, 0, 0]` for gravity.

### Joints and frames
`af.Joint(members, nodes)` rigidly connects one node of each member (negative indices
count from the end of the beam). `af.Frame(beams, joints, acc)` assembles everything.
A portal frame under gravity and a lateral load:

```python
import numpy as np
import csdl_alpha as csdl
import aframe as af

recorder = csdl.Recorder(inline=True)
recorder.start()

n = 11

def tube_beam(name, start, end, z=False):
    mesh = csdl.Variable(value=np.linspace(start, end, n))
    cs = af.CSTube(radius=csdl.Variable(value=np.full(n - 1, 0.15)),
                   thickness=csdl.Variable(value=np.full(n - 1, 0.01)))
    return af.Beam(name, mesh, E=69e9, G=26e9, density=2700, cs=cs, z=z)

left = tube_beam('left', [0, 0, 0], [0, 0, 5], z=True)    # vertical members need z=True
right = tube_beam('right', [8, 0, 0], [8, 0, 5], z=True)
top = tube_beam('top', [0, 0, 5], [8, 0, 5])
left.fix(0)
right.fix(0)

loads = np.zeros((n, 6))
loads[n // 2, 1] = 5000  # 5 kN lateral load at midspan
top.add_load(csdl.Variable(value=loads))

joints = [af.Joint(members=[left, top], nodes=[-1, 0]),    # rigid corners
          af.Joint(members=[right, top], nodes=[-1, -1])]

gravity = csdl.Variable(value=np.array([0, 0, -9.81, 0, 0, 0]))
frame = af.Frame(beams=[left, right, top], joints=joints, acc=gravity)
frame.solve()

stress = frame.compute_stress()
cg, mass = frame.compute_mass_properties()

recorder.stop()

print('midspan displacement (m):', frame.displacement['top'].value[n // 2])
print('max stress (Pa):', max(s.value.max() for s in stress.values()))
print('mass (kg):', mass.value, 'cg (m):', cg.value)
```

A beam can belong to several frames, e.g. for different joint layouts.

### Results

| Result | Description |
|---|---|
| `frame.displacement[name]`, `frame.rotation[name]` | Nodal displacements and rotations of a beam, shape `(num_nodes, 3)` (after `solve()`) |
| `frame.compute_stress(beams=None)` | Dict of per-beam stresses (see the cross-section table); pass `beams` to skip beams without a stress model |
| `frame.compute_mass_properties()` | Center of gravity `(3,)` and total mass of the beams (point masses excluded) |
| `beam.mass`, `beam.cg` | Mass and center of gravity of a beam |
| `frame.U`, `frame.K`, `frame.M`, `frame.F` | Global displacement vector, stiffness and mass matrices, and load vector |
| `frame.node_dofs(beam)` | Global dof indices of a beam's nodes, shape `(num_nodes, 6)`, e.g. for coupling to other solvers |

All results are CSDL variables: read their `.value`, differentiate them with
`csdl.derivative`, or use them as objectives and constraints.

# Optimization
Set inputs as design variables and outputs as objectives or constraints, then hand the
recorder to a CSDL simulator and an optimizer. A minimum-mass cantilever with stress and
deflection limits, using [modOpt](https://github.com/LSDOlab/modopt) and the JAX backend:

```python
import numpy as np
import csdl_alpha as csdl
import aframe as af
from modopt import CSDLAlphaProblem, SLSQP

recorder = csdl.Recorder(inline=True)
recorder.start()

num_nodes = 21
mesh = np.zeros((num_nodes, 3))
mesh[:, 0] = np.linspace(0, 10, num_nodes)

radius = csdl.Variable(value=np.full(num_nodes - 1, 0.1), name='radius')
radius.set_as_design_variable(lower=0.02, upper=0.5, scaler=10)
thickness = csdl.Variable(value=np.full(num_nodes - 1, 0.005), name='thickness')
thickness.set_as_design_variable(lower=0.001, upper=0.02, scaler=100)

beam = af.Beam(name='cantilever', mesh=csdl.Variable(value=mesh), E=69e9, G=26e9,
               density=2700, cs=af.CSTube(radius=radius, thickness=thickness))
beam.fix(0)
loads = np.zeros((num_nodes, 6))
loads[-1, 2] = -1000
beam.add_load(csdl.Variable(value=loads))

frame = af.Frame(beams=[beam])
frame.solve()

stress = frame.compute_stress()['cantilever']
stress.set_as_constraint(upper=100e6, scaler=1e-8)             # 100 MPa allowable
tip_deflection = frame.displacement['cantilever'][-1, 2]
tip_deflection.set_as_constraint(lower=-0.2, scaler=10)          # at most 0.2 m
beam.mass.set_as_objective(scaler=1e-2)

recorder.stop()

sim = csdl.experimental.JaxSimulator(recorder=recorder)
problem = CSDLAlphaProblem(problem_name='cantilever', simulator=sim)
optimizer = SLSQP(problem, solver_options={'maxiter': 200, 'ftol': 1e-8}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()
```

`csdl.experimental.PySimulator(recorder)` runs the same model without JAX. See
[`examples/single_beam_opt.py`](examples/single_beam_opt.py) for a complete script.

# Plotting
With the `plot` extra, the PyVista helpers draw beams into a `pyvista.Plotter`. For
example, the deformed cantilever from the quick start, colored by stress:

```python
import pyvista as pv

deformed = mesh + frame.displacement['cantilever'].value
plotter = pv.Plotter()
af.plot_mesh(plotter, deformed, cell_data=stress.value, line_width=10)
af.plot_points(plotter, deformed, point_size=15, color='black')
plotter.show()
```

`af.plot_cyl` and `af.plot_box` draw each element as a cylinder or box, and
`aframe.utils.plot_matplotlib` has Matplotlib equivalents.

# Modeling assumptions
- Linear statics: small displacements and rotations, linear elastic material.
- Euler-Bernoulli elements (no shear deformation); cross-section properties are constant within an element.
- Joints are rigid and share all 6 dofs.
- Stresses are evaluated at element midpoints from the element internal loads.
- The global matrices are dense and solved with a dense direct solver, so time and memory grow
  quickly with model size; models of up to a few thousand dofs (several hundred nodes) solve in
  seconds.

# Examples
The [`examples`](examples) folder has complete scripts:

| Script | What it shows |
|---|---|
| [`two_joined_beams.py`](examples/two_joined_beams.py) | Two tube beams connected by a joint |
| [`bingran_cantilever_beam.py`](examples/bingran_cantilever_beam.py) | A tube cantilever with stresses and mass properties |
| [`single_beam_opt.py`](examples/single_beam_opt.py) | Minimum-mass optimization with a displacement constraint (modOpt) |
| [`NASA_LPC_beam.py`](examples/NASA_LPC_beam.py), [`aurora_pav_vv.py`](examples/aurora_pav_vv.py) | Wing-box (`CSBox`) beams under spanwise load distributions |
| [`1_element_beam_test.py`](examples/1_element_beam_test.py) | Frames of single-element beams |
| [`ucsd_lunar_lander/`](examples/ucsd_lunar_lander) | A lunar lander frame with many joints, plotted with PyVista |
| [`bunny/`](examples/bunny) | A frame generated from the edges of an STL mesh |

# License
This project is licensed under the terms of the **MIT License**.
