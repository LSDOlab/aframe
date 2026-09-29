"""
Test models shared by the regression tests and generate_reference.py.

Each builder must be called inside an active csdl Recorder and returns
(frame, design_variables).
"""
import numpy as np
import csdl_alpha as csdl
import aframe as af

E, G, RHO = 69e9, 26e9, 2700.0


def var(value, name=None):
    return csdl.Variable(value=np.asarray(value, dtype=float), name=name)


def line(p0, p1, n):
    return np.linspace(np.asarray(p0, dtype=float), np.asarray(p1, dtype=float), n)


def joined(n=11):
    """examples/two_joined_beams.py: two tube beams joined at their tips"""
    beams, dvs = [], []
    for name, load in (('beam_1', 2000), ('beam_2', 1000)):
        loads = np.zeros((n, 6))
        loads[:, 2] = load
        r = var(np.full(n - 1, 0.5), f'{name}_r')
        t = var(np.full(n - 1, 0.001), f'{name}_t')
        beam = af.Beam(name, var(line([0, 0, 0], [20, 0, 0], n)), E, G, RHO, af.CSTube(r, t))
        beam.add_load(var(loads))
        beams.append(beam)
        dvs += [r, t]
    beams[0].fix(0)
    return af.Frame(beams, joints=[af.Joint(beams, [n - 1, n - 1])]), dvs


def tower(n=6):
    """
    3 inclined tube legs (2 fixed, 1 pinned) meeting a z-aligned tube mast in a
    4-member joint, a box-section ring joining the leg mid-points (3-member
    joints), nodal loads, gravity and extra point masses on the mast
    """
    rng = np.random.default_rng(0)
    apex = np.array([0.3, -0.2, 10.0])
    bases = [np.array([5 * np.cos(a), 5 * np.sin(a), 0.0]) for a in np.deg2rad([10, 130, 250])]
    mid = (n - 1) // 2
    dvs, legs, ring = [], [], []

    for i, base in enumerate(bases):
        r = var(0.15 + 0.02 * rng.random(n - 1), f'leg{i}_r')
        t = var(0.01 + 0.002 * rng.random(n - 1), f'leg{i}_t')
        legs.append(af.Beam(f'leg{i}', var(line(base, apex, n), f'leg{i}_mesh'), E, G, RHO, af.CSTube(r, t)))
        dvs += [r, t]
    legs[0].fix(0)
    legs[1].fix(0)
    legs[2].pin(0)
    dvs.append(legs[0].mesh)

    for i in range(3):
        p0, p1 = legs[i].mesh.value[mid], legs[(i + 1) % 3].mesh.value[mid]
        box = {k: var(v * (1 + 0.1 * rng.random(n - 1)), f'ring{i}_{k}')
               for k, v in dict(ttop=0.004, tbot=0.004, tweb=0.003, height=0.2, width=0.15).items()}
        beam = af.Beam(f'ring{i}', var(line(p0, p1, n)), E, G, RHO, af.CSBox(**box))
        loads = np.zeros((n, 6))
        loads[:, 1] = 300.0
        loads[:, 2] = -800.0
        beam.add_load(var(loads))
        ring.append(beam)
        dvs += list(box.values())

    r, t = var(np.full(n - 1, 0.12), 'mast_r'), var(np.full(n - 1, 0.008), 'mast_t')
    mast = af.Beam('mast', var(line(apex, apex + [0, 0, 8.0], n)), E, G, RHO, af.CSTube(r, t), z=True)
    loads = np.zeros((n, 6))
    loads[:, 0] = 500.0
    loads[:, 1] = -250.0
    loads[:, 5] = 40.0
    mast.add_load(var(loads))
    mast.add_inertial_mass(var(np.full(n, 3.0), 'mast_extra_mass'))
    dvs += [r, t]

    joints = [af.Joint(legs + [mast], [n - 1, n - 1, n - 1, 0])]
    joints += [af.Joint([legs[i], ring[i], ring[i - 1]], [mid, 0, n - 1]) for i in range(3)]
    acc = var([0.5, 0, -9.81, 0, 0, 0], 'acc')
    return af.Frame(legs + ring + [mast], joints=joints, acc=acc), dvs


def all_sections(n=9):
    """every cross-section type in one 3D frame, with joints, gravity and extra masses"""
    rng = np.random.default_rng(3)
    circ = af.CSCircle(var(0.1 + 0.05 * rng.random(n - 1), 'circ_r'))
    ell = af.CSEllipse(var(0.2 + 0.05 * rng.random(n - 1), 'ell_a'), var(0.1 + 0.02 * rng.random(n - 1), 'ell_b'))
    tube = af.CSTube(var(0.2 + 0.05 * rng.random(n - 1), 'tube_r'), var(0.01 + 0.002 * rng.random(n - 1), 'tube_t'))
    box = af.CSBox(var(np.full(n - 1, .004)), var(np.full(n - 1, .003)), var(np.full(n - 1, .002)),
                   var(np.full(n - 1, .2)), var(np.linspace(.3, .15, n - 1), 'box_width'))
    given = af.CSBox.from_properties(box.ttop, box.tbot, box.tweb, box.height, box.width,
                                     box.area, box.ix, box.iy, box.iz)

    b_circ = af.Beam('circ', var(line([0, 0, 0], [4, 1, 0.5], n)), E, G, RHO, circ)
    b_ell = af.Beam('ell', var(line([4, 1, 0.5], [8, -1, 1.0], n)), E, G, RHO, ell)
    b_tube = af.Beam('tube', var(line([8, -1, 1.0], [8.5, 3, 2], n)), E, G, RHO, tube)
    b_box = af.Beam('box', var(line([4, 1, 0.5], [3, 5, 1.5], n), 'box_mesh'), E, G, RHO, box)
    b_given = af.Beam('given', var(line([3, 5, 1.5], [8.5, 3, 2], n)), E, G, RHO, given)
    beams = [b_circ, b_ell, b_tube, b_box, b_given]

    b_circ.fix(0)
    b_tube.pin(n - 1)
    b_given.fix(n - 1)
    for beam in beams:
        loads = np.zeros((n, 6))
        loads[:, 2] = -500 * rng.random(n)
        loads[:, 3] = 20
        beam.add_load(var(loads))
    b_ell.add_inertial_mass(var(np.full(n, 2.0)))
    b_box.add_inertial_mass(var(np.full(n, 1.5)))

    joints = [af.Joint([b_circ, b_ell, b_box], [-1, 0, 0]),
              af.Joint([b_ell, b_tube], [-1, 0]),
              af.Joint([b_box, b_given], [-1, 0]),
              af.Joint([b_tube, b_given], [-1, -1])]
    frame = af.Frame(beams, joints=joints, acc=var([0, 0, -9.81, 0.1, 0, 0], 'acc'))
    return frame, [circ.radius, ell.semi_major_axis, tube.thickness, box.width, b_box.mesh, frame.acc]


MODELS = {'joined': joined, 'tower': tower, 'all_sections': all_sections}


def has_stress_model(beam):
    return type(beam.cs).stress is not af.CrossSection.stress


def objective(frame):
    """solve the frame; compliance plus scaled stresses, and the stresses"""
    frame.solve()
    stress = frame.compute_stress(beams=[b for b in frame.beams if has_stress_model(b)])
    obj = csdl.vdot(frame.F, frame.U)
    for s in stress.values():
        obj = obj + 1e-6 * csdl.sum(s)
    return obj, stress


def results(name):
    """build, solve and differentiate a model (inside an active inline recorder)"""
    frame, dvs = MODELS[name]()
    obj, stress = objective(frame)
    derivatives = csdl.derivative(obj, dvs)

    out = {'K': frame.K.value, 'M': frame.M.value, 'F': frame.F.value, 'U': frame.U.value, 'obj': obj.value}
    for beam in frame.beams:
        out[f'displacement_{beam.name}'] = frame.displacement[beam.name].value
        out[f'rotation_{beam.name}'] = frame.rotation[beam.name].value
        if beam.name in stress:
            out[f'stress_{beam.name}'] = stress[beam.name].value
    for i, dv in enumerate(dvs):
        out[f'd_{i}'] = derivatives[dv].value
    return out
