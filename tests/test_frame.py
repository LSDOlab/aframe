"""Frame assembly, dof numbering, validation and the cross-section contract."""
import numpy as np
import pytest
import csdl_alpha as csdl
import aframe as af
from aframe.core.frame import _scatter_add
from models import E, G, RHO, var, line

N = 5


def tube(n=N, r=0.1, t=0.01):
    return af.CSTube(var(np.full(n - 1, r)), var(np.full(n - 1, t)))


def beam(name, p0, p1, n=N, load=None, cs=None):
    b = af.Beam(name, var(line(p0, p1, n)), E, G, RHO, tube(n) if cs is None else cs)
    if load is not None:
        loads = np.zeros((n, 6))
        loads[:, 2] = load
        b.add_load(var(loads))
    return b


def star():
    """three beams meeting at (4, 0, 0); b2's first node is merged by one joint
    and is the first member of the other"""
    b1 = beam('b1', [0, 0, 0], [4, 0, 0], load=10.0)
    b2 = beam('b2', [4, 0, 0], [4, 3, 0], load=20.0)
    b3 = beam('b3', [4, 0, 0], [4, 0, 3], load=30.0)
    b1.fix(0)
    return b1, b2, b3


def displacements(frame):
    return {name: value.value.copy() for name, value in frame.displacement.items()}


def test_joint_order_does_not_matter(recorder):
    results = []
    for order in (0, 1):
        b1, b2, b3 = star()
        joints = [af.Joint([b1, b2], [-1, 0]), af.Joint([b2, b3], [0, 0])]
        frame = af.Frame([b1, b2, b3], joints=joints[::-1] if order else joints)
        frame.solve()
        assert frame.num == 3 * N - 2
        results.append(displacements(frame))

    for name in results[0]:
        np.testing.assert_array_equal(results[0][name], results[1][name])
    # the three beams share the joint node
    joint = results[0]['b1'][-1]
    np.testing.assert_array_equal(results[0]['b2'][0], joint)
    np.testing.assert_array_equal(results[0]['b3'][0], joint)


def test_negative_joint_nodes(recorder):
    b1, b2, _ = star()
    joint = af.Joint([b1, b2], [-1, 0])
    assert joint.nodes == [N - 1, 0]


def test_beam_reuse_in_several_frames(recorder):
    b1, b2, b3 = star()
    joints = [af.Joint([b1, b2], [-1, 0]), af.Joint([b2, b3], [0, 0])]
    frame_a = af.Frame([b1, b2, b3], joints=joints)
    frame_b = af.Frame([b1])                       # b1 alone, numbered differently
    frame_a.solve()
    frame_b.solve()
    stress_a = frame_a.compute_stress()

    frame_c = af.Frame([b1, b2, b3], joints=joints)  # reference: a fresh copy of frame_a
    frame_c.solve()
    stress_c = frame_c.compute_stress()

    for name, value in displacements(frame_c).items():
        np.testing.assert_array_equal(frame_a.displacement[name].value, value)
        np.testing.assert_array_equal(stress_a[name].value, stress_c[name].value)
    assert not np.allclose(frame_b.displacement['b1'].value, frame_a.displacement['b1'].value)


def test_node_dofs(recorder):
    b1, b2, b3 = star()
    frame = af.Frame([b1, b2, b3], joints=[af.Joint([b1, b2], [-1, 0]), af.Joint([b2, b3], [0, 0])])
    dofs = frame.node_dofs(b2)
    assert dofs.shape == (N, 6)
    np.testing.assert_array_equal(dofs[0], frame.node_dofs(b1)[-1])
    np.testing.assert_array_equal(dofs[0], frame.node_dofs(b3)[0])
    all_dofs = np.concatenate([frame.node_dofs(b).ravel() for b in (b1, b2, b3)])
    np.testing.assert_array_equal(np.unique(all_dofs), np.arange(frame.dim))


@pytest.mark.parametrize('make', [
    lambda b1, b2, b3: af.Frame([]),
    lambda b1, b2, b3: af.Frame([b1, b1]),
    lambda b1, b2, b3: af.Frame([b1, beam('b1', [0, 0, 0], [0, 1, 0])]),
    lambda b1, b2, b3: af.Frame([b1, b2], joints=[af.Joint([b1, b3], [-1, 0])]),
    lambda b1, b2, b3: af.Frame([b1], acc=var(np.zeros(3))),
    lambda b1, b2, b3: af.Joint([b1, b2], [0]),
    lambda b1, b2, b3: af.Joint([b1, b2], [N, 0]),
    lambda b1, b2, b3: af.Joint([b1, b2], [-N - 1, 0]),
    lambda b1, b2, b3: af.Beam('b', var(line([0, 0, 0], [1, 0, 0], N)), E, G, RHO, tube(N + 1)),
    lambda b1, b2, b3: b1.add_load(var(np.zeros((N, 3)))),
    lambda b1, b2, b3: b1.add_inertial_mass(var(np.zeros(N - 1))),
    lambda b1, b2, b3: b1.fix(N),
    lambda b1, b2, b3: b1.pin(-1),
    lambda b1, b2, b3: af.CSTube(var(np.ones(3)), var(np.ones(4))),
    lambda b1, b2, b3: af.CSBox(*[var(np.ones(3))] * 4, var(np.ones(4))),
    lambda b1, b2, b3: af.CrossSection(var(np.ones(3)), var(np.ones(3)), var(np.ones(3)), var(np.ones(4))),
], ids=['no_beams', 'duplicate_beam', 'duplicate_name', 'joint_member_not_in_frame', 'acc_shape',
        'joint_lengths', 'joint_node_high', 'joint_node_low', 'cs_shape', 'load_shape',
        'inertial_mass_shape', 'fix_range', 'pin_range', 'tube_shapes', 'box_shapes', 'property_shapes'])
def test_invalid_input(recorder, make):
    with pytest.raises(ValueError):
        make(*star())


def test_compute_stress_requires_solve(recorder):
    b1, _, _ = star()
    with pytest.raises(RuntimeError):
        af.Frame([b1]).compute_stress()


def test_sections_without_stress_model(recorder):
    b1, _, _ = star()
    circle = beam('circle', [4, 0, 0], [8, 0, 0], cs=af.CSCircle(var(np.full(N - 1, 0.05))))
    frame = af.Frame([b1, circle], joints=[af.Joint([b1, circle], [-1, 0])])
    frame.solve()
    with pytest.raises(NotImplementedError):
        frame.compute_stress()
    assert list(frame.compute_stress(beams=[b1])) == ['b1']


def test_generic_cross_section_matches_tube(recorder):
    """a CrossSection with the tube's properties gives the same structural response"""
    cs = tube()
    b_tube = beam('tube', [0, 0, 0], [3, 1, 0], load=100.0, cs=cs)
    generic = af.CrossSection(area=cs.area, ix=cs.ix, iy=cs.iy, iz=cs.iz)
    b_generic = beam('generic', [0, 0, 0], [3, 1, 0], load=100.0, cs=generic)
    assert generic.num_elements == N - 1

    results = []
    for b in (b_tube, b_generic):
        b.fix(0)
        frame = af.Frame([b])
        frame.solve()
        results.append(frame.displacement[b.name].value)
    np.testing.assert_array_equal(results[0], results[1])


def test_box_from_properties(recorder):
    dims = [var(np.full(N - 1, v)) for v in (0.004, 0.003, 0.002, 0.2, 0.15)]
    box = af.CSBox(*dims)
    given = af.CSBox.from_properties(*dims, box.area, box.ix, box.iy, box.iz)
    legacy = af.CSBoxMarius(*dims, box.area, box.ix, box.iy, box.iz)
    assert isinstance(given, af.CSBox) and isinstance(legacy, af.CSBox)

    loads = var(np.random.default_rng(0).standard_normal((N - 1, 12)) * 1e3)
    expected = box.stress(loads).value
    np.testing.assert_array_equal(given.stress(loads).value, expected)
    np.testing.assert_array_equal(legacy.stress(loads).value, expected)


def test_scatter_add(recorder):
    """entries on a position are summed (like np.add.at), -1 positions dropped,
    untouched positions keep init"""
    rng = np.random.default_rng(1)
    init = rng.standard_normal((4, 5))
    values = [rng.standard_normal((6, 2)), rng.standard_normal(7)]
    positions = rng.integers(-1, init.size, size=19)

    flat = np.concatenate([v.ravel() for v in values])
    keep = positions >= 0
    sums = np.zeros(init.size)
    np.add.at(sums, positions[keep], flat[keep])
    expected = init.ravel().copy()
    touched = np.unique(positions[keep])
    expected[touched] = sums[touched]

    out = _scatter_add(init, positions, [var(v) for v in values])
    np.testing.assert_allclose(out.value, expected.reshape(init.shape), rtol=1e-15)
