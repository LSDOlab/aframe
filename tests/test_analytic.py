"""Checks against closed-form Euler-Bernoulli cantilever solutions (exact at the nodes)."""
import numpy as np
import pytest
import aframe as af
from models import E, G, RHO, var

L, N_EL = 10.0, 10
R, T = 0.1, 0.01
A = np.pi * (R**2 - (R - T)**2)
I = np.pi * (R**4 - (R - T)**4) / 4
J = 2 * I
P = 1000.0
X_MID = (np.arange(N_EL) + 0.5) * L / N_EL


def cantilever(direction=(1, 0, 0), tip_load=None, z=False, acc=None, tip_mass=None, cs=None):
    """tube cantilever of length L, fixed at node 0, along direction"""
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    mesh = np.outer(np.linspace(0, L, N_EL + 1), d)
    if cs is None:
        cs = af.CSTube(var(np.full(N_EL, R)), var(np.full(N_EL, T)))

    beam = af.Beam('cantilever', var(mesh), E, G, RHO, cs, z=z)
    beam.fix(0)
    if tip_load is not None:
        loads = np.zeros((N_EL + 1, 6))
        loads[-1] = tip_load
        beam.add_load(var(loads))
    if tip_mass is not None:
        mass = np.zeros(N_EL + 1)
        mass[-1] = tip_mass
        beam.add_inertial_mass(var(mass))

    frame = af.Frame([beam], acc=None if acc is None else var(acc))
    frame.solve()
    return frame, beam


def tip(frame, beam):
    return frame.displacement[beam.name].value[-1], frame.rotation[beam.name].value[-1]


def assert_close(actual, expected, rtol=1e-8):
    expected = np.asarray(expected, dtype=float)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=rtol * np.max(np.abs(expected)))


@pytest.mark.parametrize('load, displacement, rotation', [
    ((P, 0, 0, 0, 0, 0), (P * L / (E * A), 0, 0), (0, 0, 0)),
    ((0, P, 0, 0, 0, 0), (0, P * L**3 / (3 * E * I), 0), (0, 0, P * L**2 / (2 * E * I))),
    ((0, 0, P, 0, 0, 0), (0, 0, P * L**3 / (3 * E * I)), (0, -P * L**2 / (2 * E * I), 0)),
    ((0, 0, 0, P, 0, 0), (0, 0, 0), (P * L / (G * J), 0, 0)),
], ids=['axial', 'bending_y', 'bending_z', 'torsion'])
def test_tip_loads(recorder, load, displacement, rotation):
    frame, beam = cantilever(tip_load=load)
    u, theta = tip(frame, beam)
    scale = max(np.max(np.abs(displacement)), np.max(np.abs(rotation)))
    np.testing.assert_allclose(u, displacement, rtol=1e-8, atol=1e-8 * scale)
    np.testing.assert_allclose(theta, rotation, rtol=1e-8, atol=1e-8 * scale)


def test_inclined_beam(recorder):
    """a tube is axisymmetric, so an inclined cantilever bends along the load"""
    d = np.array([1.0, 2.0, 0.5]) / np.linalg.norm([1.0, 2.0, 0.5])
    perpendicular = np.cross(d, [0, 0, 1])
    perpendicular /= np.linalg.norm(perpendicular)

    frame, beam = cantilever(direction=d, tip_load=(*(P * perpendicular), 0, 0, 0))
    assert_close(tip(frame, beam)[0], P * perpendicular * L**3 / (3 * E * I))


def test_inclined_beam_axial(recorder):
    d = np.array([1.0, 2.0, 0.5]) / np.linalg.norm([1.0, 2.0, 0.5])
    frame, beam = cantilever(direction=d, tip_load=(*(P * d), 0, 0, 0))
    assert_close(tip(frame, beam)[0], d * P * L / (E * A))


def test_vertical_beam(recorder):
    frame, beam = cantilever(direction=(0, 0, 1), z=True, tip_load=(P, 0, 0, 0, 0, 0))
    assert_close(tip(frame, beam)[0], (P * L**3 / (3 * E * I), 0, 0))


def test_gravity_and_tip_mass(recorder):
    """consistent mass: M @ acc is the exact load vector of the uniform self-weight"""
    g, tip_mass = 9.81, 50.0
    frame, beam = cantilever(acc=(0, 0, -g, 0, 0, 0), tip_mass=tip_mass)
    w = RHO * A * g
    expected = -w * L**4 / (8 * E * I) - tip_mass * g * L**3 / (3 * E * I)
    assert_close(tip(frame, beam)[0], (0, 0, expected))


def test_mass_properties(recorder):
    frame, beam = cantilever(direction=(1, 1, 0))
    cg, mass = frame.compute_mass_properties()
    assert_close(mass.value, [RHO * A * L])
    assert_close(beam.mass.value, [RHO * A * L])
    assert_close(cg.value, np.array([1, 1, 0]) / np.sqrt(2) * L / 2)


@pytest.mark.parametrize('axial', [5000.0, -5000.0], ids=['tension', 'compression'])
def test_tube_stress_bending_and_axial(recorder, axial):
    frame, beam = cantilever(tip_load=(axial, 0, P, 0, 0, 0))
    stress = frame.compute_stress()[beam.name].value
    assert_close(stress, abs(axial) / A + P * (L - X_MID) * R / I)


def test_tube_stress_torsion(recorder):
    frame, beam = cantilever(tip_load=(0, 0, 0, P, 0, 0))
    stress = frame.compute_stress()[beam.name].value
    assert_close(stress, np.full(N_EL, np.sqrt(3) * P * R / J))


TTOP, TWEB, HEIGHT, WIDTH = 0.004, 0.003, 0.2, 0.15


def box_cantilever():
    """symmetric box cantilever with a tip load along z"""
    cs = af.CSBox(ttop=var(np.full(N_EL, TTOP)), tbot=var(np.full(N_EL, TTOP)), tweb=var(np.full(N_EL, TWEB)),
                  height=var(np.full(N_EL, HEIGHT)), width=var(np.full(N_EL, WIDTH)))
    frame, beam = cantilever(tip_load=(0, 0, P, 0, 0, 0), cs=cs)
    return frame.compute_stress()[beam.name].value, cs.iy.value


def test_box_stress_corners(recorder):
    stress, iy = box_cantilever()
    assert stress.shape == (N_EL, 5)
    for point in range(4):
        assert_close(stress[:, point], P * (L - X_MID) * (HEIGHT / 2) / iy)


def test_box_stress_web_shear(recorder):
    stress, iy = box_cantilever()
    Q = WIDTH * TTOP * (HEIGHT / 2) + 2 * (HEIGHT / 2) * TWEB * (HEIGHT / 4)
    assert_close(stress[:, 4], np.sqrt(3) * P * Q / (iy * 2 * TWEB))
