"""The JAX backend reproduces the inline (NumPy) results."""
import numpy as np
import pytest
import csdl_alpha as csdl
import models

pytest.importorskip('jax')


def test_jax_matches_inline():
    rec = csdl.Recorder(inline=True)
    rec.start()
    frame, dvs = models.tower(n=4)
    obj, _ = models.objective(frame)
    derivatives = csdl.derivative(obj, dvs)
    rec.stop()

    inline_U, inline_obj = frame.U.value.copy(), obj.value.copy()
    inline_d = [derivatives[dv].value.copy() for dv in dvs]

    sim = csdl.experimental.JaxSimulator(rec, additional_inputs=dvs, additional_outputs=[obj, frame.U], gpu=False)
    sim.run()
    np.testing.assert_allclose(sim[frame.U], inline_U, rtol=1e-9, atol=1e-9 * np.max(np.abs(inline_U)))
    np.testing.assert_allclose(sim[obj], inline_obj, rtol=1e-9)

    totals = sim.compute_totals()
    for dv, expected in zip(dvs, inline_d):
        actual = np.asarray(totals[obj, dv]).reshape(expected.shape)
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7 * np.max(np.abs(expected)))
