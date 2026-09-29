"""
Golden-value regression tests: K, M, F, U, displacements, rotations, stresses,
objective and gradients of the test models against tests/data/reference.npz.

After an intended change of results, regenerate the reference with
``python tests/generate_reference.py`` and review the differences.
"""
import os
import numpy as np
import pytest
from models import MODELS, results

REFERENCE = np.load(os.path.join(os.path.dirname(__file__), 'data', 'reference.npz'))


@pytest.mark.parametrize('name', list(MODELS))
def test_matches_reference(recorder, name):
    out = results(name)
    keys = sorted(key.split('/', 1)[1] for key in REFERENCE.files if key.startswith(name + '/'))
    assert sorted(out) == keys

    for key in keys:
        reference = REFERENCE[f'{name}/{key}']
        # gradients go through the adjoint solve and are more sensitive to rounding
        rtol = 1e-6 if key.startswith('d_') else 1e-9
        np.testing.assert_allclose(out[key], reference, rtol=rtol, atol=rtol * np.max(np.abs(reference)),
                                   err_msg=f'{name}: {key}')
