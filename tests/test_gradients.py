"""Reverse-mode derivatives against central finite differences."""
import numpy as np
import csdl_alpha as csdl
import models


def test_gradients_match_finite_differences(recorder):
    frame, dvs = models.all_sections(n=5)
    obj, _ = models.objective(frame)
    derivatives = csdl.derivative(obj, dvs)
    # snapshot before the perturbations below re-execute the graph
    analytic_all = [derivatives[dv].value.ravel().copy() for dv in dvs]
    rng = np.random.default_rng(0)

    for dv, analytic in zip(dvs, analytic_all):
        base = dv.value.copy()

        for k in rng.choice(base.size, size=min(3, base.size), replace=False):
            h = 1e-6 * max(1.0, abs(base.flat[k]))
            values = []
            for sign in (1, -1):
                perturbed = base.copy()
                perturbed.flat[k] += sign * h
                dv.value = perturbed
                recorder.execute()
                values.append(obj.value[0])
            dv.value = base
            recorder.execute()

            fd = (values[0] - values[1]) / (2 * h)
            tol = 1e-5 * abs(analytic[k]) + 1e-7 * np.max(np.abs(analytic))
            assert abs(fd - analytic[k]) <= tol, f'{dv.name}[{k}]: finite difference {fd}, derivative {analytic[k]}'

    recorder.execute()
