"""
Regenerate the golden values in tests/data/reference.npz used by test_regression.py.

Only run this after an intended change of results, and review the change:
    python tests/generate_reference.py
"""
import os
import numpy as np
import csdl_alpha as csdl
from models import MODELS, results

here = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    data = {}
    for name in MODELS:
        rec = csdl.Recorder(inline=True)
        rec.start()
        for key, value in results(name).items():
            data[f'{name}/{key}'] = value
        rec.stop()

    os.makedirs(os.path.join(here, 'data'), exist_ok=True)
    path = os.path.join(here, 'data', 'reference.npz')
    np.savez_compressed(path, **data)
    print(f'saved {len(data)} arrays to {path}')
