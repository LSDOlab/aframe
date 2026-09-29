"""
aframe: a differentiable linear 3D beam/frame solver written in CSDL.

.. include:: ../README.md
   :start-line: 1
"""
from aframe.core.cs import CrossSection, CSTube, CSCircle, CSEllipse, CSBox, CSBoxMarius
from aframe.core.beam import Beam
from aframe.core.joint import Joint
from aframe.core.frame import Frame
from aframe.utils.meshing import mesh_from_points_and_edges

__version__ = '0.1.0'

# the PyVista plotting helpers need the optional 'plot' dependencies and are
# slow to import, so they are loaded on first use
_PLOTTING = ('plot_box', 'plot_cyl', 'plot_mesh', 'plot_points')

__all__ = ['CrossSection', 'CSTube', 'CSCircle', 'CSEllipse', 'CSBox', 'CSBoxMarius',
           'Beam', 'Joint', 'Frame', 'mesh_from_points_and_edges', *_PLOTTING]


def __getattr__(name):
    if name in _PLOTTING:
        try:
            from aframe.utils import plot_pyvista
        except ImportError as e:
            raise ImportError(f"aframe.{name} needs pyvista and matplotlib: pip install aframe[plot]") from e
        return getattr(plot_pyvista, name)

    raise AttributeError(f"module 'aframe' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
