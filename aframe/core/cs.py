"""
Beam cross-sections.

Every cross-section is a ``CrossSection``: it provides per-element section
properties as csdl Variables of shape (num_elements,)

    area : cross-sectional area
    ix   : torsion constant (polar moment for circular sections)
    iy   : second moment of area about the local y axis
    iz   : second moment of area about the local z axis

and, if it has a stress model, ``stress(element_loads)``, which maps the
(num_elements, 12) local element end loads to a stress measure.
"""
from typing import Tuple
import numpy as np
import csdl_alpha as csdl

__all__ = ['CrossSection', 'CSTube', 'CSCircle', 'CSEllipse', 'CSBox', 'CSBoxMarius']


class CrossSection:
    """
    Base class of all cross-sections, also usable directly for a section with
    known properties (it has no stress model).

    Subclasses compute their properties and pass them to ``__init__``; those
    with a stress model override ``stress``.

    Parameters
    ----------
    area, ix, iy, iz : csdl.Variable
        Section properties, shape (num_elements,).
    """

    def __init__(self, area:csdl.Variable, ix:csdl.Variable, iy:csdl.Variable, iz:csdl.Variable):

        if not area.shape == ix.shape == iy.shape == iz.shape:
            raise ValueError('section properties area, ix, iy and iz must have the same shape')

        self.area = area
        self.ix = ix
        self.iy = iy
        self.iz = iz


    @property
    def num_elements(self)->int:
        return self.area.shape[0]


    @staticmethod
    def internal_loads(element_loads:csdl.Variable)->Tuple[csdl.Variable, ...]:
        """
        internal loads at the element midspan from the local element end loads
        [F_a, M_a, F_b, M_b], shape (num_elements, 12)

        returns (N, V_y, V_z, T, M_y, M_z), each shape (num_elements,): axial
        force (tension positive), shear forces, torque and bending moments.
        The end loads act on the element, so the internal load is (end_b - end_a) / 2
        (the forces and torque are constant along an element, the moments linear).
        """
        return tuple((element_loads[:, 6 + i] - element_loads[:, i]) / 2 for i in range(6))


    def stress(self, element_loads:csdl.Variable)->csdl.Variable:
        raise NotImplementedError(f'{type(self).__name__} has no stress model')


class CSTube(CrossSection):
    """
    Hollow circular tube.

    Parameters
    ----------
    radius : csdl.Variable
        Outer radius, shape (num_elements,).
    thickness : csdl.Variable
        Wall thickness, shape (num_elements,).
    """

    def __init__(self, radius:csdl.Variable, thickness:csdl.Variable):

        if radius.shape != thickness.shape:
            raise ValueError('tube radius and thickness must have the same shape')

        self.radius = radius
        self.thickness = thickness
        self.inner_radius = radius - thickness
        self.outer_radius = radius
        self.precomp = np.pi * (self.outer_radius**4 - self.inner_radius**4)

        super().__init__(area=np.pi * (self.outer_radius**2 - self.inner_radius**2),
                         ix=self.precomp / 2,
                         iy=self.precomp / 4,
                         iz=self.precomp / 4)


    def stress(self, element_loads:csdl.Variable)->csdl.Variable:
        """
        von Mises stress at the most stressed point of the outer fiber,
        shape (num_elements,): |N| / A + M r / I combined with the torsional shear
        """
        N, _, _, T, M_y, M_z = self.internal_loads(element_loads)

        # eps keeps the absolute values/square roots differentiable at zero load
        eps = 1E-12
        axial_stress = (N**2 + eps) ** 0.5 / self.area
        shear_stress = T * self.radius / self.ix

        max_moment = (M_y**2 + M_z**2 + eps) ** 0.5
        bending_stress = max_moment * self.radius / self.iy

        normal_stress = axial_stress + bending_stress

        return (normal_stress**2 + 3*shear_stress**2 + eps) ** 0.5


class CSCircle(CrossSection):
    """
    Solid circular section (no stress model).

    Parameters
    ----------
    radius : csdl.Variable
        Radius, shape (num_elements,).
    """

    def __init__(self, radius:csdl.Variable):

        self.radius = radius
        self.precomp = np.pi * self.radius**4

        super().__init__(area=np.pi * self.radius**2,
                         ix=(1 / 2) * self.precomp,
                         iy=(1 / 4) * self.precomp,
                         iz=(1 / 4) * self.precomp)


class CSEllipse(CrossSection):
    """
    Solid elliptical section (no stress model).

    Parameters
    ----------
    semi_major_axis : csdl.Variable
        Semi-axis along the local z direction, shape (num_elements,).
    semi_minor_axis : csdl.Variable
        Semi-axis along the local y direction, shape (num_elements,).
    """

    def __init__(self, semi_major_axis:csdl.Variable, semi_minor_axis:csdl.Variable):

        a = self.semi_major_axis = semi_major_axis
        b = self.semi_minor_axis = semi_minor_axis

        area = np.pi * a * b
        # approximate torsion constant
        beta = 1 / ((1 + (b / a)**2)**0.5)
        ix = (np.pi / 2) * a * b**3 * beta
        iy = np.pi / 4 * a * b**3
        iz = np.pi / 4 * a**3 * b

        super().__init__(area=area, ix=ix, iy=iy, iz=iz)


class CSBox(CrossSection):
    """
    Thin-walled rectangular box (e.g. a wing box), with separate top/bottom
    skin and web thicknesses. Use ``CSBox.from_properties`` to supply the
    section properties instead of the thin-walled formulas.

    Parameters
    ----------
    ttop, tbot : csdl.Variable
        Top and bottom skin thicknesses, shape (num_elements,).
    tweb : csdl.Variable
        Web (side wall) thickness, shape (num_elements,).
    height, width : csdl.Variable
        Outer height (local z) and width (local y), shape (num_elements,).
    """

    def __init__(self,
                 ttop:csdl.Variable,
                 tbot:csdl.Variable,
                 tweb:csdl.Variable,
                 height:csdl.Variable,
                 width:csdl.Variable):

        self._set_dimensions(ttop, tbot, tweb, height, width)

        self.neutral_axis = self._neutral_axis()
        area = self._area()
        iy = self._iy()
        iz = self._iz()
        # thin-walled approximation
        super().__init__(area=area, ix=iy + iz, iy=iy, iz=iz)


    @classmethod
    def from_properties(cls,
                        ttop:csdl.Variable,
                        tbot:csdl.Variable,
                        tweb:csdl.Variable,
                        height:csdl.Variable,
                        width:csdl.Variable,
                        area:csdl.Variable,
                        ix:csdl.Variable,
                        iy:csdl.Variable,
                        iz:csdl.Variable)->'CSBox':
        """a box with user-supplied section properties; the dimensions are used for stress recovery"""
        box = cls.__new__(cls)
        box._set_dimensions(ttop, tbot, tweb, height, width)
        CrossSection.__init__(box, area, ix, iy, iz)
        return box


    def _set_dimensions(self, ttop, tbot, tweb, height, width):
        if not ttop.shape == tbot.shape == tweb.shape == height.shape == width.shape:
            raise ValueError('box beam ttop, tbot, tweb, height and width must have the same shape')

        self.ttop = ttop
        self.tbot = tbot
        self.tweb = tweb
        self.height = height
        self.width = width


    def _neutral_axis(self)->tuple:
        """(y, z) offset of the neutral axis from the box center"""
        Atop = self.ttop * (self.width - 2 * self.tweb)
        Abot = self.tbot * (self.width - 2 * self.tweb)

        centroid_z = (Atop - Abot) / self.width
        centroid_y = 0

        return (centroid_y, centroid_z)


    def _area(self)->csdl.Variable:
        w_i = self.width - 2 * self.tweb
        h_i = self.height - self.ttop - self.tbot
        return self.width * self.height - w_i * h_i


    def _iy(self)->csdl.Variable:
        """parallel-axis sum of the skins and webs about the neutral axis"""
        cz_neutral = self.neutral_axis[1]

        # top skin
        area_top = self.ttop * self.width
        iy_top = self.ttop ** 3 * self.width / 12
        d_top = self.height / 2 - self.ttop / 2 - cz_neutral
        iy_top_centroid = iy_top + area_top * d_top**2

        # bottom skin
        area_bot = self.tbot * self.width
        iy_bot = self.tbot ** 3 * self.width / 12
        d_bot = -self.height / 2 + self.tbot / 2 - cz_neutral
        iy_bot_centroid = iy_bot + area_bot * d_bot**2

        # front/rear webs (centered at z = 0)
        area_web = self.tweb * (self.height - self.ttop - self.tbot)
        iy_web = (self.height - self.ttop - self.tbot) ** 3 * self.tweb / 12
        iy_rear_centroid = iy_web + area_web * cz_neutral**2
        iy_front_centroid = iy_web + area_web * cz_neutral**2

        return iy_top_centroid + iy_bot_centroid + iy_rear_centroid + iy_front_centroid


    def _iz(self)->csdl.Variable:
        """parallel-axis sum of the skins and webs about the neutral axis"""
        cy_neutral = self.neutral_axis[0]

        # top/bottom skins (centered at y = 0)
        area_top = self.ttop * self.width
        iz_top = self.ttop * self.width**3 / 12
        area_bot = self.tbot * self.width
        iz_bot = self.tbot * self.width**3 / 12

        iz_top_centroid = iz_top + area_top * cy_neutral**2
        iz_bot_centroid = iz_bot + area_bot * cy_neutral**2

        # front/rear webs
        area_web = self.tweb * (self.height - self.ttop - self.tbot)
        iz_web = (self.height - self.ttop - self.tbot) * self.tweb**3 / 12

        d_front = -self.width / 2 + self.tweb / 2 - cy_neutral
        d_rear = self.width / 2 - self.tweb / 2 - cy_neutral

        iz_front_centroid = iz_web + area_web * d_front**2
        iz_rear_centroid = iz_web + area_web * d_rear**2

        return iz_top_centroid + iz_bot_centroid + iz_front_centroid + iz_rear_centroid


    def stress(self, element_loads:csdl.Variable)->csdl.Variable:
        """
        von Mises stress at 5 points of the section, shape (num_elements, 5)

            0-----------------1
            |                 |
            4                 |
            |                 |
            3-----------------2

        points 0-3 are the corners (axial + bending + torsion); point 4 is the
        web mid-height, which adds the vertical shear from V_z
        """
        N, _, V_z, T, M_y, M_z = self.internal_loads(element_loads)

        # the axial stress is common to all stress evaluation points
        axial_stress = N / self.area

        def von_mises(z, y, shear=None):
            p = (z**2 + y**2)**0.5
            torsional_stress = T * p / self.ix
            if shear is not None:
                torsional_stress = torsional_stress + shear
            normal_stress = axial_stress + M_y * y / self.iy + M_z * z / self.iz
            # 1E-8 keeps the square root differentiable at zero stress
            return (normal_stress**2 + 3*torsional_stress**2 + 1E-8)**0.5

        corners = [(-self.width / 2, self.height / 2),
                   (self.width / 2, self.height / 2),
                   (self.width / 2, -self.height / 2),
                   (-self.width / 2, -self.height / 2)]
        points = [von_mises(z, y) for z, y in corners]

        # web mid-height: add the vertical shear V_z Q / (I t), with an
        # approximate first moment of area Q and both webs (t = 2 tweb)
        tcap = (self.ttop + self.tbot) / 2
        Q = self.width * tcap * (self.height / 2) + 2 * (self.height / 2) * self.tweb * (self.height / 4)
        shear_stress = V_z * Q / (self.iy * 2 * self.tweb)
        points.append(von_mises(-self.width / 2, 0, shear_stress))

        return csdl.transpose(csdl.vstack(points))


class CSBoxMarius(CSBox):
    """
    Box with user-supplied section properties.

    Deprecated: use ``CSBox.from_properties``, which takes the same arguments.
    """

    def __init__(self,
                 ttop:csdl.Variable,
                 tbot:csdl.Variable,
                 tweb:csdl.Variable,
                 height:csdl.Variable,
                 width:csdl.Variable,
                 area:csdl.Variable,
                 ix:csdl.Variable,
                 iy:csdl.Variable,
                 iz:csdl.Variable):

        self._set_dimensions(ttop, tbot, tweb, height, width)
        CrossSection.__init__(self, area, ix, iy, iz)
