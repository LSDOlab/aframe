"""Frame: assembles beams and joints into a global linear system and solves it."""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
import numpy as np
import csdl_alpha as csdl

if TYPE_CHECKING:
    from aframe.core.beam import Beam
    from aframe.core.joint import Joint

__all__ = ['Frame']


def _concatenate(variables:List[csdl.Variable])->csdl.Variable:
    """csdl.concatenate needs at least two variables (a frame can have a single beam)"""
    if len(variables) == 1:
        return variables[0]
    return csdl.concatenate(variables)


def _scatter_add(init:np.ndarray,
                 positions:np.ndarray,
                 values:List[csdl.Variable])->csdl.Variable:
    """
    scatter the (flattened, concatenated) values into the flat positions of the
    constant init: each position that receives entries is set to their sum,
    the other positions keep init, and entries with position -1 are dropped

    csdl slices cannot contain repeated indices, so entries that share a
    position are gathered by rank (1st, 2nd, ... entry on each position) and
    summed in order, and every position is then written with a single set
    """
    values = _concatenate([value.flatten() for value in values])
    keep = np.nonzero(positions >= 0)[0]
    order = keep[np.argsort(positions[keep], kind='stable')]
    unique, first, inverse = np.unique(positions[order], return_index=True, return_inverse=True)
    rank = np.arange(order.size) - first[inverse]

    out = values[order[rank == 0].tolist()]
    for r in range(1, rank.max() + 1):
        at = inverse[rank == r].tolist()
        out = out.set(csdl.slice[at], out[at] + values[order[rank == r].tolist()])

    return csdl.Variable(value=init.ravel()).set(csdl.slice[unique.tolist()], out).reshape(init.shape)


class Frame:
    """
    A linear static frame model built from beams and joints.

    Building the Frame numbers the frame nodes and assembles the global
    stiffness and mass matrices and the load vector (with boundary conditions
    applied); call ``solve()`` for the displacements and ``compute_stress()``
    for the stresses. The dof numbering is owned by the Frame, so a Beam can
    be used in several Frames.

    Parameters
    ----------
    beams : list of Beam
        The beams (unique objects with unique names), with loads and boundary
        conditions already added.
    joints : list of Joint, optional
        Rigid connections between beams of this frame.
    acc : csdl.Variable, optional
        Frame acceleration [ax, ay, az, alpha_x, alpha_y, alpha_z], shape (6,),
        applied as inertial loads M @ acc on the structure and extra masses.

    Attributes
    ----------
    K, M : csdl.Variable
        Global stiffness and mass matrices, shape (dim, dim), dim = 6 * num.
    F : csdl.Variable
        Global load vector, shape (dim,).
    U : csdl.Variable
        Global displacement vector, set by ``solve()``.
    displacement, rotation : dict of csdl.Variable
        Per-beam nodal displacements/rotations, shape (num_nodes, 3), set by ``solve()``.
    num, dim : int
        Number of frame nodes and of global dofs.
    bc_indices : np.ndarray
        Global indices of the constrained dofs.
    """

    def __init__(self,
                 beams:List[Beam],
                 joints:Optional[List[Joint]] = None,
                 acc:Optional[csdl.Variable] = None):

        if acc is not None and acc.shape != (6,):
            raise ValueError("acc must have shape (6,)")

        if not beams:
            raise ValueError("beams must be added to the frame")

        if len(set(map(id, beams))) != len(beams):
            raise ValueError("each beam can be added to a frame only once")

        if len({beam.name for beam in beams}) != len(beams):
            raise ValueError("beam names must be unique within a frame")

        self.beams = beams
        self.joints = [] if joints is None else joints
        self.acc = acc
        self.displacement: Dict[str, csdl.Variable] = {}
        self.rotation: Dict[str, csdl.Variable] = {}
        self.U = None

        # global node number of every node of every beam
        self._nodes: Dict[Beam, np.ndarray] = self._number_nodes()
        self.num = len(np.unique(np.concatenate(list(self._nodes.values()))))
        self.dim = self.num * 6
        self.bc_indices = self._bc_indices()

        # global matrices and loads, with the boundary conditions applied
        self.K, self.M = self._global_matrices()
        self.F = self._global_loads()


    def solve(self)->None:
        """solve K U = F and extract the per-beam displacements and rotations"""
        self.U = U = csdl.solve_linear(self.K, self.F)

        for beam in self.beams:
            u = self._nodal_displacements(beam)
            self.displacement[beam.name] = u[:, :3]
            self.rotation[beam.name] = u[:, 3:]


    def compute_stress(self, beams:Optional[List[Beam]] = None)->Dict[str, csdl.Variable]:
        """
        per-beam stresses from the cross-section stress models (call solve() first)

        beams : the beams to evaluate (default: all); every one needs a
                cross-section with a stress model
        """
        if self.U is None:
            raise RuntimeError("call solve() before compute_stress()")

        stress = {}
        for beam in self.beams if beams is None else beams:
            # element displacements [u_a, u_b], shape (num_elements, 12)
            u = self._nodal_displacements(beam)
            element_displacements = csdl.concatenate([u[:-1], u[1:]], axis=1)

            stress[beam.name] = beam.cs.stress(beam.element_loads(element_displacements))

        return stress


    def compute_mass_properties(self)->Tuple[csdl.Variable, csdl.Variable]:
        """center of gravity (3,) and total mass of the beams (extra inertial masses excluded)"""
        mass, rmvec = 0, 0
        for beam in self.beams:
            mass += beam.mass
            rmvec += beam.rmvec

        cg = rmvec / mass
        return cg, mass


    def node_dofs(self, beam:Beam)->np.ndarray:
        """global dof indices of every node of a beam, shape (num_nodes, 6)"""
        return self._nodes[beam][:, None] * 6 + np.arange(6)


    def _element_dofs(self, beam:Beam)->np.ndarray:
        """global dof indices of every element of a beam, shape (num_elements, 12)"""
        dofs = self.node_dofs(beam)
        return np.hstack([dofs[:-1], dofs[1:]])


    def _nodal_displacements(self, beam:Beam)->csdl.Variable:
        """rows of U for the nodes of a beam, shape (num_nodes, 6)"""
        return self.U[self.node_dofs(beam).ravel().tolist()].reshape((beam.num_nodes, 6))


    def _number_nodes(self)->Dict[Beam, np.ndarray]:
        """
        global node numbers: every beam node gets a provisional id, joined
        nodes are merged with union-find (independent of the joint order) and
        the merged nodes are numbered contiguously
        """
        start, total = {}, 0
        for beam in self.beams:
            start[beam] = total
            total += beam.num_nodes

        parent = list(range(total))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for joint in self.joints:
            for member in joint.members:
                if member not in start:
                    raise ValueError(f"joint member {member.name!r} is not a beam of this frame")

            # the joint's first member provides the merged node's id
            root = find(start[joint.members[0]] + joint.nodes[0])
            for member, node in zip(joint.members[1:], joint.nodes[1:]):
                parent[find(start[member] + node)] = root

        # number the merged nodes in set order, which matches the numbering
        # of earlier versions (the layout of K and U is unchanged)
        ids = [find(i) for i in range(total)]
        number = {node: n for n, node in enumerate(set(ids))}

        return {beam: np.array([number[i] for i in ids[start[beam]:start[beam] + beam.num_nodes]])
                for beam in self.beams}


    def _bc_indices(self)->np.ndarray:
        """global indices of the constrained dofs (fixed: all 6, pinned: 3 translations)"""
        indices = [np.empty(0, dtype=int)]
        for beam in self.beams:
            dofs = self.node_dofs(beam)
            indices += [dofs[node] for node in beam.fixed_boundary_conditions]
            indices += [dofs[node, :3] for node in beam.pinned_boundary_conditions]

        return np.unique(np.concatenate(indices))


    def _global_matrices(self)->Tuple[csdl.Variable, csdl.Variable]:
        """
        assemble the global stiffness/mass matrices from the element matrices;
        the rows/columns of constrained dofs are never written and their
        diagonal is set to 1
        """
        dim = self.dim
        is_bc = np.isin(np.arange(dim), self.bc_indices)

        # global (row, col) of every entry of every element matrix
        dofs = np.vstack([self._element_dofs(beam) for beam in self.beams])
        rows = np.repeat(dofs, 12, axis=1).ravel()
        cols = np.tile(dofs, (1, 12)).ravel()
        positions = np.where(is_bc[rows] | is_bc[cols], -1, rows * dim + cols)

        init = np.diag(is_bc.astype(float))
        K = _scatter_add(init, positions, [beam.transformed_stiffness for beam in self.beams])
        M = _scatter_add(init, positions, [beam.transformed_mass for beam in self.beams])

        return K, M


    def _global_loads(self)->csdl.Variable:
        """
        assemble the global load vector from the nodal loads and any inertial
        loads; loads on constrained dofs are dropped
        """
        zeros = np.zeros(self.dim)

        def positions(dofs):
            dofs = np.concatenate([d.ravel() for d in dofs])
            return np.where(np.isin(dofs, self.bc_indices), -1, dofs)

        # nodal loads
        loaded = [beam for beam in self.beams if beam.loads is not None]
        if loaded:
            F = _scatter_add(zeros, positions([self.node_dofs(b) for b in loaded]), [b.loads for b in loaded])
        else:
            F = csdl.Variable(value=zeros)

        acc = self.acc
        if acc is not None:
            # structural inertial loads M @ acc, element by element: m_e @ [acc, acc]
            acc12 = csdl.concatenate([acc, acc])
            inertial = [csdl.einsum(b.transformed_mass, acc12, action='ijk,k->ij') for b in self.beams]
            F = F + _scatter_add(zeros, positions([self._element_dofs(b) for b in self.beams]), inertial)

            # extra point masses are resolved as loads
            massed = [beam for beam in self.beams if beam.extra_inertial_mass is not None]
            if massed:
                extra = [csdl.outer(b.extra_inertial_mass, acc) for b in massed]
                F = F + _scatter_add(zeros, positions([self.node_dofs(b) for b in massed]), extra)

        return F
