"""Rigid joints that connect beams at shared nodes."""
from __future__ import annotations
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from aframe.core.beam import Beam

__all__ = ['Joint']


class Joint:
    """
    Rigidly connect beams by merging one node of each into a single frame node
    (all 6 dofs are shared).

    Parameters
    ----------
    members : list of Beam
        The beams to connect.
    nodes : list of int
        The node index in each member that is connected, same length as members
        (negative indices count from the end of the beam).
    """

    def __init__(self, members:List[Beam], nodes:List[int])->None:

        if len(members) != len(nodes):
            raise ValueError('a joint needs one node index per member')

        for member, node in zip(members, nodes):
            if not -member.num_nodes <= node < member.num_nodes:
                raise ValueError(f'joint node {node} is out of range for beam {member.name!r} '
                                 f'with {member.num_nodes} nodes')

        self.members = members
        self.nodes = [int(node) % member.num_nodes for member, node in zip(members, nodes)]
