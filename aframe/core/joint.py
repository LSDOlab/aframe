import aframe as af
from typing import List


class Joint:

    def __init__(self,
                 members: List[af.Beam],
                 nodes: List[int]) -> None:

        self.members = members
        self.nodes = nodes