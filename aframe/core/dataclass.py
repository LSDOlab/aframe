




class Material:
    def __init__(self, name, E, G, rho):
        self.name = name
        self.E = E
        self.G = G
        self.rho = rho

    def __str__(self):
        return f'{self.name}: E={self.E} Pa, G={self.G} Pa, rho={self.rho} kg/m^3'


# aluminum = Material(name='aluminum', E=69E9, G=26E9, rho=2700)
# print(aluminum)



class Beam:
    def __init__(self, name, num_nodes, material, cs):
        self.name = name
        self.num_nodes = num_nodes
        self.material = material
        self.cs = cs
        self.num_elements = num_nodes - 1

    def __str__(self):
        return f'{self.name}: num_nodes={self.num_nodes}, material={self.material.name}, cs={self.cs}'


# wing = Beam(name='wing', num_nodes=10, material=aluminum)
# print(wing)

# fuselage = Beam(name='fuselage', num_nodes=10, material=aluminum)
# print(fuselage)




class Joint:
    def __init__(self, beams, nodes):
        self.beams = beams # the beams that partake in the joint
        self.nodes = nodes # the corresponding nodes for each beam

    def __str__(self):
        return f'beams={[self.beams[i].name for i in range(len(self.beams))]}, nodes={self.nodes}'


# joint_1 = Joint(beams=[wing, fuselage], nodes=[2, 5])
# print(joint_1)



class BoundaryCondition:
    def __init__(self, beam, node, dof=[True, True, True, True, True, True]):
        self.beam = beam
        self.node = node
        self.dof = dof

    def __str__(self):
        return f'boundary condition for {self.beam.name} node {self.node} with dof: {self.dof}'




# boundary_condition_1 = BoundaryCondition(beam=wing, node=0)
# print(boundary_condition_1)
    

class CSProp:
    def __init__(self, A, Ix, Iy, Iz, J, Q):
        self.A = A
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.Q = Q