import m3l

class BeamM3LDisplacements(m3l.ImplicitOperation):
    def initialize(self):        
        self.parameters.declare('connectivity', types=list)        
        self.parameters.declare('out_name', types=str)

    def evaluate_residuals(self, geo_mesh, thickness_mesh):
        return residualCSDL(geo_mesh, thickness_mesh)

    def compute_derivatives(self):
        pass


class BeamM3LNodalDisplacements(m3l.ExplicitOperation):
    def compute():
        pass

    def compute_derivatives():
        pass

class BeamM3LStrain(m3l.ExplicitOperation):
    def compute():
        pass

    def compute_derivatives(): # optional
        pass

class BeamM3LStress(m3l.ExplicitOperation):
    def compute():
        pass

    def compute_derivatives(): # optional
        pass

