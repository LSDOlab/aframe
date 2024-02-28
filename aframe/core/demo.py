from aframe import ebbeam, bc




self.add(Aframe(beams=beams, bounds=bounds, joints=joints))


aluminum = {'name': 'aluminum', 'youngs_modulus': 69E9, 'G': 26E9, }

beam_model = ebbeam(names=['wing', 'boom', 'tail'],
                    boundary_conditions={'wing': 10, 'boom': 2},
                    boundary_conditions=[bc(name='wing', node=0, x='fixed', y='fixed', z='fixed', phi='free', theta='free', psi='free'),
                                         ],
                    degrees_of_freedom=[(True, True, False, False, True, True),
                                        (True, True, True, True, True, True)]
                    joints=[('wing', 'boom')],
                    )