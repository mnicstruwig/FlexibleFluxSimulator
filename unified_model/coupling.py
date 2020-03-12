class CouplingModel:
    """Coupling between mechanical and electrical class."""

    def __init__(self):
        self.c = None

    def __repr__(self):
        return f'CouplingModel(coupling_constant={self.c})'

    def set_coupling_constant(self, c):
        self.c = c
        return self

    def get_mechanical_force(self, current):
        return self.c * current
