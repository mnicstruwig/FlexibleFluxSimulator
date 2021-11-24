from unified_model.local_exceptions import ModelError


class CouplingModel:
    """Coupling between mechanical and electrical class."""

    def __init__(self, coupling_constant):
        self.coupling_constant = coupling_constant

    def __repr__(self):
        return f"CouplingModel(coupling_constant={self.coupling_constant})"

    def get_mechanical_force(self, current):
        return self.coupling_constant * current

    def to_json(self):
        return {'coupling_constant': self.coupling_constant}
