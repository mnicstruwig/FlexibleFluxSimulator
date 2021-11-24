from unified_model.local_exceptions import ModelError


class CouplingModel:
    """Coupling between mechanical and electrical class."""

    def __init__(self):
        self.coupling_constant = None

    def __repr__(self):
        return f"CouplingModel(coupling_constant={self.coupling_constant})"

    def set_coupling_constant(self, coupling_constant):
        self.coupling_constant = coupling_constant
        return self

    def get_mechanical_force(self, current):
        return self.coupling_constant * current

    def _validate(self):
        if self.coupling_constant is None:
            raise ModelError(f'`coupling_constant` not set! Did you call `.set_coupling_constant`? {self.coupling_constant}')
