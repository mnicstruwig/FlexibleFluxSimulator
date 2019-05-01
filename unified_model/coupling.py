class ConstantCoupling:
    """Coupling between mechanical and electrical class."""

    def __init__(self, c):
        self.c = c

    def get_mechanical_force(self, current):
        return self.c * current
