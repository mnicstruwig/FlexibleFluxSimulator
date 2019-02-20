"""
Contains the unified model architecture that encapsulates the mechanical system, electrical system, the coupling
between them and the master system model that describes their interaction.
"""

from scipy import integrate
from unified_model.utils.utils import parse_output_expression


class UnifiedModel:
    def __init__(self, name):
        self.name = name
        self.mechanical_model = None
        self.electrical_model = None
        self.coupling_model = None
        self.governing_equations = None
        self.raw_solution = None
        self.t = None

    def add_mechanical_model(self, mechanical_model):
        self.mechanical_model = mechanical_model

    def add_electrical_model(self, electrical_model):
        self.electrical_model = electrical_model

    def add_coupling_model(self, coupling_model):
        self.coupling_model = coupling_model

    def add_governing_equations(self, governing_equations):
        self.governing_equations = governing_equations

    def solve(self, t_start, t_end, y0, t_max_step=1e-5, method='RK45'):
        high_level_models = {
            'mechanical_model': self.mechanical_model,
            'electrical_model': self.electrical_model,
            'coupling_model': self.coupling_model
        }

        psoln = integrate.solve_ivp(fun=lambda t, y: self.governing_equations(t, y, **high_level_models),
                                    t_span=[t_start, t_end],
                                    y0=y0,
                                    max_step=t_max_step,
                                    dense_output=True)

        self.t = psoln.t
        self.raw_solution = psoln.y

    def get_result(self, **kwargs):
        return parse_output_expression(self.t, self.raw_solution, **kwargs)
