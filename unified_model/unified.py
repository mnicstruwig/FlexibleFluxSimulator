"""Contains the unified model architecture that encapsulates the mechanical
system, electrical system, the coupling between them and the master system
model that describes their interaction.

"""
import numpy as np
from scipy import integrate
from unified_model.utils.utils import parse_output_expression
from unified_model.evaluate import MechanicalSystemEvaluator


# TODO: Add documentation
class UnifiedModel(object):
    def __init__(self, name):
        self.name = name
        self.mechanical_model = None
        self.electrical_model = None
        self.coupling_model = None
        self.governing_equations = None
        self.raw_solution = None
        self.post_processing_pipeline = {}
        self.time = None

    def add_mechanical_model(self, mechanical_model):
        self.mechanical_model = mechanical_model

    def add_electrical_model(self, electrical_model):
        self.electrical_model = electrical_model

    def add_coupling_model(self, coupling_model):
        self.coupling_model = coupling_model

    def add_governing_equations(self, governing_equations):
        self.governing_equations = governing_equations

    def add_post_processing_pipeline(self, pipeline, name):
        """Add a post-processing pipeline"""
        self.post_processing_pipeline[name] = pipeline

    def _apply_pipeline(self):
        for _, pipeline in self.post_processing_pipeline.items():
            # raw solution has dimensions d, n rather than n, d
            self.raw_solution = np.array([pipeline(y) for y in self.raw_solution.T]).T

    def solve(self, t_start, t_end, y0, t_max_step=1e-5, method='RK45'):
        high_level_models = {
            'mechanical_model': self.mechanical_model,
            'electrical_model': self.electrical_model,
            'coupling_model': self.coupling_model
        }

        psoln = integrate.solve_ivp(fun=lambda t, y: self.governing_equations(t, y, **high_level_models),
                                    t_span=[t_start, t_end],
                                    y0=y0,
                                    max_step=t_max_step)

        self.time = psoln.t
        self.raw_solution = psoln.y
        self._apply_pipeline()

    def get_result(self, **kwargs):
        return parse_output_expression(self.time, self.raw_solution, **kwargs)

    def score_mechanical_model(self,
                               metrics_dict,
                               video_labels_df,
                               labeled_video_processor,
                               prediction_expr,
                               **kwargs):
        """Evaluate the mechanical model using a selection of metrics."""

        # Prepare target and prediction data
        y_target, time_target = labeled_video_processor.fit_transform(video_labels_df,
                                                                      impute_missing_values=True)
        df_result = self.get_result(time='t',
                                    prediction=prediction_expr)
        y_predicted = df_result['prediction'].values
        time_predicted = df_result['time'].values

        # Evaluation
        mechanical_evaluator = MechanicalSystemEvaluator(y_target, time_target)
        mechanical_evaluator.fit(y_predicted, time_predicted)
        self.mechanical_evaluator = mechanical_evaluator

        mechanical_scores = mechanical_evaluator.score(**metrics_dict)
        return mechanical_scores
