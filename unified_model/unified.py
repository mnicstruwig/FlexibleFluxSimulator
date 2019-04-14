"""Contains the unified model architecture that encapsulates the mechanical
system, electrical system, the coupling between them and the master system
model that describes their interaction.

"""
import numpy as np
from scipy import integrate
from unified_model.utils.utils import parse_output_expression
from unified_model.evaluate import MechanicalSystemEvaluator, ElectricalSystemEvaluator


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
        """Evaluate the mechanical model using a selection of metrics.

        This is simply a useful helper function that makes use of the various
        evaluation tools that are present to allow for a neat and easier-to-use
        manner of evaluating the mechanical model component of the unified
        model.

        Parameters
        ----------
        metrics_dict: dict
            Metrics to compute on the predicted and target mechanical data.
            Keys must be the name of the metric returned in the Results object.
            Values must be the function used to compute the metric. The
            function must accept to numpy arrays (arr_predict, arr_target) as
            input.
        video_labels_df : dataframe
            Dataframe containing groundtruth mechanical data. This dataframe is
            produced by the OpenCV-based CLI helper script.
        labeled_video_processor : object
            Instantiated `LabeledVideoProcessor` object that will be used for
            processing the groundtruth video-labels data.
        prediction_expr : str
            Expression that is evaluated and used as the predictions for the
            mechanical system. *Any* reasonable expression is possible. You
            can refer to each of the differential equations referenced by the
            `governing_equations` using the letter `x` with the number appended.
            For example, `x1` refers to the first differential equation, and
            `x2` refers to the second differential equation. Some additional
            functions can also be applied to the differential equations. These
            are referenced in the "See Also" section below.
        **kwargs
            return_evaluator : bool
                Whether to return the evaluator used to score the electrical
                system.

        See Also
        --------
        unified_model.evaluate.LabeledVideoProcessor : class
            Class used to preprocess `video_labels_df`
        unified_model.evaluate.MechanicalSystemEvaluator.score : method
            Method that implements the scoring mechanism.
        unified_model.unified.UnifiedModel.get_result : method
            Method used to evaluate `prediction_expr`.
        unified_model.utils.utils.parse_output_expression : function
            Function that details additional functions that can be applied
            using `prediction_expr`.

        Example
        -------
        Here we use a previously created unified model
        >>> unified_model.solve(t_start=0,
        ...                     t_end=10,
        ...                     y0=initial_conditions,
        ...                     t_max_step=1e-3)
        >>> mechanical_metrics = {'mde': median_absolute_error,
        ...                       'mape': mean_absolute_percentage_err,
        ...                       'max': max_err}
        >>> pixel_scale = 0.18745
        >>> labeled_video_processor = LabeledVideoProcessor(L=125,
        ...                                                 mm=10,
        ...                                                 seconds_per_frame=3/240,
        ...                                                 pixel_scale=pixel_scale)
        >>> mech_scores = unified_model.score_mechanical_model(metrics_dict=mechanical_metrics,
        ...                                                    video_labels_df=sample.video_labels_df,
        ...                                                    labeled_video_processor=labeled_video_processor,
        ...                                                    prediction_expr='x3-x1')

        """

        # Prepare target and prediction data
        y_target, time_target = labeled_video_processor.fit_transform(video_labels_df,
                                                                      impute_missing_values=True)
        # Calculate prediction using expression
        df_result = self.get_result(time='t',
                                    prediction=prediction_expr)
        y_predict = df_result['prediction'].values
        time_predict = df_result['time'].values

        # Scoring
        mechanical_evaluator = MechanicalSystemEvaluator(y_target, time_target)
        mechanical_evaluator.fit(y_predict, time_predict)
        self.mechanical_evaluator = mechanical_evaluator

        mechanical_scores = mechanical_evaluator.score(**metrics_dict)

        if kwargs.pop('return_evaluator', None):
            return mechanical_scores, mechanical_evaluator
        return mechanical_scores

    def score_electrical_model(self,
                               metrics_dict,
                               adc_df,
                               adc_processor,
                               prediction_expr,
                               **kwargs):
        """Evaluate the electrical model using a selection of metrics.

        This is simply a useful helper function that makes use of the various
        evaluation tools that are present to allow for a neat and easier-to-use
        manner of evaluating the electrical model component of the unified
        model.

        Parameters
        ----------
        metrics_dict: dict
            Metrics to compute on the predicted and target mechanical data.
            Keys must be the name of the metric returned in the Results object.
            Values must be the function used to compute the metric. The
            function must accept to numpy arrays (arr_predict, arr_target) as
            input.
        adc_df : dataframe
            Dataframe containing groundtruth ADC data. This dataframe is
            produced by the OpenCV-based CLI helper script.
        adc_processor : object
            Instantiated `AdcProcessor` object that will be used for
            processing the groundtruth ADC data.
        prediction_expr : str
            Expression that is evaluated and used as the predictions for the
            mechanical system. *Any* reasonable expression is possible. You
            can refer to each of the differential equations referenced by the
            `governing_equations` using the letter `x` with the number appended.
            For example, `x1` refers to the first differential equation, and
            `x2` refers to the second differential equation. Some additional
            functions can also be applied to the differential equations. These
            are referenced in the "See Also" section below.
        **kwargs
            return_evaluator : bool
                Whether to return the evaluator used to score the electrical
                system.

        See Also
        --------
        unified_model.evaluate.AdcProcessor : class
            Class used to preprocess `adc_df`
        unified_model.evaluate.ElectricalSystemEvaluator.score : method
            Method that implements the scoring mechanism.
        unified_model.unified.UnifiedModel.get_result : method
            Method used to evaluate `prediction_expr`.
        unified_model.utils.utils.parse_output_expression : function
            Function that details additional functions that can be applied
            using `prediction_expr`.

        Example
        -------
        Here we use a previously created unified model
        >>> unified_model.solve(t_start=0,
        ...                     t_end=10,
        ...                     y0=initial_conditions,
        ...                     t_max_step=1e-3)
        >>> electrical_metrics = {'rms': root_mean_square}
        >>> adc_processor = AdcProcessor(voltage_division_ratio=1/0.3)
        >>> electrical_scores = unified_model.score_electrical_model(metrics_dict=electrical_metrics,
        ...                                                          adc_df=sample.adc_df,
        ...                                                          adc_processor=adc_processor,
        ...                                                          prediction_expr='g(t, x5)')

        """
        # Prepare target and prediction data
        emf_target, time_target = adc_processor.fit_transform(adc_df)
        # calculate prediction using expression
        df_result = self.get_result(time='t',
                                    prediction=prediction_expr)
        emf_predict = df_result['prediction'].values
        time_predict = df_result['time'].values

        # Scoring
        electrical_evaluator = ElectricalSystemEvaluator(emf_target, time_target)
        electrical_evaluator.fit(emf_predict, time_predict)
        electrical_scores = electrical_evaluator.score(**metrics_dict)

        if kwargs.pop('return_evaluator', None):
            return electrical_scores, electrical_evaluator
        return electrical_scores
