"""
Contains the unified model architecture that encapsulates the mechanical system,
electrical system, the coupling between them and the master system model that
describes their interaction.
"""


from __future__ import annotations

import os
import shutil
import warnings
from glob import glob
from typing import Any, Callable, Dict, List, Tuple, Union, Optional

import cloudpickle
import numpy as np
import pandas as pd
from scipy import integrate

from unified_model.coupling import CouplingModel
from unified_model.electrical_model import ElectricalModel
from unified_model.evaluate import (
    ElectricalSystemEvaluator,
    MechanicalSystemEvaluator,
    Measurement,
)
from unified_model.mechanical_model import MechanicalModel
from unified_model.utils.utils import parse_output_expression, pretty_str
from unified_model.local_exceptions import ModelError


class UnifiedModel:
    """Unified model class

    This class is used to solve your combined or unified electrical and
    mechanical models.

    Attributes
    ----------
    mechanical_model : instance of `MechanicalModel`
        The mechanical model to use as part of the unified model.
    electrical_model : instance of `ElectricalModel`
        The electrical model to use as part of the unified model.
    coupling_model : instance of `CouplingModel`
        The electro-mechanical coupling to use as part of the unified model.
    governing_equations: func
        The set of governing equations to solve using the unified model.
    raw_solution : ndarray
        The raw post-pipeline output of the solution of the governing
        equations. Intended for debugging purposes.
        Note that the dimensions of `raw_solution` is reversed: each
        row represents all the values for each differential equation
        expressed in `y` by the governing equations.
    post_processing_pipeline : dict
        Dict where keys are pipeline names and values are functions that
        accepts as a single argument the `self.raw_solution` ndarray and
        returns the processed result.
    time : ndarray
        The time steps of the solution to the governing equations.

    """

    def __init__(self) -> None:
        """Constructor."""
        self.mechanical_model: Optional[MechanicalModel] = None
        self.electrical_model: Optional[ElectricalModel] = None
        self.coupling_model: Optional[CouplingModel] = None
        self.governing_equations: Optional[Callable] = None
        self.raw_solution: Any = None
        self.post_processing_pipeline: Dict[str, Any] = {}
        self.time: Optional[np.ndarray] = None

    def __str__(self) -> str:
        """Return string representation of the UnifiedModel"""
        return f"Unified Model: {pretty_str(self.__dict__)}"

    def set_mechanical_model(
        self,
        mechanical_model: MechanicalModel,
    ) -> UnifiedModel:

        """Add a mechanical model to the unified model

        Parameters
        ----------
        mechanical_model : instance of `MechanicalModel`
            The mechanical model to add to the unified model.
            Is passed to `governing_equations` function when the `solve`
            method is called.

        """
        self.mechanical_model = mechanical_model
        return self

    def set_electrical_model(self, electrical_model: ElectricalModel) -> UnifiedModel:
        """Add an electrical model to the unified model

        Parameters
        ----------
        electrical_model : instance of `ElectricalModel`
            The electrical model to add to the unified model.
            Is passed to `governing_equations` function when the `solve`
            method is called.

        """
        self.electrical_model = electrical_model
        return self

    def set_coupling_model(self, coupling_model: CouplingModel) -> UnifiedModel:
        """Add the electro-mechanical coupling to the unified model.

        Parameters
        ----------
        coupling_model : CouplingModel
            The coupling model to add to the unified model.
            Is passed to `governing_equations` function when the `solve`
            method is called.

        """
        self.coupling_model = coupling_model
        return self

    def set_governing_equations(self, governing_equations: Callable) -> UnifiedModel:
        """Add a set of governing equations to the unified model.

        The governing equations describe the behaviour of the entire system,
        and control the manner in which the various components interact.

        Must accept arguments `t` and `y` and keyword arguments
        `mechanical_model`, `electrical_model` and `coupling_model`.The
        structure and return value of `governing_equations` must be of the same
        as functions solved by `scipy.integrate.solve_ivp` (but have the
        additional keyword arguments specified above).

        Parameters
        ----------
        governing_equations : Callable
            Set of governing equations that controls the unified model's
            behaviour.

        See Also
        --------
        scipy.integrate.solve_ivp : function
            `governing_equations` must be compatible with the class of function
            solved by `scipy.integrate.solve_ivp`.

        """
        self.governing_equations = governing_equations
        return self

    def set_post_processing_pipeline(
        self, pipeline: Callable, name: str
    ) -> UnifiedModel:
        """Add a post-processing pipeline to the unified model

        After solving the unified model, optional post-processing pipelines can
        be executed on the resulting solution data. This is useful for clipping
        certain values, resampling or filtering noise.

        The pipelines will be executed in the order that they are added.

        Parameters
        ----------
        pipeline : Callable
            Function that accepts an ndarray of dimensions (N, d), where
            N is the number of time points for which a solution has been
            computed, and d is the dimension of the solution vector `y`
            that is passed into the governing equations.
        name : str
            Name of the pipeline.

        See Also
        --------
        self.set_governing_equations : function that adds the governing
            equations to the unified model.

        """
        self.post_processing_pipeline[name] = pipeline
        return self

    def solve(
        self,
        t_start: float,
        t_end: float,
        y0: List[float],
        t_eval: Union[List, np.ndarray],
        t_max_step: float = 1e-5,
    ) -> None:
        """Solve the unified model.

        Parameters
        ----------
        t_start : float
            The start time of the simulation.
        t_end : float
            The end time of the simulation
        y0 : ndarray
            The initial values of `y`, or the result vector that is passed
            to the governing equations.
        t_eval : ndarray[float]
            Times at which to store the computed solution.
        t_max_step : float, optional
            The maximum time step (in seconds) to be used when solving the
            unified model. Default value is 1e-5.

        See Also
        --------
        scipy.integrate.solve_ivp : Function used to solve the governing
            equations of the unified model.

        """
        self._validate()

        high_level_models = {
            "mechanical_model": self.mechanical_model,
            "electrical_model": self.electrical_model,
            "coupling_model": self.coupling_model,
        }

        psoln = integrate.solve_ivp(
            fun=lambda t, y: self.governing_equations(t, y, **high_level_models),  # type: ignore # noqa
            t_span=[t_start, t_end],
            y0=y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-4,
            max_step=t_max_step,
        )

        self.time = psoln.t
        self.raw_solution = psoln.y
        self._apply_pipeline()

    def get_result(self, **kwargs) -> pd.DataFrame:
        """Get a dataframe of the results using expressions.

        *Any* reasonable expression is possible. You can refer to each of the
        differential equations that is represented by the governing equations
        using the letter 'x' with the number appended. For example `x1` refers
        to the first differential equation, `x2` to the second, etc.

        Each expression is available as a column in the returned pandas
        dataframe, with the column name being the key of the passed expression.

        Parameters
        ----------
        **kwargs
            Each key is the name of the column of the returned dataframe.
            Each value is the expression to be evaluated.

        Returns
        -------
        pandas dataframe
            Output dataframe containing the evaluated expressions.

        See Also
        --------
        unified_model.utils.utils.parse_output_expression : helper function
            that contains the parsing logic.

        Example
        --------
        >>> # Here we use previously-built and solved unified model
        >>> unified_model
        <unified_model.unified.UnifiedModel at 0x7fa9e45a83c8>
        >>> print(unified_model.raw_solution)
        [[1 2 3 4 5]
         [1 1 1 1 1]]
        >>> unified_model.get_result(an_expr='x1',
                                     another_expr='x2-x1',
                                     third_expr='x1*x2')
           an_expr  another_expr  third_expr
        0        1             0           1
        1        2            -1           2
        2        3            -2           3
        3        4            -3           4
        4        5            -4           5

        """
        try:
            return parse_output_expression(self.time, self.raw_solution, **kwargs)
        except AssertionError as e:
            raise ValueError(
                "Raw solution is not found. Did you run .solve?"
            ) from e  # noqa

    def get_quick_results(self) -> pd.DataFrame:
        """Get a table of commonly used results.

        Return a DataFrame containing the time, relative magnet position,
        relative magnet velocity and load voltage.
        """
        return self.get_result(
            time="t", rel_pos_mag="x3-x1", rel_pos_vel="x4-x2", v_load="g(t, x5)"
        )

    def score_mechanical_model(
        self,
        y_target: np.ndarray,
        time_target: np.ndarray,
        metrics_dict: Dict[str, Callable],
        prediction_expr: str,
        warp: bool = False,
        **kwargs,
    ):
        warnings.warn(
            "`score_mechanical_model has been deprecated.` Please use `score_measurement` instead.",  # noqa
            DeprecationWarning,
        )

        return self._score_mechanical_model(
            y_target,
            time_target,
            metrics_dict,
            prediction_expr,
            warp,
            **kwargs,
        )

    def _score_mechanical_model(
        self,
        y_target: np.ndarray,
        time_target: np.ndarray,
        metrics_dict: Dict[str, Callable],
        prediction_expr: str,
        warp: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, float], Any]:
        """Evaluate the mechanical model using a selection of metrics.

        This is a useful helper function that makes use of the various
        evaluation tools that are present to allow for a neat and easier-to-use
        manner of evaluating the mechanical model component of the unified
        model.

        Parameters
        ----------
        y_target : np.ndarray
            The target mechanical values.
        time_target : np.ndarray
            The corresponding target time values.
        metrics_dict: dict
            Metrics to compute on the predicted and target mechanical data.
            Keys must be the name of the metric returned in the Results object.
            Values must be the function used to compute the metric. The
            function must accept to numpy arrays (arr_predict, arr_target) as
            input.
        prediction_expr : str
            Expression that is evaluated and used as the predictions for the
            mechanical system. *Any* reasonable expression is possible. You
            can refer to each of the differential equations referenced by the
            `governing_equations` using the letter `x` with the number appended.
            For example, `x1` refers to the first differential equation, and
            `x2` refers to the second differential equation. Some additional
            functions can also be applied to the differential equations. These
            are referenced in the "See Also" section below.
        warp : bool, optional
            Score using after dynamically time-warping the prediction and
            target signals.
            Default value is False.
        **kwargs
            Keyword arguments passed to to the underlying evaluator class.

        Returns
        -------
        Dict
            A dictionary where the keys match the keys of `metrics_dict`, and
            the values contain the corresponding computed function specified in
            `metrics_dict`.

        See Also
        --------
        unified_model.evaluate.LabeledVideoProcessor : class
            Helper class used to preprocess labeled mechanical video data into
            `time_target` and `y_target`.
        unified_model.evaluate.MechanicalSystemEvaluator.score : method
            Method that implements the scoring mechanism.
        unified_model.unified.UnifiedModel.get_result : method
            Method used to evaluate `prediction_expr`.

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
        >>> labeled_video_processor = LabeledVideoProcessor(
                L=125,
        ...     mm=10,
        ...     seconds_per_frame=3/240,
        ...     pixel_scale=pixel_scale
        ... )
        >>> y_target, y_time_target = labeled_video_processor.fit_transform(
        ...     video_labels_df,
        ...     impute_missing_values=True
        ... )
        >>> mech_scores = unified_model.score_mechanical_model(
        ...     time_target=y_time_target,
        ...     y_target=y_target,
        ...     metrics_dict=mechanical_metrics,
        ...     video_labels_df=sample.video_labels_df,
        ...     labeled_video_processor=labeled_video_processor,
        ...     prediction_expr='x3-x1'
        ... )

        """

        # Calculate prediction using expression
        df_result = self.get_result(time="t", prediction=prediction_expr)
        y_predict = df_result["prediction"].values
        time_predict = df_result["time"].values

        # Scoring
        mechanical_evaluator = MechanicalSystemEvaluator(
            y_target,
            time_target,
            metrics=metrics_dict,
            clip=kwargs.get("clip", True),
            warp=warp,
        )
        mechanical_evaluator.fit(y_predict, time_predict)
        mechanical_scores = mechanical_evaluator.score()

        return mechanical_scores, mechanical_evaluator

    def score_electrical_model(
        self,
        emf_target: np.ndarray,
        time_target: np.ndarray,
        metrics_dict: Dict,
        prediction_expr: str,
        warp: bool = False,
        **kwargs,
    ):
        warnings.warn(
            "`score_electrical_model has been deprecated.` Please use `score_measurement` instead.",  # noqa
            DeprecationWarning,
        )

        return self._score_electrical_model(
            emf_target,
            time_target,
            metrics_dict,
            prediction_expr,
            warp,
            **kwargs,
        )

    def _score_electrical_model(
        self,
        emf_target: np.ndarray,
        time_target: np.ndarray,
        metrics_dict: Dict,
        prediction_expr: str,
        warp: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, float], Any]:
        """Evaluate the electrical model using a selection of metrics.

        This is simply a useful helper function that makes use of the various
        evaluation tools that are present to allow for a neat and easier-to-use
        manner of evaluating the electrical model component of the unified
        model.

        Parameters
        ----------
        emf_target: np.ndarray
            The groundtruth load power values.
        time_target : np.ndarray
            The groundtruth time values.
        metrics_dict: dict
            Metrics to compute on the predicted and target electrical data.
            Keys will be used to set the attributes of the Score object.  Values
            must be the function used to compute the metric. Each function must
            accept arguments (arr_predict, arr_target) as input, where
            `arr_predict` and `arr_target` are numpy arrays that contain the
            predicted values and target values, respectively. The return value
            of the functions can have any shape.
        prediction_expr : str
            Expression that is evaluated and used as the predictions for the
            electrical system. *Any* reasonable expression is possible. You
            can refer to each of the differential equations referenced by the
            `governing_equations` using the letter `x` with the number appended.
            For example, `x1` refers to the first differential equation, and
            `x2` refers to the second differential equation. Some additional
            functions can also be applied to the differential equations. These
            are referenced in the "See Also" section below.
        **kwargs
            Keyword arguments passed to the underlying evaluator class.

        Returns
        -------
        Dict
            A dictionary where the keys match the keys of `metrics_dict`, and
            the values contain the corresponding computed function specified in
            `metrics_dict`.

        Example
        -------
        Here we use a previously created unified model
        >>> unified_model.solve(t_start=0,
        ...                     t_end=10,
        ...                     y0=initial_conditions,
        ...                     t_max_step=1e-3)
        >>> electrical_metrics = {'rms': root_mean_square}
        >>> adc_processor = AdcProcessor(voltage_division_ratio=1/0.3)
        >>> electrical_scores = unified_model.score_electrical_model(
        ... metrics_dict=electrical_metrics,
        ... adc_df=sample.adc_df,
        ... adc_processor=adc_processor,
        ... prediction_expr='g(t, x5)')

        """
        # calculate prediction using expression
        df_result = self.get_result(time="t", prediction=prediction_expr)
        emf_predict = df_result["prediction"].values
        time_predict = df_result["time"].values

        # Scoring
        electrical_evaluator = ElectricalSystemEvaluator(
            emf_target, time_target, metrics_dict, warp
        )

        electrical_evaluator.fit(emf_predict, time_predict)

        electrical_scores = electrical_evaluator.score()

        return electrical_scores, electrical_evaluator

    def score_measurement(
        self,
        measurement: Measurement,
        solve_kwargs: Dict,
        mech_pred_expr: Optional[str] = None,
        mech_metrics_dict: Optional[Dict[str, Callable]] = None,
        elec_pred_expr: Optional[str] = None,
        elec_metrics_dict: Optional[Dict[str, Callable]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Score against a single measurement using its input excitation.

        The mechanical and electrical (or both) can be scored against. This
        function is a convenience wrapper around the `_score_mechanical_system`
        and `_score_electrical_system` methods.

        Note, that this explicitly calls a `solve` after updating the model to
        use the `measurement` input excitation.

        Parameters
        ----------
        measurement : Measurement
            The measurement object containing the groundtruth data.
        solve_kwargs : Dict
            The keyword arguments to pass to the `solve` method.
        mech_pred_expr : str
            Optional. The mechanical expression to score.
        mech_metrics_dict : Dict[str, Callable]
            The metrics to calculate on `mech_pred_expr` and the mechanical
            ground truth data. Keys are a user-chosen name to give each metric.
            Values are callables that accept two positional arguments; the first
            being the predicted output (calculated from `mech_pred_expr`), and
            the second being the ground truth measurement.
        elec_pred_expr : str
            Optional. The electrical expression to score.
        elec_metrics_dict : Dict[str, Callable]
            The metrics to calculate on `elec_pred_expr` and the electrical
            ground truth data. Keys are a user-chosen name to give each metric.
            Values are callables that accept two positional arguments; the first
            being the predicted output (calculated from `elec_pred_expr`), and
            the second being the ground truth measurement.

        Examples
        --------

        >>> model.score_measurement(
        ...    measurement=measurement,
        ...    mech_pred_expr='x3-x1',
        ...    mech_metrics_dict={
        ...    'y_diff_dtw_distance': metrics.dtw_euclid_distance
        ...    },
        ...    elec_pred_expr='g(t, x5)',
        ...    elec_metrics_dict={
        ...        'emf_dtw_distance': metrics.dtw_euclid_distance,
        ...        'rms_perc_diff': metrics.root_mean_square_percentage_diff
        ...    }
        ... )

        Returns
        -------
        Tuple[Dict[str, float], Dict[str, Any]]
            The calculated metrics and evaluators in a tuple. For the calculated
            metrics, the keys correspond to the names given in the
            `*_metrics_dict` dictionaries and values correspond to the
            calculated Callables.

        """

        result = {}
        evaluators: Dict[str, Any] = {"mech": None, "elec": None}

        mech_result: Dict[str, float] = {}
        elec_result: Dict[str, float] = {}

        if self.mechanical_model:
            self.mechanical_model.set_input(measurement.input_)
        else:
            raise ValueError("Mechanical model has not been specified.")

        # Run the solver
        self.reset()
        self.solve(**solve_kwargs)

        # Do the scoring
        if mech_pred_expr is not None and mech_metrics_dict is not None:
            mech_result, mech_eval = self._score_mechanical_model(
                y_target=measurement.groundtruth.mech["y_diff"],
                time_target=measurement.groundtruth.mech["time"],
                metrics_dict=mech_metrics_dict,
                prediction_expr=mech_pred_expr,
                return_evaluator=True,
            )
            evaluators["mech"] = mech_eval

        if elec_pred_expr is not None and elec_metrics_dict is not None:
            elec_result, elec_eval = self._score_electrical_model(
                emf_target=measurement.groundtruth.elec["emf"],
                time_target=measurement.groundtruth.elec["time"],
                metrics_dict=elec_metrics_dict,
                prediction_expr=elec_pred_expr,
                return_evaluator=True,
            )
            evaluators["elec"] = elec_eval

        result.update(mech_result)
        result.update(elec_result)

        return result, evaluators

    def calculate_metrics(self, prediction_expr: str, metric_dict: Dict) -> Dict:
        """Calculate metrics on a prediction expressions."""

        df_result = self.get_result(expr=prediction_expr)

        results = {}
        for name, metric_func in metric_dict.items():
            results[name] = metric_func(df_result["expr"].values)
        return results

    def reset(self) -> None:
        """Clear all computed results from the unified model."""

        self.time = None
        self.raw_solution = None

    def update_params(self, config: List[Tuple[str, Any]]):
        """Update the parameters of the unified model."""
        for path, value in config:
            sub_paths = path.split(".")
            if len(sub_paths) == 2:  # TODO: Make this less hardcoded
                self.__dict__[sub_paths[0]][sub_paths[1]] = value
            if len(sub_paths) == 3:  # TODO: Make this less hardcoded
                # Check we're not setting something that doesn't exist
                try:
                    assert (
                        sub_paths[-1]
                        in self.__dict__[sub_paths[0]].__dict__[sub_paths[1]].__dict__
                    )  # noqa
                except AssertionError as e:
                    raise ValueError(
                        f'The path to the parameter "{path}" does not exist.'
                    ) from e  # noqa

                self.__dict__[sub_paths[0]].__dict__[sub_paths[1]].__dict__[
                    sub_paths[2]
                ] = value

    def save_to_disk(self, path: str, overwrite=False) -> None:
        """Persists a unified model to disk"""
        if overwrite:
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass

        if not os.path.exists(path):
            os.makedirs(path)
        else:
            raise FileExistsError("The path already exists.")

        for key, val in self.__dict__.items():
            component_path = path + key + ".pkl"
            with open(component_path, "wb") as f:
                cloudpickle.dump(val, f)

    @staticmethod
    def load_from_disk(path: str) -> UnifiedModel:
        """Load a unified model from disk."""
        unified_model = UnifiedModel()

        try:
            assert os.path.isdir(path)
        except AssertionError:
            raise FileNotFoundError("Path to model does not exist")

        files = glob(path + "*")
        # TODO: Use regex instead
        keys = [f.split(".pkl")[0].split("/")[-1] for f in files]

        for key, file_ in zip(keys, files):
            with open(file_, "rb") as f:
                unified_model.__dict__[key] = cloudpickle.load(f)

        return unified_model

    def _validate(self) -> None:
        try:
            assert self.mechanical_model is not None
            self.mechanical_model._validate()
        except AssertionError:
            raise ModelError("A mechanical model must be specified.")
        try:
            assert self.electrical_model is not None
            self.electrical_model._validate()
        except AssertionError:
            raise ModelError("A electrical model must be specified.")
        try:
            assert self.coupling_model is not None
        except AssertionError:
            raise ModelError("A coupling model must be specified.")

    def _apply_pipeline(self) -> None:
        """Execute the post-processing pipelines on the raw solution.."""
        for _, pipeline in self.post_processing_pipeline.items():
            # raw solution has dimensions d, n rather than n, d
            self.raw_solution = np.array([pipeline(y) for y in self.raw_solution.T]).T
