"""
Contains the unified model architecture that encapsulates the mechanical system,
electrical system, the coupling between them and the master system model that
describes their interaction.
"""

from __future__ import annotations

import copy
import json
import os
import warnings
from glob import glob
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cloudpickle
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.signal import savgol_filter

from .coupling import CouplingModel
from .electrical_components.coil import CoilConfiguration
from .electrical_components.flux import FluxModelPretrained
from .electrical_components.load import SimpleLoad
from .evaluate import ElectricalSystemEvaluator, Measurement, MechanicalSystemEvaluator
from .local_exceptions import ModelError
from .mechanical_components import magnet_assembly
from .mechanical_components.damper import MassProportionalDamper
from .mechanical_components.input_excitation.accelerometer import AccelerometerInput
from .mechanical_components.magnetic_spring import MagneticSpringInterp
from .mechanical_components.mechanical_spring import MechanicalSpring
from .utils.paint import paint_device
from .utils.utils import parse_output_expression, pretty_str


def _has_update_method(obj):
    """Return `True` if the object has an `update` method."""
    try:
        update_attr = getattr(obj, "update")
        return callable(update_attr)
    except AttributeError:
        return False


def send_notification(fn):
    """Causes the decorated function to send a notification to all Observers."""
    from functools import wraps

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        # Call the underlying function, which must return
        # a new model object
        new_model = fn(self, *args, **kwargs)
        # Send out notifications to all the observers
        new_model._notify()

        # Return our new model
        return new_model

    # Decorators must return functions, so we return our wrapper
    return wrapper


class UnifiedModel:
    """Unified model class

    This class is used to solve your combined or unified electrical and
    mechanical models.

    Attributes
    ----------
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

    # TODO: Type hint attributes
    def __init__(self) -> None:
        """Constructor."""
        self.height: Optional[float] = None

        self.governing_equations: Optional[Callable] = None

        self.time: Optional[np.ndarray] = None
        self.raw_solution: Optional[np.ndarray] = None
        self.post_processing_pipeline: Dict[str, Any] = {}

        # Mechanical components
        self.magnetic_spring: Optional[Any] = None
        self.magnet_assembly = None
        self.mechanical_spring = None
        self.mechanical_damper = None
        self.input_excitation = None

        # Electrical components
        self.flux_model: Optional[Any] = None
        self.coil_configuration: Optional[Any] = None
        self.rectification_drop: Optional[float] = None
        self.load_model: Optional[Any] = None

        # Coupling model
        self.coupling_model: Optional[Any] = None

        # Extra, non-standard components
        self.extra_components: Dict[str, Any] = {}

        # Observers
        self._observers: Dict[str, Any] = {}

    def _attach_if_observer(self, name, candidate):
        """
        Attach the component as an observer if it has a valid `update` method.
        """
        if _has_update_method(candidate):
            self._observers[name] = candidate

    def _notify(self):
        """Notify all observers of state changes."""
        for observer in self._observers.values():
            observer.update(self)

    def __str__(self) -> str:
        """Return string representation of the UnifiedModel"""
        return f"Unified Model: {pretty_str(self.__dict__)}"

    @send_notification
    def with_magnetic_spring(self, magnetic_spring):
        self.magnetic_spring = magnetic_spring
        self._attach_if_observer("magnetic_spring", self.magnetic_spring)

        return self

    @send_notification
    def with_magnet_assembly(self, magnet_assembly):
        self.magnet_assembly = magnet_assembly
        self._attach_if_observer("magnet_assembly", self.magnet_assembly)

        return self

    @send_notification
    def with_mechanical_spring(self, mechanical_spring):
        self.mechanical_spring = mechanical_spring
        self._attach_if_observer("mechanical_spring", self.mechanical_spring)

        return self

    @send_notification
    def with_mechanical_damper(self, mechanical_damper):
        self.mechanical_damper = mechanical_damper
        self._attach_if_observer("mechanical_damper", self.mechanical_damper)

        return self

    @send_notification
    def with_input_excitation(self, input_excitation):
        self.input_excitation = input_excitation
        self._attach_if_observer("input_excitation", self.input_excitation)

        return self

    @send_notification
    def with_flux_model(self, flux_model):
        self.flux_model = flux_model
        self._attach_if_observer("flux_model", self.flux_model)

        return self

    @send_notification
    def with_coil_configuration(self, coil_configuration):
        self.coil_configuration = coil_configuration
        self._attach_if_observer("coil_configuration", self.coil_configuration)

        return self

    @send_notification
    def with_rectification_drop(self, rectification_drop):
        self.rectification_drop = rectification_drop
        self._attach_if_observer("rectification_drop", self.rectification_drop)

        return self

    @send_notification
    def with_load_model(self, load_model):
        self.load_model = load_model
        self._attach_if_observer("load_model", self.load_model)

        return self

    @send_notification
    def with_coupling_model(self, coupling_model):
        self.coupling_model = coupling_model
        self._attach_if_observer("coupling_model", self.coupling_model)

        return self

    def with_extra(self, name, extra_component):
        """Attach extra, non-standard components as part of the model."""
        self.extra_components[name] = extra_component
        self._attach_if_observer(name, extra_component)

        return self

    @send_notification
    def with_height(self, height: float) -> UnifiedModel:
        """Constrain the microgenerator to a maximum vertical height.

        This constraint will be validated before solving for a solution.

        Parameters
        ----------
        height : float
            The height of the device, in metres.

        Returns
        -------
        UnifiedModel
            An updated UnifiedModel.

        """
        self.height = height
        return self

    def with_governing_equations(self, governing_equations: Callable) -> UnifiedModel:
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

    def with_post_processing_pipeline(
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

    def _print_device(self):
        ma = self.magnet_assembly
        mag_spring = self.magnetic_spring
        cc = self.coil_configuration

        # We must compensate for the fact that the hover height and
        # coil center are relative to the *top* edge of the floating magnet.
        fixed_magnet_offset = ma.l_m_mm
        l_bth = 3
        offset = fixed_magnet_offset + l_bth

        paint_device(
            step=5,
            m=ma.m,
            l_m_mm=ma.l_m_mm,
            l_mcd_mm=ma.l_mcd_mm,
            l_hover=mag_spring.get_hover_height(ma) * 1000 + offset,
            c=cc.c,
            l_c_mm=cc.get_height() * 1000,
            l_ccd_mm=cc.l_ccd_mm,
            l_center=cc.coil_center_mm + offset,
            l_L=self.height * 1000,
        )

    def summarize(self) -> None:
        """Summarize and validate the microgenerator design."""
        ma = self.magnet_assembly
        cc = self.coil_configuration
        ms = self.mechanical_spring
        mag_spring = self.magnetic_spring
        load = self.load_model

        try:
            assert self.height is not None
        except AssertionError as e:
            raise ModelError(
                "Please set the height first using the .set_height method!"
            ) from e

        self._print_device()

        top_rule = "====================\n"
        mid_rule = "-----------\n"

        magnet_assembly_str = (
            f"🧲 The magnet assembly consists of {ma.m} magnet(s)"
            f" that are {ma.l_m_mm}mm long and have a diameter of {ma.dia_magnet_mm}mm.\n"
            f"🧲 The magnets' centers are {ma.l_mcd_mm}mm apart.\n"
            f"🧲 The magnet assembly has a weight of {np.round(ma.get_weight(), 4)}N.\n"
            f"🧲 The magnet assembly hovers {np.round(mag_spring.get_hover_height(ma) * 1000, 3)}mm above the fixed magnet.\n"
        )

        coil_config_str = (
            f"⚡ There are {cc.c} coils,"
            f" each with {cc.n_z * cc.n_w} windings ({cc.n_z} vertical X {cc.n_w} horizontal).\n"
            f"⚡ This gives each a coil an estimated height of ~{np.round(cc.get_height() * 1000, 2)}mm"
            f" and width of ~{np.round(cc.get_width() * 1000, 2)}mm.\n"
            f"⚡ The coils' centers are {cc.l_ccd_mm}mm apart.\n"
            f"⚡ The first coil's center is {cc.coil_center_mm}mm above the fixed magnet.\n"
            f"⚡ The total microgenerator resistance is {cc.get_coil_resistance()}Ω.\n"
        )

        mech_spring_str = (
            f"📏 The device has a height of {ms.position * 1000}mm.\n"
            f"📏 The minimum required height is {np.round(self._calculate_required_vertical_space() * 1000, 3)}mm.\n"
        )

        load_str = f"🎯 The device is powering a {load.R}Ω load.\n"

        final_str = (
            top_rule
            + magnet_assembly_str
            + mid_rule
            + coil_config_str
            + mid_rule
            + mech_spring_str
            + mid_rule
            + load_str
            + top_rule
        )

        print(final_str)

    def solve(
        self,
        t_start: float,
        t_end: float,
        y0: List[float],
        t_eval: Union[List, np.ndarray],
        t_max_step: float = 1e-4,
        method: str = "RK45",
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
            unified model. Default value is 1e-4.
        method : str, optional
            The solver to use. Default value is 'RK45'. See the documentation
            for `scipy.integrate.solve_ivp` for possible solution methods.

        See Also
        --------
        scipy.integrate.solve_ivp : Function used to solve the governing
            equations of the unified model.

        """

        psoln = integrate.solve_ivp(
            fun=lambda t, y: self.governing_equations(t, y, self),  # type: ignore # noqa
            t_span=[t_start, t_end],
            y0=y0,
            t_eval=t_eval,
            method=method,
            max_step=t_max_step,
            rtol=1e-3,  # default 1e-3,
            atol=1e-6,  # default 1e-6
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
        ffs.utils.utils.parse_output_expression : helper function
            that contains the parsing logic.

        Example
        --------
        >>> # Here we use previously-built and solved unified model
        >>> ffs
        <ffs.unified.UnifiedModel at 0x7fa9e45a83c8>
        >>> print(ffs.raw_solution)
        [[1 2 3 4 5]
         [1 1 1 1 1]]
        >>> ffs.get_result(an_expr='x1',
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
        ffs.evaluate.LabeledVideoProcessor : class
            Helper class used to preprocess labeled mechanical video data into
            `time_target` and `y_target`.
        ffs.evaluate.MechanicalSystemEvaluator.score : method
            Method that implements the scoring mechanism.
        ffs.unified.UnifiedModel.get_result : method
            Method used to evaluate `prediction_expr`.

        Example
        -------
        Here we use a previously created unified model
        >>> ffs.solve(t_start=0,
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
        >>> mech_scores = ffs.score_mechanical_model(
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
        >>> ffs.solve(t_start=0,
        ...                     t_end=10,
        ...                     y0=initial_conditions,
        ...                     t_max_step=1e-3)
        >>> electrical_metrics = {'rms': root_mean_square}
        >>> adc_processor = AdcProcessor(voltage_division_ratio=1/0.3)
        >>> electrical_scores = ffs.score_electrical_model(
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

        Note: this emthod explicitly calls the `.solve` method after updating
        the model to use the `measurement` input excitation.

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

        self.with_input_excitation(measurement.input_)

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
        """Calculate metrics on a prediction expression."""

        df_result = self.get_result(expr=prediction_expr)

        results = {}
        for name, metric_func in metric_dict.items():
            results[name] = metric_func(df_result["expr"].values)
        return results

    def reset(self) -> None:
        """Clear all computed results from the unified model."""

        self.time = None
        self.raw_solution = None

    # TODO: Docstring
    def update_params(self, config: List[Tuple[str, Any]]) -> UnifiedModel:
        """Update the parameters and return a new copy of the unified model."""

        new_model = copy.deepcopy(self)

        for full_path, new_value in config:
            try:
                if "." in full_path:
                    component, param_path = full_path.split(".")
                    new_model.__dict__[component].__dict__[param_path] = new_value
                else:
                    component = full_path
                    new_model.__dict__[component] = new_value
            except KeyError as e:
                raise ValueError(f"Unable to find parameter path: {full_path}") from e
            except AttributeError as e:
                raise ValueError(f"Unable to find parameter path: {full_path}") from e

        new_model._notify()  # Notify our components of the update
        return new_model

    def get_config(self, kind: str = "dict") -> Union[Dict[str, Any], str]:
        """Get the configuration of the unified model."""

        output = {
            "height": None,
            "magnet_assembly": None,
            "magnetic_spring": None,
            "mechanical_spring": None,
            "mechanical_damper": None,
            "input_excitation": None,
            "coil_configuration": None,
            "flux_model": None,
            "rectification_drop": None,
            "load_model": None,
            "coupling_model": None,
            "extra_components": None,
            "governing_equations": None,
        }

        # Mechanical components
        if self.height:
            output["height"] = self.height  # type: ignore
        if self.magnetic_spring:
            output["magnetic_spring"] = self.magnetic_spring.to_json()
        if self.magnet_assembly:
            output["magnet_assembly"] = self.magnet_assembly.to_json()
        if self.mechanical_spring:
            output["mechanical_spring"] = self.mechanical_spring.to_json()
        if self.mechanical_damper:
            output["mechanical_damper"] = self.mechanical_damper.to_json()
        if self.input_excitation:
            output["input_excitation"] = self.input_excitation.to_json()

        # Electrical components
        if self.coil_configuration:
            output["coil_configuration"] = self.coil_configuration.to_json()
        if self.flux_model:
            output["flux_model"] = self.flux_model.to_json()
        if self.rectification_drop:
            output["rectification_drop"] = self.rectification_drop
        if self.load_model:
            output["load_model"] = self.load_model.to_json()
        if self.coupling_model:
            output["coupling_model"] = self.coupling_model.to_json()

        # Governing equations
        if self.governing_equations:
            output["governing_equations"] = {  # type: ignore
                "module_path": self.governing_equations.__module__,
                "func_name": self.governing_equations.__qualname__,
            }

        if kind == "dict":
            return output
        elif kind == "json":
            return json.dumps(output)
        else:
            raise ValueError('`kind` must be "dict" or "json"!')

    @staticmethod
    def from_config(config: Dict[Any, Any]):

        model = UnifiedModel()

        for comp, comp_config in config.items():
            if comp_config is None:
                continue  # Skip
            if comp == "height":  # Handle height separately
                model.with_height(comp_config)
                continue
            if comp == "rectification_drop":  # Handle the rectification drop separately
                model.with_rectification_drop(comp_config)
                continue

            # First loop through all component kwargs, and inject missing
            # dependencies where necessary. We are assuming that the
            # `param_name` of the dependency exists in as a key in the `config`
            # dict passed into `from_config`, and that there are no circular
            # dependencies -- things will just break, there is no warning yet.
            for param_name, param_value in comp_config.items():
                if param_value == "dep:magnet_assembly":
                    dep = magnet_assembly.MagnetAssembly(
                        **config[param_name]
                    )  # Lookup dependency config and instantiate it
                    comp_config[param_name] = dep  # Inject dependency
                if param_value == "dep:coil_configuration":
                    dep = CoilConfiguration(**config[param_name])
                    comp_config[param_name] = dep

            # And then build the component, depending on what it is
            if comp == "mechanical_damper":
                damper = MassProportionalDamper(**comp_config)
                model.with_mechanical_damper(damper)

            if comp == "magnet_assembly":
                model.with_magnet_assembly(
                    magnet_assembly.MagnetAssembly(**comp_config)
                )

            if comp == "magnetic_spring":
                # Some additional set-up required for the filter callable
                if comp_config["filter_callable"] == "auto":
                    comp_config["filter_callable"] = lambda x: savgol_filter(x, 11, 7)
                model.with_magnetic_spring(MagneticSpringInterp(**comp_config))

            if comp == "mechanical_spring":
                model.with_mechanical_spring(MechanicalSpring(**comp_config))

            if comp == "input_excitation":
                model.with_input_excitation(AccelerometerInput(**comp_config))

            if comp == "coil_configuration":
                model.with_coil_configuration(CoilConfiguration(**comp_config))

            if comp == "flux_model":
                model.with_flux_model(FluxModelPretrained(**comp_config))

            if comp == "governing_equations":
                import importlib  # TODO: Move?

                module_ = importlib.import_module(comp_config["module_path"])
                governing_equations = getattr(module_, comp_config["func_name"])
                model.with_governing_equations(governing_equations)

            if comp == "load_model":
                model.with_load_model(SimpleLoad(**comp_config))
            if comp == "coupling_model":
                model.with_coupling_model(CouplingModel(**comp_config))

        return model

    @staticmethod
    def load_from_disk(path: str) -> UnifiedModel:
        """Load a unified model from disk."""
        ffs = UnifiedModel()

        try:
            assert os.path.isdir(path)
        except AssertionError:
            raise FileNotFoundError("Path to model does not exist")

        files = glob(path + "*")
        # TODO: Use regex instead
        keys = [f.split(".pkl")[0].split("/")[-1] for f in files]

        for key, file_ in zip(keys, files):
            with open(file_, "rb") as f:
                ffs.__dict__[key] = cloudpickle.load(f)

        return ffs

    def validate(self, verbose=True) -> None:
        def _fail_if_true(bool_or_func, message, err_message=""):
            good = " OK!"
            bad = " ERROR -- "
            exception_message = ""

            if isinstance(bool_or_func, bool):
                result = bool_or_func
                if not result:  # If the test passes
                    message += good
                else:  # If the test fails
                    message += bad
                    message += err_message

            elif callable(bool_or_func):
                try:
                    result = bool_or_func()
                    # We assume that if nothing gets returned, the test passed
                    if result is None:
                        result = False
                except ModelError as e:
                    exception_message = str(e)
                    result = True

                if not result:  # If the test passes
                    message += good
                else:  # If the test fails
                    message += bad
                    message += err_message
                    message += exception_message

            else:
                raise ValueError(
                    "Must specify either a boolean variable or a callable!"
                )

            return result, message

        messages = ["Model validation report:"]
        error_messages = ["Model validation failed:"]

        def _do_validation(
            test,
            message,
            err_message,
            messages=messages,
            error_messages=error_messages,
        ):
            did_fail, message = _fail_if_true(test, message, err_message)
            if did_fail:
                error_messages.append(message)
            messages.append(message)

            return messages, error_messages

        # Basic checks
        messages, error_messages = _do_validation(
            self.mechanical_model is None,
            "Checking if mechanical model is present...",
            "No mechanical model.",
            messages,
            error_messages,
        )

        messages, error_messages = _do_validation(
            self.mechanical_model._validate,
            "Validating mechanical model...",
            "",
            messages,
            error_messages,
        )

        messages, error_messages = _do_validation(
            self.electrical_model is None,
            "Checking if electrical model is present...",
            "No electrical model.",
            messages,
            error_messages,
        )

        messages, error_messages = _do_validation(
            self.electrical_model._validate,
            "Validating electrical model...",
            "",
            messages,
            error_messages,
        )

        messages, error_messages = _do_validation(
            self.coupling_model is None,
            "Checking if coupling model is present...",
            "No coupling model.",
            messages,
            error_messages,
        )

        messages, error_messages = _do_validation(
            self.coupling_model._validate,
            "Validating coupling model...",
            "",
            messages,
            error_messages,
        )

        messages, error_messages = _do_validation(
            self.height is None,
            "Checking if device height has been specified...",
            "Height of the device has not been set.",
            messages,
            error_messages,
        )

        messages, error_messages = _do_validation(
            self.mechanical_model.mechanical_spring.position is None,
            "Checking if the mechanical spring position has been set...",
            "Mechanical spring position not set. Did you call `.set_height` on the UnifiedModel after setting the mechanical spring?",
            messages,
            error_messages,
        )

        messages, error_messages = _do_validation(
            self.governing_equations is None,
            "Checking if governing equations have been specified...",
            "Governing equations have not been set.",
            messages,
            error_messages,
        )

        # More involved checks

        l_bth = 3 / 1000  # Bottom part of the tube
        # We use the same fixed magnet length as in the assembly
        fixed_mag_offset = self.mechanical_model.magnet_assembly.l_m_mm / 1000
        offset = fixed_mag_offset + l_bth

        # Check if the coils are within bounds
        # Find the top edge of the uppermost coil
        cc = self.electrical_model.coil_config
        coil_top_edge = (
            cc.coil_center_mm / 1000
            + (cc.c - 1) * cc.l_ccd_mm / 1000
            + cc.get_height() / 2
        )

        # Coil's position is relative to the top edge of the fixed magnet, while
        # the height of the device is the absolute height. So we must compensate
        # for this offset
        messages, error_messages = _do_validation(
            bool(coil_top_edge + offset > self.height),
            "Check if coil configuration fits onto device...",
            f"The top edge of the top coil is {(coil_top_edge + offset) * 1000}mm, which exceeds the set device height of {self.height * 1000}mm.",  # type:ignore # noqa
            messages,
            error_messages,
        )

        # Find the top edge of the uppermost magnet assembly
        ma = self.mechanical_model.magnet_assembly
        ms = self.mechanical_model.magnetic_spring
        mag_top_edge = ms.get_hover_height(ma) + ma.get_length() / 1000

        magnet_top_edge_is_outside = bool(
            np.round(mag_top_edge + offset, 3) >= self.height
        )
        messages, error_messages = _do_validation(
            magnet_top_edge_is_outside,
            "Checking if the magnet assembly fits into device...",
            f"The top edge of the magnet assembly is {(mag_top_edge + offset) * 1000}mm, which exceeds the set device height of {self.height * 1000}mm.",  # type:ignore # noqa
            messages,
            error_messages,
        )

        if verbose:
            print("\n".join(messages))

        # If we failed anything, raise a ModelError.
        if len(error_messages) > 1:
            raise ModelError("\n".join(error_messages))

    def _apply_pipeline(self) -> None:
        """Execute the post-processing pipelines on the raw solution.."""
        for _, pipeline in self.post_processing_pipeline.items():
            # raw solution has dimensions d, n rather than n, d
            self.raw_solution = np.array([pipeline(y) for y in self.raw_solution.T]).T

    def _calculate_required_vertical_space(self) -> float:
        """Calculate the vertical space that the microgenerator will require.

        Returns
        -------
        float
            The required vertical height of the device in metres.

        """
        # TODO: Make these overridable from args.
        l_th = 0
        l_bth = 3 / 1000
        l_eps = 5 / 1000
        mag_assembly: magnet_assembly.MagnetAssembly = self.magnet_assembly
        coil_config: CoilConfiguration = self.coil_configuration
        l_hover = self.magnetic_spring.get_hover_height(magnet_assembly=mag_assembly)

        l_L = (
            l_bth
            + l_hover
            + 2 * l_eps
            + 2 * (mag_assembly.l_mcd_mm / 1000) * (mag_assembly.m - 1)
            + 2 * (mag_assembly.m * mag_assembly.l_m_mm / 1000)
            + coil_config.l_ccd_mm / 1000 * (coil_config.c - 1)
            + coil_config.get_height()
            + l_th
        )

        return l_L
