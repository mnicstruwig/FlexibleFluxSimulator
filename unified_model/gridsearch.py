import logging
from collections import defaultdict
from copy import copy
from itertools import product
from typing import (Any, Dict, Generator, List, NamedTuple,
                    Union, Tuple)

import numpy as np
import pandas as pd
import ray

from unified_model import (ElectricalModel, MechanicalModel, UnifiedModel,
                           pipeline)

logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(message)s',
                    level=logging.INFO)


class MechanicalGroundtruth(NamedTuple):
    y_diff: Any
    time: Any


class ElectricalGroundtruth(NamedTuple):
    emf: Any
    time: Any


class Groundtruth(NamedTuple):
    mech: MechanicalGroundtruth
    elec: ElectricalGroundtruth


class AbstractUnifiedModelFactory:
    """An abstract unified model factory that interpolates across all params.

    This abstract factory generates UnifiedModelFactory-ies for each possible
    permutation in `mechanical_components`, `electrical_components`,
    `coupling_models` and `governing_equations`.

    This is a useful property when performing a parameter search using the
    gridsearch method. As such, this class is usually passed as input to the
    `GridSearchBatchExecutor` class.

    Parameters
    ----------
    mechanical_components

    """
    def __init__(self,
                 mechanical_components: List[Any],
                 electrical_components: List[Any],
                 coupling_models: List[Any],
                 governing_equations: List[Any],
                 ) -> None:
        """Constructor"""
        self.mechanical_components = mechanical_components
        self.electrical_components = electrical_components
        self.coupling_models = coupling_models
        self.governing_equations = governing_equations

        self.combined: Dict[Any, Any] = {}
        self.combined.update(mechanical_components)
        self.combined.update(electrical_components)
        self.combined['coupling_model'] = coupling_models
        self.combined['governing_equations'] = self.governing_equations

    def generate(self) -> Generator:
        """Generate a UnifiedModelFactory for a combination of all components

        Each yielded UnifiedModelFactory can produce a UnifiedModel that
        consists of one of all the possible combination of `mechanical`-
        `electrical`-, and `coupling`-models and governing equations specified
        in the constructor.

        Yields
        ------
        UnifiedModelFactory
            A unified model factory containing one combination of parameters.

        """
        param_product = product(*[v for v in self.combined.values()])
        for pp in param_product:
            new_kwargs = {k: v for k, v in zip(self.combined.keys(), pp)}
            yield UnifiedModelFactory(**new_kwargs)


class UnifiedModelFactory:
    """A factory that produces UnifiedModel object.

    Not designed to be used directly -- use the `AbstractUnifiedModelFactory`
    instead.
    """

    def __init__(self,
                 damper: Any = None,
                 magnet_assembly: Any = None,
                 magnetic_spring: Any = None,
                 mechanical_spring: Any = None,
                 input_excitation: Any = None,
                 coil_resistance: Any = None,
                 rectification_drop: Any = None,
                 load_model: Any = None,
                 flux_model: Any = None,
                 dflux_model: Any = None,
                 coupling_model: Any = None,
                 governing_equations: Any = None) -> None:
        """Constructor"""
        self.damper = damper
        self.magnet_assembly = magnet_assembly
        self.magnetic_spring = magnetic_spring
        self.mechanical_spring = mechanical_spring
        self.input_excitation = input_excitation
        self.coil_resistance = coil_resistance
        self.rectification_drop = rectification_drop
        self.load_model = load_model
        self.flux_model = flux_model
        self.dflux_model = dflux_model
        self.coupling_model = coupling_model
        self.governing_equations = governing_equations

    def get_args(self, param_filter: List[str] = None) -> Dict[str, float]:
        """Get arguments args that are be passed to `UnifiedModel`.

        The arguments can be filtered using `param_filter`.

        Parameters
        ----------
        param_filter : list[str]
            A list of "parameter paths" (see example) to retrieve.

        Example
        -------
        >>> umf = UnifiedModelFactory(load_model=SimpleLoad(R=10))
        >>> umf.get_args(['load_model.R])
        {'load_model.R': 10}

        Returns
        -------
        dict
            Dictionary containing all arguments if `param_filter` is not
            specified, or only the arguments specified by `param_filter`.

        """
        if param_filter is None:
            return self.__dict__
        return _get_params_of_interest(self.__dict__, param_filter)

    def make(self) -> UnifiedModel:
        """Make and return a `UnifiedModel`."""
        mechanical_model = (
            MechanicalModel()
            .set_input(self.input_excitation)
            .set_damper(self.damper)
            .set_magnet_assembly(self.magnet_assembly)
            .set_mechanical_spring(self.mechanical_spring)
            .set_magnetic_spring(self.magnetic_spring)
        )
        electrical_model = (
            ElectricalModel()
            .set_coil_resistance(self.coil_resistance)
            .set_rectification_drop(self.rectification_drop)
            .set_load_model(self.load_model)
            .set_flux_model(self.flux_model, self.dflux_model)
        )
        unified_model = (
            UnifiedModel()
            .set_mechanical_model(mechanical_model)
            .set_electrical_model(electrical_model)
            .set_coupling_model(self.coupling_model)
            .set_governing_equations(self.governing_equations)
            .set_post_processing_pipeline(pipeline.clip_x2, name='clip x2')
        )
        return unified_model


def _get_nested_param(obj: Any, path: str) -> Any:
    """Get a parameter from `obj` with a tree-path `path`.

    Works with nested parameters on objects and dicts.

    Parameters
    ----------
    obj : Any
        The target object. Can be a dict or object.
    path : str
        The path to the nested parameter.

    Returns
    -------
    Any
        The value at the end of `path`.

    Example
    -------

    >>> Class A:
    ...       def __init__(self):
    ...       self.x = 'some value'

    >>> Class B:
    ...     def __init__(self):
    ...         self.some_class = A()

    >>> _get_nested_param(B(), 'some_class.x')
    'some value'

    """
    split_ = path.split('.')
    temp = obj[split_[0]]
    for s in split_[1:]:
        if isinstance(temp, dict):  # If we have a dict
            temp = temp[s]
        else:  # If we have an object
            temp = temp.__dict__[s]
    return temp


def _get_params_of_interest(param_dict: Dict[Any, Any],
                            params_of_interest: List[str]) -> Dict[str, Any]:
    """Get a list of parameters from `param_dict`

    Parameters
    ----------
    param_dict : dict
        Dictionary of parameters. Can be nested.
    params_of_interest : List[str]
        List of "parameter paths" to be retrieved.

    Returns
    -------
    Dict[str, Any]
        The value of the parameters specified in `params_of_interest`.

    See Also
    --------
    _get_nested_param : function
        The underlying function that is used.

    """
    if not isinstance(param_dict, dict):
        raise TypeError('param_dict is not a dict')
    if not isinstance(params_of_interest, list):
        raise TypeError('params_of_interest is not a list')

    result: Dict[str, Any] = {}
    for param in params_of_interest:
        result[param] = _get_nested_param(param_dict, param)
    return result


def _parse_score_dict(score_dict: Dict[str, NamedTuple]) -> Dict:
    """Parse a single `score_dict` that is part of a result of a gridsearch.

    Parameters
    ----------
    score_dict : Dict[str, NamedTuple]
        A dict with keys indicating the score category (eg. 'mechanical' or
        'electrical') and values being a `Score` namedtuple containing scores.

    Returns
    -------
    Dict
        A flattened dictionary that contains all the scores from the various
        categories.

    """
    parsed_score = {}
    for category, score_named_tuple in score_dict.items():
        for metric, value in score_named_tuple._asdict().items():
            metric_name = category[:4] + '_' + metric
            parsed_score[metric_name] = value
    return parsed_score


def _scores_to_dataframe(grid_scores: List[Dict]) -> pd.DataFrame:
    """Parse scores from a grid search into a pandas dataframe.

    Parameters
    ----------
    grid_scores : List[Dict]
        The grid scores, which are a list of `score_dict`s where the keys
        indicate the score category, and the values are a `Score` namedtuple.

    Returns
    -------
    pandas dataframe
        Pandas dataframe containing the scores.

    """
    parsed_scores = [_parse_score_dict(score_dict)
                     for score_dict
                     in grid_scores]

    result = defaultdict(list)
    for i, score_dict in enumerate(parsed_scores):
        for key, val in score_dict.items():
            result[key].append(val)

        result['param_set_id'].append(i)
    return pd.DataFrame(result)


def _curves_to_dataframe(grid_curves: List[Dict],
                         sample_rate: int) -> pd.DataFrame:
    """Parse the curve waveforms from a grid search into a pandas dataframe.

    Parameters
    ----------
    grid_curves : List[Dict]
        The grid curves, which are a list of Dicts where the keys indicate
        the waveform, and the values are a List that holds the values.
    sample_rate : int
        The rate at which to *subsample* the curves, in order to save space.

    Returns
    -------
    pandas dataframe
        A pandas dataframe containing the curves.

    """
    result: Dict = defaultdict(lambda: np.empty(0))

    for i, curve_dict in enumerate(grid_curves):
        for key, values in curve_dict.items():
            subsampled_values = values[::sample_rate]
            result[key] = np.concatenate([result[key], subsampled_values])
            values_length = len(subsampled_values)

        param_set_id = np.full(values_length, i)
        result['param_set_id'] = np.concatenate([result['param_set_id'],
                                                 param_set_id])

    return pd.DataFrame(result)


def _param_dict_list_to_dataframe(param_dict_list: List[Dict]) -> pd.DataFrame:
    """Parse the parameter list into a pandas dataframe.

    Useful for parsing a lit of parameter values into a dataframe for later
    analysis.

    Parameters
    ----------
    param_dict_list : List[Dict]
        List of dictionaries whose keys are the name of the parameter to track
        and whose values is the value of the parameter.

    Returns
    -------
    pandas dataframe
        A pandas dataframe containing the parameter values.

    """
    result = defaultdict(list)
    for i, param_dict in enumerate(param_dict_list):
        for key, value in param_dict.items():
            result[key].append(value)
        result['param_set_id'].append(i)

    return pd.DataFrame(result)


def _chunk(array_like: Union[List, np.ndarray],
           chunk_size: int) -> Generator:
    """Chunk up an array-like yielded chunks."""
    total_size = len(array_like)
    indexes = list(range(0, total_size, chunk_size))

    # Make sure we get the final _chunk:
    if indexes[-1] < total_size:
        indexes.append(total_size)

    for start, stop in zip(indexes, indexes[1:]):
        yield array_like[start:stop]


@ray.remote
def run_cell(unified_model_factory: UnifiedModelFactory,
             groundtruth: Groundtruth,
             metrics: Dict[str, Dict]) -> Tuple[Dict, Dict]:
    """Execute a single cell of a grid search.

    This is designed to be executed in parallel using Ray.

    Parameters
    ----------
    unified_model_factory : UnifiedModelFactory
        The unified model factory that is used to make the unified model that
        will be simulated.
    groundtruth : NamedTuple
        A Groundtruth namedtuple that contains the mechanical groundtruth under
        the `MechanicalGroundtruth` attribute and the electrical groundtruth
        under the `ElectricalGroundtruth` attribute.
    metrics : Dict[str, Dict]
        The metrics used to score the unified model. Keys are 'mechanical' and
        `electrical`. The values are Dicts, where the key is the name of the
        metric and the value is the function to compute.

    Returns
    -------
    Tuple[Dict, Dict]
        The scores and respective curves of the unified_model's simulation.

    See Also
    --------
    Ray : library
        The module the we use to execute.
    unified_model.metrics : module
        A module containing a number of metrics that can be used to score the
        model.

    """
    model = unified_model_factory.make()
    model.solve(t_start=0,
                t_end=8,
                t_max_step=1e-3,
                y0=[0.0, 0.0, 0.04, 0.0, 0.0])

    mech_score, mech_eval = model.score_mechanical_model(
        time_target=groundtruth.mech.time,
        y_target=groundtruth.mech.y_diff,
        metrics_dict=metrics['mechanical'],
        prediction_expr='x3-x1',
        return_evaluator=True
    )

    elec_score, elec_eval = model.score_electrical_model(
        time_target=groundtruth.elec.time,
        emf_target=groundtruth.elec.emf,
        metrics_dict=metrics['electrical'],
        prediction_expr='g(t, x5)',
        return_evaluator=True,
        clip_threshold=1e-1
    )

    scores = {'mechanical': mech_score,
              'electrical': elec_score}

    curves = {'time': mech_eval.time_,
              'y_diff': mech_eval.y_predict_,
              'emf': elec_eval.emf_predict_}

    return scores, curves


class GridsearchBatchExecutor:
    """Execute a batch grid search using Ray.

    Parameters
    ----------
    abstract_unified_model_factory : AbstractUnifiedModelFactory
        An abstract factory that produces unified model factories. One
        simulation will be run for each of the factories produces by
        abstract_unified_model_factory.
    groundtruth : Groundtruth
        Groundtruth used to as a basis for scoring each unified model.
    metrics : Dict[str, Dict]
        The metrics used to score the unified model. Keys are 'mechanical' and
        `electrical`. The values are Dicts, where the key is the name of the
        metric and the value is the function to compute.
    parameters_to_track : List[str]
        A list of "parameter paths" to be tracked for each unified model. For
        example, `['load_model.R', 'damper.damping_coefficient]`.

    """
    def __init__(self,
                 abstract_unified_model_factory: AbstractUnifiedModelFactory,
                 groundtruth: Groundtruth,
                 metrics: Dict[str, Dict],
                 parameters_to_track: List[str],
                 **ray_kwargs) -> None:
        """Constructor"""

        self.metrics = metrics
        self.abstract_unified_model_factory = abstract_unified_model_factory
        self.groundtruth = groundtruth
        self.parameters_to_track = parameters_to_track
        ray_kwargs.setdefault('ignore_reinit_error', True)
        self.ray_kwargs = ray_kwargs
        self.raw_grid_results = None
        self.result = None

    def _start_ray(self, ray_init_kwargs: Dict = None) -> None:
        """Initialize Ray"""
        if ray_init_kwargs is None:
            ray_init_kwargs = {}
        ray.init(**ray_init_kwargs)

    def _kill_ray(self) -> None:
        "Kill Ray."
        ray.shutdown()

    def _execute_grid_search(self, batch_size: int = 8) -> Tuple:
        """Execute the gridsearch."""

        # This doesn't make me happy, but I'm out of clean ideas for now If I
        # want this to work and _also_ preserve order.
        model_factories = list(self.abstract_unified_model_factory.generate())
        total_completed = 0
        total_tasks = len(model_factories)

        grid_scores: List[Dict] = []
        grid_curves: List[Dict] = []
        grid_params: List[Dict] = []

        for model_factory_batch in _chunk(model_factories, batch_size):
            task_queue = []
            for model_factory in model_factory_batch:
                grid_params.append(model_factory.get_args(
                    self.parameters_to_track
                ))
                task_id = run_cell.remote(model_factory,
                                          self.groundtruth,
                                          self.metrics)
                task_queue.append(task_id)

            ready: List[Any] = []
            while len(ready) < len(task_queue):
                ready, remaining = ray.wait(task_queue,
                                            num_returns=len(task_queue),
                                            timeout=5.)
                logging.info(f'Progress: {len(ready)+total_completed}/{total_tasks}')  # noqa

            # Once all tasks are completed...
            total_completed += len(ready)  # noqa: increment the total completed counter
            results = [ray.get(task_id) for task_id in task_queue]  # noqa: ... and fetch results

            # Parse the results
            for result in results:
                grid_scores.append(copy(result[0]))
                grid_curves.append(copy(result[1]))

            del results  # Remove reference so Ray can free memory as needed
        return grid_scores, grid_curves, grid_params

    def _process_results(self,
                         grid_results: Tuple[List, List, List],
                         curve_subsample_rate: int = 3) -> pd.DataFrame:
        """Process the gridsearch results into a single pandas Dataframe."""
        grid_scores, grid_curves, grid_params = grid_results

        # Some basic validation
        assert(len(grid_scores) == len(grid_curves))
        assert(len(grid_scores) == len(grid_params))

        df_params = _param_dict_list_to_dataframe(grid_params)
        df_scores = _scores_to_dataframe(grid_scores)
        df_curves = _curves_to_dataframe(grid_curves,
                                         sample_rate=curve_subsample_rate)

        # Merge dataframes
        result = df_scores.merge(df_params, on='param_set_id')
        result = result.merge(df_curves, on='param_set_id')

        return result

    def run(self, batch_size: int = 8) -> pd.DataFrame:
        """Run the grid search and returns the results.

        Parameters
        ----------
        batch_size : int
            The number of batches to use.
            Default is 8.

        Returns
        -------
        pandas dataframe
            Pandas dataframe containing the scores, curves and tracked
            parameters.

        """
        logging.info('Starting Ray...')
        self._start_ray(self.ray_kwargs)
        logging.info('Running grid search...')
        self.raw_grid_result = self._execute_grid_search(batch_size=batch_size)
        self.result = self._process_results(self.raw_grid_result)
        logging.info('Gridsearch complete. Shutting down Ray...')
        self._kill_ray()
        return self.result

    def save(self, path: str) -> None:
        """Save the result to disk in parquet format.

        Parameters
        ----------
        path : str
            The file path to save the results to.

        """
        if self.result is None:
            raise ValueError('Nothing to save. Have you called `run` yet?')
        logging.info(f'Saving to result to file {path}')
        self.result.to_parquet(path)
        logging.info('Save complete!')
