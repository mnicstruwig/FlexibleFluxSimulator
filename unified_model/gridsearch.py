import logging
import warnings
from collections import defaultdict
from copy import copy
from itertools import product
from typing import (Any, Dict, Generator, List, NamedTuple,
                    Union, Tuple, Callable)

import numpy as np
import pandas as pd
import ray
import pyarrow.parquet as pq
import pyarrow as pa

from unified_model import (ElectricalModel, MechanicalModel, UnifiedModel,
                           pipeline)
from unified_model.mechanical_components import AccelerometerInput

logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(message)s',
                    level=logging.INFO)


class EvaluatorFactory:

    def __init__(self,
                 evaluator_cls: Any,
                 expr_targets: List,
                 time_targets: List,
                 metrics: Dict[str, Callable],
                 **kwargs) -> None:
        """Build a factory that produces fitted Evaluators

        Parameters
        ----------
        evaluator_cls : Any
            The class of evaluator to construct.
        expr_targets : List
            Each element in the list must contain the "target" values.
        time_targets : List
            Each element in the list must contain the corresponding time values
            of the target values.
        metrics : Dict[str, Callable]
            A dictionary where the key is a name given to the metric. The
            values are Callables that accept two array-likes (the first being
            target data, the second being predicted data) and returns a scalar
            representing the metric.
        **kwargs :
            Additional kwargs passed through to the `evaulator_cls`.

        See Also
        ---------
        unified_model.evaluate : module
            Module that houses the evaluator classes.
        unified_model.metrics : module
            Module that houses a lot of useful metrics.
        """

        # Do some validation
        assert len(expr_targets) == len(time_targets)

        self.evaluator_cls = evaluator_cls
        self.expr_targets = expr_targets
        self.time_targets = time_targets
        self.metrics = metrics
        self.evaluator_kwargs = kwargs

    def make(self) -> np.ndarray:
        """Return an array of Evaluator objects."""
        evaluator_list = []
        for expr_target, time_target in zip(self.expr_targets, self.time_targets):  # noqa
            evaluator = self.evaluator_cls(expr_target,
                                           time_target,
                                           self.metrics,
                                           **self.evaluator_kwargs)
            evaluator_list.append(evaluator)
        return np.array(evaluator_list)

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

        if 'input_excitation' in mechanical_components:
            warnings.warn('input_excitation was specified in mechanical_components! This will break when `generate` is called!')  # noqa

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
        for i, pp in enumerate(param_product):
            new_kwargs = {k: v for k, v in zip(self.combined.keys(), pp)}
            new_kwargs['model_id'] = i
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
                 coil_resistance: Any = None,
                 rectification_drop: Any = None,
                 load_model: Any = None,
                 flux_model: Any = None,
                 dflux_model: Any = None,
                 coupling_model: Any = None,
                 governing_equations: Any = None,
                 model_id: int = None) -> None:
        """Constructor"""
        self.damper = damper
        self.magnet_assembly = magnet_assembly
        self.magnetic_spring = magnetic_spring
        self.mechanical_spring = mechanical_spring
        self.coil_resistance = coil_resistance
        self.rectification_drop = rectification_drop
        self.load_model = load_model
        self.flux_model = flux_model
        self.dflux_model = dflux_model
        self.coupling_model = coupling_model
        self.governing_equations = governing_equations
        self.model_id = model_id  # <-- used to keep track of a set of parameters

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

    def make(self, input_excitation: AccelerometerInput) -> UnifiedModel:
        """Make and return a `UnifiedModel`."""
        mechanical_model = (
            MechanicalModel()
            .set_input(input_excitation)  # <-- set from parameter!
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


def _scores_to_dataframe(grid_scores: List[Dict[str, Any]],
                         model_ids: List[int]) -> pd.DataFrame:  # TODO: Docstring
    """Parse scores from a grid search into a pandas dataframe.

    Parameters
    ----------
    grid_scores : List[Dict[str, Any]]
        The grid scores, which are a list of `score_dict`s where the keys
        indicate the score name, and the values are the calculated score.

    Returns
    -------
    pandas dataframe
        Pandas dataframe containing the scores.

    """
    if len(grid_scores[0]) == 0:
        return None

    result = defaultdict(list)
    for i, score_dict in enumerate(grid_scores):
        for key, val in score_dict.items():
            result[key].append(val)

        result['grid_cell_id'].append(i)
        result['model_id'].append(model_ids[i])
    return pd.DataFrame(result)


def _calc_metrics_to_dataframe(
        grid_calcs: List[Dict[str, Any]],
        model_ids: List[int]  # TODO: Docstring
) -> pd.DataFrame:
    """Parse calculated metrics from a grid search into a pandas dataframe.

    Parameters
    ----------
    grid_scores : List[Dict[str, Any]]
        The grid's calculated metrics, which are a list of dicts, where the key
        is the name of the calculated metric and the values are the calculated
        metric.

    Returns
    -------
    pandas dataframe
        Pandas dataframe containing the calculated metrics.

    """
    if len(grid_calcs[0]) == 0:
        return None
    result = defaultdict(list)
    for i, calc_dict in enumerate(grid_calcs):
        for key, val in calc_dict.items():
            result[key].append(val)

        result['grid_cell_id'].append(i)
        result['model_id'].append(model_ids[i])
    return pd.DataFrame(result)


def _curves_to_dataframe(grid_curves: List[Dict[str, Any]],
                         sample_rate: int,
                         model_ids: List[int]) -> pd.DataFrame:
    """Parse the curve waveforms from a grid search into a pandas dataframe.

    Parameters
    ----------
    grid_curves : List[Dict[str, Any]]
        The grid curves, which are a list of Dicts where the keys indicate
        the waveform, and the values are a List that holds the values.
    sample_rate : int
        The rate at which to *subsample* the curves, in order to save space.

    Returns
    -------
    pandas dataframe
        A pandas dataframe containing the curves.

    """
    if len(grid_curves[0]) == 0:
        return None

    result: Dict = defaultdict(lambda: np.empty(0))

    for i, curve_dict in enumerate(grid_curves):
        for key, values in curve_dict.items():
            subsampled_values = values[::sample_rate]
            result[key] = np.concatenate([result[key], subsampled_values])
            values_length = len(subsampled_values)

        grid_cell_id = np.full(values_length, i, dtype=int)
        model_id = np.full(values_length, model_ids[i], dtype=int)

        result['grid_cell_id'] = np.concatenate([result['grid_cell_id'],
                                                 grid_cell_id])
        result['model_id'] = np.concatenate([result['model_id'], model_id])

    df = pd.DataFrame(result)
    df['model_id'] = df['model_id'].astype('int')  # Force `model_id` to be int
    return df


def _param_dict_list_to_dataframe(param_dict_list: List[Dict],
                                  model_ids: List[int],
                                  input_excitations: List[int]) -> pd.DataFrame:
    """Parse the parameter list into a pandas dataframe.

    Useful for parsing a list of parameter values into a dataframe for later
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
    if len(param_dict_list[0]) == 0:
        return None

    result = defaultdict(list)
    for i, param_dict in enumerate(param_dict_list):
        for key, value in param_dict.items():
            result[key].append(value)

        result['grid_cell_id'].append(i)
        result['model_id'].append(model_ids[i])
        # TODO: A bit of a hack. Find another place to append the input excitations.
        result['input_excitation'].append(input_excitations[i])

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
             input_excitation: AccelerometerInput,
             curve_expressions: Dict[str, str] = None,
             score_metrics: Dict[str, Any] = None,
             calc_metrics: Dict[str, Callable] = None) -> Tuple[Dict, Dict, Dict]:  # noqa
    """Execute a single cell of a grid search.

    This is designed to be executed in parallel using Ray.

    Parameters
    ----------
    unified_model_factory : UnifiedModelFactory
        The unified model factory that is used to make the unified model that
        will be simulated.
    curve_expressions : Dict[str, str]
        The expressions of the curves to return of the unified model after
        simulation. Each key is the string prediction expression. Each value is
        the name given to the curve.
    score_metrics : Dict[str, Dict[str, Callable]]
        The metrics used to score the unified model. Keys are 'mechanical' and
        `electrical`. The values are Dicts, where the key is the name of the
        metric and the value is the function to compute.
    calc_metrics : Dict[str, Dict[str, Callable]]
       The metrics calculated on the results of the unified model. Keys are the
       prediction expression of the result to be calculated, and the value is a
       Dict whose key is the name given to the metric, and whose value is a
       function that calculates the metric. This function must accept a numpy
       array as input.

    Returns
    -------
    Tuple[Dict, Dict]
        The scores and respective curves of the unified_model's simulation.

    See Also
    --------
    Ray : library
        The module the we use to execute.
    unified_model.get_result : method
        Method used to parse the curve expressions.
    unified_model.metrics : module
        A module containing a number of metrics that can be used to score the
        model.
    UnifiedModel.calculate_metrics : method
        The method used to evaluate the `calc_metrics` parameter.

    """

    model = unified_model_factory.make(input_excitation)
    model.solve(t_start=0,
                t_end=8,
                t_max_step=1e-3,
                y0=[0.0, 0.0, 0.04, 0.0, 0.0])

    curves: Dict[str, Any] = {}
    metric_scores: Dict[str, Any] = {}
    metric_calcs: Dict[str, Any] = {}

    if curve_expressions:
        # We need to do a key-value swap to match the `.get_result` method
        # interface
        swapped = {v: k for k, v in curve_expressions.items()}
        df_result = model.get_result(**swapped)
        curves = df_result.to_dict(orient='list')

    if score_metrics:
        expression_kwargs = {}
        for i, expression in enumerate(score_metrics.keys()):
            expression_kwargs[str(i)] = expression
        df_result = model.get_result(time='t', **expression_kwargs)

        # Ok, let's calculate the score metrics
        metric_scores = {}
        for i, (_, evaluator) in enumerate(score_metrics.items()):
            evaluator.fit(df_result[str(i)].values, df_result['time'].values)
            score: Dict[str, Any] = evaluator.score()  # type: ignore
            metric_scores.update(score)  # Score and update the table

    if calc_metrics:  # TODO: Convert this to helper function (DRY)
        metric_calcs = {}
        for expression, metric_dict in calc_metrics.items():
            # We use the model's helper function to calculate the metrics
            result = model.calculate_metrics(expression, metric_dict)
            metric_calcs.update(result)

    return curves, metric_scores, metric_calcs


class GridsearchBatchExecutor:
    """Execute a batch grid search using Ray.

    Parameters
    ----------
    abstract_unified_model_factory : AbstractUnifiedModelFactory
        An abstract factory that produces unified model factories. One
        simulation will be run for each of the factories produces by
        abstract_unified_model_factory.
    curve_expressions : Dict[str, str]
        The curves to capture from the simulated unified model. Keys are a
        prediction expression, and values are the name given to the curve.
    score_metrics : Dict[str, Any]
        Metrics used to score the unified model's predictions. Keys are the
        prediction expressions to be evaluated. Values are the instantiated
        Evaluator objects that will score the prediction expression at its key.
    calc_metrics : Dict[str, Callable]
        Additional metrics to be calculated. Keys are the prediction
        expressions to be evaluated. Values are Callables that will calculate
        the metric using the prediction expressions at its key.
    parameters_to_track : List[str]
        A list of "parameter paths" to be tracked for each unified model. For
        example, `['load_model.R', 'damper.damping_coefficient]`.
    **ray_kwargs :
        Additional parameters to be passed to `ray.init`.

    """
    def __init__(self,
                 abstract_unified_model_factory: AbstractUnifiedModelFactory,
                 input_excitations: List[AccelerometerInput],  # TODO: Docstring
                 curve_expressions: Dict[str, str] = None,
                 score_metrics: Dict[str, List[Any]] = None,
                 calc_metrics: Dict[str, Callable] = None,
                 parameters_to_track: List[str] = None,
                 **ray_kwargs) -> None:
        """Constructor"""

        if score_metrics:
            try:
                for _, v in score_metrics.items():
                    assert len(input_excitations) == len(v)
            except AssertionError:
                raise AssertionError('len(input excitations) != len(score_metrics[x]) ')  # noqa

        self.input_excitations = input_excitations
        self.curve_expressions = curve_expressions
        self.score_metrics = score_metrics
        self.calc_metrics = calc_metrics
        self.abstract_unified_model_factory = abstract_unified_model_factory
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

    def _execute_grid_search(
            self,
            output_file: str,
            batch_size: int = 8
    ):
        """Execute the gridsearch."""

        # Convert to list since we may iterate over it multiple times
        model_factories = list(self.abstract_unified_model_factory.generate())
        total_tasks = len(model_factories)

        for input_number, input_ in enumerate(self.input_excitations):


            # Build score metric that needs to be calculated for the input
            # excitation. Remember: each groundtruth evaluator is directly
            # linked to only one input!
            linked_score_metric = None
            if self.score_metrics:
                # Fetch the evaluator with the correct groundtruth data
                linked_score_metric = {k: evaluators[input_number]
                                       for k, evaluators
                                       in self.score_metrics.items()}

            total_completed = 0
            for model_factory_batch in _chunk(model_factories, batch_size):

                grid_curves: List[Dict] = []
                grid_scores: List[Dict] = []
                grid_calcs: List[Dict] = []
                grid_params: List[Dict] = []
                model_ids: List[int] = []
                input_numbers: List[int] = []

                task_queue = []
                for model_factory in model_factory_batch:
                    # Get the model id
                    model_ids.append(model_factory.model_id)
                    # Record grid parameters
                    grid_params.append(model_factory.get_args(self.parameters_to_track))  # noqa
                    # Record which input excitation
                    input_numbers.append(input_number)

                    # Queue a simulation
                    task_id = run_cell.remote(model_factory,
                                              input_excitation=input_,
                                              curve_expressions=self.curve_expressions,  # noqa
                                              score_metrics=linked_score_metric,
                                              calc_metrics=self.calc_metrics)

                    task_queue.append(task_id)

                ready: List[Any] = []
                while len(ready) < len(task_queue):
                    ready, remaining = ray.wait(task_queue,
                                                num_returns=len(task_queue),
                                                timeout=1.)
                    # Log output
                    log_base = 'Progress: '
                    input_log = f':: Input: {input_number+1}/{len(self.input_excitations)} '  # noqa
                    grid_progress_log = f':: {len(ready)+total_completed}/{total_tasks}'  # noqa
                    logging.info(log_base + input_log + grid_progress_log)

                # Once all tasks are completed...
                total_completed += len(ready)  # noqa: increment the total completed counter
                results = [ray.get(task_id) for task_id in task_queue]  # noqa: ... and fetch results

                # Parse the results
                for result in results:
                    grid_curves.append(copy(result[0]))
                    grid_scores.append(copy(result[1]))
                    grid_calcs.append(copy(result[2]))

                # Process results to dataframe
                df_results = self._process_results(
                    grid_results=(grid_curves,
                                  grid_scores,
                                  grid_calcs,
                                  grid_params,
                                  model_ids,
                                  input_numbers)
                )

                # Write out results to file
                logging.info(f'Writing chunk to :: {output_file} ...')
                self._write_out_results(df_results, output_file, ['input_excitation'])
                del results  # Remove reference so Ray can free memory as needed
                ray.internal.free(task_queue)

    def _write_out_results(self,
                           df_results,
                           path,
                           partition_cols):
        table = pa.Table.from_pandas(df_results)
        pq.write_to_dataset(table, path, partition_cols=partition_cols, compression='brotli')

    def _process_results(self,
                         grid_results: Tuple[List, List, List, List, List, List],
                         curve_subsample_rate: int = 3) -> pd.DataFrame:
        """Process the gridsearch results into a single pandas Dataframe."""
        grid_curves, grid_scores, grid_calcs, grid_params, model_ids, input_excitations = grid_results  # noqa

        # Some basic validation
        assert(len(grid_calcs) == len(grid_curves))
        assert(len(grid_scores) == len(grid_curves))
        assert(len(grid_scores) == len(grid_params))

        df_curves = _curves_to_dataframe(grid_curves,
                                         sample_rate=curve_subsample_rate,
                                         model_ids=model_ids)
        df_scores = _scores_to_dataframe(grid_scores, model_ids=model_ids)
        df_calcs = _calc_metrics_to_dataframe(grid_calcs, model_ids=model_ids)
        df_params = _param_dict_list_to_dataframe(grid_params,
                                                  model_ids=model_ids,
                                                  input_excitations=input_excitations)

        # Merge dataframes
        results = []
        result: pd.DataFrame = None
        for df in [df_curves, df_scores, df_calcs, df_params]:
            # Get the first defined dataframe
            if result is None:
                if df is not None:
                    result = df

            else:  # If we have our first defined dataframe ...
                if df is not None:  # ... and the next one is defined ...
                    result = result.merge(df, on=['grid_cell_id', 'model_id'])  # noqa ...merge

        results.append(result)

        return pd.concat(results)

    def _print_dict(self, dict_):
        if dict_ is not None:
            for key, value in dict_.items():
                print(f'{key} --> {value}')
        else:
            print('None')

    def _print_list(self, list_: List[Any]):
        for x in list_:
            print(x)

    def preview(self) -> None:
        """Preview the gridsearch."""

        factory_copy = copy(self.abstract_unified_model_factory)
        num_models = len(list(factory_copy.generate()))

        print('Gridsearch Preview')
        print('==================')
        print()

        print('Number of inputs:')
        print('----------------')
        print(f'Total # inputs --> {len(self.input_excitations)}')
        print()

        print('Tracking the following parameters:')
        print('----------------------------------')
        if self.parameters_to_track:
            self._print_list(self.parameters_to_track)
        print()

        print('Saving the following curves:')
        print('----------------------------')
        self._print_dict(self.curve_expressions)
        print()

        print('Scoring on the following expressions:')
        print('---------------------------------------------')
        if self.score_metrics:
            self._print_list(list(self.score_metrics.keys()))
        print()

        print('Calculating the following metrics:')
        print('----------------------------------')
        self._print_dict(self.calc_metrics)
        print()

        print('Model parameters:')
        print('---------------------------')

        print(f'Total # models per input --> {num_models}')
        print()
        for param_path, param_values_list in factory_copy.combined.items():
            print(f'{param_path} --> number: {len(param_values_list)}')
        print()
        print('==================')
        print(f'Total # simulations --> {num_models*len(self.input_excitations)}')  # noqa
        print('==================')

    def run(self, output_file, batch_size: int = 8) -> pd.DataFrame:
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
        self._execute_grid_search(output_file, batch_size=batch_size)
        logging.info('Gridsearch complete. Shutting down Ray...')
        self._kill_ray()

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
        self.result.to_parquet(path, engine='pyarrow', compression='brotli')
        logging.info('Save complete!')
