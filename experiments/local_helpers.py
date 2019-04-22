import numpy as np
from unified_model.electrical_system.flux.utils import FluxDatabase


def get_flux_models_from_db(flux_db,
                            reference_keys,
                            coil_centers,
                            mm,
                            **query_kwargs):
    """Get flux models from a flux database.

    Each argument (besides `flux_db`) is a list of values. The ith element of
    each list is used to create each flux model. The flux models are returned
    as a dictionary, whose keys specified by `reference_keys`.

    Parameters
    ----------
    flux_db : FluxDatabase
        Instantiated `FluxDatabase` object. This object is queried for the
        various models.
    reference_keys : list(str)
        User-selectable keys to use as reference for the returned dictionary
        that contains the flux models.
    coil_centers : list(float)
        The coil centers where the flux curve should be centered around
        for each flux model.
    mm : list(float)
        The magnet height for each flux model.
    query_kwargs :
        The query keyword arguments that will be passed to the `query_to_model`
        method of `flux_db`. These keyword arguments must correspond to the
        same keyword arguments that were used to reference each model when it
        was added to `flux_db`.

    Returns
    -------
    dict
        Dictionary containing each flux model. The models can be accessed
        using the keys specified in `reference_keys`.

    See Also
    --------
    unified_model.electrical_system.flux.utils.FluxDatabase
        The flux database class that `flux_db` should be. This is the object
        that is queried for the flux models.
    unified_model.electrical_system.flux.utils.FluxDatabase.query_to_model
        Method used to query `flux_db`.

    Examples
    --------
    >>> flux_db = FluxDatabase(csv_database_path='myfluxdatabasse.csv',
    ...                        fixed_velocity=0.35)
    >>> get_flux_models(flux_database,
    ...                 ['A', 'B'],
    ...                 coil_centers=[59/1000, 61/1000],
    ...                 mm=[10, 10],
    ...                 winding_num_z=['17', '33'],
    ...                 winding_num_r=['15', '15'],
    ...                 coil_height=['0.008meter', '0.012meter'])
    {'B': [<scipy.interpolate.fitpack2.InterpolatedUnivariateSpline at 0x7f74668fb7f0>],
    'A': [<scipy.interpolate.fitpack2.InterpolatedUnivariateSpline at 0x7f74668fb710>]}

    """

    try:
        assert(len(set(len(x) for x in [reference_keys, coil_centers, mm, *query_kwargs.values()])) <= 1)
    except AssertionError:
        raise AssertionError('Not all elements have the same length.')

    flux_models = {}
    for i, ref_key in enumerate(reference_keys):
        # Extract flux_db query parameters
        query = {key: value[i] for key, value in query_kwargs.items()}

        model = flux_db.query_to_model(flux_model_type='unispline',
                                       coil_center=coil_centers[i],
                                       mm=mm[i],
                                       **query)

        flux_models[ref_key] = model
    return flux_models
