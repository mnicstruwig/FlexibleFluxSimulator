import pandas as pd
import numpy as np


def _parse_raw_flux_input(raw_flux_input):
    if isinstance(raw_flux_input, str):
        return pd.read_csv(raw_flux_input)
    if isinstance(raw_flux_input, pd.DataFrame):
        return raw_flux_input


def _extract_parameter_from_str(str_):
    split_ = str_.split()
    unprocessed_params = [s for s in split_ if '=' in s]
    param_names = [param.split('=')[0] for param in unprocessed_params]
    param_values = [param.split('=')[1].replace("'", '') for param in unprocessed_params]

    param_dict = {}
    for name, value in zip(param_names, param_values):
        param_dict[name] = value
    return param_dict


class FluxDatabase(object):
    """Convert table produced by Maxwell parametric simulation into flux database"""
    def __init__(self, csv_database_path, fixed_velocity):
        self.raw_database = pd.read_csv(csv_database_path)
        self.velocity = fixed_velocity
        self.params = None
        self.lut = None
        self.database = {}

        self._produce_database()

    def _produce_database(self):
        """Build the flux database."""
        self.time = self.raw_database.iloc[:, 0].values/1000
        self.z = self.time * self.velocity
        self._create_index(_extract_parameter_from_str(self.raw_database.columns[1]).keys())
        for col in self.raw_database.columns[1:]:  # First column is time information
            key_dict = _extract_parameter_from_str(col)
            self.add(key_dict, value=self.raw_database[col].values)

    def _build_db_key(self, **kwargs):
        """Build a database key using the internal look-up table."""
        db_key = [None]*len(self.lut)
        for key in kwargs:
            db_key[self.lut[key]] = kwargs[key]
        if None in db_key:
            raise KeyError('Not all keys specified')
        return tuple(db_key)

    def add(self, key_dict, value):
        """Add an entry to the database."""
        db_key = self._build_db_key(**key_dict)
        self.database[db_key] = value

    def query(self, **kwargs):
        db_key = self._build_db_key(**kwargs)
        return self.database[db_key]

    def _create_index(self, key_list):
        if self.lut is None:
            self.lut = {}
            for i, k in enumerate(key_list):
                self.lut[k] = i
        else:
            raise ValueError('Index cannot be created more than once.')

    def itervalues(self):
        for key, value in self.database.items():
            yield key, value
