import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score, median_absolute_error
from tqdm import tqdm

import warnings

from unified_model.coupling import ConstantCoupling
from unified_model.electrical_model import ElectricalModel
from unified_model.electrical_system.flux.utils import FluxDatabase
from unified_model.electrical_system.load import SimpleLoad
from unified_model.governing_equations import unified_ode
from unified_model.mechanical_model import MechanicalModel
from unified_model.mechanical_system.damper import Damper
from unified_model.mechanical_system.input_excitation.accelerometer import AccelerometerInput
from unified_model.mechanical_system.magnet_assembly import MagnetAssembly
from unified_model.mechanical_system.spring.magnetic_spring import \
    MagneticSpring
from unified_model.unified import UnifiedModel
from unified_model.evaluate import ElectricalSystemEvaluator, MechanicalSystemEvaluator, LabeledVideoProcessor, AdcProcessor
from unified_model.pipeline import clip_x2
from unified_model.utils.utils import collect_samples
from unified_model.metrics import max_err, mean_absolute_percentage_err, corr_coeff, root_mean_square, root_mean_square_percentage_diff

warnings.simplefilter('ignore', FutureWarning)

# Constants
coil_center = {'A': 59/1000,
               'B': 61/1000,
               'C': 63/1000}

winding_num_z = {'A': '17',
                 'B': '33',
                 'C': '66'}

winding_num_r = {'A': '15',
                 'B': '15',
                 'C': '5'}

coil_height = {'A': '0.008meter',
               'B': '0.012meter',
               'C': '0.014meter'}

# Path handling
fea_data_path = './unified_model/mechanical_system/spring/data/10x10alt.csv'

# MECHANICAL MODEL
# Note: The accelerometer input is deliberately left out (it is added in later)
spring = MagneticSpring(fea_data_file=fea_data_path,
                        model='savgol_smoothing',
                        model_type='interp')

magnet_assembly = MagnetAssembly(n_magnet=1,
                                 h_magnet=10,
                                 h_spacer=0,
                                 dia_magnet=10,
                                 dia_spacer=10,
                                 mat_magnet='NdFeB',
                                 mat_spacer='iron')

damper = Damper(model='constant', model_kwargs={'damping_coefficient': 0.05})
mechanical_model = MechanicalModel(name='mech_system')
mechanical_model.set_magnetic_spring(spring)
mechanical_model.set_magnet_assembly(magnet_assembly)
mechanical_model.set_damper(damper)

# ELECTRICAL MODEL
flux_database = FluxDatabase(csv_database_path='/home/michael/Dropbox/PhD/Python/Research/fea-flux-curves-numr[5,15]-numz[17,33,66]-wdiam[0.15]-cheight[8,12,14]-2019-04-11.csv', fixed_velocity=0.35)

flux_models = {}
for device in ['A', 'B', 'C']:
    flux_models[device] = flux_database.query_to_model(flux_model_type='unispline',
                                                       coil_center=coil_center[device],
                                                       mm=10,
                                                       winding_num_z=winding_num_z[device],
                                                       winding_num_r=winding_num_r[device],
                                                       coil_height=coil_height[device])

# LOAD
load_model = SimpleLoad(np.inf)

electrical_model = ElectricalModel(name='elec_system')
electrical_model.set_load_model(load_model)

# COUPLING MODEL
coupling_model = ConstantCoupling(c=0)

# SYSTEM MODEL
governing_equations = unified_ode

# Ground-truth files
base_groundtruth_path = './experiments/data/2018-12-20/'

a_samples = collect_samples(base_path=base_groundtruth_path,
                            acc_pattern='A/*acc*.csv',
                            adc_pattern='A/*adc*.csv',
                            labeled_video_pattern='A/*labels*.csv')

b_samples = collect_samples(base_path=base_groundtruth_path,
                            acc_pattern='B/*acc*.csv',
                            adc_pattern='B/*adc*.csv',
                            labeled_video_pattern='B/*labels*.csv')

c_samples = collect_samples(base_path=base_groundtruth_path,
                            acc_pattern='C/*acc*.csv',
                            adc_pattern='C/*adc*.csv',
                            labeled_video_pattern='C/*labels*.csv')
samples_dict = {'A': a_samples,
                'B': b_samples,
                'C': c_samples}


def _build_quick_unified_model(accelerometer_input, flux_model):
    """Build a unified model quickly with some hard-coded values. Debugging only."""
    mechanical_model.set_input(accelerometer_input)
    electrical_model.set_flux_model(flux_model)

    # Build unified model from components
    unified_model = UnifiedModel(name='Unified')
    unified_model.add_mechanical_model(mechanical_model)
    unified_model.add_electrical_model(electrical_model)
    unified_model.add_coupling_model(coupling_model)
    unified_model.add_governing_equations(governing_equations)
    unified_model.add_post_processing_pipeline(clip_x2, name='clip tube velocity')

    return unified_model


# BATCH SOLVE
mechanical_evals = {'A': [],
                    'B': [],
                    'C': []}

electrical_evals = {'A': [],
                    'B': [],
                    'C': []}

mechanical_scores = {'A': [],
                     'B': [],
                     'C': []}

electrical_scores = {'A': [],
                     'B': [],
                     'C': []}

unified_models = {'A': [],
                  'B': [],
                  'C': []}

for sample_collection in ['A']:
    for i, sample in enumerate(tqdm(samples_dict[sample_collection])):

        # Set Accelerometer
        accelerometer = AccelerometerInput(sample.acc_df,
                                           accel_column='z_G',
                                           time_column='time(ms)',
                                           smooth=True,
                                           interpolate=True)

        flux_model = flux_models[sample_collection]

        # The accelerometer and flux model need to be set for each experiment
        mechanical_model.set_input(accelerometer)
        electrical_model.set_flux_model(flux_model)

        # Build unified model from components
        unified_model = UnifiedModel(name='Unified')
        unified_model.add_mechanical_model(mechanical_model)
        unified_model.add_electrical_model(electrical_model)
        unified_model.add_coupling_model(coupling_model)
        unified_model.add_governing_equations(governing_equations)
        unified_model.add_post_processing_pipeline(clip_x2, name='clip tube velocity')

        # Solve
        initial_conditions = [0, 0, 0.04, 0, 0]
        unified_model.solve(t_start=0, t_end=15,
                            y0=initial_conditions,
                            t_max_step=1e-3)

        # Mechanical scoring
        pixel_scale = 0.18745
        labeled_video_processor = LabeledVideoProcessor(L=125,
                                                        mm=10,
                                                        seconds_per_frame=3/240,
                                                        pixel_scale=pixel_scale)

        mechanical_metrics = {'mde': median_absolute_error,
                              'mape': mean_absolute_percentage_err,
                              'max': max_err}

        mech_scores, m_eval = unified_model.score_mechanical_model(metrics_dict=mechanical_metrics,
                                                                   video_labels_df=sample.video_labels_df,
                                                                   labeled_video_processor=labeled_video_processor,
                                                                   prediction_expr='x3-x1',
                                                                   return_evaluator=True)
        mechanical_scores[sample_collection].append(mech_scores)

        # EMF scoring
        voltage_division_ratio = 1/0.342
        adc_processor = AdcProcessor(voltage_division_ratio=voltage_division_ratio,
                                     smooth=True,
                                     critical_frequency=1 / 8)

        electrical_metrics = {'rms': root_mean_square,
                              'rms_err_perc': root_mean_square_percentage_diff}

        elec_scores, e_eval = unified_model.score_electrical_model(metrics_dict=electrical_metrics,
                                                                   adc_df=sample.adc_df,
                                                                   adc_processor=adc_processor,
                                                                   prediction_expr='g(t, x5)',
                                                                   return_evaluator=True)
        electrical_scores[sample_collection].append(elec_scores)

        # Meta
        electrical_evals[sample_collection].append(e_eval)
        mechanical_evals[sample_collection].append(m_eval)
        unified_models[sample_collection].append(unified_model)


def pretty_print_scores(score_dict):
    for key, score_collection in score_dict.items():
        print(key)
        for score in score_collection:
            print(score)

pretty_print_scores(mechanical_scores)
pretty_print_scores(electrical_scores)
