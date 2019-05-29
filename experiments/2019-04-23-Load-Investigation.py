
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
from unified_model.pipeline import clip_x2
from unified_model.utils.utils import collect_samples
from experiments.local_helpers import get_flux_models_from_db


warnings.simplefilter('ignore', FutureWarning)


def pretty_print_scores(score_dict):
    for key, score_collection in score_dict.items():
        print(key)
        for score in score_collection:
            print(score)


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


# Path handling
mag_spring_data_path = './unified_model/mechanical_system/spring/data/10x10alt.csv'

# Note: The accelerometer input is deliberately left out (it is added in later)
spring = MagneticSpring(fea_data_file=mag_spring_data_path,
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
mechanical_model.set_spring(spring)
mechanical_model.set_magnet_assembly(magnet_assembly)
mechanical_model.set_damper(damper)


# ELECTRICAL
R = 10.
load_model = SimpleLoad(R)
electrical_model = ElectricalModel(name='elec_system')

flux_database = FluxDatabase(csv_database_path='/home/michael/Dropbox/PhD/Python/Research/fea-flux-curves-numr[5,15]-numz[17,33,66]-wdiam[0.15]-cheight[8,12,14]-2019-04-11.csv', fixed_velocity=0.35)

flux_models = get_flux_models_from_db(flux_database,
                                      ['A', 'B', 'C'],
                                      coil_centers=[59/1000, 61/1000, 63/1000],
                                      mm=[10, 10, 10],
                                      winding_num_z=['17', '33', '66'],
                                      winding_num_r=['15', '15', '5'],
                                      coil_height=['0.008meter', '0.012meter', '0.014meter'])

# COUPLING MODEL
coupling_model = ConstantCoupling(c=0.5)


def _build_quick_unified_model(accelerometer_input,
                               flux_model,
                               load_model,
                               coupling_model,
                               governing_equations=unified_ode):
    """Build a unified model quickly with some hard-coded values.

    This function is intended for debugging and experimentation only.

    """
    mechanical_model.set_input(accelerometer_input)

    electrical_model.set_flux_model(flux_model, precompute_gradient=True)
    electrical_model.set_load_model(load_model)

    # Build unified model from components
    unified_model = UnifiedModel(name='Unified')
    unified_model.add_mechanical_model(mechanical_model)
    unified_model.add_electrical_model(electrical_model)
    unified_model.add_coupling_model(coupling_model)
    unified_model.add_governing_equations(governing_equations)
    unified_model.add_post_processing_pipeline(clip_x2, name='clip tube velocity')

    return unified_model


###
sample_number = 1
###

accelerometer = AccelerometerInput(a_samples[sample_number].acc_df,
                                   accel_column='z_G',
                                   time_column='time(ms)',
                                   smooth=True,
                                   interpolate=True)
flux_model = flux_models['A']
unified_model_l = _build_quick_unified_model(accelerometer,
                                             flux_model,
                                             load_model,
                                             coupling_model,
                                             governing_equations=unified_ode)

unified_model_nl = _build_quick_unified_model(accelerometer,
                                              flux_model,
                                              load_model,
                                              ConstantCoupling(c=0),
                                              governing_equations=unified_ode)
# Solve
initial_conditions = [0, 0, 0.04, 0, 0]

unified_model_l.solve(t_start=0, t_end=15,
                    y0=initial_conditions,
                    t_max_step=1e-3)


unified_model_nl.solve(t_start=0, t_end=15,
                    y0=initial_conditions,
                    t_max_step=1e-3)

# adc_processor = AdcProcessor(voltage_division_ratio=1/0.342,
#                              smooth=True,
#                              critical_frequency=1/8)

# metrics = {'rms': root_mean_square,
#            'rms_diff': root_mean_square_percentage_diff}

# e_score, e_eval = unified_model_l.score_electrical_model(metrics,
#                                                        a_samples[sample_number].adc_df,
#                                                        adc_processor,
#                                                        prediction_expr='g(t, x5)',
#                                                        return_evaluator=True)

df_l = unified_model_l.get_result(time='t', x='x3-x1', x_dot='x4-x2', v='g(t, x5)')
df_nl = unified_model_nl.get_result(time='t', x='x3-x1', x_dot='x4-x2', v='g(t, x5)')


