import numpy as np
import matplotlib.pyplot as plt
import warnings

from scipy.integrate import solve_ivp
from unified_model.mechanical_components.mechanical_spring import MechanicalSpring

def elastic_spring(x, push_direction='up', pos=0, strength=1000, sharpness=0.01):
    if push_direction is 'down':
        direction_modifier = -1
    elif push_direction is 'up':
        direction_modifier = 1
    else:
        direction_modifier = 0
        warnings.warn('Incorrect spring direction specified: {}. Using "up"'.format(push_direction))

    return strength*np.exp(direction_modifier*(pos-x)/sharpness)

def inelastic_spring(x, v, c, push_direction='up', pos=0, strength=1000, sharpness=0.01):
    if push_direction is 'down':
        direction_modifier = -1
    elif push_direction is 'up':
        direction_modifier = 1
    else:
        raise ValueError(f'Incorrect spring direction specified: {push_direction}. \
        Possible values include "up" and "down".')

        direction_modifier = 0
        warnings.warn('Incorrect spring direction specified: {}. Using "up"'.format(push_direction))

    return strength*np.exp(direction_modifier*(pos-x)/sharpness) - c*v

def bouncing_ball(t, y, mass_ball, mechanical_spring):
    x1, x2 = y  # <-- displacement, velocity
    g = -9.81

    dx1 = x2
    # spring_force = mechanical_spring.get_force(x1, x2)
    spring_force = mechanical_spring.get_force(x1, x2)
    dx2 = (-spring_force - g*mass_ball)/mass_ball

    return [dx1, dx2]


mechanical_spring = MechanicalSpring(push_direction='down',
                                     position=0.1,
                                     strength=1000,
                                     sharpness=0.001,
                                     pure=True,
                                     damper_constant=3)

mass_ball =  1 # kg

t_span = [0, 5]
y0 = [0.0, 0]
sol = solve_ivp(fun=lambda t,y: bouncing_ball(t,
                                              y,
                                              mass_ball=mass_ball,
                                              mechanical_spring=mechanical_spring),
                t_span=t_span,
                y0=y0,
                max_step=1e-3)

x1 = sol.y[0]
x2 = sol.y[1]


plt.plot(x1)
plt.show()
