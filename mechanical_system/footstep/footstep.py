import numpy as np


class Footstep(object):
    """
    The footstep class that can simulate a constant-acceleration footstep
    for use in a mechanical model
    """

    def __init__(self, accelerations, acceleration_couple_time_separation, positive_footstep_displacement,
                 t_footstep_start=0):
        """
        Initializes the Footstep object
        :param accelerations: Array [acc_up, acc_dec, acc_down, acc_impact] of accelerations specified in m/s^2
        :param acceleration_couple_time_separation: Time between the up/dec and down/impact acceleration couples.
        Represents the time taken for the horizontal motion of the footstep.
        :param positive_footstep_displacement: The maximum desired positive displacement (what is the maximum height
        the foot reaches above the ground)
        """
        self.acc_up = None
        self.acc_dec = None
        self.acc_down = None
        self.acc_impact = None

        self.t_acc_up = None
        self.t_acc_dec = None
        self.t_acc_down = None
        self.t_acc_impact = None

        self.t_interval_acc_up = None
        self.t_interval_acc_dec = None
        self.t_interval_acc_down = None
        self.t_interval_acc_impact = None

        self.t_footstep_start = t_footstep_start
        self.acceleration_couple_time_separation = acceleration_couple_time_separation
        self.positive_footstep_displacement = positive_footstep_displacement

        self._set_accelerations(accelerations)
        self._set_acceleration_times()
        self._set_acceleration_time_intervals()

    def _set_accelerations(self, accelerations):
        """
        Sets the acceleration values
        :param accelerations: Array [acc_up, acc_dec, acc_down, acC_impact] of accelerations specified in m/s^2.
        """
        self.acc_up = accelerations[0]
        self.acc_dec = accelerations[1]
        self.acc_down = accelerations[2]
        self.acc_impact = accelerations[3]

    def _set_acceleration_times(self):
        """
        Sets the amount of time each acceleration acts for
        """

        self.t_acc_up, self.t_acc_dec = self._calculate_acceleration_couple_times(self.acc_up, self.acc_dec,
                                                                                  self.positive_footstep_displacement)
        self.t_acc_down, self.t_acc_impact = self._calculate_acceleration_couple_times(self.acc_down, self.acc_impact,
                                                                                       self.positive_footstep_displacement)

    def _calculate_acceleration_couple_times(self, acc_a, acc_b, displacement_delta):
        """
        Calculates the times that two opposing acceleration couples, `acc_a` and `acc_b` must act for in order to
         bring an object to rest with a change in displacement `displacement_delta`.
        :param acc_a: The first acceleration
        :param acc_b: The second acceleration
        :param displacement_delta: The total change in displacement that must occur
        :return: The displacement at which `acc_a` should stop acting and `acc_b` should start acting
        """
        acc_switch_displacement = (-displacement_delta * acc_b) / (acc_a - acc_b)
        time_acc_a = np.sqrt(2 * acc_switch_displacement / acc_a)

        velocity_at_switch = acc_a * time_acc_a
        time_acc_b = 2 * (displacement_delta - acc_switch_displacement) / velocity_at_switch

        return time_acc_a, time_acc_b

    def _set_acceleration_time_intervals(self):
        """
        Calculate the time-intervals for each of the accelerations
        :return: Array containing a tuple giving the start and end time of each acceleration for
        [acc_up, acc_dec, acc_down, acc_impact]
        """

        self.t_interval_acc_up = (self.t_footstep_start, self.t_footstep_start + self.t_acc_up)
        self.t_interval_acc_dec = (self.t_interval_acc_up[1], self.t_interval_acc_up[1] + self.t_acc_dec)
        self.t_interval_acc_down = (self.t_interval_acc_dec[1] + self.acceleration_couple_time_separation,
                                    self.t_interval_acc_dec[
                                        1] + self.acceleration_couple_time_separation + self.t_acc_down)
        self.t_interval_acc_impact = (self.t_interval_acc_down[1], self.t_interval_acc_down[1] + self.t_acc_impact)

    def _create_footstep_waveform(self):
        """
        Creates the footstep waveform
        """

    def get_footstep_acceleration(self, t):
        """
        Get the footstep acceleration at time `t`
        :param t: Time point where footstep acceleration will be calculated.
        :return: Footstep acceleration in m/s^2
        """
        if self.t_interval_acc_up[0] <= t <= self.t_interval_acc_up[1]:
            return self.acc_up
        elif self.t_interval_acc_dec[0] < t <= self.t_interval_acc_dec[1]:
            return self.acc_dec
        elif self.t_interval_acc_down[0] < t <= self.t_interval_acc_down[1]:
            return self.acc_down
        elif self.t_interval_acc_impact[0] < t <= self.t_interval_acc_impact[1]:
            return self.acc_impact
        else:
            return 0
