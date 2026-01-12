import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.signal import find_peaks, welch, correlate
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time

from utils import get_config

def is_oscillating(arr):
    if len(arr) < 4:
        return False, {}
    methods = {}
    # Method 1: Sign changes
    sign_changes = np.sum(np.diff(np.sign(arr)) != 0)
    methods['sign_changes'] = sign_changes
    if sign_changes >= 4:
        return True, methods

class LongitudinalControllerBase:
    '''
    Longitudinal Control using a PID Controller

    functions:
        PID_step()
        control()
    '''
    def __init__(self, KP=0.6, KD=0.01, KI=0.05, max_sum_error=1000, controller_type='pid'):
        self.last_error = 0
        self.sum_error = 0
        self.max_sum_error = max_sum_error
        self.last_control = 0
        self.speed_history = []
        self.target_speed_history = []
        self.step_history = []

        self.filtered_derivative = 0

        self.controller_type = controller_type

        # PID parameters
        self.KP = KP
        self.KI = KI
        self.KD = KD

    def PID_step(self, speed, target_speed):
        '''
        Perform one step of the PID control
        - Implement the descretized control law.
        - Implement a maximum value for the sum of error you are using for the intgral term

        args:
            speed
            target_speed

        output:
            control (u)
        '''

        # define error from set point target_speed to speed
        error = target_speed - speed
        self.sum_error = error + 0.7 * self.sum_error
        self.sum_error = np.clip(self.sum_error, -self.max_sum_error, self.max_sum_error)

        controller_p = self.KP * error

        derivate = error - self.last_error
        self.filtered_derivative = 0.8 * self.filtered_derivative + 0.2 * derivate
        controller_d = self.KD * self.filtered_derivative

        controller_i = self.KI * self.sum_error
        if abs(self.sum_error) > self.max_sum_error * 0.7:
            controller_i = controller_i * 1.5

        control = controller_p + controller_i + controller_d

        self.last_error = error
        return control

    def MPC_step(self, speed, target_speed, dt=0.1):
        '''
        Perform one step of the MPC control
        - Implement the descretized control law.
        - Implement a maximum value for the sum of error you are using for the intgral term

        args:
            speed
            target_speed
        output:
            control (u)
        '''
        # use last 50 speed values to estimate oscillation
        if len(self.speed_history) >= 50:
            recent_speeds = self.speed_history[-50:]
            oscillating, methods = is_oscillating(recent_speeds)
            if oscillating:
                print("Oscillation detected by methods:", methods)
                # reduce last control to dampen oscillations
                self.last_control *= 0.7
                return self.last_control
        def cost_function(u_seq):
            sim_speed = speed
            total_cost = 0
            for u in u_seq:
                sim_speed += u * dt
                error = target_speed - sim_speed
                total_cost += 1.3 * error**2 + 0.2 * u**2
            return total_cost
        initial_guess = [0.5] * 10
        bounds = [(-0.8, 0.9)] * 10
        result = minimize(cost_function, initial_guess, bounds=bounds)
        optimal_u = result.x[0]
        return optimal_u


    def control(self, speed, target_speed, time_step=0):
        '''
        Derive action values for gas and brake via the control signal
        using PID controlling

        Args:
            speed (float)
            time_step (int)
            target_speed (float)

        output:
            gas
            brake
        '''
        if time_step < 50:
            return 0.9, 0.0
        if self.controller_type == 'pid':
            control = self.PID_step(speed, target_speed)
        elif self.controller_type == 'mpc':
            control = self.MPC_step(speed, target_speed)
        brake = 0
        gas = 0

        # translate the signal from the PID controller
        # to the action variables gas and brake
        if control >= 0:
            gas = np.clip(control, 0, 0.9)
        else:
            brake = np.clip(-1*control, 0, 0.85)
        return gas, brake

    def plot_speed(self, speed, target_speed, step, fig):
        self.speed_history.append(speed)
        self.target_speed_history.append(target_speed)
        self.step_history.append(step)
        ax = fig.gca()
        ax.clear()
        ax.plot(self.step_history, self.speed_history, c="green")
        ax.plot(self.step_history, self.target_speed_history)

        # plot error as well
        error = np.array(self.target_speed_history) - np.array(self.speed_history)
        ax.fill_between(self.step_history, self.target_speed_history, self.speed_history, color='red', alpha=0.2)

        # show a circle with color depending on the self.sum_error with text as self.sum_error
        # color is a scale from green (low error) to red (high error)
        max_error = self.max_sum_error
        error_ratio = np.clip(self.sum_error / max_error, 0, 1)
        color = (error_ratio, 1 - error_ratio, 0)
        circle = Circle((0.9, 0.1), 0.05, color=color, transform=plt.gca().transAxes)
        # add text to circle
        ax.text(0.9, 0.1, f"{self.sum_error:.1f}", color='black', fontsize=8, ha='center', va='center', transform=plt.gca().transAxes)
        ax.add_patch(circle)

        try:
            fig.canvas.flush_events()
            return False
        except:
            return True

config = get_config()
if config['longitudinal_controller']['module'] in ['pid', 'mpc']:
    params = config['longitudinal_controller']['params']
    LongitudinalController = lambda: LongitudinalControllerBase(**params)
else:
    raise ValueError('Unknown lane detection module')
