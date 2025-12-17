import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time

class LongitudinalController:
    '''
    Longitudinal Control using a PID Controller

    functions:
        PID_step()
        control()
    '''
    def __init__(self, KP=0.0175, KI=1e-5, KD=0.07, max_sum_error=1000):
        self.last_error = 0
        self.sum_error = 0
        self.max_sum_error = max_sum_error
        self.last_control = 0
        self.speed_history = []
        self.target_speed_history = []
        self.step_history = []

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
        self.sum_error += error
        if np.abs(self.sum_error) > self.max_sum_error:
            self.sum_error = np.clip(self.sum_error, -self.max_sum_error / 4, self.max_sum_error / 4)

        controller_p = self.KP * error
        controller_d = self.KD * (error - self.last_error)
        controller_i = self.KI * self.sum_error

        self.last_error = error
        # derive PID elements
        control = controller_p + controller_i + controller_d
        return control

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

        control = self.PID_step(speed, target_speed)
        brake = 0
        gas = 0

        # translate the signal from the PID controller
        # to the action variables gas and brake
        if control >= 0:
            gas = np.clip(control, 0, 0.8)
        else:
            brake = np.clip(-1*control, 0, 0.8)

        return gas, brake

    def plot_speed(self, speed, target_speed, step, fig):
        self.speed_history.append(speed)
        self.target_speed_history.append(target_speed)
        self.step_history.append(step)
        plt.gcf().clear()
        plt.plot(self.step_history, self.speed_history, c="green")
        plt.plot(self.step_history, self.target_speed_history)

        # show a circle with color depending on the self.sum_error with text as self.sum_error
        # color is a scale from green (low error) to red (high error)
        max_error = self.max_sum_error
        error_ratio = (self.sum_error + max_error) / (2 * max_error)
        color = (error_ratio, 1 - error_ratio, 0)
        circle = Circle((0.9, 0.1), 0.05, color=color, transform=plt.gca().transAxes)
        plt.gca().add_patch(circle)

        try:
            fig.canvas.flush_events()
            return False
        except:
            return True
