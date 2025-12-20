import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time

from utils import distance_point_to_line

class LateralController:
    '''
    Lateral control using the Stanley controller

    functions:
        stanley

    init:
        gain_constant (default=5)
        damping_constant (default=0.6)
    '''


    def __init__(self, gain_constant=6, damping_constant=0.6):

        self.gain_constant = gain_constant
        self.damping_constant = damping_constant
        self.previous_steering_angle = 0


    def stanley(self, waypoints, speed, time_step=0, car_pos=(48, 0)):
        '''
        one step of the stanley controller with damping
        args:
            waypoints (np.array) [2, num_waypoints]
            time_step (int)
            speed (float)
        '''
        if time_step < 50:
            return 0.0
        car_position = np.array(car_pos)

        path_vector = waypoints[:, 1] - waypoints[:, 0]
        orientation_error = np.arctan2(path_vector[0], path_vector[1])

        # derive stanley control law
        effective_gain = self.gain_constant
        cross_track_error = distance_point_to_line(car_position, waypoints.T)
        if np.abs(cross_track_error) < 5:
            cte_effect = 0
        else:
            cte_effect = np.arctan2(effective_gain * cross_track_error, max(speed, 1.0))

        stanley_term = orientation_error + cte_effect
        stanley_term = np.clip(stanley_term, -np.pi / 4, np.pi / 4)
        damping_term = self.damping_constant * (stanley_term - self.previous_steering_angle)

        steering_angle = stanley_term - damping_term
        steering_angle = np.clip(steering_angle, -0.4, 0.4)
        self.previous_steering_angle = steering_angle
        return steering_angle

def test_lc(LatC_module, waypoints, speed, car_initial_pos):
    car_pos = np.array(car_initial_pos)
    car_trajectory = [car_initial_pos]
    distances = [np.linalg.norm(car_pos - waypoints[:, 0])]
    # run the lateral controller untill the end of the waypoints (same y coordinate)
    for t in range(1, waypoints.shape[1]):
        steering_angle = LatC_module.stanley(waypoints[:, t:t+5], speed, time_step=50+t, car_pos=car_pos)
        car_pos = car_pos + np.array([speed * np.sin(steering_angle), speed * np.cos(steering_angle)])
        distance_from_waypoint_at_t = np.linalg.norm(car_pos - waypoints[:, t])
        car_trajectory.append(car_pos)
        distances.append(distance_from_waypoint_at_t)
        if car_pos[1] >= waypoints[1, -1]:
            break
    return car_trajectory, distances

if __name__ == "__main__":
    # define car speed
    speed = 0.2  # m/s
    car_initial_pos = (48, 0)

    # define a set of waypoints (a curve)
    t = np.linspace(0, 10, 100)
    x = 48 + np.sin(t)
    y = 0 + t + x - 48
    waypoints = np.vstack((x, y))

    plt.figure()
    plt.plot(waypoints[0, :], waypoints[1, :], label='Waypoints')
    for damping_constant in [0.3, 0.5, 0.8]:
        LatC_module = LateralController(damping_constant=damping_constant)
        car_trajectory, distances = test_lc(LatC_module, waypoints, speed, car_initial_pos)
        car_trajectory = np.array(car_trajectory)
        mean_distance = np.mean(distances)
        plt.plot(car_trajectory[:, 0], car_trajectory[:, 1], linewidth=2, label=f'Car Trajectory (damping={damping_constant}), mean dist={mean_distance:.2f}px')

    # for gain in [2.5, 5.0, 7.5, 10]:
    #     LatC_module = LateralController(gain_constant=gain)
    #     car_trajectory, distances = test_lc(LatC_module, waypoints, speed, car_initial_pos)
    #     car_trajectory = np.array(car_trajectory)
    #     mean_distance = np.mean(distances)
    #     plt.plot(car_trajectory[:, 0], car_trajectory[:, 1], linewidth=2, label=f'Car Trajectory (gain={gain}), mean dist={mean_distance:.2f}px')
    #
    # plot car starting car_position
    plt.plot(car_initial_pos[0], car_initial_pos[1], 'ro', label='Car Start Position')
    plt.axis('equal')
    plt.legend()
    # plot colorbar
    # sm = plt.cm.ScalarMappable(cmap='viridis_r', norm=plt.Normalize(vmin=distances.min(), vmax=distances.max()))
    # sm.set_array([])
    # plt.colorbar(sm, label='Distance from Waypoints (m)')
    plt.title('Lateral Controller Test')
    plt.show()
