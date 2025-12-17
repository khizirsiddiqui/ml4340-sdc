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


    def __init__(self, gain_constant=3, damping_constant=0.6):

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
        # print("\nTime step: ", time_step)
        # print("Car speed (px/s): ", speed)
        car_position = np.array(car_pos)

        # derive orientation error as the angle of the first path segment to the car orientation
        path_vector = waypoints[:, 1] - waypoints[:, 0]
        orientation_error = np.arctan2(path_vector[0], path_vector[1])
        # in_degrees = orientation_error * 180 / np.pi
        # print("Orientation error (deg): ", in_degrees)

        # derive cross track error as distance between desired waypoint at spline parameter equal zero ot the car position
        cross_track_error = distance_point_to_line(car_position, waypoints.T)
        # scale the error to meters (assuming waypoints are in pixels and 10 pixels = 1 meter)
        # print("Cross track error (px): ", cross_track_error)

        # derive stanley control law
        effective_gain = self.gain_constant
        stanley_term = orientation_error + np.arctan2(effective_gain * cross_track_error, max(speed, 1.0))
        stanley_term = np.clip(stanley_term, -np.pi / 4, np.pi / 4)
        # print("Stanley term (deg): ", stanley_term * 180 / np.pi)

        # derive damping term
        if time_step == 0:
            damping_term = 0
        else:
            epsilon = 1e-6
            damping_term = self.damping_constant * (stanley_term - self.previous_steering_angle) / (time_step + epsilon)
        # print("Damping term (deg): ", damping_term * 180 / np.pi)
        steering_angle = stanley_term - damping_term
        # clip to the maximum stering angle (0.4) and rescale the steering action space
        steering_angle = np.clip(steering_angle, -0.4, 0.4) / 0.4
        # print("Steering angle (deg): ", steering_angle * 0.4 * 180 / np.pi)
        self.previous_steering_angle = steering_angle
        return steering_angle

if __name__ == "__main__":
    # simple test of the lateral controller
    LatC_module = LateralController()

    # define a set of waypoints (a curve)
    t = np.linspace(0, 8, num=100)
    x = 20 * np.cos(t) + 48
    y = 10 * t
    waypoints = np.vstack((x, y))
    total_time_steps = waypoints.shape[1]

    # define car speed
    speed = 1  # m/s
    car_initial_pos = (48, 0)
    car_pos = np.array(car_initial_pos)
    car_trajectory = [car_initial_pos]
    distances = [np.linalg.norm(car_pos - waypoints[:, 0])]
    # run the lateral controller
    for t in range(1, total_time_steps):
        steering_angle = LatC_module.stanley(waypoints, speed, time_step=t, car_pos=car_pos)
        car_pos = car_pos + np.array([speed * np.sin(steering_angle), speed * np.cos(steering_angle)])
        distance_from_waypoint_at_t = np.linalg.norm(car_pos - waypoints[:, t])
        car_trajectory.append(car_pos)
        distances.append(distance_from_waypoint_at_t)

    # convert distances as a color scale for the trajectory
    distances = np.array(distances)
    colors = plt.colormaps.get_cmap('viridis_r')(distances / distances.max())

    # plot the waypoints and the car position
    plt.figure()
    plt.plot(waypoints[0, :], waypoints[1, :], label='Waypoints')
    plt.quiver(48, 0, np.sin(steering_angle * 0.4), np.cos(steering_angle * 0.4), angles='xy', scale_units='xy', scale=1, color='r', label='Steering Direction')
    car_trajectory = np.array(car_trajectory)
    for i in range(car_trajectory.shape[0]-1):
        plt.plot(car_trajectory[i:i+2, 0], car_trajectory[i:i+2, 1], color=colors[i], linewidth=2)
    plt.axis('equal')
    plt.legend()
    # plot colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis_r', norm=plt.Normalize(vmin=distances.min(), vmax=distances.max()))
    sm.set_array([])
    plt.colorbar(sm, label='Distance from Waypoints (m)')
    plt.title('Lateral Controller Test')
    plt.show()
