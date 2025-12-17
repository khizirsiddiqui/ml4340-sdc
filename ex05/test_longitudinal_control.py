import gymnasium as gym
import pygame

from sdc_wrapper import SDC_Wrapper

from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction
from lateral_control import LateralController
from longitudinal_control import LongitudinalController
import matplotlib.pyplot as plt
import numpy as np


class ControlStatus:
    """
    Class to keep track of key presses while recording demonstrations.
    """

    def __init__(self):
        self.stop = False
        self.quit = False

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.quit = True

            if event.type == pygame.KEYDOWN:
                self.key_press(event)

        keys = pygame.key.get_pressed()

    def key_press(self, event):
        if event.key == pygame.K_ESCAPE:    self.quit = True
        if event.key == pygame.K_SPACE:     self.stop = True

# action variables
a = np.array( [0.0, 0.0, 0.0] )

# init environement
env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode='human'), remove_score=True, return_linear_velocity=True)
try:
    _, _ = env.reset()
except:
    print("Please note that you can't test longitudinal control on the cluster")
    exit(1)

# define variables
total_reward = 0.0
steps = 0

# init modules of the pipeline
LD_module = LaneDetection()
LatC_module = LateralController()
LongC_module = LongitudinalController()

# init extra plot
fig = plt.figure()
plt.ion()
plt.show()

# init ControlStatus
control_status = ControlStatus()

while not control_status.quit:
    # perform step
    control_status.update()
    s, r, done, trunc, info = env.step(a)
    speed = info['speed']

    # lane detection
    lane1, lane2 = LD_module.lane_detection(s)

    # waypoint and target_speed prediction
    waypoints = waypoint_prediction(lane1, lane2)
    target_speed = target_speed_prediction(waypoints, max_speed=60, exp_constant=4.5)

    # control
    a[0] = LatC_module.stanley(waypoints, speed)
    a[1], a[2] = LongC_module.control(speed, target_speed)

    # reward
    total_reward += r

    # outputs during training
    if steps % 2 == 0 or done:
        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        print("speed {:+0.2f} targetspeed {:+0.2f}".format(speed, target_speed))

        #LD_module.plot_state_lane(s, steps, fig, waypoints=waypoints)
        closed_pygame_window = LongC_module.plot_speed(speed, target_speed, steps, fig)
        if closed_pygame_window:
            break

    steps += 1

    if done or control_status.stop:
        print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        control_status.stop = False
        s, info = env.reset()
        speed = info['speed']

env.close()