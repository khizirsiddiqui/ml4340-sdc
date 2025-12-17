import gymnasium as gym
import pygame

from sdc_wrapper import SDC_Wrapper

from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction
import matplotlib.pyplot as plt
import numpy as np

class ControlStatus:
    """
    Class to keep track of key presses while recording demonstrations.
    """

    def __init__(self):
        self.stop = False
        self.quit = False

        self.steer = 0.0
        self.accelerate = 0.0
        self.brake = 0.0

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.quit = True

            if event.type == pygame.KEYDOWN:
                self.key_press(event)

        keys = pygame.key.get_pressed()
        self.accelerate = 1 if keys[pygame.K_UP] else 0
        self.brake = 0.8 if keys[pygame.K_DOWN] else 0  # set 1.0 for wheels to block to zero rotation
        self.steer = 1 if keys[pygame.K_RIGHT] else (-1 if keys[pygame.K_LEFT] else 0)

    def key_press(self, event):
        if event.key == pygame.K_ESCAPE:    self.quit = True
        if event.key == pygame.K_SPACE:     self.stop = True

# init environement
env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode='human'), remove_score=True, return_linear_velocity=False)
try:
    _, _ = env.reset(seed=int(np.random.randint(0, 1e6)))
except:
    print("Please note that you can't test waypoint prediction on the cluster")
    exit(1)

# define variables
total_reward = 0.0
steps = 0

# init modules of the pipeline
LD_module = LaneDetection()

# init ControlStatus
control_status = ControlStatus()

# init extra plot
fig = plt.figure()
plt.ion()
plt.show()

while not control_status.quit:
    # perform step
    control_status.update()
    a = [control_status.steer, control_status.accelerate, control_status.brake]
    s, r, done, speed, info = env.step(a)

    # lane detection
    lane1, lane2 = LD_module.lane_detection(s)

    # waypoint and target_speed prediction
    waypoints = waypoint_prediction(lane1, lane2)
    target_speed = target_speed_prediction(waypoints)

    # reward
    total_reward += r

    # outputs during training
    if steps % 2 == 0 or done:
        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        print("step {} total_reward {:+0.2f}".format(steps, total_reward))

        closed_pygame_window = LD_module.plot_state_lane(s, steps, fig, waypoints=waypoints)
        if closed_pygame_window:
            break

    steps += 1

    # check if stop
    if done or control_status.stop:
        print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        control_status.stop = False
        s, _ = env.reset()

env.close()