import gymnasium as gym
import pygame

from sdc_wrapper import SDC_Wrapper

class ControlStatus:
    """
    Class to keep track of key presses while recording demonstrations.
    """
    def __init__(self):
        self.stop = False
        self.save = False
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
        self.accelerate = 0.5 if keys[pygame.K_UP] else 0
        self.brake = 0.8 if keys[pygame.K_DOWN] else 0
        self.steer = 1 if keys[pygame.K_RIGHT] else (-1 if keys[pygame.K_LEFT] else 0)

    def key_press(self, event):
        if event.key == pygame.K_ESCAPE:    self.quit = True
        if event.key == pygame.K_SPACE:     self.stop = True
        if event.key == pygame.K_TAB:       self.save = True

def drive():
    """
    Function to drive in the environment.

    The controls are:
    arrow keys:         control the car; steer left, steer right, gas, brake
    ESC:                quit and close
    SPACE:              restart on a new track
    """
    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode='human'))
    status = ControlStatus()
    total_reward = 0.0

    status.quit = False

    while not status.quit:
        # get an observation from the environment
        observation, _ = env.reset()

        while True:
            status.update()

            # submit the users' action to the environment and get the reward
            # for that step as well as the new observation (status)
            observation, reward, done, trunc, info = env.step([status.steer,
                                                        status.accelerate,
                                                        status.brake])
            total_reward += reward

            if status.stop or status.save or status.quit:
                break

        status.stop = False
        status.save = False

    env.close()

if __name__ == "__main__":
    drive()
