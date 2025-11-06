import os
import numpy as np
import gymnasium as gym
import time
import pygame

from sdc_wrapper import SDC_Wrapper

def load_demonstrations(data_folder):
    """
    1.1 a)
    Given the folder containing the expert demonstrations, the data gets loaded and
    stored it in two lists: observations and actions.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    return:
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    observations = []
    actions = []
    for fd in os.listdir(data_folder):
        if fd.endswith('.npy'):
            if fd.startswith('observation_'):
                data = np.load(os.path.join(data_folder, fd))
                observations.append(data)
                idx = fd.split('_')[1].split('.')[0]
                data = np.load(os.path.join(data_folder, f'action_{idx}.npy'))
                actions.append(data)
    assert len(observations) == len(actions), "Number of observations and actions do not match."
    return observations, actions


def save_demonstrations(data_folder, actions, observations):
    """
    1.1 f)
    Save the lists actions and observations in numpy .npy files that can be read
    by the function load_demonstrations.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    index = 0
    for f in os.listdir(data_folder):
        if f.endswith('.npy') and f.startswith('observation_'):
            idx = f.split('_')[1].split('.')[0]
            index = max(index, int(idx) + 1)
    for observation, action in zip(observations, actions):
        save_path = os.path.join(data_folder, f'observation_{index:05d}.npy')
        np.save(save_path, observation)
        save_path = os.path.join(data_folder, f'action_{index:05d}.npy')
        np.save(save_path, action)
        index += 1


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


def record_demonstrations(demonstrations_folder):
    """
    Function to record own demonstrations by driving the car in the gym car-racing
    environment.
    demonstrations_folder:  python string, the path to where the recorded demonstrations
                        are to be saved

    The controls are:
    arrow keys:         control the car; steer left, steer right, gas, brake
    ESC:                quit and close
    SPACE:              restart on a new track
    TAB:                save the current run
    """

    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode='human'), remove_score=True, return_linear_velocity=False)
    try:
        _, _ = env.reset(seed=int(np.random.randint(0, 1e6)))
    except:
        print("Please note that you can't collect data on the cluster.")
        return

    status = ControlStatus()
    total_reward = 0.0

    while not status.quit:
        observations = []
        actions = []
        # get an observation from the environment
        observation, _ = env.reset()

        while not status.stop and not status.save and not status.quit:
            status.update()

            # collect all observations and actions
            observations.append(observation.copy())
            actions.append(np.array([status.steer, status.accelerate,
                                    status.brake]))
            # submit the users' action to the environment and get the reward
            # for that step as well as the new observation (status)
            observation, reward, done, trunc, info = env.step([status.steer,
                                                           status.accelerate,
                                                           status.brake])

            total_reward += reward
            time.sleep(0.01)

        if status.save:
            save_demonstrations(demonstrations_folder, actions, observations)
            status.save = False

        status.stop = False

    env.close()

if __name__ == "__main__":
    print("Idempotency Test of the save and load functions:")
    print("Test1: Loading existing demonstrations from ./data")
    data_folder = "./data"
    obs, act = load_demonstrations(data_folder)
    print("Size observations = ", len(obs), "x", obs[0].shape)
    print("Size actions = ", len(act), "x", act[0].shape)
    assert obs[0].shape == (96, 96, 3), "Loaded observations do not have the correct shape."
    assert act[0].shape == (3,), "Loaded actions do not have the correct shape."
    print("Test passed!")

    print("Test2: Saving and loading dummy demonstrations")
    dummy_folder = "/tmp/sdc-dummy-data"
    if not os.path.exists(dummy_folder):
        os.makedirs(dummy_folder)
    for f in os.listdir(dummy_folder):
        os.remove(os.path.join(dummy_folder, f))
    obs, act = [ obs[0] ], [ act[0] ]
    print("Saving single demonstration")
    save_demonstrations(dummy_folder, act, obs)
    print("Size observations = ", 1, "x", obs[0].shape)
    print("Size actions = ", 1, "x", act[0].shape)
    print("Loading the demonstration")
    obs2, act2 = load_demonstrations(dummy_folder)
    print("Size observations = ", len(obs2), "x", obs2[0].shape)
    print("Size actions = ", len(act2), "x", act2[0].shape)
    assert 1 == len(obs2), f"Saved (1) and loaded ({len(obs2)}) observations do not match in length."
    assert 1 == len(act2), f"Saved (1) and loaded ({len(act2)}) actions do not match in length."
    assert obs[0].shape == obs2[0].shape, "Saved and loaded observations do not match in shape. (" + str(obs.shape) + " vs " + str(obs2[0].shape) + ")"
    assert act[0].shape == act2[0].shape, "Saved and loaded actions do not match in shape. (" + str(act.shape) + " vs " + str(act2[0].shape) + ")"
    assert np.array_equal(obs[0], obs2[0]), "Saved and loaded observations do not match in content."
    assert np.array_equal(act[0], act2[0]), "Saved and loaded actions do note match in content."
    print("Test passed!")
