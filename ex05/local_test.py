import gymnasium as gym

from sdc_wrapper import SDC_Wrapper

from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction
from lateral_control import LateralController
from longitudinal_control import LongitudinalController
import numpy as np
import argparse

def seed_evaluate(env, seed):
    # action variables
    a = np.array([0.0, 0.0, 0.0])

    # init environement
    try:
        s, info = env.reset(seed=seed)
        speed = info['speed']
    except:
        print("Please note that you can't use the window on the cluster")
        exit(1)

    # init modules of the pipeline
    LD_module = LaneDetection()
    LatC_module = LateralController()
    LongC_module = LongitudinalController()
    reward_per_episode = 0

    for t in range(600):
        # lane detection
        lane1, lane2 = LD_module.lane_detection(s, t)

        # waypoint and target_speed prediction
        waypoints = waypoint_prediction(lane1, lane2, t)
        target_speed = target_speed_prediction(waypoints, t)

        # control
        a[0] = LatC_module.stanley(waypoints, speed, t)
        a[1], a[2] = LongC_module.control(speed, target_speed, t)

        # perform step
        s, r, done, trunc, info = env.step(a)
        speed = info['speed']

        # reward
        reward_per_episode += r

    print("Reward: ", reward_per_episode)

def evaluate(env, num_episodes=100):
    reward_list = []
    for episode in range(num_episodes):
        # action variables
        a = np.array([0.0, 0.0, 0.0])

        # init environement
        try:
            s, info = env.reset()
            speed = info['speed']
        except:
            print("Please note that you can't use the window on the cluster")
            exit(1)

        # init modules of the pipeline
        LD_module = LaneDetection()
        LatC_module = LateralController()
        LongC_module = LongitudinalController()
        reward_per_episode = 0

        for t in range(600):
            # lane detection
            lane1, lane2 = LD_module.lane_detection(s, t)

            # waypoint and target_speed prediction
            waypoints = waypoint_prediction(lane1, lane2, t)
            target_speed = target_speed_prediction(waypoints, t)

            # control
            a[0] = LatC_module.stanley(waypoints, speed, t)
            a[1], a[2] = LongC_module.control(speed, target_speed, t)

            # perform step
            s, r, done, trunc, info = env.step(a)
            speed = info['speed']

            # reward
            reward_per_episode += r

    # print stats of rewards
    print("#episodes: ", num_episodes)
    print("mean:      ", np.mean(reward_list))
    print("std:       ", np.std(reward_list))
    print("max:       ", np.max(reward_list))
    print("min:       ", np.min(reward_list))

def calculate_score_for_leaderboard(env):
    """
    DO NOT CHANGE

    Evaluate the performance of the agent. This is the function to be used for
    the final ranking on the course-wide leaderboard, only with a different set
    of seeds. Better not change it.
    """
    # These are not the final evaluation seeds, do not overfit on these tracks!
    seeds = [68583770, 61441673, 35379582, 71567864, 15016296,
            14114821, 92318023, 76046482, 89620946, 72686959]

    total_reward = 0
    for episode, seed in enumerate(seeds):
    # action variables
        a = np.array([0.0, 0.0, 0.0])

        # init environement
        try:
            s, info = env.reset(seed=seed)
            speed = info['speed']
        except:
            print("Please note that you can't use the window on the cluster")
            exit(1)

        # init modules of the pipeline
        LD_module = LaneDetection()
        LatC_module = LateralController()
        LongC_module = LongitudinalController()

        reward_per_episode = 0
        for t in range(600):
            # lane detection
            lane1, lane2 = LD_module.lane_detection(s, t)

            # waypoint and target_speed prediction
            waypoints = waypoint_prediction(lane1, lane2, t)
            target_speed = target_speed_prediction(waypoints, t)

            # control
            a[0] = LatC_module.stanley(waypoints, speed, t)
            a[1], a[2] = LongC_module.control(speed, target_speed, t)

            # perform step
            s, r, done, trunc, info = env.step(a)
            speed = info['speed']

            # reward
            reward_per_episode += r

        print('episode %d \t seed %d \t reward %f' % (episode, seed, reward_per_episode))
        total_reward += np.clip(reward_per_episode, 0, np.infty)

    print('---------------------------')
    print(' total score: %f' % (total_reward / len(seeds)))
    print('---------------------------')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument( '--score', action="store_true", help='a flag to evaluate the pipeline over set seeds' )
    parser.add_argument( '--seed', type=int, default=None, help='random seed for reproducibility' )
    args = parser.parse_args()

    assert not (args.score and args.seed), "Please provide either --score or --seed, not both."

    if args.seed:
        render_mode = 'human'
    else:
        render_mode = 'rgb_array'
    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode=render_mode), remove_score=True, return_linear_velocity=True)

    if args.score:
        calculate_score_for_leaderboard(env)
    elif args.seed:
        seed_evaluate(env, args.seed)
    else:
        evaluate(env)

    env.close()
