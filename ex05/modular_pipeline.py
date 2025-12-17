import gymnasium as gym

from sdc_wrapper import SDC_Wrapper

from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction
from lateral_control import LateralController
from longitudinal_control import LongitudinalController
import numpy as np
import argparse


def evaluate(env):
    """
    """
    for episode in range(5):
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
            #print('lane1, lane2 ',lane1, lane2)

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

        print('episode %d \t reward %f' % (episode, reward_per_episode))

def calculate_score_for_leaderboard(env):
    """
    DO NOT CHANGE

    Evaluate the performance of the agent. This is the function to be used for
    the final ranking on the course-wide leaderboard, only with a different set
    of seeds. Better not change it.
    """
    # These are not the final evaluation seeds, do not overfit on these tracks!
    seeds = [97657630, 47460391, 22619914, 76925063, 84647422, 
            83470445, 77482096, 94017676, 99341122, 58134947]

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

        print('episode %d \t reward %f' % (episode, reward_per_episode))
        total_reward += np.clip(reward_per_episode, 0, np.infty)

    print('---------------------------')
    print(' total score: %f' % (total_reward / len(seeds)))
    print('---------------------------')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument( '--score', action="store_true", help='a flag to evaluate the pipeline for the leaderboard' )
    parser.add_argument("--no_display", action="store_true", default=False, help='a flag indicating whether training runs in the cluster')

    args = parser.parse_args()

    render_mode = 'rgb_array' if args.no_display else 'human'
    env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode=render_mode), remove_score=True, return_linear_velocity=True)

    if args.score:
        calculate_score_for_leaderboard(env)
    else:
        evaluate(env)

    env.close()