import gymnasium as gym

from sdc_wrapper import SDC_Wrapper

env = SDC_Wrapper(gym.make('CarRacing-v2', render_mode='rgb_array'))

obs, _ = env.reset()
done = False
ep_rew = 0.0
while not done:
   observation, reward, done, trunc, info = env.step([0.0, 1.0, 0.0])
   ep_rew += reward
   done = done or trunc
print("Reward:", ep_rew)
