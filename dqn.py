"""
This is my attempt to implement DQN on the cartpole-v0 environment from OpenAI gym.

To force myself to get better at version control, I will develop it all in this one file instead
of making backups each time I change something.

I have a trello board https://trello.com/b/iQUDEFxL/dqn
and a github repo https://github.com/naimenz/DQN

Building on the general structure I used for Sarsa/REINFORCE-type algorithms, I'll write a class to hold
all the important bits and share parameters and stuff.
"""

import torch
import gym

env = gym.make('CartPole-v0')

class DQN():
    """
    DQN class specifically for solving Cartpole. 
    Might work on other simple continuous obs, discrete act environments too if it works on Cartpole.
    """
    # we initialise with an environment for now, might need to add architecture once
    # I get the NN working.
    def __init__(self, env):
        self.env = env
        self.n_acts = env.action_space.n
        self.obs_dim = env.observation_space.shape # we won't actually be using these observations though

dqn = DQN(env)
env.reset()
done = False
while not done:
    _, _, done, _ = env.step(0)
    import time
    tic = time.perf_counter()
    x = env.render(mode='rgb_array')
    toc = time.perf_counter()
    print(f"render took {toc - tic:0.4f} seconds")
    print(x)
env.close()

