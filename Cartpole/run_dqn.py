"""
File for running the experiments, kept separate from class definitions.

TODO: see if there are issues with random number generation between files etc.
"""
import numpy as np # using numpy as sparingly as possible, mainly for random numbers but also some other things
import torch
import torch.nn as nn
import gym
import matplotlib.pyplot as plt
from dqn import DQN

# set a seed for reproducibility during implementation
SEED = 42
torch.manual_seed(SEED)
# using the new recommended numpy random number routines
rng = np.random.default_rng(SEED)

# make the environment and seed it
env = gym.make('CartPole-v0')
env.seed(SEED)

# initialise agent
dqn = DQN(env, gamma=1., eval_eps=0.05)

# train
n_frames = 1000000
lr = 1e-5
n_holdout = 1000
# save output
directory = 'run7'
ep_rets, holdout_scores = dqn.train(N=n_frames, lr=lr, n_holdout=n_holdout, directory=directory)

np.save(f"{directory}/DQNrets.npy", np.array(ep_rets))
np.save(f"{directory}/DQNh_scores.npy", np.array(holdout_scores))
dqn.save_params(f"{directory}/DQNparams.dat")
dqn.load_params(f"{directory}/DQNparams.dat")
plt.plot(ep_rets)
plt.title("Episode returns during training")
plt.show()
plt.plot(holdout_scores)
plt.title(f"Holdout scores evaluated on {n_holdout} states")
plt.show()
env.close()
