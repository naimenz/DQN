"""
File for running the experiments, kept separate from class definitions.
NOTE: Rewritten for pausable training

TODO: see if there are issues with random number generation between files etc.
"""
import numpy as np # using numpy as sparingly as possible, mainly for random numbers but also some other things
import torch
import torch.nn as nn
import gym
import matplotlib.pyplot as plt
from dqn import DQN
# for making directory if it doesn't exist
from pathlib import Path

# set a seed for reproducibility during implementation
SEED = 42
torch.manual_seed(SEED)
# using the new recommended numpy random number routines
rng = np.random.default_rng(SEED)

# make the environment and seed it
env = gym.make('Pong-v0', frameskip=4)
env.seed(SEED)

# initialise agent
dqn = DQN(env, gamma=0.999, eval_eps=0.05)

# train
n_frames = 10000
lr = 1e-5
n_holdout = 100
n_eval_eps = 2
# save output
directory = 'experiments/run8'
# make directory if it doesn't exist (if it does exist, throw an error so I have to switch lines if I wnat to overwrite
# Path(directory).mkdir(parents=True, exist_ok=False)
Path(directory).mkdir(parents=True, exist_ok=True)

# TESTING TRAIN FROM STATE
state = torch.load(f"{directory}/saved_state.tar")
print(state['current_time'])
# ep_rets, holdout_scores = dqn.train_from_state(state)
ep_rets, holdout_scores = dqn.train(N=n_frames, lr=lr, n_holdout=n_holdout, n_eval_eps=n_eval_eps, directory=directory)

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
