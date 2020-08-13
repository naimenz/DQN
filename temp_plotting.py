"""
Temp file for taking a look at the results of the run
"""
import numpy as np
import matplotlib.pyplot
import gym

# pull DQN over
from dqn import DQN

env = gym.make('CartPole-v0')
dqn = DQN(env, gamma=0.99, eval_eps=0.05)
dqn.load_params('run1/DQNparams.dat')



