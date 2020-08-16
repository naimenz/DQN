"""
Temp file for taking a look at the results of the run
"""
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
# pull DQN over
from dqn import DQN

def rets_from_rews(ep_rews, gamma):
    T = len(ep_rews) 
    rets = torch.tensor(ep_rews, dtype=torch.float)
    for i in reversed(range(T)):
        if i < T-1:
            rets[i] += gamma*rets[i+1]
    # return for final timestep is just 0
    return rets

# DIRECTORY = 'run7'

# env = gym.make('CartPole-v0')
# dqn = DQN(env, gamma=1., eval_eps=0.05)
# dqn.load_params(f'{DIRECTORY}/DQNparams.dat')
# # evaluating the learned model
# ep_rets  = []
N = 2000
# import time
# tic = time.perf_counter()

# for i in range(N):
#     ep_states, ep_acts, ep_rews = dqn.evaluate()
#     rets = rets_from_rews(ep_rews, 1.)
#     # states = torch.stack(ep_states)
#     # Qs = dqn.compute_Qs(states)
#     # print(Qs)
#     ep_rets.append(rets[0])
#     print(i, "not random",rets[0])
# toc = time.perf_counter()
# print(f"{N} episodes{N} episodes took {toc - tic:0.4f} seconds")
# print(f"On {N} episodes, mean return is {np.mean(ep_rets)}, std is {np.std(ep_rets)}")
# np.save(f"{DIRECTORY}/evaluation_rets.npy", ep_rets)
# ep_rets = np.load(f"{DIRECTORY}/evaluation_rets.npy")
ep_rets = np.load(f"temp_rets.npy")
print(f"On {N} episodes, mean return is {np.mean(ep_rets)}, std is {np.std(ep_rets)}")
print(f"On {N} episodes, median return is {np.median(ep_rets)}")
print(np.sum(ep_rets == 200))

# env.close()

# plt.scatter(range(len(ep_rets)),ep_rets, s=2)
# plt.show()
# plt.hist(ep_rets, bins=200)
# plt.show()
# ep_rets = np.load(f'{DIRECTORY}/DQNrets.npy')
# def running_mean(x, N):
#     cumsum = np.cumsum(np.insert(x, 0, 0)) 
#     return (cumsum[N:] - cumsum[:-N]) / float(N)

# smoothed_rets = running_mean(ep_rets, 20)

# plt.plot(smoothed_rets)
# plt.show()

# hscores = np.load(f'{DIRECTORY}/DQNh_scores.npy', allow_pickle=True)
# plt.plot(hscores)
# plt.show()
# # x = 0
# # for i in range(200):
#     # x = 0.99 * x + 1
