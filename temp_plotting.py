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


env = gym.make('CartPole-v0')
dqn = DQN(env, gamma=0.99, eval_eps=0.0)
dqn.load_params('run3/DQNparams.dat')
# evaluating the learned model
ep_rets  = []
for i in range(100):
    ep_states, ep_acts, ep_rews = dqn.evaluate()
    rets = rets_from_rews(ep_rews, 0.99)
    states = torch.stack(ep_states)
    Qs = dqn.compute_Qs(states)
    print(Qs)
    ep_rets.append(rets[0])
    print(i, "not random",rets[0])
print(f"Mean return on 100 episodes is {np.mean(ep_rets)}")
env.close()

# plt.plot(ep_rets)
# plt.show()
# ep_rets = np.load('run3/DQNrets.npy')
# def running_mean(x, N):
#     cumsum = np.cumsum(np.insert(x, 0, 0)) 
#     return (cumsum[N:] - cumsum[:-N]) / float(N)

# smoothed_rets = running_mean(ep_rets, 100)

# plt.plot(smoothed_rets)
# plt.show()

