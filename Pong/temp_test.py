""" 
Testing file, trying to see if my 'get phi' function is broken
Get phi structure seems sound

Also testing buffer
"""
import torch
from temp_dqn import DQN
import gym

li = []
old = torch.tensor([[[0.,0.]],[[1.,1.]],[[2.,2.]],[[3.,3.]]])
li.append(old)
print(f"Old in list: {li[0]}")
extra = torch.tensor([[4.,4.]])

# takes in 'old' state, returns a new state with 'extra' on the end 
def test(old, extra):
    new = torch.cat((old[1:], extra.unsqueeze(0)))
    return new

old = test(old, extra)
print(f"Old after reassignment: {old}")
print(f"Old in list after operation: {li[0]}")


# Add to a buffer and see what happens
env = gym.make('Pong-v0', frameskip=4)
dqn = DQN(env, gamma=0.99, eval_eps=0.05)
dqn.train(1000, lr=1e-4, n_holdout=1, directory=None)
# max_size = 10
# im_dim = dqn.processed_dim
# state_depth = dqn.state_depth
# max_ep_len = dqn.max_ep_len

# buf = Buffer(max_size, im_dim, state_depth, max_ep_len)
# print(buf)
# # fill up the buffer with transitions
# N = 100
