"""
This is my attempt to implement DQN on the cartpole-v0 environment from OpenAI gym.

To force myself to get better at version control, I will develop it all in this one file instead
of making backups each time I change something.

I have a trello board https://trello.com/b/iQUDEFxL/dqn
and a github repo https://github.com/naimenz/DQN

Building on the general structure I used for Sarsa/REINFORCE-type algorithms, I'll write a class to hold
all the important bits and share parameters and stuff.
"""

import numpy as np # using numpy as sparingly as possible, mainly for random numbers
import torch
import gym
import matplotlib.pyplot as plt

# set a seed for reproducibility during implementation
torch.manual_seed(42)
# using the new recommended numpy random number routines
rng = np.random.default_rng(42)

class DQN():
    """
    DQN class specifically for solving Cartpole. 
    Might work on other simple continuous obs, discrete act environments too if it works on Cartpole.
    """
    # we initialise with an environment for now, might need to add architecture once
    # I get the NN working.
    def __init__(self, env, gamma, init_eps, eval_eps):
        self.eps = init_eps # initial epsilon for use in eps-greedy policy: random action is chosen eps of the time
        self.eval_eps = eval_eps # epsilon to be used at evaluation time (typically lower than training eps)
        self.env = env
        self.gamma = gamma # discount rate (I think they used 0.99)
        self.n_acts = env.action_space.n
        # quickly get the dimensions of the images we will be using
        # unfortunately this will flicker the image on the screen at the moment
        env.reset()
        x = env.render(mode='rgb_array')
        env.close()
        self.im_dim = x.shape # we won't actually be using these observations though
        self.state_dim = self.im_dim + (4,) # we will use 4 most recent frames as the states

        """
         TODO:
         - Initialise Q network
         - Initialise optimiser for the Q network
        """

    # takes a batch of states of shape (batch, self.state_dim) as input and returns Q values as outputs
    def compute_Qs(self, s):
        """
         TODO:
         - Compute Qs with neural network
        """
        # TODO For now, just return 0s across the board
        batch = s.shape[0]
        return torch.zeros((batch, self.n_acts))
    
    # get action for a state based on a given eps value)
    # NOTE does not work with batches of states
    def get_act(self, s, eps):
        # 1 - eps of the time we pick the action with highest Q
        if rng.uniform() > eps:
            Qs = self.compute_Qs(s.unsqueeze(0)) # we have to add a batch dim
            act = torch.argmax(Qs)
        # rest of the time we just pick randomly among the n_acts actions
        else:
            act = torch.randint(0, self.n_acts, (1,))
        return act

    # preprocess a frame for use in constructing states
    # NOTE TODO At the moment I am not actually using state information (random agent)
    # so just return the frame unaltered
    def preprocess_frame(self, frame):
        """
         TODO:
         - Convert to greyscale
         - Downsample/crop as needed
        """
        # TODO placeholder: just return frame
        return frame

    # encode the observation into a state based on the previous state and current obs
    def get_state(self, s, obs):
        processed_frame = self.preprocess_frame(obs)
        # bump oldest frame and add new one
        # we do this by dropping the first element of s and concatenating the new frame
        # note that we have to unsqueeze the new frame so it has the same dimensions as s
        sp = torch.cat((s[1:], processed_frame.unsqueeze(0)))
        return sp
    
    # create an initial state by stacking four copies of the first frame
    def initial_state(self, obs):
        f = self.preprocess_frame(obs).unsqueeze(0)
        s = torch.cat((f,f,f,f))
        return s

    # run an episode in evaluation mode 
    def evaluate(self):
        # lists for logging
        ep_states = []
        ep_acts = []
        ep_rews = []

        # reset environment
        done = False
        _ = env.reset()
        # get frame of the animation
        obs = torch.from_numpy(env.render(mode='rgb_array').copy())
        # because we only have one frame so far, just make the initial state 4 copies of it
        s = self.initial_state(obs)

        # loop over steps in the episode
        while not done:
            act = self.get_act(s, self.eval_eps) # returns a 1-element tensor
            _, reward, done, info = env.step(act.item()) # throw away their obs

            # log state, act, reward
            ep_states.append(s)
            ep_acts.append(act.item())
            ep_rews.append(reward)

            # get frame of the animation
            obs = torch.from_numpy(env.render(mode='rgb_array').copy())
            # construct new state from obsp
            s = self.get_state(s, obs)

        return ep_states, ep_acts, ep_rews

# make the environment
env = gym.make('CartPole-v0')
# initialise agent
dqn = DQN(env, gamma=0.99, init_eps=1., eval_eps=1.)
ep_states, ep_acts, ep_rews = dqn.evaluate()
for s in ep_states:
    plt.imshow(s[-1])
    plt.show()
env.close()
