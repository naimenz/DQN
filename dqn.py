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
import torch.nn as nn
import gym
import matplotlib.pyplot as plt

# set a seed for reproducibility during implementation
torch.manual_seed(42)
# using the new recommended numpy random number routines
rng = np.random.default_rng(42)

# test TO AVOID DRAWING, code from https://stackoverflow.com/a/61694644
# It works but it's still slow as all hell
def disable_view_window():
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor
disable_view_window()

# To start with, follow a very similar architecture to what they did for Atari games
# NOTE I'm going to extend the Module class here: I've never really extended classes before so this is
# a possible failure point
class QNet(nn.Module):
    """
    This class defines the Deep Q-Network that will be used to predict Q-values of Cartpole states.

    I have defined this OUTSIDE the main class. I'll hardcode the parameters for now.
    TODO: Don't hardcode the parameters and make it good.
    """
    def __init__(self):
        super(QNet, self).__init__()
        # defining the necessary layers to be used in forward()
        # note we have four frames in an input state
        # all other parameters are basically arbitrary, but following similarly to the paper's Atari architecture
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(8,8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4,4), stride=(1,2))
        # NOTE TODO: figure out how exactly it converts from input to output channels; it's not multiplicative
        self.linear1 = nn.Linear(2112, 256)
        # NOTE: for now just hardcoding number of actions TODO: pass this in
        self.linear2 = nn.Linear(256, 2)

    # define how the net handles input
    def forward(self, x):
        conv1_relu = self.conv1(x).clamp(min=0)
        conv2_relu = self.conv2(conv1_relu).clamp(min=0)
        # flattening the output of the convs to be passed to linear
        # BUT I don't want to flatten the batch dimension, so I'll say start_dim=1
        flat = torch.flatten(conv2_relu, start_dim=1)
        # appling the linear layers 
        linear1_relu = self.linear1(flat).clamp(min=0)
        output = self.linear2(linear1_relu)
        # should be a (batch, n_acts) output of Q values
        return output

class Buffer():
    """
    Writing a class to store and sample experienced transitions.

    I am implementing it with a tensor and a set maximum size. This way
    I can keep a running counter and use modulo arithmetic to keep the buffer
    filled up with new transitions automatically.
    """
    def __init__(self, max_size):
        self.max_size = max_size# maximum number of elements to store
        self.data = torch.zeros((max_size,), dtype=torch.float)# tensor to store the actual transitions

        # I will keep track of the next index to insert at
        # Note that because this doesn't actually keep track of the state of
        # the buffer, it'll be internal and you should call .count() to see
        # how full it is
        self._counter = 0 

        # because the counter will loop, to know how many experiences we have 
        # i need to know if we've gone round already
        self.filled = False 

    # add a transition to the buffer
    # TODO: Store transitions more efficiently (currently, most frames are stored 8 times(!) )
    def add(self, transition):
        self.data[self.counter] = transition
        self._counter += 1
        # handle wrap-around
        if self._counter == self.max_size:
            self._counter = 0
            self.filled = True

    # get how many elements are in the buffer
    def count(self):
        # if we have filled already, then return max_size
        if filled:
            return self.max_size
        # else counter hasn't wrapped around yet so return it instead
        else:
            return self._counter

    # sample a random batch of experiences
    def sample(self, batch_size):
        # largest index to consider
        max_ix = self.count()
        # sample batch random indices 
        indices = torch.randint(low=0, high=max_ix, size=(batch_size,))
        samples = self.data[indices]
        return samples

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
        self.n_acts = env.action_space.n # get the number of discrete actions possible

        # quickly get the dimensions of the images we will be using
        env.reset()
        x = env.render(mode='rgb_array')
        env.close()
        self.im_dim = x.shape # this is before processing
        self.processed_dim = self.preprocess_frame(torch.as_tensor(x.copy(), dtype=torch.float)).shape
        self.state_dim = self.processed_dim + (4,) # we will use 4 most recent frames as the states

        # initialising a list to act
        self.buffer

        self.qnet = self.initialise_network()


        """
         TODO:
         - Initialise optimiser for the Q network
        """
    
    # Function to initialise the convolutional NN used to predict Q values
    def initialise_network(self):
        """
         TODO:
         - Make this more useful - i.e. calculate the values to pass in to the network
        """
        # Our images are currently 40 x 100 (height, width)
        h = 40 # height
        w = 100 # width
        n = 4 # number of frames in a state
        return QNet()


    # takes a batch of states of shape (batch, self.state_dim) as input and returns Q values as outputs
    def compute_Qs(self, s):
        # NOTE TEST
        return self.qnet(s)
    
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
    # TODO tidy up this processing, but it seems to work alright
    # Currently takes 600x400 images and gives 40x100
    def preprocess_frame(self, frame):
        # converting to greyscale(?) by just averaging pixel colors in the last dimension(is this right?)
        # print(f"Frame information: type {type(frame)}, shape {frame.shape}, dtype {frame.dtype}")
        grey_frame = torch.mean(frame, dim=-1)
        # Now I want to downsample the image in both dimensions
        downsampled_frame = grey_frame[::4, ::6]
        # Trying a crop because a lot of the vertical space is unused
        cropped_frame = downsampled_frame[40:80, :]

        return cropped_frame

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
        obs = torch.as_tensor(env.render(mode='rgb_array').copy(), dtype=torch.float)
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
            obs = torch.as_tensor(env.render(mode='rgb_array').copy(), dtype=torch.float)
            # construct new state from obsp
            s = self.get_state(s, obs)

        return ep_states, ep_acts, ep_rews

# make the environment
env = gym.make('CartPole-v0')
# initialise agent
dqn = DQN(env, gamma=0.99, init_eps=1., eval_eps=0.05)
rets = []
for i in range(10):
    ep_states, ep_acts, ep_rews = dqn.evaluate()
    rets.append(np.sum(ep_rews))
plt.plot(rets)
plt.show()
# for s in ep_states:
#     plt.imshow(s[-1])
#     plt.show()
# env.close()
