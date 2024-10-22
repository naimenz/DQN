"""
TEMPORARY FILE FOR TESTING THE BUFFER, DO NOT MAKE PROPER CHANGES HERE
"""

import numpy as np # using numpy as sparingly as possible, mainly for random numbers but also some other things
import torch
import torch.nn as nn
import gym
import matplotlib.pyplot as plt

# set a seed for reproducibility during implementation
SEED = 42
torch.manual_seed(SEED)
# using the new recommended numpy random number routines
rng = np.random.default_rng(SEED)

# To start with, follow a very similar architecture to what they did for Atari games
# NOTE I'm going to extend the Module class here: I've never really extended classes before so this is
# a possible failure point
class QNet(nn.Module):
    """
    This class defines the Deep Q-Network that will be used to predict Q-values of PONG states.

    I have defined this OUTSIDE the main class. I'll hardcode the parameters for now.
    TODO: Don't hardcode the parameters and make it good.
    """
    def __init__(self, n_outputs):
        super(QNet, self).__init__()
        # defining the necessary layers to be used in forward()
        # note we have four frames in an input state
        # parameters are (ALMOST) the same as used in the original DQN paper
        # NOTE Because I have 80*80 images instead of 84 by 84, I will pad by 2 to
        # get the same shape moving forward
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(8,8), stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4,4), stride=2)
        # NOTE TODO: figure out how exactly it converts from input to output channels; it's not multiplicative
        self.linear1 = nn.Linear(2592, 256)
        # NOTE NOTE: Pong has SIX legal actions but only TWO actually do anything useful
        self.linear2 = nn.Linear(256, n_outputs)

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

    NOTE: I am completely rewriting this because it's just using too much memory.
    Instead of storing full transitions, I will store action, rewards, and terminal flags
    as before but store frames individually and reconstruct states as needed.
    This will be slightly more fiddly but be roughly 8x more efficient
    """

    def __init__(self, max_size, im_dim, state_depth, max_ep_len):
        self.max_size = max_size # maximum number of elements to store
        self.im_dim = im_dim # dimensions of each frame
        self.state_depth = state_depth # number of frames in a state
        # storing FRAMES instead of STATES
        # Actually we need to store enough to keep an entire extra episode because otherwise we could 
        # overwrite transitions in a current episode
        self.frame_size = max_size + max_ep_len
        self.frame_tensor = torch.empty(size=(self.frame_size,) + im_dim, dtype=torch.float)

        self.a_tensor = torch.LongTensor(size=(max_size,))
        self.r_tensor = torch.empty(size=(max_size,), dtype=torch.float)
        self.d_tensor = torch.empty(size=(max_size,), dtype=torch.bool)
        
        # keep track of which frames we need to construct the states
        self.s_ix_tensor = torch.LongTensor(size=(max_size, state_depth))
        self.sp_ix_tensor = torch.LongTensor(size=(max_size, state_depth))

        # I will keep track of the next index to insert at
        # Note that because this doesn't actually keep track of the state of
        # the buffer, it'll be internal and you should call .count() to see
        # how full it is
        self._counter = 0 

        # because the counter will loop, to know how many experiences we have 
        # i need to know if we've gone round already
        self.filled = False 

    # add a transition to the buffer
    # We pass the state, the action, the reward, the next state, and then three extras:
    # done: is sp terminal?
    # t: time from start of episode (if it's greater than state_depth then it doesn't matter)
    # fcount: the frame of training we are on currently
    def add(self, transition):
        s, a, r, sp, done, t, fcount = transition
        # the index of a, r, and done tensors to fill up, and what we aim to reconstruct
        counter = self._counter
        # writing to the easy buffers
        self.a_tensor[counter] = a
        self.r_tensor[counter] = r
        self.d_tensor[counter] = done
        # handling the hard frame buffer differently depending on if this is the initial state
        # if this is the first transition, we need to store frames for initial obs and first step
        # these will be the LAST two frames of sp
        if t == 0:
            # NOTE: This only works because the done flag tells us if we care about the sp state
            # and if we don't it doesn't matter what's in the buffer there so we can overwrite fcount
            self.frame_tensor[fcount % self.frame_size] = s[-1]
            self.frame_tensor[(fcount + 1) % self.frame_size] = sp[-1]
        # if this isn't the first transition, we only need to store the new frame sp[-1]
        else:
            self.frame_tensor[(fcount + 1) % self.frame_size] = sp[-1]

        # NOW we have to construct the tuples that tell us which frames to access
        # if we are near the beginning of the episode, we need multiple copies of the first frame
        s_ix = []
        if t < self.state_depth - 1:
            # first frame will be fcount - t
            first_frame_ix = fcount - t
            # we need state_depth - t of these 
            s_ix += [first_frame_ix] * (self.state_depth - t)
            # we'll need the t next frames as well
            for i in range(1, t+1):
                s_ix.append(first_frame_ix + i)
        # otherwise we just need the 'state_depth' states from fcount-self.state_depth+1 up to 
        # and including fcount
        else:
            for i in range(1, self.state_depth+1):
                s_ix.append(fcount - self.state_depth + i)
        # in both cases, sp_ix is just all but the first of s_ix with fcount + 1 o nthe front
        sp_ix = s_ix[1:] + [fcount+1]
        self.s_ix_tensor[counter] = torch.tensor(s_ix, dtype=torch.long)
        self.sp_ix_tensor[counter] = torch.tensor(sp_ix, dtype=torch.long)

        self._counter += 1
        # handle wrap-around
        if self._counter == self.max_size:
            self._counter = 0
            self.filled = True

    # get how many elements are in the buffer
    def count(self):
        # if we have filled already, then return max_size
        if self.filled:
            return self.max_size
        # else counter hasn't wrapped around yet so return it instead
        else:
            return self._counter

    # sample a random batch of experiences
    # NOTE this is tricky now because I need to reconstruct the states
    def sample(self, batch_size):
        # largest index to consider
        max_ix = self.count()
        # sample batch random indices 
        indices = torch.randint(low=0, high=max_ix, size=(batch_size,))
        a_sample = self.a_tensor[indices]
        r_sample = self.r_tensor[indices]
        d_sample = self.d_tensor[indices]
        # now I need to reconstruct s and sp
        s_ix_sample = self.s_ix_tensor[indices]
        sp_ix_sample = self.sp_ix_tensor[indices]
        # get the corresponding frames
        ftens = self.frame_tensor
        # TODO: make this not look so ABSOLUTELY horrible
        s_sample = torch.stack([torch.stack([ftens[i % self.frame_size] for i in ix]) for ix in s_ix_sample])
        sp_sample = torch.stack([torch.stack([ftens[i % self.frame_size] for i in ix]) for ix in sp_ix_sample])

        return s_sample, a_sample, r_sample, sp_sample, d_sample

class DQN():
    """
    DQN class specifically for solving Pong
    Might work on other Atari continuous obs, discrete act environments too if it works on Pong.
    """
    # we initialise with an environment for now, might need to add architecture once
    # I get the NN working.
    def __init__(self, env, gamma, eval_eps):
        self.eval_eps = eval_eps # epsilon to be used at evaluation time (typically lower than training eps)
        self.env = env
        self.gamma = gamma # discount rate (I think they used 0.99)
        # NOTE TEST: replacing the in-built actions with just THREE (including no-op) that are actually useful in Pong
        # NOTE TEST: declaring an action set to to use
        self.action_set = [0,2,5]
        self.n_acts = len(self.action_set)
        # self.n_acts = env.action_space.n # get the number of discrete actions possible

        # NOTE TEST: setting a maximum episode length of 2000 (will print episode lengths as I go though)
        self.max_ep_len = 2000
        # function to convert to greyscale (takes np frames)
        self.to_greyscale = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 

        # quickly get the dimensions of the images we will be using
        x = env.reset()
        self.im_dim = x.shape # this is before processing
        self.processed_dim = self.preprocess_frame(x).shape
        self.state_depth = 4 # number of frames in a state, hardcoded for now
        self.state_dim = (self.state_depth,) + self.processed_dim # we will use 4 most recent frames as the states

        # initialise the network TODO pass in params
        # now passing in n_acts at least
        self.qnet = self.initialise_network(n_outputs=self.n_acts)
    
    # Function to initialise the convolutional NN used to predict Q values
    def initialise_network(self, n_outputs):
        """
         TODO:
         - Make this more useful - i.e. calculate the values to pass in to the network
        """
        return QNet(n_outputs=n_outputs)

    # takes a batch of states of shape (batch,)+ self.state_dim as input and returns Q values as outputs
    # simple wrapper really
    def compute_Qs(self, s):
        return self.qnet(s)
    
    # get action for a state based on a given eps value
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
    # NOTE TEST: shifting color values so that the background is 0
    def preprocess_frame(self, frame):
        # greyscale is actually best done with brightness 
        frame = self.to_greyscale(frame)
        # Now I want to downsample the image in both dimensions
        # NOTE: maybe this isn't the best way to downsample
        frame = frame[::2, ::2]
        # Trying a crop because a lot of the vertical space is unused
        frame = frame[17:97, :]
        # NOTE TEST: subtracting background grey colour (87.258)
        frame = frame - 87.258
        # I'm going to rescale so that min and max values are at most 1 apart
        frame = frame / 255
        return torch.as_tensor(frame, dtype=torch.float)

    # encode the observation into a state based on the previous state and current obs
    def get_phi(self, s, obs):
        processed_frame = self.preprocess_frame(obs)
        # bump oldest frame and add new one
        # we do this by dropping the first element of s and concatenating the new frame
        # note that we have to unsqueeze the new frame so it has the same dimensions as s
        sp = torch.cat((s[1:], processed_frame.unsqueeze(0)))
        return sp
    
    # create an initial state by stacking four (or self.state_depth) copies of the first frame
    def initial_phi(self, obs):
        f = self.preprocess_frame(obs).unsqueeze(0)
        s = torch.cat(self.state_depth * (f,))
        return s

    # run an episode in evaluation mode 
    def evaluate(self):
        # alias env
        env = self.env
        
        # lists for logging
        ep_states = []
        ep_acts = []
        ep_rews = []

        # reset environment
        done = False
        obs = env.reset()
        # because we only have one frame so far, just make the initial state 4 copies of it
        s = self.initial_phi(obs)

        # loop over steps in the episode
        while not done:
            act = self.get_act(s, self.eval_eps) # returns a 1-element tensor
            # NOTE TEST: converting an action in (0,1,2) into 0,2,5 (stay still, up and down in atari)
            av = self.action_set[act]
            obs, reward, done, info = env.step(av) 

            # log state, act, reward
            ep_states.append(s)
            ep_acts.append(act.item())
            ep_rews.append(reward)

            # construct new state from new obs
            s = self.get_phi(s, obs)

        return ep_states, ep_acts, ep_rews

    # generate a set of holdout states randomly for use as validation
    def generate_holdout(self, N):
        # we will store N states, each of size state_dim
        states = torch.empty(size=(N,) + self.state_dim, dtype=torch.float)

        t = 0 # frame counter
        done = True # indicate that we should restart episode immediately

        # alias env
        env = self.env
        
        # while we haven't seen enough frames
        while t < N:
            import time
            time.sleep(0.1)
            if done: # reset environment for a new episode
                done = False
                obs = env.reset()
                # because we only have one frame so far, just make the initial state 4 copies of it
                s = self.initial_phi(obs)

            # save the state
            states[t] = s
            # generate a random action given the current state
            act = self.get_act(s, 1.)
            # NOTE TEST: converting an action in (0,1,2) into 0,2,5 (stay still, up and down in atari)
            av = self.action_set[act]
            # act in the environment
            obs, reward, done, _ = env.step(av)

            # get the next state
            s = self.get_phi(s, obs)
            t += 1
        return states

    # evaluate the current Q function on a set of holdout states
    # we return the mean of the maximum Q for each state
    def evaluate_holdout(self, holdout):
        Qmax = torch.max(self.compute_Qs(holdout), dim=1)[0]
        return torch.mean(Qmax)

    # functions to save and load a model
    def save_params(self, filename):
        s_dict = self.qnet.state_dict()
        torch.save(s_dict, filename)
        print(f"Parameters saved to {filename}")

    def load_params(self, filename):
        s_dict = torch.load(filename)
        self.qnet.load_state_dict(s_dict)
        print(f"Parameters loaded from {filename}")

    # compute loss on a batch of transitions
    # gradient of this should be what we need
    def compute_loss(self, s, a, r, sp, d):
        # don't need gradients except for Q(s,a)
        all_q = self.compute_Qs(s)
        q = all_q[range(len(s)), a]
        with torch.no_grad():
            # get the 'values' part of the max function and drop the 'indices'
            qsp = torch.max(self.compute_Qs(sp), dim=1)[0]
            # setting terminal states to 0
            qsp[d] = 0
            # getting Q values for actions
            targets = r + self.gamma * qsp - q
        # loss is mean of targets * Q
        loss = -torch.mean(targets * q)
        return loss

    # given a minibatch of transitions, compute the sample gradient and take
    # the step
    def update_minibatch(self, minibatch, optim):
        optim.zero_grad()
        loss = self.compute_loss(*minibatch)
        loss.backward()
        optim.step()

    # run the entire training algorithm for N FRAMES, not episodes
    # in line with their parameters, we will decrease eps from 1 to 0.1 linearly over the first
    # 10% of frames, and keep a buffer of 10% of frames
    def train(self, N=10000, lr=1e-6, n_holdout=100, directory=None):
        n_evals = 50 # number of times to evaluate during training
        # at the beginning of training we generate a set of 100(?) holdout states
        # to be used to estimate the performance of the algorithm based
        # on its average Q on these states
        tenth_N = int(N/10)
        buf_size = tenth_N
        eps_epoch = tenth_N
        # NOTE: episodes can go a lot longer in Pong
        buf = Buffer(max_size=buf_size, im_dim=self.processed_dim, state_depth=self.state_depth, max_ep_len=self.max_ep_len)

        # alias env
        env = self.env

        print("=============================")
        print(f"BEGINNING TRAINING with N={N}, lr={lr}, n_holdout={n_holdout}")
        print(f"buffer size={buf_size}, epsilon epoch={eps_epoch}")
        print("=============================")
        import time
        tic = time.perf_counter()
        holdout = self.generate_holdout(N=n_holdout)
        toc = time.perf_counter()
        print(f"Generating holdout took {toc - tic:0.4f} seconds")


        # starting and ending epsilon
        eps0 = 1.
        eps1 = 0.1
        epstep = (eps1 - eps0)/eps_epoch # the quantity to add to eps every frame
        get_eps = lambda t: eps0 + t*epstep if t < eps_epoch else eps1

        # I'm going to try initialising the optimiser in this function,
        # as it isn't needed outside of training.
        # I'm using RMSProp because they used RMSProp in the paper and i'm lazy
        # NOTE TEST: Using Adam instead just to see what happens
        optim = torch.optim.Adam(self.qnet.parameters(), lr=lr)

        # NOTE LOG: I'm going to track return per episode for testing
        ep_ret = 0
        ep_rets = []
        holdout_scores = []
        recent_eps = 0

        t = 0 # frame counter
        done = True # indicate that we should restart episode immediately
        
        # NOTE LOG: timing
        import time
        bigtic = time.perf_counter()
        tic = time.perf_counter()
        
        # while we haven't seen enough frames
        while t < N:
            env.render()
            time.sleep(0.1)
            # NOTE LOG: evaluating 50 times throughout training
            if n_evals*t % N == 0 and t > 0:
                liltic = time.perf_counter()
                h_score = self.evaluate_holdout(holdout)
                liltoc = time.perf_counter()
                holdout_scores.append(h_score)
                toc = time.perf_counter()
                print(
                f"""============== FRAME {t}/{N} ============== 
Last {N/n_evals} frames took {toc - tic:0.4f} seconds.
Mean of recent episodes is {np.mean(ep_rets[recent_eps:])}.
Score on holdout is {h_score}.
                """)
                print(f"computing Qs on {n_holdout} holdout states took {liltoc - liltic:0.4f} seconds")

                # set new recent eps
                recent_eps = len(ep_rets)
                # NOTE LOG saving the stats so far 
                if not directory is None:
                    np.save(f"{directory}/DQNrets.npy", np.array(ep_rets))
                    np.save(f"{directory}/DQNh_scores.npy", np.array(holdout_scores))
                    # NOTE LOG I will overwrite parameters each time because they are big
                    self.save_params(f"{directory}/DQNparams.dat")
                    # save parameters separately 10 times
                    if 10*t % N == 0:
                        self.save_params(f"{directory}/{t}DQNparams.dat")
                tic = time.perf_counter()

            if done: # reset environment for a new episode
                # NOTE LOG: tracking episode return
                if t > 0: # if this isn't the first episode
                    ep_rets.append(ep_ret)
                    # NOTE LOG: printing the length of the previous episode
                    print(f"Episode {len(ep_rets)} had length {ep_t}")
                ep_ret = 0

                # NOTE tracking episode time 
                ep_t = 0

                done = False
                start = True
                obs = env.reset()
                # because we only have one frame so far, just make the initial state 4 copies of it
                s = self.initial_phi(obs)

            # generate an action given the current state
            eps = get_eps(t)
            act = self.get_act(s, eps)
            # NOTE TEST: converting an action in (0,1,2) into 0,2,5 (stay still, up and down in atari)
            av = self.action_set[act]

            # act in the environment
            obsp, reward, done, _ = env.step(av)

            # NOTE LOG: tracking episode return
            ep_ret = ep_ret + (self.gamma**ep_t) * reward

            # get the next state
            sp = self.get_phi(s, obsp)
            # print("DRAWING NEXT STATE")
            # print(sp)
            # plt.subplot(221)
            # plt.imshow(sp[0] + 0.34218823529, vmin=0, vmax=1, cmap='gray')
            # plt.colorbar()
            # plt.title("s frame 0")
            # plt.subplot(222)
            # plt.imshow(sp[1] + 0.34218823529, vmin=0, vmax=1, cmap='gray')
            # plt.title("s frame 1")
            # plt.subplot(223)
            # plt.imshow(sp[2] + 0.34218823529, vmin=0, vmax=1, cmap='gray')
            # plt.title("s frame 2")
            # plt.subplot(224)
            # plt.imshow(sp[3] + 0.34218823529, vmin=0, vmax=1, cmap='gray')
            # plt.title("s frame 3")
            # plt.show()

            # add all this to the experience buffer
            # PLUS the done flag so I know if sp is terminal
            # AND the various times
            buf.add((s, act, reward, sp, done, ep_t, t))
            # if we have added to the buffer then we are no longer in the first state
            start = False

            # NOW WE SAMPLE A MINIBATCH and update on that
            minibatch = buf.sample(batch_size=32)
            # wait until first frames are mostly out of buffer
            if t > 50:
                ss = minibatch[0]
                sps = minibatch[3]
                for i in range(32):
                    print("===================")
                    print(f"DRAWING {i} S AND SP OF MINIBATCH")
                    action_i = minibatch[1][i]
                    print(f"Action: {action_i}")
                    print(f"Reward: {minibatch[2][i]}")
                    print(f"Done: {minibatch[4][i]}")

                    if action_i == 1:
                        plt.subplot(221)
                        plt.imshow(ss[i][0] + 0.34218823529, vmin=0, vmax=1, cmap='gray')
                        plt.colorbar()
                        plt.title("s frame 0")
                        plt.subplot(222)
                        plt.imshow(ss[i][1] + 0.34218823529, vmin=0, vmax=1, cmap='gray')
                        plt.title("s frame 1")
                        plt.subplot(223)
                        plt.imshow(ss[i][2] + 0.34218823529, vmin=0, vmax=1, cmap='gray')
                        plt.title("s frame 2")
                        plt.subplot(224)
                        plt.imshow(ss[i][3] + 0.34218823529, vmin=0, vmax=1, cmap='gray')
                        plt.title("s frame 3")
                        plt.show()
                        plt.subplot(221)
                        plt.imshow(sps[i][0] + 0.34218823529, vmin=0, vmax=1, cmap='gray')
                        plt.title("sp frame 0")
                        plt.subplot(222)
                        plt.imshow(sps[i][1] + 0.34218823529, vmin=0, vmax=1, cmap='gray')
                        plt.title("sp frame 1")
                        plt.subplot(223)
                        plt.imshow(sps[i][2] + 0.34218823529, vmin=0, vmax=1, cmap='gray')
                        plt.title("s frame 2")
                        plt.subplot(224)
                        plt.imshow(sps[i][3] + 0.34218823529, vmin=0, vmax=1, cmap='gray')
                        plt.title("sp frame 3")
                        plt.show()
            self.update_minibatch(minibatch, optim)

            # prepare for next frame
            t += 1
            ep_t += 1 # updating the episode time as well
            s = sp

        bigtoc = time.perf_counter()
        print(f"ALL TRAINING took {bigtoc - bigtic:0.4f} seconds")
        # NOTE LOG: tracking episode returns
        return ep_rets, holdout_scores
