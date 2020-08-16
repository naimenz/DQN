"""
Running a random agent on Pong using OpenAI gym and timing it.
"""

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as tv

env = gym.make('Pong-v0')
print(f"{env.action_space.n} differnet actions")

done = True
frames = []
import time
tic = time.perf_counter()
t = 0
while t < 100:
    # env.render()
    if done:
        obs = env.reset()
        done = False
    else:
        obs, reward, done, info = env.step(a)
        if reward != 0:
            print(f"Reward: {reward}")
    a = env.action_space.sample()
    frames.append(obs)
    t += 1
toc = time.perf_counter()
print(f"1000 frames of Pong took {toc - tic:0.4f} seconds")
print(obs.shape)

to_greyscale = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 

def preprocess_frame(frame):
    # converting to greyscale(?) by just averaging pixel colors in the last dimension(is this right?)
    # print(f"Frame information: type {type(frame)}, shape {frame.shape}, dtype {frame.dtype}")
    frame = to_greyscale(frame)
    # Now I want to downsample the image in both dimensions 
    # NOTE TEST USING torch.transforms.resize to try this (this ndidn't work)
    frame = frame[::2, ::2]
    # Trying a crop because a lot of the vertical space is unused
    frame = frame[17:97, :]
    # I'm going to rescale so that values are in [0,1]
    frame = frame / 255
    return frame

frame = preprocess_frame(obs)
print(frame)
print(frame.shape)
plt.subplot(121)
plt.imshow(frame, cmap='gray', vmin=0, vmax=1)
plt.subplot(122)
plt.imshow(obs)
plt.show()
env.close()

