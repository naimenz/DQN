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
while t < 1000:
    env.render()
    # time.sleep(.1)
    if done:
        obs = env.reset()
        done = False
    else:
        obs, reward, done, info = env.step(a)
        if reward != 0:
            print(f"Reward: {reward} at frame {t}")
    # a = 
    a = env.action_space.sample()
    frames.append(obs)
    t += 1

to_greyscale = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
def preprocess_frame(frame):
    # greyscale is actually best done with brightness 
    frame = to_greyscale(frame)
    # Now I want to downsample the image in both dimensions
    # NOTE: maybe this isn't the best way to downsample
    frame = frame[::2, ::2]
    # Trying a crop because a lot of the vertical space is unused
    frame = frame[17:97, :]
    print(frame - 87.258)
    # I'm going to rescale so that values are in [0,1]
    frame = frame / 255
    return torch.as_tensor(frame, dtype=torch.float)



# toc = time.perf_counter()
# print(f"1000 frames of Pong took {toc - tic:0.4f} seconds")
# print(obs.shape)

frame = preprocess_frame(obs)
# print(frame)
# print(frame.shape)
# plt.subplot(121)
# plt.imshow(frame, cmap='gray', vmin=0, vmax=1)
# plt.subplot(122)
# plt.imshow(obs)
# plt.show()
# env.close()

