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
    time.sleep(.1)
    if done:
        obs = env.reset()
        done = False
    else:
        obs, reward, done, info = env.step(a)
        if reward != 0:
            print(f"Reward: {reward}")
    a = 5
    # a = env.action_space.sample()
    frames.append(obs)
    t += 1
# toc = time.perf_counter()
# print(f"1000 frames of Pong took {toc - tic:0.4f} seconds")
# print(obs.shape)

# frame = preprocess_frame(obs)
# print(frame)
# print(frame.shape)
# plt.subplot(121)
# plt.imshow(frame, cmap='gray', vmin=0, vmax=1)
# plt.subplot(122)
# plt.imshow(obs)
# plt.show()
# env.close()

