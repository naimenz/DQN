from dqn import DQN
import matplotlib.pyplot as plt
import numpy as np
import torch
import gym

def rets_from_rews(ep_rews, gamma):
    T = len(ep_rews) 
    rets = torch.tensor(ep_rews, dtype=torch.float)
    for i in reversed(range(T)):
        if i < T-1:
            rets[i] += gamma*rets[i+1]
    # return for final timestep is just 0
    return rets

directory = 'experiments/run7'

ep_rets = np.load(f"{directory}/DQNrets.npy")
plt.plot(ep_rets)
plt.show()

gamma = 0.999
env = gym.make('Pong-v0', frameskip=4)
dqn = DQN(env, gamma=gamma, eval_eps=0.05)
# making a dqn agent that always acts randomly
random = DQN(env, gamma=gamma, eval_eps=1.)
dqn.load_params(f"{directory}/DQNparams.dat")
undisc_rets = []
disc_rets = []
random_undisc = []
random_disc = []

# evaluate on N episodes, and print idscounted and undiscounted return
N = 30
# for i in range(N):
#     ep_states, ep_acts, ep_rews = dqn.evaluate(render=0.1)
#     # print("Qs:",dqn.compute_Qs(torch.stack(ep_states)))
#     # print("ACTS:",ep_acts)
#     # undisc_rets.append(rets_from_rews(ep_rews, 1.)[0])
#     # disc_rets.append(rets_from_rews(ep_rews, gamma)[0])
#     # ep_states, ep_acts, ep_rews = random.evaluate(render=0.1)
#     # random_undisc.append(rets_from_rews(ep_rews, 1.)[0])
#     # random_disc.append(rets_from_rews(ep_rews, gamma)[0])
#     print("=========================")
#     # print(f"Episode {i}, our agent got undiscounted:{undisc_rets[-1]}, discounted:{disc_rets[-1]}")
#     print(f"Episode {i}, random got undiscounted:{random_undisc[-1]}, discounted:{random_disc[-1]}")

# np.save('temp_disc.npy', disc_rets)
# np.save('temp_undisc.npy', undisc_rets)
# np.save('random_disc.npy', random_disc)
# np.save('random_undisc.npy', random_undisc)

disc_rets = np.load('temp_disc.npy')
undisc_rets = np.load('temp_undisc.npy')
random_disc = np.load('random_disc.npy')
random_undisc = np.load('random_undisc.npy')
print(f" our agent got undiscounted:{np.mean(undisc_rets)}, discounted:{np.mean(disc_rets)}")
print(f" our agent got std deviations undiscounted:{np.std(undisc_rets)}, discounted:{np.std(disc_rets)}")
print(f" random got undiscounted:{np.mean(random_undisc)}, discounted:{np.mean(random_disc)}")

plt.subplot(121)
plt.plot(undisc_rets, label="Learned")
plt.plot(random_undisc, label="Random")
plt.legend()
plt.title("Undiscounted eval returns")
plt.subplot(122)
plt.plot(disc_rets, label="Learned")
plt.plot(random_disc, label="Random")
plt.legend()
plt.title("Discounted eval returns")
plt.show()
