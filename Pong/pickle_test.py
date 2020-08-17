"""
Testing pickling objects in Python
"""
from dqn import Buffer
import pickle
import torch
import os

class Foo():
    def __init__(self, data):
        self.data = data

    def sum(self):
        return sum(self.data)
# a = [1,2,3]
# a = Foo([1,2,3])
max_size = 10000
im_dim = (80,80)
state_depth=4
max_ep_len=2000

# a = Buffer(max_size, im_dim, state_depth, max_ep_len)
# s = torch.tensor([[[0.]], [[1.]], [[2.]], [[3.]]])
# act = 10
# r = 0.5
# sp = torch.tensor([[[1.]], [[2.]], [[3.]], [[4.]]])
# done = False
# ep_t = 0
# fcount = 0
# for i in range(1000):
#     a.add((s,act,r,sp,done,ep_t,fcount))
#     ep_t += 1
#     fcount += 1
a = torch.optim.Adam([torch.tensor([1.])], lr=1e-6)
print(a.param_groups[0]['lr'])
stuff = {'buffer': a}
# with open('dumpfile.dat', 'wb') as f:
#     pickle.dump(stuff, f)

# with open('dumpfile.dat', 'rb') as f:
#     b = pickle.load(f)
torch.save(stuff, "torchfile.dat")
# size = os.path.getsize('dumpfile.dat')
b = torch.load('torchfile.dat')
torchsize = os.path.getsize('torchfile.dat')

def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0

# print("pickle",convert_bytes(size))
print("torhc",convert_bytes(torchsize))
print(b['buffer'])
print(b['buffer'].sample(1))

# TESTING CATCHING CTRL+C
import signal
import sys

# def signal_handler(sig, frame):
#     print('You pressed Ctrl+C!')
#     sys.exit(0)
# signal.signal(signal.SIGINT, signal_handler)
# print('Press Ctrl+C')
# import time
# while 1:
#     try:
#         print("looping")
#         time.sleep(.5)
#     except (KeyboardInterrupt, SystemExit):
#         print("shutting down")
#         input("Press Ctrl-C to exit, enter key to resume")
