"""
I want to test the buffer and see if it is reconstructing states properly.
I'll try a simplified version here
"""

from dqn import Buffer
import torch

buf = Buffer(max_size=10, im_dim=(2,), state_depth=3, max_ep_len=2)

f = 0
print(f"count before: {buf.count()}")
# adding a sample transition to the buffer
t = 0
s = torch.tensor([[0.,1.], [0.,1.], [0.,1.]])
a = torch.tensor(0)
r = torch.tensor(1)
sp = torch.tensor([[0.,1.], [0.,1.], [2.,3.]])
done = False
start = True
buf.add((s, a, r, sp, done, t, f))

# adding another sample transition
f += 1
t = 1
s = torch.tensor([[0.,1.], [0.,1.], [2.,3.]])
a = torch.tensor(1)
r = torch.tensor(2)
sp = torch.tensor([[0.,1.], [2.,3.], [5.,5.]])
done = True
start = False
buf.add((s, a, r, sp, done, t, f))


print(f"count after: {buf.count()}")

# filling the buffer
# NOTE: the buffer relies on states being sequential
# so has to be a valid trajectory
for i in range(301):
    f += 1
    t = 0
    # ADDING A START STATE
    s = torch.tensor([[0.,1.], [0.,1.], [0.,1.]])
    a = torch.tensor(0)
    r = torch.tensor(1)
    sp = torch.tensor([[0.,1.], [0.,1.], [2.,3.]])
    done = False
    buf.add((s, a, r, sp, done, t, f))
    # ADDING AN END STATE
    f += 1
    t = 1
    s = torch.tensor([[0.,1.], [0.,1.], [2.,3.]])
    a = torch.tensor(1)
    r = torch.tensor(2)
    sp = torch.tensor([[0.,1.], [2.,3.], [5.,5.]])
    done = True
    buf.add((s, a, r, sp, done, t, f))

print(f"count after: {buf.count()}")
print(buf.frame_tensor)
# sampling from the buffer
n = 10
s, a, r, sp, d = buf.sample(n)
print(len(buf.frame_tensor))
for i in range(n):
    print(f"Sample {i}: s={s[i]}, a={a[i]}, r={r[i]}, sp={sp[i]}, d={d[i]}")
