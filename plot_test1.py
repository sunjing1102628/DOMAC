import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

'''dt = 0.01
t = np.arange(0, 30, dt)
nse1 = np.random.randn(len(t))                 # white noise 1
nse2 = np.random.randn(len(t))                 # white noise 2

# Two signals with a coherent part at 10Hz and a random part
s1 = np.sin(2 * np.pi * 10 * t) + nse1
s2 = np.sin(2 * np.pi * 10 * t) + nse2

fig, axs = plt.subplots(figsize=(10,5))
base_line1, = axs.plot(t, s1, 'r')
base_line2, = axs.plot(t, s2, 'b')  
#axs.fill_between(x, y - z/2, y + z/2, facecolor=base_line.get_color(), alpha=0.5)
#axs.plot(t, s1, t, s2)
axs.set_xlim(0, 1)
axs.set_xlabel('time')
axs.set_ylabel('s1 and s2')
axs.grid(True)

# cxy, f = axs[1].cohere(s1, s2, 256, 1. / dt)
# axs[1].set_ylabel('coherence')

#fig.tight_layout()
plt.show()'''
import torch
import torch.nn.functional as F
P = torch.Tensor([0.36, 0.48, 0.16])
Q = torch.Tensor([0.333, 0.333, 0.333])
est_dist1 = torch.Tensor([0.2,0.2,0.2,0.2,0.2])
est_dist2 = torch.Tensor([0.15,0.15,0.15,0.15,0.4])
est_dist3 = torch.Tensor([0.155,0.155,0.155,0.155,0.38])
est_dist4 = torch.Tensor([0.16,0.16,0.16,0.16,0.36])
est_dist5 = torch.Tensor([0.1625,0.1625,0.1625,0.1625,0.35])
target_dist = torch.Tensor([0.175, 0.175, 0.175, 0.175, 0.3])
(P * (P / Q).log()).sum()
# tensor(0.0863), 10.2 µs ± 508

#a=F.kl_div(Q.log(), P, None, None, 'sum')
a= F.kl_div(target_dist.log(),est_dist1,None, None, 'sum')
b= F.kl_div(target_dist.log(),est_dist2,None, None, 'sum')
c= F.kl_div(target_dist.log(),est_dist3,None, None, 'sum')
d= F.kl_div(target_dist.log(),est_dist4,None, None, 'sum')
e= F.kl_div(target_dist.log(),est_dist5,None, None, 'sum')
print('a',a)
print('b',b)
print('c',c)
print('d',d)
print('e',e)