import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torch.distributions import Categorical
import random

# random.seed(5)
# print(random.random())
# define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,seed=0):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #print('F.softmax(self.fc3(x), dim=-1)',F.softmax(self.fc3(x), dim=-1))
        return F.softmax(self.fc3(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, agent_num, state_dim, action_dim,seed=0):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        input_dim = 1 + state_dim * agent_num + agent_num
       # input_dim = 1 + state_dim * agent_num
        #print('input_dim',input_dim)

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #print('self.fc3(x)',self.fc3(x))
        return self.fc3(x)
class TwinnedQNetwork(nn.Module):
    def __init__(self, agent_num, state_dim, action_dim,seed=0):
        super().__init__()
        self.Q1 = Critic(agent_num,  state_dim, action_dim,seed)
        self.Q2 = Critic(agent_num,  state_dim, action_dim,seed)

    def forward(self, x):
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        # print('q1',q1)
        # print('q2',q2)
        return q1, q2