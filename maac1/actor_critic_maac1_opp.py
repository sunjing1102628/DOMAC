import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torch.distributions import Categorical
import random

random.seed(5)
print(random.random())
# define the actor network
class Opp_Actor(nn.Module):
    def __init__(self, state_dim, action_dim,seed=0):
        super(Opp_Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        opp_action_prob= F.softmax(self.fc3(x), dim=-1)
        #print('opp_action_prob',opp_action_prob)
        return opp_action_prob
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,seed=0):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_dim+action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.opp_actor=Opp_Actor(state_dim,action_dim)

    def forward(self, x):
        opponent_num = torch.arange(0, 5)
        # print('opp_num',opponent_num)

        opponent_action_num = F.one_hot(opponent_num, num_classes=len(opponent_num))
        # print('opponent_action_num',opponent_action_num)
        #print('x is',x)
        #print('x.repeat(1, 5).reshape(len(x),5,28)',x.repeat(1, 5).reshape(len(x),5,28))

       # print('x.repeat(1, 5)',opponent_action_num.repeat(1, len(x)).reshape(len(x),5, 5))
        #print('opponent_action_num',opponent_action_num.repeat(1, len(x)).reshape(5,len(x),5).size())
        '''a = torch.cat([x.repeat(1, 5).reshape(5, len(x)),
                       opponent_action_num.to(x.device)],
                      dim=1)'''


        a = torch.cat([x.repeat(1, 5).reshape(len(x),5,28),
                       opponent_action_num.repeat(1, len(x)).reshape(len(x),5, 5).to(x.device)],
                      dim=2).squeeze(1)
       # print('a is',a.size())
        a = F.relu(self.fc1(a))

        a = F.relu(self.fc2(a))
        agent_action_probs= F.softmax(self.fc3(a), dim=-1)
       # print('agent_action_prob',agent_action_probs.size())
        opp_actions_probs = self.opp_actor(x)
        #print('opp',opp_actions_probs.size())
        opp_action_entropy = Categorical(opp_actions_probs).entropy().squeeze(0)
        opp_action_dist_tets = opp_actions_probs.detach()
        prey_move_probs = torch.tensor([0.175, 0.175, 0.175, 0.175, 0.3])
        acc = F.kl_div(opp_action_dist_tets.log(), prey_move_probs, None, None, 'sum')


        actions_probs = torch.matmul(opp_actions_probs,agent_action_probs)[:,0]
       # print('actions_probs', actions_probs.size())
       # print('actions_probs',actions_probs)


        return actions_probs,acc,opp_action_entropy

class Critic(nn.Module):
    def __init__(self, agent_num, state_dim, action_dim,seed):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        input_dim = 1 + state_dim * agent_num +agent_num*action_dim
        #print('input_dim',input_dim)

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        # print('x is',x)
        # print('x_size',x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #print('self.fc3(x)',self.fc3(x))
        return self.fc3(x)


