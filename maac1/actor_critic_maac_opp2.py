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
    def __init__(self, state_dim, action_dim,opp_agents,seed=0):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)


        self.fc1 = nn.Linear(state_dim+action_dim*2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        #self.opp_actor=Opp_Actor(state_dim,action_dim)
        #print('opp_agents',opp_agents)
        self.opp_actors = nn.ModuleList([Opp_Actor(state_dim, action_dim) for _ in range(opp_agents)])



    def forward(self, x):
        actions=[]
        opp_actions_probs=[]
        acc = []
        opp_actions_entropy = []


        for opp_actor in self.opp_actors:
           #print('!')
            #print('x.size',x.size())
            opp_action_dist = opp_actor(x)
            opp_action_dist_tets = opp_action_dist.detach()
            # print('opp_actions_dist', opp_action_dist_tets)
            prey_move_probs = torch.tensor([0.175, 0.175, 0.175, 0.175, 0.3])
            #print('opp_actions_dist', opp_action_dist.size())
            #print('111', Categorical(opp_action_dist).sample([10]))
            opp_action = Categorical(opp_action_dist).sample([10]).reshape(len(opp_action_dist),10) #torch.Size([5])
            opp_action_entropy = Categorical(opp_action_dist).entropy()
            opp_actions_entropy.append(opp_action_entropy)
            acc1 = F.kl_div(prey_move_probs.log(), opp_action_dist_tets, None, None, 'sum')
            acc.append(acc1)
            opp_action_prob=torch.gather(opp_action_dist, dim=-1, index=opp_action)
            #print('opp_action_probs0',opp_action_prob)
            actions.append(opp_action)
            opp_actions_probs.append(opp_action_prob)
        #print('opp_action0',opp_actions_probs[0])
        #print('opp_action1',opp_actions_probs[1])
        opp_actions_probs=opp_actions_probs[0]*opp_actions_probs[1]

        #actions=torch.cat(actions,dim=0).reshape(2,10).t()
        actions = torch.cat(actions,dim=0).reshape(len(opp_actions_probs),2,10).transpose(1,2)
        opp_actions= torch.nn.functional.one_hot(actions, 5).view(len(opp_actions_probs),10, 10)
       # print('opp_actions',opp_actions.size())

        #opponent_num = torch.arange(0, 5)
        # print('opp_num',opponent_num)

        #opponent_action_num = F.one_hot(opponent_num, num_classes=len(opponent_num))
        # print('opponent_action_num',opponent_action_num)
        # print('x is',x)
        #print('len(x)',x.repeat(1, 10).reshape(10,len(x)))
        '''a = torch.cat([x.repeat(1, 10).reshape(10,len(x)),
                       opp_actions.to(x.device)],
                      dim=1)'''
        a = torch.cat([x.repeat(1, 10).reshape(len(x), 10, 28),
                       opp_actions.to(x.device)],
                      dim=2).squeeze(1)
        #print('a is',a)
        a = F.relu(self.fc1(a))

        a = F.relu(self.fc2(a))
        agent_action_probs= F.softmax(self.fc3(a), dim=-1)
        #print('agent_action_prob',agent_action_probs)
        #opp_actions_probs = self.opp_actor(x)
        # print('opp',opp_actions_probs)



        actions_probs = torch.matmul(opp_actions_probs,agent_action_probs)[:,0]
       # print('ac',actions_probs)


        return actions_probs, sum(acc)/2, sum(opp_actions_entropy)/2


class Critic(nn.Module):
    def __init__(self, agent_num, state_dim, action_dim,seed):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        input_dim = 1+ state_dim * agent_num+action_dim*agent_num
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


