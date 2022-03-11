import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
import random

random.seed(5)
print('random_seed',random.random())

class Memory:
    def __init__(self, agent_num, action_dim):
        self.agent_num = agent_num
        # print('self.agent_num',self.agent_num)
        self.action_dim = action_dim
        # print('self.action_dim',self.action_dim)

        self.actions = []
        self.observations = []
        self.pi = [[] for _ in range(agent_num)]
        self.reward = []
        self.done = [[] for _ in range(agent_num)]

    def get(self):
        actions = torch.tensor(self.actions)
        observations = self.observations

        pi = []
        for i in range(self.agent_num):
            # print('pi 0 is',self.pi[i])
            # print('len(self.pi[i])',len(self.pi[i]))
            # print('torch.cat(self.pi[i])',len(torch.cat(self.pi[i])))
            pi.append(torch.cat(self.pi[i]).view(len(self.pi[i]), self.action_dim))

        # print('pi is',pi)
        reward = torch.tensor(self.reward)
        done = self.done

        return actions, observations, pi, reward, done

    def clear(self):
        self.actions = []
        self.observations = []
        self.pi = [[] for _ in range(self.agent_num)]
        self.reward = []
        self.done = [[] for _ in range(self.agent_num)]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # print('state_dim', state_dim)
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #print('F.softmax(self.fc3(x), dim=-1)',F.softmax(self.fc3(x), dim=-1))
        return F.softmax(self.fc3(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, agent_num, state_dim, action_dim):
        super(Critic, self).__init__()

        input_dim = 1 + state_dim * agent_num + agent_num
        #print('input_dim',input_dim)

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        print('self.fc3(x)',self.fc3(x))
        return self.fc3(x)


class COMA:
    def __init__(self, agent_num, state_dim, action_dim, lr_c, lr_a, gamma, target_update_steps):
        self.agent_num = agent_num
       # print('self.agent_num',self.agent_num)
        self.state_dim = state_dim
        #print('self.state_dim',self.state_dim)
        self.action_dim = action_dim
        #print('self.action_dim',self.action_dim)

        self.gamma = gamma
        #print('self.gamma',self.gamma)

        self.target_update_steps = target_update_steps
        #print('self.target_update_steps',self.target_update_steps)

        self.memory = Memory(agent_num, action_dim)


        self.actors = [Actor(state_dim, action_dim) for _ in range(agent_num)]
        #print('self.actors',self.actors)
        self.critic = Critic(agent_num, state_dim, action_dim)
        #print('self.critic',self.critic)

        self.critic_target = Critic(agent_num, state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # print('self.critic.state_dict()',self.critic.state_dict())
        # print('111 is',self.critic_target.load_state_dict(self.critic.state_dict()))

        self.actors_optimizer = [torch.optim.Adam(self.actors[i].parameters(), lr=lr_a) for i in range(agent_num)]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

        self.count = 0

    def get_actions(self, observations):
        #print('obs1',observations)
        observations = torch.tensor(observations)
        #print('obs2',observations)

        actions = []

        for i in range(self.agent_num):
            dist = self.actors[i](observations[i])
            #print('dist',dist)
            action = Categorical(dist).sample()
            #print('action is',action)

            self.memory.pi[i].append(dist)

            actions.append(action.item())

        self.memory.observations.append(observations)
        self.memory.actions.append(actions)

        return actions

    def train(self):
        #print('########')
        actor_optimizer = self.actors_optimizer
        critic_optimizer = self.critic_optimizer

        actions, observations, pi, reward, done = self.memory.get()
        #print('obs_train',observations)
        # print('actions_memory',actions)



        for i in range(self.agent_num):
            # train actor


            input_critic = self.build_input_critic(i, observations, actions)
            Q_target = self.critic_target(input_critic).detach()
            #print('Q_target is',Q_target)

            action_taken = actions.type(torch.long)[:, i].reshape(-1, 1)

            # print('pi[i]',pi[i])

            baseline = torch.sum(pi[i] * Q_target, dim=1).detach()
            #print('baseline',baseline)
            Q_taken_target = torch.gather(Q_target, dim=1, index=action_taken).squeeze()
            #print('Q_taken_target',Q_taken_target)
            advantage = Q_taken_target - baseline
            print('adv',advantage)

            log_pi = torch.log(torch.gather(pi[i], dim=1, index=action_taken).squeeze())

            actor_loss = - torch.mean(advantage * log_pi)

            actor_optimizer[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 5)
            actor_optimizer[i].step()

            # train critic

            Q = self.critic(input_critic)
            #print('Q is',Q)

            action_taken = actions.type(torch.long)[:, i].reshape(-1, 1)
            # print('action_taken',action_taken.size())
            Q_taken = torch.gather(Q, dim=1, index=action_taken).squeeze()
            # print('Q_taken',Q_taken.size())

            # TD(0)
            r = torch.zeros(len(reward[:, i]))
           # print('r is',r)

            for t in range(len(reward[:, i])):
                #print('done',done[i][t])
                if done[i][t]:
                    #print('reward[:, i][t]',reward[:, i][t])
                    r[t] = reward[:, i][t]
                    #print('r[t]',r[t])
                else:
                    #print('Q_taken_target[t + 1]',Q_taken_target[t + 1])
                    r[t] = reward[:, i][t] + self.gamma * Q_taken_target[t + 1]

            #print('r is',r)
            critic_loss = torch.mean((r - Q_taken) ** 2)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
            critic_optimizer.step()
            # print('actor_loss',actor_loss)
            # print('critic_loss',critic_loss)

        if self.count == self.target_update_steps:
            # print('+++++++')
            #print('self.critic.state_dict()',self.critic.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            # print('self.critic_target',self.critic_target)
            self.count = 0
        else:
            self.count += 1

        self.memory.clear()

    def build_input_critic(self, agent_id, observations, actions):
        batch_size = len(observations)

        ids = (torch.ones(batch_size) * agent_id).view(-1, 1)

        observations = torch.cat(observations).view(batch_size, self.state_dim * self.agent_num)
        input_critic = torch.cat([observations.type(torch.float32), actions.type(torch.float32)], dim=-1)
        input_critic = torch.cat([ids, input_critic], dim=-1)

        return input_critic
