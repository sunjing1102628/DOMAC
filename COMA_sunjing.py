import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import ma_gym
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Categorical
import random

random.seed(5)
print(random.random())

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
            pi.append(torch.cat(self.pi[i]).view(len(self.pi[i]), self.action_dim))

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

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        print('x is',x)
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
        #print('self.fc3(x)',self.fc3(x))
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

        self.actors_optimizer = [torch.optim.Adam(self.actors[i].parameters(), lr=lr_a) for i in range(agent_num)]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

        self.count = 0

    def get_actions(self, observations):
        observations = torch.tensor(observations)

        actions = []

        for i in range(self.agent_num):
            dist = self.actors[i](observations[i])
            #print('dist',dist)
            action = Categorical(dist).sample()
            #print('action1',action)

            self.memory.pi[i].append(dist)
            actions.append(action.item())
        #print('actions',actions)

        self.memory.observations.append(observations)
        self.memory.actions.append(actions)

        return actions

    def train(self):
        actor_optimizer = self.actors_optimizer
        critic_optimizer = self.critic_optimizer

        actions, observations, pi, reward, done = self.memory.get()
        # print('obs',len(observations))
        # print('actions_memory',actions)
        # print('observations', observations)
        # print('pi is',pi)
        # print('pi is',pi[0].size())


        for i in range(self.agent_num):
            # train actor


            input_critic = self.build_input_critic(i, observations, actions)
            print('onput)critic',input_critic)
            Q_target = self.critic_target(input_critic).detach()
            print('Q_target is',Q_target)

            action_taken = actions.type(torch.long)[:, i].reshape(-1, 1)
            #print('action_taken', action_taken)

            baseline = torch.sum(pi[i] * Q_target, dim=1).detach()
            #print('baseline',baseline)
            Q_taken_target = torch.gather(Q_target, dim=1, index=action_taken).squeeze()
            #print('Q_taken_target',Q_taken_target)
            advantage = Q_taken_target - baseline

            log_pi = torch.log(torch.gather(pi[i], dim=1, index=action_taken).squeeze())

            actor_loss = - torch.mean(advantage * log_pi)

            actor_optimizer[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 5)
            actor_optimizer[i].step()

            # train critic

            Q = self.critic(input_critic)
            # print('Q is',Q.size())

            action_taken = actions.type(torch.long)[:, i].reshape(-1, 1)
            # print('action_taken',action_taken.size())
            Q_taken = torch.gather(Q, dim=1, index=action_taken).squeeze()
            # print('Q_taken',Q_taken.size())

            # TD(0)
            r = torch.zeros(len(reward[:, i]))
            for t in range(len(reward[:, i])):
                if done[i][t]:
                    r[t] = reward[:, i][t]
                else:
                    r[t] = reward[:, i][t] + self.gamma * Q_taken_target[t + 1]

            critic_loss = torch.mean((r - Q_taken) ** 2)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
            critic_optimizer.step()

        if self.count == self.target_update_steps:
            self.critic_target.load_state_dict(agents.critic.state_dict())
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




import matplotlib.pyplot as plt
import numpy as np
def moving_average(x, N):
    return np.convolve(x, np.ones((N,)) / N, mode='valid')
agent_num = 2

state_dim = 2
action_dim = 5

gamma = 0.99
lr_a = 0.0001
lr_c = 0.005

target_update_steps = 10

# agent initialisation

agents = COMA(agent_num, state_dim, action_dim, lr_c, lr_a, gamma, target_update_steps)

env = gym.make("Switch2-v0")

obs = env.reset()

episode_reward = 0
episodes_reward = []

# training loop

n_episodes = 10000
episode = 0

while episode < n_episodes:
    actions = agents.get_actions(obs)
    next_obs, reward, done_n, _ = env.step(actions)

    agents.memory.reward.append(reward)
    for i in range(agent_num):
        agents.memory.done[i].append(done_n[i])

    episode_reward += sum(reward)

    obs = next_obs

    if all(done_n):

        episodes_reward.append(episode_reward)
        #print('episodes_reward',episodes_reward)
        episode_reward = 0

        episode += 1

        obs = env.reset()

        if episode % 10 == 0:
            agents.train()

        if episode % 100 == 0:
            print(f"episode: {episode}, average reward: {sum(episodes_reward[-100:]) / 100}")
plt.plot(moving_average(episodes_reward, 100))
plt.title('Learning curve')
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()
import gym
# import ma_gym
# from ma_gym.wrappers import Monitor
import matplotlib.pyplot as plt
import glob
import io
import base64
# from IPython.display import HTML
# from IPython import display as ipythondisplay

#from pyvirtualdisplay import Display
# display = Display(visible=0, size=(1400, 900))
# display.start()

"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""

# def show_video():
#   mp4list = glob.glob('video/*.mp4')
#   if len(mp4list) > 0:
#     mp4 = mp4list[0]
#     video = io.open(mp4, 'r+b').read()
#     encoded = base64.b64encode(video)
#     ipythondisplay.display(HTML(data='''<video alt="test" autoplay
#                 loop controls style="height: 400px;">
#                 <source src="data:video/mp4;base64,{0}" type="video/mp4" />
#              </video>'''.format(encoded.decode('ascii'))))
#   else:
#     print("Could not find video")
# def wrap_env(env):
#   env = Monitor(env, './video', force=True)
#   return env
# env = wrap_env(gym.make("PredatorPrey5x5-v0"))
# obs_n = env.reset()
# reward = 0
#
# while True:
#   obs_n, reward_n, done_n, info = env.step(agents.get_actions(obs_n))
#
#   reward += sum(reward_n)
#
#   env.render()
#
#   if all(done_n):
#     agents.memory.clear()
#     break
#
# env.close()
#
# print(reward)



