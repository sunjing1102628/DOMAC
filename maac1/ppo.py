import torch
import os
from maac1.actor_critic_categorical import Actor, Critic
#from maac1.actor_critic_mdac1 import Actor, Critic
import numpy as np
import torch.nn as nn
import random
from common.replay_buffer_ppo import Memory
random.seed(5)
# print(random.random())
#from .distribution import BaseAgent
import torch.nn.functional as F
from torch.distributions import Categorical
torch.cuda.is_available()
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 3
class PPO:
    def __init__(self, args,seed):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args=args
        self.agent_num = args.n_agents
        # print('self.agent_num',self.agent_num)
        self.state_dim = args.state_dim
        # print('self.state_dim',self.state_dim)
        self.action_dim = args.action_dim
        # print('self.action_dim',self.action_dim)

        self.gamma = args.gamma
        self.seed = seed
        random.seed(seed)
        # print('self.gamma',self.gamma)

        self.target_update_steps = args.target_update_interval
        # print('self.target_update_steps',self.target_update_steps)


        self.memory = Memory(args.n_agents, args.action_dim,seed)

        self.actors = [Actor(args.state_dim, args.action_dim,seed).to(device) for _ in range(self.agent_num)]
        # print('self.actors',self.actors)
        self.critic = Critic(args.n_agents,  args.state_dim, args.action_dim,seed).to(device)
        # print('self.critic',self.critic)

        self.critic_target = Critic(args.n_agents,  args.state_dim, args.action_dim,seed).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # print('self.critic.state_dict()',self.critic.state_dict())
        # print('111 is',self.critic_target.load_state_dict(self.critic.state_dict()))

        self.actors_optimizer = [torch.optim.Adam(self.actors[i].parameters(), lr=args.lr_actor) for i in range(args.n_agents)]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr_critic)

        self.count = 0

       # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        # path to save the model
        '''self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        for agent_id in range(self.agent_num):
            self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
            else:
                if self.args.load_existing:
                    #self.actor_network.load_state_dict(torch.load(self.model_path + '/' + str(self.args.load_network_index) + '_actor_params.pkl'))
                    #self.critic_target.load_state_dict(self.critic.state_dict())
                    self.critic_network.load_state_dict(torch.load(self.model_path + '/' + str(self.args.load_network_index) + '_critic_params.pkl'))
                    print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,self.model_path + '/' + str(self.args.load_network_index) + '_actor_params.pkl'))
                    print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,self.model_path + '/' + str(self.args.load_network_index) + '_critic_params.pkl'))
                    #self.actor_target_network.load_state_dict(self.actor_network.state_dict())
                    self.critic_target_network.load_state_dict(self.critic_network.state_dict())'''

                # 加载模型
        '''if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))'''

    # soft update

    def _soft_update_target_network(self):
        # for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
        #     target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    def get_actions(self, observations):
        observations = torch.tensor(observations).to(device)

        actions = []
        dist_entropys = []



        for i in range(self.agent_num):
            dist = self.actors[i](observations[i])
            #print('dist',dist)
            action = Categorical(dist).sample()
            dist_entropy = Categorical(dist).entropy()

            #pi_action = dist[action.item()].item()
            #print('@@@@@',pi_action)

            #print('action is',action)

            self.memory.pi[i].append(dist)

            actions.append(action.item())
            dist_entropys.append(dist_entropy)

        self.memory.observations.append(observations)
        self.memory.actions.append(actions)
        #self.memory.pi_a.append(pi_a)

        return actions,dist_entropys
    def train(self):
        # print('########')
        actor_optimizer = self.actors_optimizer
        critic_optimizer = self.critic_optimizer

        actions, observations, pi,  reward, done = self.memory.get()
        # print('obs_train',observations)

        for i in range(self.agent_num):
            for j in range(K_epoch):
                input_critic = self.build_input_critic(i, observations, actions).to(device)
                Q_target = self.critic_target(input_critic).detach()
                action_taken = actions.type(torch.long)[:, i].reshape(-1, 1)

                # baseline = torch.sum(pi[i] * Q_target.to(device), dim=1).detach()

                Q_taken_target = torch.gather(Q_target.to(device), dim=1, index=action_taken.to(device)).squeeze()

                log_pi = torch.log(
                    torch.gather(pi[i].to(device), dim=1, index=action_taken.to(device)).squeeze()).detach()
                # train critic

                Q = self.critic(input_critic)
                # print('Q is',Q.size())

                action_taken = actions.type(torch.long)[:, i].reshape(-1, 1)
                # print('action_taken',action_taken.size())
                Q_taken = torch.gather(Q.to(device), dim=1, index=action_taken.to(device)).squeeze()
                # TD(0)
                r = torch.zeros(len(reward[:, i])).to(device)
                # print('r is',r)

                for t in range(len(reward[:, i])):
                    # print('done',done[i][t])
                    if done[i][t]:
                        # print('reward[:, i][t]',reward[:, i][t])
                        r[t] = reward[:, i][t]
                        # print('r[t]',r[t])
                    else:
                        # print('Q_taken_target[t + 1]',Q_taken_target[t + 1])
                        r[t] = reward[:, i][t] + self.gamma * Q_taken_target[t + 1]
                with torch.no_grad():
                    adv = r - Q_taken
                    # print('adv',adv.size())
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # train actor

                # calculate pi_new
                batch_size = len(observations)
                observations1 = torch.cat(observations).view(batch_size, self.agent_num, self.state_dim).to(
                        device)  # torch.Size([647, 56])
                # print('observations[:,i]',observations[:,i].size())
                pi_new = self.actors[i](observations1[:, i])

                pi_new_a = torch.gather(pi_new.to(device), dim=1, index=action_taken.to(device)).squeeze()

                ratio = torch.exp(torch.log(pi_new_a) - log_pi)  # a/b == exp(log(a)-log(b))

                # print('ratio is',ratio)
                surr1 = ratio * adv.squeeze(-1)
                surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv.squeeze(-1)
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_optimizer[i].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 5)
                actor_optimizer[i].step()

                # train critic
                critic_loss = torch.mean((r - Q_taken) ** 2).to(device)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
                critic_optimizer.step()

        if self.count == self.target_update_steps:
            # print('+++++++')
            # print('self.critic.state_dict()',self.critic.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            # print('self.critic_target',self.critic_target)
            self.count = 0
        else:
            self.count += 1

        self.memory.clear()

    def build_input_critic(self, agent_id, observations,actions):

        batch_size = len(observations)


        ids = (torch.ones(batch_size) * agent_id).view(-1, 1).to(device)

        action = torch.nn.functional.one_hot(actions, 5)

        actions = action.view(batch_size, self.action_dim * self.agent_num).to(device)


        observations = torch.cat(observations).view(batch_size, self.state_dim * self.agent_num).to(device)
        input_critic = torch.cat([observations.type(torch.float32).to(device), actions.type(torch.float32).to(device)],
                                 dim=-1)
        input_critic = torch.cat([ids.to(device), input_critic.to(device)], dim=-1)


        return input_critic

    def save_model_best(self):
        # save actors
        for agent_id, actor_net in enumerate(self.actors):
            model_path = os.path.join(self.args.save_dir7b, self.args.algorithm_name1)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_path = os.path.join(model_path, 'agent_%d' % agent_id)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(actor_net.state_dict(), model_path + '/' + 'actor_params.pkl')
            # save shared critic
            torch.save(self.critic.state_dict(), model_path + '/' + 'critic_params.pkl')

    def save_model(self, train_step):  # old save fn
        num = str(train_step // self.args.save_rate)
        for agent_id, actor_net in enumerate(self.actors):
            model_path = os.path.join(self.args.save_dir7a, self.args.algorithm_name1)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_path = os.path.join(model_path, 'agent_%d' % agent_id)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(actor_net.state_dict(), model_path + '/' + num + 'actor_params.pkl')
            # save shared critic
            torch.save(self.critic.state_dict(), model_path + '/' + num + 'critic_params.pkl')
