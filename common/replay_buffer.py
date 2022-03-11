
import torch
import random
class Memory:
    def __init__(self, agent_num, action_dim,seed):
        self.agent_num = agent_num
        #print('self.agent_num',self.agent_num)
        self.action_dim = action_dim
        #print('self.action_dim',self.action_dim)

        self.actions = []
        self.observations = []
        self.pi = [[] for _ in range(agent_num)]
        self.reward = []
        self.done = [[] for _ in range(agent_num)]
        self.seed = random.seed(seed)

    def get(self):
        #print('LLLLLLL')
        actions = torch.tensor(self.actions)
        #print('actions', actions)
        observations = self.observations

        pi = []
        for i in range(self.agent_num):
            #print('len(self.pi[i]',self.pi[i])

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
