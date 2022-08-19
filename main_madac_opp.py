#from runner_1005 import Runner
from common.arguments import get_args

import matplotlib.pyplot as plt
import numpy as np
from common.arguments import get_args
import gym
import ma_gym
import matplotlib.pyplot as plt
import numpy as np
from maac1.madac1_opp import MADAC_OPP
from common.replay_buffer import Memory
import torch
import random


# print('random_seed',random.random())
torch.cuda.is_available()
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
def moving_average(x, N):
    return np.convolve(x, np.ones((N,)) / N, mode='valid')
# num_seeds = 6
# seeds = [i for i in range(num_seeds)]
num_seeds = [2,3,4,5,6,8,10,12]
seeds = [i for i in num_seeds]
print('seeds',seeds)
import time
if __name__ == "__main__":
    for seed in seeds:
        args = get_args()

        # agent initialisation
        start = time.time()

        agents = MADAC_OPP(args,seed)

        memory = Memory(args.n_agents, args.action_dim,seed)
        #env = gym.make("PredatorPrey5x5-v0")
        env = gym.make("PredatorPrey7x7-v0")
        opp_agents = args.opp_agents
       # print('opp_agent', args.opp_agents)
        opp_sample_num = args.opp_sample_num
      #  print('self.opp_sample_num', opp_sample_num)

        obs = env.reset()

        episode_reward = 0
        episodes_reward = []

        # training loop
        lr_madac_opp = 0.04

        n_episodes = 15000
        episode = 0
        acc_test = []
        log = []
        log_acc_mean = []

        while episode < n_episodes:
            # print('~~~~~')
            actions,acc = agents.get_actions(obs)
            # print('obs0',obs)
            acc_test.append(sum(acc) / 4)
            next_obs, reward, done_n, _ = env.step(actions)

            agents.memory.reward.append(reward)
            for i in range(args.n_agents):
                agents.memory.done[i].append(done_n[i])

            episode_reward += sum(reward)
            # print('done',done_n)

            obs = next_obs
            # print('next_obs',obs)

            if all(done_n):
                # if episode % 10 == 0:
                # print('!!!!!')
                episodes_reward.append(episode_reward)
                episode_reward = 0

                episode += 1

                obs = env.reset()

                # print('episode',episode)

                if episode % 10 == 0:
                    # print('aaaa')
                    agents.train()
                    #log.append([episode, sum(episodes_reward[-10:]) / 10])

                if episode % 100 == 0:
                    log.append([episode, sum(episodes_reward[-100:]) / 100])
                    # log acc mean and std
                    acc_std = np.array(acc_test[-100:]).std()
                    log_acc_mean.append([episode, sum(acc_test[-100:]) / 100])
                    print(f"episode: {episode}, average reward: {sum(episodes_reward[-100:]) / 100}")
                np.savetxt('./results/DOMAC_PP7_maskobs/train_score_seed_{}.csv'.format(seed), np.array(log),
                           delimiter=";")
                np.savetxt('./results/DOMAC_PP7_maskobs/train_scoreacc_seed_{}.csv'.format(seed), np.array(log_acc_mean),
                           delimiter=";")

                # np.save('./log/training_log_'
                #     '{}'  # environment parameter
                #     '{}.npy'  # training parameter
                #     .format(args.gamma, lr_madac_opp,
                #             ),
                #     np.array(log))

        end = time.time()
        print("Execution time of the program is- ", end - start)

    # import cProfile
    #
    # cProfile.run('Runner(args, env)', filename='restates')
    #             #

