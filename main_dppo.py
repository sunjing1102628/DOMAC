#from runner_1005 import Runner
from common.arguments import get_args

#import matplotlib.pyplot as plt
import numpy as np
from common.arguments import get_args
import gym
import ma_gym
import matplotlib.pyplot as plt
import numpy as np
from maac1.DPPO import DPPO
from common.replay_buffer import Memory
import torch
from torch.distributions import Categorical

import random

torch.cuda.is_available()
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
# random.seed(5)
# print('random_seed',random.random())
# num_seeds = 10
# seeds = [i for i in range(num_seeds)]
def moving_average(x, N):
    return np.convolve(x, np.ones((N,)) / N, mode='valid')
num_seeds = [3]
seeds = [i for i in num_seeds]
print('seeds',seeds)
def eval(agents, env):
    done = False
    episode_reward = 0
    obs = env.reset()
    while not done:
        observations = torch.tensor(obs).to(device)
        actions = []
        for i in range(agents.agent_num):
            dist = agents.actors[i](observations[i])
            action = Categorical(dist).sample()
            actions.append(action.item())
        obs, reward, done_n, _ = env.step(actions)
        episode_reward += sum(reward)
        done = all(done_n)
        if done:
            return episode_reward


import time
if __name__ == "__main__":
    for seed in seeds:
        args = get_args()
        # # Hyperparameters

        # agent initialisation
        start = time.time()

        agents = DPPO(args,seed)
        memory = Memory(args.n_agents, args.action_dim,seed)
        env = gym.make("PredatorPrey7x7-v0")
        env_eval = gym.make("PredatorPrey7x7-v0")
        eval_times = 10
        #env = gym.make("PredatorPrey7x7-v0")

        obs = env.reset()
        #print('obs',obs)

        episode_reward = 0
        episodes_reward = []
        episodes_entropy = []

        # training loop

        n_episodes = 8000
        episode = 0
        log_mean = []
        log_std = []
        log_dist_entropy = []
        log_dist_entropy_std = []

        validation_return = - np.inf

        while episode < n_episodes:
            # print('~~~~~')
            actions,dist_entropys1 = agents.get_actions(obs)
            dist_entropy = sum(dist_entropys1) / 4
            episodes_entropy.append(dist_entropy.tolist())

            # print('obs0',obs)
            # print('action is',actions)
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
                    log_mean.append([episode, sum(episodes_reward[-100:]) / 100])
                    #print('episodes_reward[-100:]',episodes_reward[-100:])
                    episode_reward_std= np.array(episodes_reward[-100:]).std()
                    log_std.append([episode, episode_reward_std])
                    print('type',type(episodes_reward[-100:]))
                    print('episode_std',episode_reward_std)
                    print(f"episode: {episode}, average reward: {sum(episodes_reward[-100:]) / 100}")
                    # log episode entropy mean and std
                    log_dist_entropy.append([episode, sum(episodes_entropy[-100:]) / 100])
                    episodes_entropy_std = np.array(episodes_entropy[-100:]).std()
                    log_dist_entropy_std.append([episode, episodes_entropy_std])
                    # print(f"episode: {episode}, average reward: {sum(episodes_reward[-100:]) / 100}")
                    # print(f"episode: {episode}, std: {episode_reward_std}")
                    # validate agents performance
                    agents.save_model(episode)
                    current_returns = []
                    for i in range(eval_times):
                        current_validation_return = eval(agents, env_eval)

                        current_returns.append(current_validation_return)
                    current_mean_return = sum(current_returns) / eval_times
                    if current_mean_return > validation_return:
                        print("Found better agents: current return {} vs previous best return {}".format(
                            current_mean_return, validation_return))
                        agents.save_model_best()
                        validation_return = current_mean_return

                np.savetxt('./results/dppo4v2_seed3/train_score_seed_{}.csv'.format(seed), np.array(log_mean),
                           delimiter=";")
                np.savetxt('./results/dppo4v2_seed3/train_score_std_seed_{}.csv'.format(seed), np.array(log_std),
                           delimiter=";")
                np.savetxt('./results/dppo4v2_seed3/train_entropyscore_seed_{}.csv'.format(seed),
                           np.array(log_dist_entropy), delimiter=";")
                np.savetxt('./results/dppo4v2_seed3/train_entropystd_seed_{}.csv'.format(
                    seed), np.array(log_dist_entropy_std), delimiter=";")

                # np.save('./log/training_log_'
                #         '{}'  # environment parameter
                #         '{}.npy'  # training parameter
                #         .format(args.gamma, lr_a1,
                #                 ),
                #         np.array(log))
        # plt.plot(moving_average(episodes_reward, 100))
        # plt.title('Learning curve')
        # plt.xlabel("Episodes")
        # plt.ylabel("Reward")
        # plt.show()
        end = time.time()
        print("Execution time of the program is- ", end - start)
        # import cProfile
        #

    # cProfile.run('Runner(args, env)', filename='restates')
    #             #

