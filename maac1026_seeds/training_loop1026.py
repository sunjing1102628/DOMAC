"""
Training loop for the Coma framework on Switch2-v0 the ma_gym
"""

import gym
import numpy as np
from maac1026_seeds.COMA1026 import COMA
import torch

#random.seed(5)
num_seeds = 6
seeds = [i for i in range(num_seeds)]
torch.cuda.is_available()
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
# print('random_seed',random.random())

def moving_average(x, N):
    return np.convolve(x, np.ones((N,)) / N, mode='valid')


if __name__ == "__main__":
    for seed in seeds:
        # Hyperparameters
        agent_num = 2

        state_dim = 28
        action_dim = 5

        gamma = 0.99
        lr_a = 0.0001
        lr_c = 0.005

        target_update_steps = 10

        # agent initialisation

        agents = COMA(agent_num, state_dim, action_dim, lr_c, lr_a, gamma, target_update_steps,seed)

        env = gym.make("PredatorPrey5x5-v0")
        #env.seed(seed=seed)

        obs = env.reset()

        episode_reward = 0
        episodes_reward = []
        lr_training_loop=0.06
        log=[]

        # training loop

        n_episodes = 10000
        episode = 0

        while episode < n_episodes:
            # print('~~~~~')
            actions = agents.get_actions(obs)
            # print('obs0',obs)
            # print('action is',actions)
            next_obs, reward, done_n, _ = env.step(actions)

            agents.memory.reward.append(reward)
            for i in range(agent_num):
                agents.memory.done[i].append(done_n[i])

            episode_reward += sum(reward)
            # print('done',done_n)

            obs = next_obs
            # print('next_obs',obs)

            if all(done_n):
                # if episode % 10 == 0:
                #print('!!!!!')
                episodes_reward.append(episode_reward)
                episode_reward = 0

                episode += 1

                obs = env.reset()

                # print('episode',episode)

                if episode % 10 == 0:
                   # print('aaaa')
                    agents.train()
                    log.append([episode, sum(episodes_reward[-10:]) / 10])

                if episode % 100 == 0:
                    # log.append([episode, sum(episodes_reward[-100:]) / 100])
                    #env.render()
                    print(f"episode: {episode}, average reward: {sum(episodes_reward[-100:]) / 100}")
                np.savetxt('./results/qmix_pp5/train_score_seed_{}.csv'.format(seed), np.array(log),delimiter = ";")
                # np.save('./log/training_log_'
                #         '{}'  # environment parameter
                #         '{}.npy'  # training parameter
                #         .format(args.gamma, lr_maac_opp,
                #                 ),
                #         np.array(log))
        # plt.plot(moving_average(episodes_reward, 100))
        # plt.title('Learning curve')
        # plt.xlabel("Episodes")
        # plt.ylabel("Reward")
        # plt.show()
        #



