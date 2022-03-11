"""
Training loop for the Coma framework on Switch2-v0 the ma_gym
"""

import gym
import numpy as np

from maac_distributional import MADAC


#random.seed(5)
# print('random_seed',random.random())

def moving_average(x, N):
    return np.convolve(x, np.ones((N,)) / N, mode='valid')
#num_seeds = 6
num_seeds = [3,4,5,6,7,8]
seeds = [i for i in num_seeds]
print('seeds',seeds)
if __name__ == "__main__":
    for seed in seeds:
        # Hyperparameters
        agent_num = 2

        state_dim = 28
        action_dim = 5
        num_quant = 2

        gamma = 0.95
        lr_a = 0.0001
        lr_c = 0.005
        lr_madc = 0.03
        log = []

        target_update_steps = 10

        # agent initialisation

        agents = MADAC(agent_num, state_dim, action_dim, num_quant, lr_c, lr_a, gamma, target_update_steps,seed)

        env = gym.make("PredatorPrey5x5-v0")

        obs = env.reset()

        episode_reward = 0
        episodes_reward = []

        # training loop

        n_episodes = 12000
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
                # print('!!!!!')
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
                    # env.render()
                    #log.append(sum(episodes_reward[-100:]) / 100)
                    print(f"episode: {episode}, average reward: {sum(episodes_reward[-100:]) / 100}")
                np.savetxt('./results/final_results03/train_score_seed_{}.csv'.format(seed), np.array(log),
                           delimiter=";")
                # np.save('./log/training_log_'
                #         '{}'  # environment parameter
                #         '{}.npy'  # training parameter
                #         .format(gamma, lr_madc,
                #                 ),
                #         np.array(log))

    # plt.plot(moving_average(episodes_reward, 100))
    # plt.title('Learning curve')
    # plt.xlabel("Episodes")
    # plt.ylabel("Reward")
    # plt.show()
                #

