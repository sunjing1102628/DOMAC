# File name: model.py
# Author: Zachry
# Time: 2019-12-04
# Description: The enter of this project
import sys
import time
import gym
import ma_gym
import torch 
import numpy as np
#from smac.env import StarCraft2Env
import random
from maac1.QR_QMIX import QR_QMIX_Agent
from common.arguments_qmix import parse_args
from env.multiagentenv import MultiAgentEnv
seed =3
def train(env, args,seed):
    """ step1: init the env and par """
    env_info = env.get_env_info()

    #num_agents = env_info["n_agents"]
    num_agents = args.n_agents
    random.seed(seed)


    shape_obs = env_info['obs_shape'] + num_agents # add agent_idx bits
    shape_state = env_info['state_shape']
    num_actions_set = [env_info["n_actions"]]
    obs_0_idx = np.eye(num_agents)

    """ step2: init the QMIX agent """
    qmix_agent = QR_QMIX_Agent(shape_obs, shape_state, num_agents, num_actions_set, args,seed)
    qmix_agent.init_trainers(args)

    """ step3: interact with the env and learn """
    step_cnt = 0
    done_cnt = 0
    log_mean=[]
    log_std =[]
    for epi_cnt in range(args.max_episode):
        # evaluation: for check the progress of the model
        if epi_cnt > 0 and epi_cnt % args.fre_epi4evaluation == 0: 
            env.reset()
            mean_reward, episode_reward_std =evaluation(env, args, qmix_agent)
            log_mean.append([epi_cnt, mean_reward])
            log_std.append([epi_cnt,episode_reward_std])

        # init the episode data
        env.reset()
        episode_reward = 0
        #actions_last = env.last_action
        qmix_agent.memory.create_new_episode()
        hidden_last = np.zeros((num_agents, args.q_net_hidden_size))

        for epi_step_cnt in range(args.per_episode_max_len):
            step_cnt += 1 # update the cnt every time

            # get obs state for select action
            state = env.get_state()
            obs = np.concatenate([obs_0_idx, np.array(env.get_obs())], axis=1)
            avail_actions = np.array(env.get_avail_actions())
            #avail_actions = env.get_avail_actions()

            # interact with the env and get new state obs
            #actions, hidden = qmix_agent.select_actions(avail_actions, obs, actions_last, hidden_last, args)
            actions, hidden = qmix_agent.select_actions(avail_actions, obs, hidden_last, args)

            reward, done, _ = env.step(actions)
            reward = reward*args.reward_scale_par # normalize the reward
            if epi_step_cnt == args.per_episode_max_len-1: done = True # max len of episode
            state_new = env.get_state()
            obs_new = np.concatenate([obs_0_idx, np.array(env.get_obs())], axis=1)
            avail_actions_new = np.array(env.get_avail_actions())
            #actions_now_onehot = env.last_action # the env do the things for us

            # update the date and save experience to memory
            if done == True: done_cnt += 1

            # concatenate the obs and actions_last for speed up the train
            qmix_agent.save_memory(np.concatenate([obs], axis=-1), state, \
                actions.reshape(1, -1), avail_actions_new, np.concatenate([obs_new], axis=-1), \
                    state_new, reward, done)
            #actions_last = env.last_action
            hidden_last = hidden

            # agents learn
            loss = qmix_agent.learn(step_cnt, epi_cnt, args)
            print(' '*80, 'loss is', loss, end='\r')

            # if done, end the episode
            episode_reward += reward
            if done: break

        if epi_cnt % args.print_fre == 0:
            print("episode_cnt:{} episode_len:{} epsilon: {} reward in episode {} ".format( \
                epi_cnt, epi_step_cnt, round(qmix_agent.epsilon, 3), round(episode_reward, 3), \
                ))
        np.savetxt('./results/qmix_pp5/train_score_seed_{}.csv'.format(3), np.array(log_mean),
                   delimiter=";")
        np.savetxt('./results/qmix_pp5/train_score_std_seed_{}.csv'.format(3), np.array(log_std),
                   delimiter=";")

    """ Note: close the env """
    env.close()

def evaluation(env, args, qmix_agent):
    """ step1: init the env and par """
    env_info = env.get_env_info()
    num_agents = env_info["n_agents"]
    shape_obs = env_info['obs_shape'] + num_agents # first bit is agent_idx
    shape_state = env_info['state_shape']
    num_actions_set = [env_info["n_actions"]]
    obs_0_idx = np.eye(num_agents)
    rewards_list = []
    with torch.no_grad():
        for _ in range(args.num_epi4evaluation):
            env.reset()
            episode_reward = 0
            #actions_last = env.last_action
            hidden_last = np.zeros((num_agents, args.q_net_hidden_size))
            for epi_step_cnt in range(args.per_episode_max_len):
                # get obs state for select action
                state = env.get_state()
                obs = np.concatenate([obs_0_idx, np.array(env.get_obs())], axis=1)
                avail_actions = np.array(env.get_avail_actions())

                # interact with the env and get new state obs
                actions, hidden = qmix_agent.select_actions(avail_actions, obs,  hidden_last, args, eval_flag=True)
                reward, done, _ = env.step(actions)
                reward = reward*args.reward_scale_par # normalize the reward
                if epi_step_cnt == args.per_episode_max_len-1: done = True # max len of episode

                #actions_last = env.last_action
                hidden_last = hidden

                # if done, end the episode
                episode_reward += reward
                if done: break

            # record the reward for final evaluation
            rewards_list.append(episode_reward)
        episode_reward_std = np.array(rewards_list[-100:]).std()
        #print('The evaluation mean(all/{}) reward is'.format(args.num_epi4evaluation), round(sum(rewards_list)/rewards_list.__len__(), 3))
        return round(sum(rewards_list)/rewards_list.__len__(), 3), episode_reward_std
if __name__ == '__main__':
    args = parse_args()
    #env = StarCraft2Env(map_name=args.map_name, difficulty=args.difficulty)
    env1 =  gym.make("PredatorPrey5x5-v0")
    env= MultiAgentEnv(env1, args)
    #print('env',env)
    """ run the main """
    train(env, args,seed)
