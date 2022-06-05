from gym.spaces import flatdim

import numpy as np
class MultiAgentEnv(object):

    def __init__(self, env, args, **kwargs):
        self.n_agents = args.n_agents
        self._obs = None
        self.env= env
        self.longest_action_space = max(self.env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self.env.observation_space, key=lambda x: x.shape
        )

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        self._obs, reward, done, info = self.env.step(actions)
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]

        return float(sum(reward)), all(done), {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        #print('flatdim(self.longest_observation_space)', flatdim(self.longest_observation_space))
        return flatdim(self.longest_observation_space)

    def get_state(self):

        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        self.n_agents * flatdim(self.longest_observation_space)
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self.env.action_space[agent_id]) * [1]  # [1, 1, 1, 1, 1, 1]

        invalid = [0] * (self.longest_action_space.n - len(valid))  # []

        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    '''def reset(self):
        """ Returns initial observations and states"""
        raise NotImplementedError'''
    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self.env.reset()
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        return self.get_obs(), self.get_state()

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents}
        return env_info
